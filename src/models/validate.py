"""Quality gate: is the selected model safe to serve?   uv run python -m src.models.validate [--strict]

Runs after `select` (dvc stage `validate`) on models/model.joblib and checks it against the
thresholds in params.yaml → validate. Writes:

    reports/validation.json   every check with value, threshold and pass/fail (DVC metric)
    reports/model_card.md     human-readable summary: what the model is, how good, where it fails

Checks (model-risk view, not just "is accuracy high"):
    performance   top-3 on 5-skill inputs, margin over plain profile matching, 2019 drift floor
    calibration   expected calibration error of the top-1 confidence; the app SHOWS this %
    slices        worst role family and worst single role (top-3 accuracy)
    behaviour     hand-written obvious cases (configs/behaviour_cases.yaml) land in the top 3
    invariance    skill order, duplicated skills and unknown skills must not change the output
    robustness    empty skills / missing experience still give a valid probability distribution
    operational   model file size and single-request latency

register.py refuses to promote a model whose report says passed = false.
--strict exits with code 1 when the gate fails (for CI); the DVC stage records the report either way.
"""

from __future__ import annotations

import operator
import sys
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from src.features.build import EXPERIENCE, SKILLS
from src.models.evaluate import inputs, read_split
from src.utils import CONFIGS, MODELS, REPORTS, get_logger, load_params, load_yaml, read_json, write_json

log = get_logger("validate")

# (value name, params.yaml → validate key, rule). A check passes when `value <rule> threshold`.
THRESHOLD_CHECKS = [
    ("top3_5skills", "min_top3_5skills", ">="),
    ("margin_over_profile_matching", "min_margin_over_profile", ">="),
    ("ref2019_top3_5skills", "min_ref2019_top3_5skills", ">="),
    ("ece_top1_5skills", "max_ece_top1", "<="),
    ("min_family_top3_5skills", "min_family_top3", ">="),
    ("min_role_top3_5skills", "min_role_top3", ">="),
    ("behaviour_pass_rate", "min_behaviour_pass_rate", ">="),
    ("model_size_mb", "max_model_size_mb", "<="),
    ("p95_latency_ms", "max_p95_latency_ms", "<="),
]
# Must simply be true.
BOOLEAN_CHECKS = [
    "invariant_to_skill_order",
    "invariant_to_duplicate_skills",
    "ignores_unknown_skills",
    "robust_inputs",
]
OPS = {">=": operator.ge, "<=": operator.le}


# ── individual measurements ──────────────────────────────────────────────────


def ece_top1(proba: np.ndarray, classes: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error of the top-1 confidence: |accuracy - confidence|, averaged over bins."""
    conf = proba.max(axis=1)
    correct = classes[proba.argmax(axis=1)] == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def top3_hits(proba: np.ndarray, classes: np.ndarray, y: np.ndarray) -> np.ndarray:
    top3 = classes[np.argsort(-proba, axis=1)[:, :3]]
    return (top3 == y[:, None]).any(axis=1)


def slice_scores(hits: np.ndarray, groups: pd.Series) -> pd.Series:
    """Top-3 accuracy per group, worst first."""
    return pd.Series(hits, index=groups.to_numpy()).groupby(level=0).mean().sort_values()


def p95_latency_ms(pipe, X: pd.DataFrame, n: int) -> float:
    """95th-percentile time for ONE request (a single-row predict_proba), like the API makes."""
    rows = [X.iloc[[i % len(X)]] for i in range(n)]
    pipe.predict_proba(rows[0])  # warm-up
    times = []
    for row in rows:
        t = time.perf_counter()
        pipe.predict_proba(row)
        times.append(time.perf_counter() - t)
    return round(float(np.percentile(times, 95)) * 1000, 2)


def model_vocabulary(pipe) -> set[str]:
    return set(pipe.named_steps["features"].named_transformers_["skills"].vocabulary_)


def behaviour(pipe, cases: list[dict]) -> tuple[float, list[dict]]:
    """Share of hand-written cases whose expected role is in the top 3, plus the failures."""
    vocab, classes = model_vocabulary(pipe), pipe.classes_
    unknown = sorted({s for c in cases for s in c["skills"] if s not in vocab})
    if unknown:
        raise ValueError(f"behaviour cases use skills the model doesn't know: {unknown}")
    missing_roles = sorted({c["expected"] for c in cases} - set(classes))
    if missing_roles:
        raise ValueError(f"behaviour cases expect roles the model can't predict: {missing_roles}")
    X = pd.DataFrame({SKILLS: [c["skills"] for c in cases], EXPERIENCE: [c["experience"] for c in cases]})
    proba = pipe.predict_proba(X)
    failures = []
    for case, row in zip(cases, proba, strict=True):
        top3 = classes[np.argsort(-row)[:3]].tolist()
        if case["expected"] not in top3:
            failures.append({"expected": case["expected"], "skills": case["skills"], "got_top3": top3})
    return 1 - len(failures) / len(cases), failures


def invariance(pipe, X: pd.DataFrame) -> dict[str, bool]:
    """Things that must NOT change the answer. Trivially true for a bag-of-skills model today;
    these guard against a future feature change (e.g. sequence models, TF-IDF) silently breaking it."""
    base = pipe.predict_proba(X)

    def same(transform) -> bool:
        changed = X.assign(**{SKILLS: X[SKILLS].map(transform)})
        return bool(np.allclose(pipe.predict_proba(changed), base, atol=1e-9))

    return {
        "invariant_to_skill_order": same(lambda s: list(reversed(s))),
        "invariant_to_duplicate_skills": same(lambda s: list(s) + list(s[:2])),
        "ignores_unknown_skills": same(lambda s: list(s) + ["zz-not-a-real-skill-1", "zz-not-a-real-skill-2"]),
    }


def robust_inputs(pipe) -> bool:
    """No skills the model knows and/or no experience → still a valid probability distribution."""
    X = pd.DataFrame({SKILLS: [[], ["zz-unknown"], ["zz-unknown"]], EXPERIENCE: [np.nan, 3.0, np.nan]})
    proba = pipe.predict_proba(X)
    return bool(np.isfinite(proba).all() and np.allclose(proba.sum(axis=1), 1))


# ── the gate ─────────────────────────────────────────────────────────────────


def gate(values: dict, cfg: dict) -> dict[str, dict]:
    """Compare every measured value with its threshold. Pure function (unit-tested)."""
    checks = {}
    for name, key, rule in THRESHOLD_CHECKS:
        v, t = values[name], cfg[key]
        checks[name] = {"value": v, "rule": rule, "threshold": t, "passed": bool(OPS[rule](v, t))}
    for name in BOOLEAN_CHECKS:
        checks[name] = {"value": bool(values[name]), "rule": "==", "threshold": True, "passed": bool(values[name])}
    return checks


def run(pipe, test: pd.DataFrame, selected: dict, cases: list[dict], cfg: dict) -> dict:
    """Measure everything and apply the gate. `selected` is reports/metrics.json from select.py."""
    X, y, classes = inputs(test, "skills_eval"), test["role"].to_numpy(), pipe.classes_
    proba = pipe.predict_proba(X)
    hits = top3_hits(proba, classes, y)
    by_family, by_role = slice_scores(hits, test["family"]), slice_scores(hits, test["role"])
    pass_rate, failures = behaviour(pipe, cases)
    m = selected["best_model_metrics"]
    values = {
        "top3_5skills": m["top3_5skills"],
        "margin_over_profile_matching": selected["margin_over_profile_matching"],
        "ref2019_top3_5skills": m["ref2019_top3_5skills"],
        "ece_top1_5skills": round(ece_top1(proba, classes, y), 4),
        "mean_top1_confidence": round(float(proba.max(axis=1).mean()), 4),
        "top1_accuracy": round(float((classes[proba.argmax(axis=1)] == y).mean()), 4),
        "min_family_top3_5skills": round(float(by_family.iloc[0]), 4),
        "min_role_top3_5skills": round(float(by_role.iloc[0]), 4),
        "behaviour_pass_rate": round(pass_rate, 4),
        "model_size_mb": m["model_size_mb"],
        "p95_latency_ms": p95_latency_ms(pipe, X, cfg["latency_requests"]),
        **invariance(pipe, X.head(500)),
        "robust_inputs": robust_inputs(pipe),
    }
    checks = gate(values, cfg)
    return {
        "model": selected["best_model"],
        "passed": all(c["passed"] for c in checks.values()),
        "failed_checks": [k for k, c in checks.items() if not c["passed"]],
        "checks": checks,
        "context": {
            "mean_top1_confidence": values["mean_top1_confidence"],
            "top1_accuracy": values["top1_accuracy"],
            "worst_families": by_family.head(5).round(4).to_dict(),
            "worst_roles": by_role.head(5).round(4).to_dict(),
            "behaviour_failures": failures,
            "n_test": len(test),
            "n_classes": len(classes),
        },
    }


def model_card(report: dict, selected: dict) -> str:
    m, ctx = selected["best_model_metrics"], report["context"]
    verdict = "PASSED" if report["passed"] else f"FAILED ({', '.join(report['failed_checks'])})"
    rows = "\n".join(
        f"| {k} | {c['value']} | {c['rule']} {c['threshold']} | {'✅' if c['passed'] else '❌'} |"
        for k, c in report["checks"].items()
    )
    worst_f = "\n".join(f"- {k}: {v:.1%}" for k, v in ctx["worst_families"].items())
    worst_r = "\n".join(f"- {k}: {v:.1%}" for k, v in ctx["worst_roles"].items())
    fails = "\n".join(f"- expected **{f['expected']}**, got {f['got_top3']}" for f in ctx["behaviour_failures"])
    return f"""# Model card: {report["model"]}

_Generated by `src/models/validate.py` on {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Do not edit by hand._

**Quality gate: {verdict}**

## What it does
Given at least 5 skills (from a fixed ~6,000-skill vocabulary) and years of experience, ranks
{ctx["n_classes"]} specific job roles in the Indian job market. The app shows the top 3 with confidence %.

- **Intended use:** career exploration for students and job seekers in India; a starting point, not a hiring decision.
- **Not for:** screening or ranking candidates; roles outside the {ctx["n_classes"]} classes; markets outside India.
- **Training data:** Naukri 2025 postings (labels from regex rules on job titles), with skill-dropout augmentation.
- **MLflow run:** `{selected.get("mlflow_run_id")}`

## Performance (held-out 2025 test set, {ctx["n_test"]:,} postings cut to 5 skills)
| Metric | Value |
|---|---|
| Top-1 / Top-3 / Top-5 | {m["top1_5skills"]:.1%} / {m["top3_5skills"]:.1%} / {m["top5_5skills"]:.1%} |
| Macro-F1 | {m["macro_f1_5skills"]:.3f} |
| Right family (top-1) | {m["family_top1_5skills"]:.1%} |
| 2019 reference set, top-3 (drift) | {m["ref2019_top3_5skills"]:.1%} |
| Mean shown confidence vs. actual top-1 accuracy | {ctx["mean_top1_confidence"]:.1%} vs {ctx["top1_accuracy"]:.1%} |

## Quality gate
| Check | Value | Rule | Pass |
|---|---|---|---|
{rows}

## Where it is weakest
Role families (top-3 accuracy):
{worst_f}

Single roles:
{worst_r}

Behavioural cases that failed:
{fails or "- none"}

## Known limitations
- Labels come from job-title rules, so vague roles (e.g. "Software Engineer (general)") overlap with specific ones.
- Accuracy drops on 2019 postings: the job market drifts, so the model needs retraining on new snapshots.
- Confidence % is only meaningful if calibration (ECE) passes; otherwise show the ranking only.
"""


def main() -> None:
    p = load_params()
    cfg = p["validate"]
    selected = read_json(REPORTS / "metrics.json")
    pipe = joblib.load(MODELS / "model.joblib")
    cases = load_yaml(CONFIGS / "behaviour_cases.yaml")["cases"]
    report = run(pipe, read_split("test"), selected, cases, cfg) | {"mlflow_run_id": selected.get("mlflow_run_id")}
    write_json(report, REPORTS / "validation.json")
    (REPORTS / "model_card.md").write_text(model_card(report, selected), encoding="utf-8", newline="\n")
    for name, c in report["checks"].items():
        log.info("%s %-32s %s %s %s", "PASS" if c["passed"] else "FAIL", name, c["value"], c["rule"], c["threshold"])
    log.info("%s: quality gate %s", report["model"], "PASSED" if report["passed"] else "FAILED")
    if "--strict" in sys.argv and not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
