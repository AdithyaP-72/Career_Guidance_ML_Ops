"""Real temporal drift: score the frozen 2025 champion on real 2019 postings.

The synthetic 2026 replay shows *forward* drift under a controlled scenario. This report adds
evidence from real data: the same frozen model and the same PSI reference, applied to the
2019 reference split (a different market era) next to the 2025 test split as a control.

It reports three things for both splits:
  * input drift: PSI per feature against the 2025 training reference (src.models.drift)
  * performance: top-1 / top-3 / macro-F1 / ECE on the five-skill view (src.models.validate)
  * which roles lost the most top-3 accuracy and which skills grew or faded between 2019 and 2025

Writes reports/phase3/real_drift/latest.json and per_role_top3.csv. Grafana reads the JSON
through /metrics (career_phase3_real_drift_score).

    uv run python -m scripts.real_drift_2019
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from src.features.build import SKILLS
from src.models.drift import build_reference, compare_frame
from src.models.evaluate import read_split
from src.models.validate import evaluate_labelled
from src.utils import MODELS, REPORTS, ROOT, get_logger, read_json, write_json

log = get_logger("real_drift_2019")
REFERENCE = ROOT / ".phase3" / "bundle" / "drift_reference.json"
OUT = REPORTS / "phase3" / "real_drift"
MIN_ROLE_ROWS = 30  # ignore roles too rare in either split for a stable top-3 rate
MIN_PREVALENCE = 0.005  # a skill must appear in 0.5% of postings in one split to be listed


def prevalence(frame: pd.DataFrame) -> pd.Series:
    counts = Counter(skill for skills in frame[SKILLS] for skill in set(skills))
    return pd.Series(counts, dtype=float) / len(frame)


def skill_shift(new: pd.DataFrame, old: pd.DataFrame, top: int = 15) -> dict:
    """Change in share of postings mentioning a skill, 2025 minus 2019 (positive = grew)."""
    p_new, p_old = prevalence(new), prevalence(old)
    both = pd.concat({"p2025": p_new, "p2019": p_old}, axis=1).fillna(0.0)
    both = both[(both["p2025"] >= MIN_PREVALENCE) | (both["p2019"] >= MIN_PREVALENCE)]
    both["change"] = both["p2025"] - both["p2019"]

    def rows(frame: pd.DataFrame) -> list[dict]:
        return [
            {"skill": s, "share_2019": round(r.p2019, 4), "share_2025": round(r.p2025, 4), "change": round(r.change, 4)}
            for s, r in frame.iterrows()
        ]

    return {
        "grew_since_2019": rows(both.nlargest(top, "change")),
        "faded_since_2019": rows(both.nsmallest(top, "change")),
        "skills_only_in_2025": len(set(p_new.index) - set(p_old.index)),
        "skills_only_in_2019": len(set(p_old.index) - set(p_new.index)),
    }


def summary(metrics: dict, drift: dict) -> dict:
    return {
        "rows": metrics["samples"],
        **{k: round(metrics[k], 4) for k in ("top1", "top3", "macro_f1", "ece")},
        "psi_status": drift["status"],
        "max_psi": drift["overall_psi"],
        "top_drifted": [{"feature": f, "psi": round(drift["features"][f]["psi"], 4)} for f in drift["top_drifted"][:8]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()

    pipe = joblib.load(MODELS / "model.joblib")
    run_id = read_json(REPORTS / "metrics.json").get("mlflow_run_id")
    if REFERENCE.is_file() and read_json(REFERENCE).get("model_run_id") == run_id:
        reference = read_json(REFERENCE)
    else:
        log.info("no matching Compose reference; building one from the 2025 training split")
        reference = build_reference(read_split("train"), pipe, model_run_id=run_id)

    test, old = read_split("test"), read_split("reference_2019")
    # Only roles the model knows can be scored; the 2019 split shares the taxonomy.
    old = old[old["role"].isin(pipe.classes_)]
    m_test, _ = evaluate_labelled(pipe, test)
    m_old, _ = evaluate_labelled(pipe, old)
    d_test, d_old = compare_frame(reference, test, pipe), compare_frame(reference, old, pipe)

    n_test, n_old = test["role"].value_counts(), old["role"].value_counts()
    per_role = pd.DataFrame({"top3_2025": m_test["top3_by_role"], "top3_2019": m_old["top3_by_role"]}).dropna()
    per_role["n_2025"], per_role["n_2019"] = n_test.reindex(per_role.index), n_old.reindex(per_role.index)
    per_role = per_role[(per_role["n_2025"] >= MIN_ROLE_ROWS) & (per_role["n_2019"] >= MIN_ROLE_ROWS)]
    per_role["change"] = per_role["top3_2019"] - per_role["top3_2025"]
    per_role = per_role.sort_values("change").rename_axis("role")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_run_id": run_id,
        "note": "Same frozen model and PSI reference on both splits; 2025 test is the no-drift control.",
        "real_2025_test": summary(m_test, d_test),
        "real_2019_reference": summary(m_old, d_old),
        "top3_change_points": round((m_old["top3"] - m_test["top3"]) * 100, 2),
        "worst_roles_top3": [
            {"role": role, "top3_2025": round(r.top3_2025, 3), "top3_2019": round(r.top3_2019, 3)}
            for role, r in per_role.head(10).iterrows()
        ],
        "skills": skill_shift(test, old),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(report, args.output / "latest.json")
    per_role.round(4).to_csv(args.output / "per_role_top3.csv", lineterminator="\n")
    t, o = report["real_2025_test"], report["real_2019_reference"]
    print(f"2025 test (control): top-3 {t['top3']:.1%}, max PSI {t['max_psi']:.3f} ({t['psi_status']})")
    print(f"2019 reference:      top-3 {o['top3']:.1%}, max PSI {o['max_psi']:.3f} ({o['psi_status']})")
    print(f"Written {args.output / 'latest.json'}")


if __name__ == "__main__":
    main()
