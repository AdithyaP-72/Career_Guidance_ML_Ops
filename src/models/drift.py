"""Reference and batch comparisons for a frozen career classifier.

The reference uses API-eligible 2025 training rows. Skills are independent binary
presence distributions, not multinomial categories. PSI thresholds are operational
heuristics. PSI measures distribution change, not accuracy or concept drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from src.features.build import EXPERIENCE, SKILLS
from src.utils import MODELS, PROCESSED, REPORTS, load_aliases, normalise_skill, read_json, write_json

SKILL_LIMIT = 60
EXPERIENCE_EDGES = (1, 3, 5, 8, 12, 20)
CONFIDENCE_EDGES = tuple(i / 10 for i in range(1, 10))
GROUPS = {
    "ai_skills": frozenset(
        {
            "ai",
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "generative ai",
            "genai",
            "llm",
            "llms",
            "large language models",
            "large language model",
            "rag",
            "retrieval-augmented generation",
            "retrieval augmented generation",
            "ai agents",
            "agentic ai",
            "llmops",
            "mlops",
            "vector databases",
            "vector database",
            "model serving",
            "prompt engineering",
            "langchain",
            "hugging face",
        }
    ),
    "cloud_skills": frozenset(
        {
            "cloud computing",
            "cloud",
            "aws",
            "azure",
            "gcp",
            "amazon web services",
            "google cloud platform",
            "kubernetes",
            "docker",
            "terraform",
            "cloud architecture",
            "devops",
            "cloud engineering",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class Observation:
    """Redacted data only: vocabulary IDs, aggregate counts, bins and learned labels."""

    skill_ids: tuple[int, ...]
    groups: tuple[str, ...]
    unknown_count: int
    experience_bucket: int
    role: str | None = None
    confidence: float | None = None


def observation(known: list[str], unknown: list[str], experience: float | None, vocab: dict) -> Observation:
    ids = tuple(sorted({int(vocab[s]) for s in known if s in vocab}))
    terms = set(known) | set(unknown)
    groups = tuple(name for name, members in GROUPS.items() if terms & members)
    exp = (
        len(EXPERIENCE_EDGES) + 1
        if experience is None or not math.isfinite(experience)
        else bisect_right(EXPERIENCE_EDGES, experience)
    )
    return Observation(ids, groups, len(unknown), exp)


def vocabulary_hash(vocab: dict) -> str:
    data = json.dumps(sorted((key, int(value)) for key, value in vocab.items()), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _probabilities(values: list[float], smoothing: float) -> list[float]:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("distribution must have finite, non-negative values")
    total = sum(values)
    if total <= 0:
        raise ValueError("distribution has no observations")
    floored = [max(value / total, smoothing) for value in values]
    mass = sum(floored)
    return [value / mass for value in floored]


def population_stability_index(reference: list[float], current: list[float], smoothing: float = 0.001) -> float:
    """Normalize counts, floor probabilities at epsilon, renormalize, and apply PSI."""
    if len(reference) != len(current) or not 0 < smoothing < 1:
        raise ValueError("matching bins and a probability floor between 0 and 1 are required")
    ref, cur = _probabilities(reference, smoothing), _probabilities(current, smoothing)
    return sum((actual - expected) * math.log(actual / expected) for expected, actual in zip(ref, cur, strict=True))


def severity(psi: float, warning: float, critical: float) -> str:
    if psi >= critical:
        return "critical"
    return "warning" if psi >= warning else "normal"


def _make_observations(pipe, frame, skills_column: str, min_known: int) -> list[Observation]:
    import numpy as np
    import pandas as pd

    vocab = pipe.named_steps["features"].named_transformers_["skills"].vocabulary_
    aliases = load_aliases()
    accepted = []
    rows = []
    for raw, exp in zip(frame[skills_column], frame[EXPERIENCE], strict=True):
        submitted = list(raw)[:SKILL_LIMIT]
        experience = None if pd.isna(exp) else float(exp)
        if any(not 1 <= len(skill) <= 100 for skill in submitted) or (
            experience is not None and not 0 <= experience <= 60
        ):
            continue
        cleaned = list(dict.fromkeys(normalise_skill(s, aliases) for s in submitted))
        known = [s for s in cleaned if s in vocab]
        if len(known) < min_known:
            continue
        unknown = [s for s in cleaned if s and s not in vocab]
        accepted.append(observation(known, unknown, experience, vocab))
        rows.append({SKILLS: known, EXPERIENCE: float("nan") if experience is None else experience})
    for start in range(0, len(rows), 1024):
        probabilities = pipe.predict_proba(pd.DataFrame(rows[start : start + 1024]))
        # Match how the API breaks score ties when ranking roles.
        winners = np.argsort(-probabilities, axis=1)[:, 0]
        for offset, (scores, best) in enumerate(zip(probabilities, winners, strict=True)):
            index = start + offset
            accepted[index] = replace(
                accepted[index], role=str(pipe.classes_[best]), confidence=round(float(scores[best]), 4)
            )
    return accepted


def _distributions(rows: list[Observation], tracked: dict, roles: list[str]) -> dict:
    skill_counts = Counter(skill for row in rows for skill in row.skill_ids)
    group_counts = Counter(group for row in rows for group in row.groups)
    role_counts = Counter(row.role for row in rows)
    exp_counts = Counter(row.experience_bucket for row in rows)
    confidence_counts = Counter(bisect_right(CONFIDENCE_EDGES, row.confidence) for row in rows)
    size = len(rows)
    fields = {
        f"skill:{name}": (["absent", "present"], [size - skill_counts[idx], skill_counts[idx]])
        for name, idx in tracked.items()
    }
    fields.update({name: (["absent", "present"], [size - group_counts[name], group_counts[name]]) for name in GROUPS})
    fields.update(
        {
            "unknown_skill_rate": (
                ["known", "unknown"],
                [sum(len(r.skill_ids) for r in rows), sum(r.unknown_count for r in rows)],
            ),
            "experience": (
                ["0–<1", "1–<3", "3–<5", "5–<8", "8–<12", "12–<20", "20+", "missing"],
                [exp_counts[i] for i in range(len(EXPERIENCE_EDGES) + 2)],
            ),
            "predicted_role": (roles, [role_counts[role] for role in roles]),
            "confidence": (
                [f"{i / 10:.1f}–{(i + 1) / 10:.1f}" for i in range(10)],
                [confidence_counts[i] for i in range(len(CONFIDENCE_EDGES) + 1)],
            ),
        }
    )
    return {name: {"labels": labels, "counts": counts} for name, (labels, counts) in fields.items()}


def build_reference(
    frame,
    pipe,
    *,
    skills_column: str = SKILLS,
    min_samples: int = 100,
    warning: float = 0.10,
    critical: float = 0.25,
    min_known: int = 5,
    model_run_id: str | None = None,
    source_sha256: str | None = None,
) -> dict:
    if not (0 < warning < critical and math.isfinite(critical) and min_samples >= 1 and 1 <= min_known <= SKILL_LIMIT):
        raise ValueError("reference thresholds, minimum sample size, or minimum known skills are invalid")
    vocab = pipe.named_steps["features"].named_transformers_["skills"].vocabulary_
    records = _make_observations(pipe, frame, skills_column, min_known)
    if len(records) < min_samples:
        raise ValueError(f"only {len(records)} eligible reference rows, need {min_samples}")
    counts = Counter(skill for row in records for skill in row.skill_ids)
    names = {int(idx): name for name, idx in vocab.items()}
    # A fixed bounded set chosen from the baseline controls metric cardinality.
    tracked = {names[idx]: idx for idx, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:48]}
    roles = [str(role) for role in pipe.classes_]
    return {
        "schema_version": 1,
        "source": "real_2025_train",
        "source_sha256": source_sha256,
        "source_rows": len(frame),
        "skills_column": skills_column,
        "model_run_id": model_run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "sample_count": len(records),
        "min_samples": min_samples,
        "min_known": min_known,
        "thresholds": {"warning": warning, "critical": critical},
        "smoothing": 0.001,
        "vocabulary_sha256": vocabulary_hash(vocab),
        "roles": roles,
        "tracked_skills": tracked,
        "distributions": _distributions(records, tracked, roles),
    }


def compare_observations(reference: dict, records: list[Observation]) -> dict:
    now = time.time()
    out = {
        "status": "insufficient",
        "overall_psi": None,
        "aggregation": "maximum feature PSI",
        "sample_count": len(records),
        "reference_sample_count": reference["sample_count"],
        "min_samples": reference["min_samples"],
        "thresholds": reference["thresholds"],
        "smoothing": reference["smoothing"],
        "model_run_id": reference["model_run_id"],
        "reference_source": reference["source"],
        "reference_sha256": reference["source_sha256"],
        "generated_at": datetime.fromtimestamp(now, UTC).isoformat(),
        "generated_at_seconds": now,
        "features": {},
        "top_drifted": [],
    }
    if len(records) < reference["min_samples"]:
        return out
    current = _distributions(records, reference["tracked_skills"], reference["roles"])
    limits = reference["thresholds"]
    floor = reference["smoothing"]
    for key, value in current.items():
        base = reference["distributions"][key]
        if value["labels"] != base["labels"]:
            raise ValueError(f"incompatible reference bins for {key}")
        psi = population_stability_index(base["counts"], value["counts"], floor)
        out["features"][key] = {
            "psi": psi,
            "status": severity(psi, **limits),
            "labels": value["labels"],
            "reference": [n / sum(base["counts"]) for n in base["counts"]],
            "current": [n / sum(value["counts"]) for n in value["counts"]],
        }
    ranked = sorted(out["features"], key=lambda key: (-out["features"][key]["psi"], key))
    out["top_drifted"] = ranked[:10]
    out["overall_psi"] = out["features"][ranked[0]]["psi"]
    out["status"] = severity(out["overall_psi"], **limits)
    return out


def compare_frame(reference: dict, frame, pipe, *, skills_column: str = SKILLS) -> dict:
    vocab = pipe.named_steps["features"].named_transformers_["skills"].vocabulary_
    if skills_column != reference.get("skills_column", SKILLS):
        raise ValueError("reference and current comparisons require the same skill column protocol")
    if reference["vocabulary_sha256"] != vocabulary_hash(vocab) or reference["roles"] != list(pipe.classes_):
        raise ValueError("reference vocabulary or classes do not match the frozen model")
    return compare_observations(reference, _make_observations(pipe, frame, skills_column, reference["min_known"])) | {
        "submitted_rows": len(frame),
    }


class DriftArgumentParser(argparse.ArgumentParser):
    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: {message}\n")


def _main() -> int:
    import joblib
    import pandas as pd

    cli = DriftArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("baseline")
    baseline.add_argument("--train", type=Path, default=PROCESSED / "train.parquet")
    baseline.add_argument("--min-samples", type=int, default=100)
    baseline.add_argument("--min-known", type=int, default=5)
    baseline.add_argument("--warning", type=float, default=0.10)
    baseline.add_argument("--critical", type=float, default=0.25)
    compare = commands.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--current", type=Path, required=True)
    for command in (baseline, compare):
        command.add_argument("--model", type=Path, default=MODELS / "model.joblib")
        command.add_argument("--metadata", type=Path, default=REPORTS / "metrics.json")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--skills-column", default=SKILLS)
    args = cli.parse_args()
    try:
        pipe = joblib.load(args.model)
        run_id = read_json(args.metadata).get("mlflow_run_id") if args.metadata.exists() else None
        if args.command == "baseline":
            with args.train.open("rb") as file:
                source_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
            report = build_reference(
                pd.read_parquet(args.train),
                pipe,
                skills_column=args.skills_column,
                min_samples=args.min_samples,
                min_known=args.min_known,
                warning=args.warning,
                critical=args.critical,
                model_run_id=run_id,
                source_sha256=source_sha256,
            )
        else:
            reference = read_json(args.reference)
            if reference["model_run_id"] != run_id:
                raise ValueError("reference run does not match model metadata")
            report = compare_frame(reference, pd.read_parquet(args.current), pipe, skills_column=args.skills_column)
        write_json(report, args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Drift execution error: {error}", file=sys.stderr)
        return 1
    status = report.get("status", "reference_created")
    print(
        json.dumps({"status": status, "sample_count": report["sample_count"], "overall_psi": report.get("overall_psi")})
    )
    return {"warning": 2, "critical": 2, "insufficient": 3}.get(status, 0)


if __name__ == "__main__":
    raise SystemExit(_main())
