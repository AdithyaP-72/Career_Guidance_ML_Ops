"""Deterministic labelled 2026 SIMULATION, preserving real data and unseen evaluation sources.

Run from the repository root: uv run python scripts/generate_2026_drift.py
Synthetic train uses real train sources; synthetic eval/holdout use disjoint real test sources.
These class-conditional perturbations are controlled assumptions, not observed 2026 statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORIGIN = "synthetic_2026"
SPLITS = ("train", "eval", "holdout")
LEGACY = {"jquery", "angularjs", "visual basic", "svn", "struts", "hadoop", "sqoop", "pig", "flash"}
MODERN = {
    "ML / AI Engineer": [
        "large language models",
        "retrieval-augmented generation",
        "generative ai",
        "ai agents",
        "llmops",
        "mlops",
        "vector databases",
        "model serving",
    ],
    "Data Scientist": ["large language models", "generative ai", "vector databases", "mlops"],
    "Data Engineer": ["apache iceberg", "dbt", "cloud computing", "vector databases", "delta lake"],
    "Cloud Engineer": ["kubernetes", "cloud computing", "infrastructure as code", "platform engineering"],
    "DevOps Engineer": ["kubernetes", "cloud computing", "mlops", "model serving", "gitops"],
    "ServiceNow Developer": ["servicenow", "servicenow now assist", "servicenow flow designer", "ai agents"],
}
FAMILY_SKILLS = {
    "Software Development": ["ai-assisted development", "cloud computing", "generative ai"],
    "Data & Analytics": ["modern data engineering", "cloud computing", "generative ai"],
    "IT Infra & Support": ["cloud computing", "zero trust security", "platform engineering"],
    "ERP / Enterprise Apps": ["cloud integration", "workflow automation", "enterprise ai"],
    "QA & Testing": ["ai-assisted testing", "test automation", "cloud testing"],
    "Accounting & Finance": ["financial process automation", "ai-assisted financial analysis"],
    "Marketing": ["generative ai content", "marketing analytics"],
    "Content & Media": ["ai-assisted editing", "generative ai content"],
    "HR & Recruitment": ["hr analytics", "recruitment automation"],
}
FAMILY_WEIGHTS = {
    "Data & Analytics": 2.6,
    "IT Infra & Support": 2.0,
    "ERP / Enterprise Apps": 1.8,
    "Software Development": 1.3,
    "Office Admin & Data Entry": 0.6,
}
ROLE_WEIGHTS = {
    "ML / AI Engineer": 3.0,
    "Data Engineer": 2.0,
    "ServiceNow Developer": 2.5,
    "Mainframe Developer": 0.35,
    "PHP Developer": 0.6,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_group_id(role: str, family: str, skills, exp_min) -> str:
    """Group identical model inputs even if posting IDs or noisy role labels differ.

    Role/family arguments are accepted for convenient use with source rows, but are
    deliberately excluded: different labels must not hide identical feature leakage.
    """
    exp = None if pd.isna(exp_min) else float(exp_min)
    value = [sorted(set(map(str, skills))), exp]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _prepare(frame: pd.DataFrame, roles: dict[str, str]) -> pd.DataFrame:
    required = {"posting_id", "skills", "exp_min", "role", "family"}
    if not required <= set(frame):
        raise ValueError(f"source missing columns: {sorted(required - set(frame))}")
    frame = frame.copy()
    if frame.empty or frame["role"].isna().any() or frame["family"].isna().any():
        raise ValueError("source must contain labelled, nonempty postings")
    if not all(roles.get(str(r)) == str(f) for r, f in zip(frame.role, frame.family, strict=True)):
        raise ValueError("source roles/families do not match the training taxonomy")
    if not frame.skills.map(lambda values: isinstance(values, (list, np.ndarray)) and len(values) >= 1).all():
        raise ValueError("source skills must be nonempty lists")
    frame["source_group_id"] = [source_group_id(r.role, r.family, r.skills, r.exp_min) for r in frame.itertuples()]
    return (
        frame.sort_values(["source_group_id", "posting_id"]).drop_duplicates("source_group_id").reset_index(drop=True)
    )


def _sample(pool: pd.DataFrame, count: int, roles: dict[str, str], rng) -> pd.DataFrame:
    if count < len(roles) or count > len(pool):
        raise ValueError(f"requested {count} rows from {len(pool)} source groups; need at least {len(roles)}")
    chosen = []
    for role in sorted(roles):
        indices = pool.index[pool.role == role].to_numpy()
        if not len(indices):
            raise ValueError(f"source pool has no independent group for {role!r}")
        chosen.append(int(rng.choice(indices)))
    rest = pool.drop(index=chosen)
    weights = np.array([FAMILY_WEIGHTS.get(r.family, 1.0) * ROLE_WEIGHTS.get(r.role, 1.0) for r in rest.itertuples()])
    if count > len(chosen):
        chosen.extend(rng.choice(rest.index, count - len(chosen), replace=False, p=weights / weights.sum()).tolist())
    return pool.loc[chosen].reset_index(drop=True)


def validate_split(frame: pd.DataFrame, roles: dict[str, str], split: str) -> None:
    """Validate labels, feature values, synthetic markers, uniqueness and source provenance."""
    required = {
        "posting_id",
        "snapshot",
        "title",
        "company",
        "skills",
        "n_skills",
        "exp_min",
        "exp_max",
        "role",
        "family",
        "skills_eval",
        "data_origin",
        "split",
        "source_posting_id",
        "source_group_id",
        "row_id",
        "generation_seed",
    }
    if split not in SPLITS or frame.empty or not required <= set(frame):
        raise ValueError("invalid split, empty dataset, or missing synthetic schema columns")
    if not (frame.data_origin.eq(ORIGIN).all() and frame.split.eq(split).all() and frame.snapshot.eq(2026).all()):
        raise ValueError("synthetic markers do not match the split")
    if not all(roles.get(str(r)) == str(f) for r, f in zip(frame.role, frame.family, strict=True)):
        raise ValueError("invalid role/family label")
    for row in frame.itertuples():
        skills, sample = list(row.skills), list(row.skills_eval)
        if (
            not 5 <= len(skills) <= 60
            or len(set(skills)) != len(skills)
            or not all(isinstance(s, str) and s.strip() == s and s and len(s) <= 100 for s in skills)
        ):
            raise ValueError("invalid skills list")
        if len(sample) != min(5, len(skills)) or len(set(sample)) != len(sample) or not set(sample) <= set(skills):
            raise ValueError("skills_eval must contain five distinct skills from skills")
        if row.n_skills != len(skills) or not (math.isfinite(row.exp_min) and 0 <= row.exp_min <= row.exp_max <= 20):
            raise ValueError("invalid skill count or experience")
        if len(str(row.source_group_id)) != 64 or not row.source_posting_id:
            raise ValueError("invalid source provenance")
    if any(frame[col].duplicated().any() for col in ("row_id", "posting_id", "source_group_id")):
        raise ValueError("duplicate synthetic record/source group")


def distributions(frame: pd.DataFrame, vocab: set[str]) -> dict:
    counts = Counter(s for values in frame.skills for s in set(values))
    unknown = sum(s not in vocab for values in frame.skills for s in values)
    total = int(frame.n_skills.sum())
    return {
        "roles": frame.role.value_counts().sort_index().to_dict(),
        "families": frame.family.value_counts().sort_index().to_dict(),
        "skill_presence": {skill: round(count / len(frame), 6) for skill, count in sorted(counts.items())},
        "experience": {
            "mean": round(float(frame.exp_min.mean()), 4),
            "quantiles": {str(q): round(float(frame.exp_min.quantile(q)), 4) for q in (0, 0.25, 0.5, 0.75, 1)},
        },
        "unknown_skill_token_rate": round(unknown / total, 6) if total else 0.0,
        "mean_skills": round(float(frame.n_skills.mean()), 4),
    }


def generate(
    source: Path = ROOT / "data/processed/train.parquet",
    evaluation_source: Path = ROOT / "data/processed/test.parquet",
    output_dir: Path = ROOT / "data/drift",
    rows: int = 12000,
    seed: int = 42,
    vocabulary_path: Path | None = ROOT / "models/skill_vocab.json",
) -> dict:
    source, evaluation_source, output_dir = Path(source), Path(evaluation_source), Path(output_dir)
    if rows < 15 or seed < 0:
        raise ValueError("rows must be at least 15 and seed must be nonnegative")
    if output_dir.resolve() in {source.resolve().parent, evaluation_source.resolve().parent}:
        raise ValueError("write synthetic outputs in a separate directory")
    raw = pd.read_parquet(source)
    mapping = raw[["role", "family"]].drop_duplicates()
    if mapping.role.duplicated().any():
        raise ValueError("each role must have one family")
    roles = dict(zip(mapping.role.astype(str), mapping.family.astype(str), strict=True))
    train_pool = _prepare(raw, roles)
    eval_pool = _prepare(pd.read_parquet(evaluation_source), roles)
    original_eval_groups = len(eval_pool)
    eval_pool = eval_pool[~eval_pool.source_group_id.isin(train_pool.source_group_id)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    eval_indices, holdout_indices = [], []
    for role in sorted(roles):
        indices = eval_pool.index[eval_pool.role == role].to_numpy()
        if len(indices) < 2:
            raise ValueError(f"need at least two unseen evaluation groups for {role!r}")
        rng.shuffle(indices)
        half = len(indices) // 2
        eval_indices.extend(indices[:half])
        holdout_indices.extend(indices[half:])
    pools = {"train": train_pool, "eval": eval_pool.loc[eval_indices], "holdout": eval_pool.loc[holdout_indices]}
    counts = {"train": int(rows * 0.6), "eval": int(rows * 0.2)}
    counts["holdout"] = rows - counts["train"] - counts["eval"]
    vocab = (
        set(json.loads(Path(vocabulary_path).read_text()))
        if vocabulary_path and Path(vocabulary_path).exists()
        else set(s for values in raw.skills for s in values)
    )
    role_skills = {
        role: [
            s
            for s, _ in Counter(s for vals in group.skills for s in vals if s in vocab and len(s) <= 100).most_common()
        ]
        for role, group in raw.groupby("role")
    }
    all_fingerprints, source_sets, output = set(), {}, {}
    for split in SPLITS:
        records = []
        selected = _sample(pools[split], counts[split], roles, rng)
        for i, row in enumerate(selected.itertuples()):
            for _attempt in range(100):
                base = list(dict.fromkeys(str(s).strip() for s in row.skills if 0 < len(str(s).strip()) <= 100))
                skills = [s for s in base if s not in LEGACY or rng.random() >= 0.7]
                known = {s for s in skills if s in vocab}
                for skill in role_skills[row.role]:
                    if len(known) >= 5:
                        break
                    if skill not in known:
                        skills.append(skill)
                        known.add(skill)
                options = MODERN.get(row.role, FAMILY_SKILLS.get(row.family, []))
                if options and rng.random() < (0.95 if row.role in MODERN else 0.75):
                    n = min(len(options), int(rng.integers(2, 5)))
                    skills.extend(rng.choice(options, n, replace=False).tolist())
                skills = list(dict.fromkeys(skills))[:60]
                # A modest experience shift, with stronger requirements in the targeted technical sectors.
                initial = float(row.exp_min) if pd.notna(row.exp_min) else float(raw.exp_min.median())
                shift = 1.5 if row.family in {"Data & Analytics", "IT Infra & Support"} else 0.5
                experience = round(float(np.clip(initial + rng.normal(shift, 0.8), 0, 18)), 2)
                fingerprint = source_group_id(row.role, row.family, skills, experience)
                if fingerprint not in all_fingerprints:
                    all_fingerprints.add(fingerprint)
                    break
            else:
                raise ValueError("could not generate unique feature rows")
            sample = [skills[j] for j in sorted(rng.choice(len(skills), 5, replace=False))]
            row_id = f"{ORIGIN}-{split}-{i:06d}"
            records.append(
                {
                    "posting_id": row_id,
                    "snapshot": 2026,
                    "title": f"Simulated {row.role}",
                    "company": "synthetic employer",
                    "skills": skills,
                    "n_skills": len(skills),
                    "exp_min": experience,
                    "exp_max": round(min(20.0, experience + 2.0), 2),
                    "role": row.role,
                    "family": row.family,
                    "skills_eval": sample,
                    "data_origin": ORIGIN,
                    "split": split,
                    "source_posting_id": str(row.posting_id),
                    "source_group_id": row.source_group_id,
                    "row_id": row_id,
                    "generation_seed": seed,
                }
            )
        frame = pd.DataFrame(records)
        validate_split(frame, roles, split)
        source_sets[split] = set(frame.source_group_id)
        output[split] = frame
    if any(source_sets[a] & source_sets[b] for a, b in (("train", "eval"), ("train", "holdout"), ("eval", "holdout"))):
        raise ValueError("source leakage across synthetic partitions")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "dataset_origin": ORIGIN,
        "seed": seed,
        "generator_sha256": sha256_file(Path(__file__)),
        "assumptions": [
            "Controlled simulation; not observed or independently labelled real-world 2026 market outcomes.",
            "Class-conditional modern skills and sector/role weights are experimental assumptions, not forecasts.",
            "Labels/families retained from rule-labelled sources; perturbations do not verify real-world suitability.",
            "Synthetic train uses real train sources. Eval/holdout use disjoint unseen real test groups.",
            "Real train/test feature-identical groups removed before eval/holdout allocation.",
            "No generated holdout features/labels may be used until after challenger selection.",
        ],
        "inputs": {
            "train": {"path": str(source), "sha256": sha256_file(source)},
            "evaluation": {"path": str(evaluation_source), "sha256": sha256_file(evaluation_source)},
        },
        "vocabulary": {
            "path": str(vocabulary_path) if vocabulary_path else None,
            "size": len(vocab),
            "sha256": sha256_file(vocabulary_path) if vocabulary_path and Path(vocabulary_path).exists() else None,
        },
        "parameters": {
            "legacy_removal_probability": 0.7,
            "family_weights": FAMILY_WEIGHTS,
            "role_weights": ROLE_WEIGHTS,
            "targeted_skill_probability": 0.95,
            "other_modern_skill_probability": 0.75,
            "technical_experience_mean_shift": 1.5,
            "other_experience_mean_shift": 0.5,
            "experience_noise_sd": 0.8,
        },
        "reference_distributions": distributions(raw.assign(n_skills=raw.skills.map(len)), vocab),
        "evaluation_overlap_groups_removed": original_eval_groups - len(eval_pool),
        "splits": {},
    }
    for split, frame in output.items():
        path = output_dir / f"{ORIGIN}_{split}.parquet"
        frame.to_parquet(path, index=False)
        manifest["splits"][split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(frame),
            "source_group_ids": sorted(source_sets[split]),
            "distributions": distributions(frame, vocab),
        }
    (output_dir / f"{ORIGIN}_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/processed/train.parquet")
    parser.add_argument("--evaluation-source", type=Path, default=ROOT / "data/processed/test.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/drift")
    parser.add_argument("--vocabulary", type=Path, default=ROOT / "models/skill_vocab.json")
    parser.add_argument("--rows", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = generate(args.source, args.evaluation_source, args.output_dir, args.rows, args.seed, args.vocabulary)
    print(
        json.dumps(
            {"origin": ORIGIN, "seed": args.seed, "rows": {k: v["rows"] for k, v in manifest["splits"].items()}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
