"""Per-role skill profiles: what the skill gap, readiness score and learning path read from.

Built from the TRAIN split only, because the "profile matching" baseline in Step 5 uses
these profiles to predict, and test postings must not leak into them.

    models/skill_profiles.json   per role: core skills + shares, top skills, common titles,
                                 experience range; plus skill share 2019 vs 2025 (growth)
    models/skill_cooccurrence.npz + models/skill_vocab.json
                                 how often two skills appear in the same posting. Used to
                                 order the learning path: learn first what usually goes
                                 with the skills you already have.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from src.features.build import identity
from src.utils import INTERIM, MODELS, PROCESSED, get_logger, load_params, read_json, write_json

log = get_logger("profiles")

QUANTILES = {"p25": 0.25, "median": 0.5, "p75": 0.75}


def skill_shares(skills: pd.Series) -> pd.Series:
    """Share of postings that list each skill."""
    return skills.explode().dropna().value_counts() / len(skills)


def main() -> None:
    p = load_params()
    pp, top_k, min_skills = p["profiles"], p["features"]["top_k_skills"], p["data"]["min_skills"]
    cls = read_json(PROCESSED / "classes.json")
    status = {c["role"]: c["status"] for c in cls["classes"]}
    train = pd.read_parquet(PROCESSED / "train.parquet")
    train["skills"] = train["skills"].map(list)

    roles = {}
    for role, g in train.groupby("role"):
        shares = skill_shares(g["skills"])
        core = shares[shares >= pp["core_min_share"]].head(pp["max_core_skills"])
        roles[role] = {
            "family": g["family"].iloc[0],
            "status": status[role],
            "n_postings": len(g),
            "core_skills": {s: round(float(v), 4) for s, v in core.items()},
            "top_skills": {s: round(float(v), 4) for s, v in shares.head(pp["top_n_skills"]).items()},
            "common_titles": g["title"].value_counts().head(pp["top_n_titles"]).index.tolist(),
            "experience_years": {q: float(g["exp_min"].quantile(v)) for q, v in QUANTILES.items()},
        }

    # Growth: share of all usable postings that list a skill, reference year vs train year.
    lab = pd.read_parquet(INTERIM / "labelled.parquet", columns=["snapshot", "skills", "n_skills"])
    lab = lab[lab["n_skills"] >= min_skills]
    ref_y, train_y = cls["reference_snapshot"], cls["train_snapshot"]
    old = skill_shares(lab.loc[lab["snapshot"] == ref_y, "skills"].map(list))
    new = skill_shares(lab.loc[lab["snapshot"] == train_y, "skills"].map(list))
    tracked = sorted({s for r in roles.values() for s in r["top_skills"]})
    growth = {
        s: {f"share_{ref_y}": round(float(old.get(s, 0.0)), 5), f"share_{train_y}": round(float(new.get(s, 0.0)), 5)}
        for s in tracked
    }

    write_json(
        {"roles": roles, "skill_growth": growth, "params": {**pp, "built_from": "data/processed/train.parquet"}},
        MODELS / "skill_profiles.json",
    )

    vec = CountVectorizer(analyzer=identity, binary=True, max_features=top_k, lowercase=False)
    X = vec.fit_transform(train["skills"]).astype(np.int32)
    co = (X.T @ X).tocsr()
    sp.save_npz(MODELS / "skill_cooccurrence.npz", co)
    write_json(vec.get_feature_names_out().tolist(), MODELS / "skill_vocab.json")
    log.info("profiles for %d roles; co-occurrence %s with %d non-zeros", len(roles), co.shape, co.nnz)


if __name__ == "__main__":
    main()
