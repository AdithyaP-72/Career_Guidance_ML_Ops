"""A tiny but complete model bundle (3 roles, ~15 skills), for tests and the CI Docker smoke test.

    uv run python -m tests.toy_bundle bundle      # writes ./bundle, no DVC or MLflow access needed

Same files and format as the real bundle (src/serving/bundle.py), so the API, the image and
the tests exercise exactly the code paths production uses, just with a toy model.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from src.features.build import identity
from src.models.train import build_pipeline
from src.serving.bundle import write_local
from src.utils import load_params, write_json

ROLES = {
    "Accountant": ("Accounting & Finance", ["accounting", "tally", "gst", "tds", "excel", "book keeping"]),
    "Frontend Developer": ("Software Development", ["javascript", "react.js", "html", "css", "typescript", "git"]),
    "Staff Nurse": ("Healthcare & Pharma", ["nursing", "gnm", "icu", "patient care", "hospital", "anm"]),
}


def toy_postings(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for role, (family, skills) in ROLES.items():
        for _ in range(8):
            k = int(rng.integers(3, len(skills) + 1))
            picked = [skills[i] for i in sorted(rng.choice(len(skills), size=k, replace=False))]
            rows.append({"skills": picked, "exp_min": float(rng.integers(0, 8)), "role": role, "family": family})
    return pd.DataFrame(rows)


def make_toy_bundle(out: Path) -> Path:
    df = toy_postings()
    pipe = build_pipeline("logreg", load_params()).fit(df[["skills", "exp_min"]], df["role"])
    roles = {}
    for role, g in df.groupby("role"):
        shares = g["skills"].explode().value_counts() / len(g)
        roles[role] = {
            "family": g["family"].iloc[0],
            "status": "stable",
            "n_postings": len(g),
            "core_skills": {s: round(float(v), 4) for s, v in shares.items()},
            "top_skills": {s: round(float(v), 4) for s, v in shares.items()},
            "common_titles": [role],
            "experience_years": {"p25": 1.0, "median": 2.0, "p75": 4.0},
        }
    growth = {"gst": {"share_2019": 0.01, "share_2025": 0.03}, "html": {"share_2019": 0.05, "share_2025": 0.02}}
    vec = CountVectorizer(analyzer=identity, binary=True, lowercase=False)
    X = vec.fit_transform(df["skills"]).astype(np.int32)
    with tempfile.TemporaryDirectory() as tmp:
        lookups = Path(tmp)
        write_json({"roles": roles, "skill_growth": growth, "params": {}}, lookups / "skill_profiles.json")
        sp.save_npz(lookups / "skill_cooccurrence.npz", (X.T @ X).tocsr())
        write_json(vec.get_feature_names_out().tolist(), lookups / "skill_vocab.json")
        meta = {"source": "toy", "model_name": "career-role-classifier", "version": "toy", "algorithm": "logreg"}
        write_local(out, pipe, lookups, meta)
    return out


if __name__ == "__main__":
    make_toy_bundle(Path(sys.argv[1] if len(sys.argv) > 1 else "bundle"))
