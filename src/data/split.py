"""Labelled postings → train / test (2025) and the 2019 drift reference set.

    data/processed/train.parquet           2025, class roles, stratified by role
    data/processed/test.parquet            same, held out; adds `skills_eval`
    data/processed/reference_2019.parquet  2019 postings in *stable* classes; adds `skills_eval`

`skills_eval` is a fixed random sample of `features.eval_n_skills` skills per posting
(all of them if it has fewer). Users type ~5 skills while postings list ~8, so the
headline metric is computed on `skills_eval`, not on the full list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.build import sample_skills
from src.utils import INTERIM, PROCESSED, REPORTS, get_logger, load_params, read_json, write_json

log = get_logger("split")

KEEP = ["posting_id", "snapshot", "title", "company", "skills", "n_skills", "exp_min", "exp_max", "role", "family"]


def add_eval_skills(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return df.assign(skills_eval=[sample_skills(s, n, rng) for s in df["skills"]])


def main() -> None:
    p = load_params()
    min_skills, sp, n_eval = p["data"]["min_skills"], p["split"], p["features"]["eval_n_skills"]
    cls = read_json(PROCESSED / "classes.json")
    roles = {c["role"] for c in cls["classes"]}
    stable = {c["role"] for c in cls["classes"] if c["status"] == "stable"}

    df = pd.read_parquet(INTERIM / "labelled.parquet")
    base = df[(df["n_skills"] >= min_skills) & df["role"].notna()][KEEP]

    cur = base[(base["snapshot"] == cls["train_snapshot"]) & base["role"].isin(roles)]
    train, test = train_test_split(cur, test_size=sp["test_size"], stratify=cur["role"], random_state=sp["seed"])
    test = add_eval_skills(test, n_eval, sp["seed"])

    ref = base[(base["snapshot"] == cls["reference_snapshot"]) & base["role"].isin(stable)]
    ref = add_eval_skills(ref, n_eval, sp["seed"])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED / "train.parquet", index=False)
    test.to_parquet(PROCESSED / "test.parquet", index=False)
    ref.to_parquet(PROCESSED / "reference_2019.parquet", index=False)

    sizes = train["role"].value_counts()
    summary = {
        "train": len(train),
        "test": len(test),
        "reference_2019": len(ref),
        "classes": int(sizes.size),
        "stable_classes_in_reference": int(ref["role"].nunique()),
        "smallest_class_train": int(sizes.min()),
        "largest_class_train": int(sizes.max()),
        "largest_class_share": round(float(sizes.max() / len(train)), 4),
        "largest_class": sizes.index[0],
    }
    write_json(summary, REPORTS / "split_summary.json")
    log.info("%s", summary)


if __name__ == "__main__":
    main()
