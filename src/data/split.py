"""Stratified split. The test set is always a held-out part of Naukri 2025 (comparable across experiments).
Extra sources (Naukri 2022, Internshala) only ever add to the TRAIN side, and anything that duplicates a test
posting (same title + skill set) is dropped from them, so there is no leakage."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.common import ROOT, load_params

P = ROOT / "data/processed"


def _key(df: pd.DataFrame) -> pd.Series:
    return df["title"].str.lower() + "|" + df["skills"].map(lambda s: ",".join(sorted(s)))


def main() -> None:
    p = load_params()["split"]
    base = pd.read_parquet(P / "labelled_2025.parquet")
    train, test = train_test_split(base, test_size=p["test_size"], random_state=p["seed"], stratify=base["role"])
    test_keys = set(_key(test))
    parts = [train]
    for src in p.get("extra_sources", []):
        ex = pd.read_parquet(P / f"labelled_{src}.parquet")
        ex = ex[~_key(ex).isin(test_keys)]
        parts.append(ex)
        print(f"extra source {src}: +{len(ex)} training postings")
    train = pd.concat(parts, ignore_index=True)
    train = train[~_key(train).duplicated()].reset_index(drop=True)
    train.to_parquet(P / "train.parquet")
    test.reset_index(drop=True).to_parquet(P / "test.parquet")
    print(f"train={len(train)} test={len(test)} roles={base['role'].nunique()}")


if __name__ == "__main__":
    main()
