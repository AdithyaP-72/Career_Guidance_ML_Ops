"""Exploratory numbers behind the design choices (README Step 3) -> reports/eda.json + reports/vocab_coverage.png"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.common import ROOT  # noqa: E402

INTERIM, PROCESSED, REPORTS = ROOT / "data/interim", ROOT / "data/processed", ROOT / "reports"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    out = {}
    for name in ("2019", "2022", "2025", "internshala"):
        df = pd.read_parquet(INTERIM / f"postings_{name}.parquet")
        n = df["skills"].map(len)
        out[name] = {"postings": int(len(df)), "median_skills": float(n.median()), "mean_skills": round(float(n.mean()), 2),
                     "median_min_experience": float(df["experience"].median())}
    d25 = pd.read_parquet(INTERIM / "postings_2025.parquet")
    freq = Counter(s for sk in d25["skills"] for s in set(sk))
    ranked = [s for s, _ in freq.most_common()]
    out["distinct_skills_2025"] = len(ranked)
    sizes = [100, 300, 500, 1000, 1500, 3000, 6000, 10000]
    cover = {}
    for k in sizes:
        v = set(ranked[:k])
        cover[k] = round(float((d25["skills"].map(lambda s: sum(x in v for x in s)) >= 3).mean()), 4)
    out["share_postings_with_3plus_known_skills_by_vocab_size"] = cover
    lab = pd.read_parquet(PROCESSED / "labelled_2025.parquet")
    vc = lab["role"].value_counts()
    out["class_balance"] = {"roles": int(len(vc)), "largest": {vc.index[0]: int(vc.iloc[0])}, "smallest": {vc.index[-1]: int(vc.iloc[-1])},
                            "largest_family_share": round(float(lab["family"].value_counts(normalize=True).iloc[0]), 3)}
    (REPORTS / "eda.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(list(cover), list(cover.values()), marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("vocabulary size (top-K skills)")
    ax.set_ylabel("postings with >=3 known skills")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "vocab_coverage.png", dpi=110)
    print(json.dumps(out["share_postings_with_3plus_known_skills_by_vocab_size"]))


if __name__ == "__main__":
    main()
