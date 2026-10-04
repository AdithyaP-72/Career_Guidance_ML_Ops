"""Per-role skill profiles (pandas, no model): core skills, readiness weights, co-occurrence, growth, titles."""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations

import pandas as pd

from src.data.common import ROOT, load_params

P = ROOT / "data/processed"
CO_TOP = 30  # co-occurrence kept among each role's top-N skills


def skill_share(df: pd.DataFrame) -> dict[str, float]:
    c = Counter(s for sk in df["skills"] for s in set(sk))
    n = max(len(df), 1)
    return {k: v / n for k, v in c.items()}


def main() -> None:
    pp = load_params()["profiles"]
    d25 = pd.read_parquet(P / "labelled_2025.parquet")
    d19 = pd.read_parquet(P / "labelled_2019.parquet")
    all19 = skill_share(d19) if len(d19) else {}
    all25 = skill_share(d25)

    growth = {s: round(all25[s] - all19.get(s, 0.0), 5) for s in all25 if all25[s] >= 0.002}

    profiles = {}
    for role, g in d25.groupby("role"):
        share = skill_share(g)
        ranked = sorted(share.items(), key=lambda kv: -kv[1])
        top = ranked[: pp["top_n_skills"]]
        core = [(s, v) for s, v in ranked if v >= pp["core_skill_min_share"]][:12] or top[:10]
        top_names = [s for s, _ in ranked[:CO_TOP]]
        pair, single = Counter(), Counter()
        for sk in g["skills"]:
            have = [s for s in set(sk) if s in top_names]
            single.update(have)
            pair.update(combinations(sorted(have), 2))
        cooc = {}
        for (a, b), n in pair.items():
            cooc.setdefault(a, {})[b] = round(n / single[a], 4)  # P(b | a)
            cooc.setdefault(b, {})[a] = round(n / single[b], 4)
        profiles[role] = {
            "family": g["family"].iloc[0],
            "n_postings": int(len(g)),
            "core_skills": [{"skill": s, "share": round(v, 4)} for s, v in core],
            "top_skills": [{"skill": s, "share": round(v, 4)} for s, v in top],
            "cooccurrence": cooc,
            "growth": {s: growth.get(s, 0.0) for s, _ in core},
            "top_titles": g["title"].str.lower().value_counts().head(pp["top_n_titles"]).index.tolist(),
            "experience": {
                "median": float(g["experience"].median()),
                "p25": float(g["experience"].quantile(0.25)),
                "p75": float(g["experience"].quantile(0.75)),
            },
        }
    out = ROOT / "models/skill_profiles.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"profiles": profiles, "skill_growth_2019_2025": growth,
                    "reference_skill_share": {k: round(v, 5) for k, v in sorted(all25.items(), key=lambda kv: -kv[1])[:5000]}}), encoding="utf-8")
    print(f"wrote {len(profiles)} role profiles")


if __name__ == "__main__":
    main()
