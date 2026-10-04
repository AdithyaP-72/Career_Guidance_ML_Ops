"""Give every posting a candidate role + family from its title (configs/role_rules.yaml).

Output: data/interim/labelled.parquet (postings + role, family, fa_family).

How good are the rules? 2017 and 2019 also carry Naukri's own functional-area label,
mapped to our families via configs/functional_area_map.yaml (`fa_family`). Where both
exist we measure agreement. Results:
    reports/label_quality.json          coverage + agreement per snapshot (DVC metric)
    reports/labels/disagreements.csv    most common (Naukri family, our family) mismatches
    reports/labels/unlabelled_titles.csv most common titles no rule catches (2025)
"""

from __future__ import annotations

import re

import pandas as pd

from src.utils import CONFIGS, INTERIM, REPORTS, get_logger, load_yaml, write_json

log = get_logger("label")


def compile_rules(path=CONFIGS / "role_rules.yaml") -> list[tuple[str, str, re.Pattern]]:
    cfg = load_yaml(path)
    families = set(cfg["families"])
    rules = []
    for r in cfg["rules"]:
        if r["family"] not in families:
            raise ValueError(f"rule for {r['role']!r} uses unknown family {r['family']!r}")
        rules.append((r["role"], r["family"], re.compile(r["pattern"])))
    return rules


def compile_area_map(path=CONFIGS / "functional_area_map.yaml") -> list[tuple[re.Pattern, str]]:
    return [(re.compile(a["pattern"]), a["family"]) for a in load_yaml(path)["areas"]]


def assign_role(title: str, rules) -> tuple[str | None, str | None]:
    t = str(title).lower()
    for role, family, pattern in rules:
        if pattern.search(t):
            return role, family
    return None, None


def assign_area_family(area, area_map) -> str | None:
    if pd.isna(area):
        return None
    a = str(area).lower().strip()
    for pattern, family in area_map:
        if pattern.search(a):
            return family
    return None


def main() -> None:
    postings = pd.read_parquet(INTERIM / "postings.parquet")
    rules, area_map = compile_rules(), compile_area_map()

    # Titles repeat a lot, so label each unique title once.
    uniq = pd.Series(postings["title"].unique())
    labels = pd.DataFrame(uniq.map(lambda t: assign_role(t, rules)).tolist(), columns=["role", "family"])
    labels["title"] = uniq
    df = postings.merge(labels, on="title", how="left")
    df["fa_family"] = df["functional_area"].map(lambda a: assign_area_family(a, area_map))
    for col in ["role", "family", "fa_family"]:
        df[col] = df[col].astype("string")

    quality = {}
    for snap, g in df.groupby("snapshot"):
        q = {"rows": len(g), "coverage": round(float(g["role"].notna().mean()), 4)}
        both = g.dropna(subset=["family", "fa_family"])
        if len(both):
            q["family_agreement"] = round(float((both["family"] == both["fa_family"]).mean()), 4)
            q["n_compared"] = len(both)
        quality[str(snap)] = q
        log.info("%s: %s", snap, q)
    write_json(quality, REPORTS / "label_quality.json")

    out = REPORTS / "labels"
    out.mkdir(parents=True, exist_ok=True)
    both = df.dropna(subset=["family", "fa_family"])
    mism = both[both["family"] != both["fa_family"]]
    (
        mism.groupby(["fa_family", "family"])
        .agg(n=("title", "size"), example_titles=("title", lambda t: " | ".join(t.value_counts().index[:5])))
        .sort_values("n", ascending=False)
        .head(60)
        .reset_index()
        .rename(columns={"fa_family": "naukri_family", "family": "our_family"})
        .to_csv(out / "disagreements.csv", index=False)
    )
    (
        df[(df["snapshot"] == 2025) & df["role"].isna()]["title"]
        .str.lower()
        .value_counts()
        .head(300)
        .rename_axis("title")
        .reset_index(name="n")
        .to_csv(out / "unlabelled_titles.csv", index=False)
    )

    df.to_parquet(INTERIM / "labelled.parquet", index=False)
    log.info("wrote data/interim/labelled.parquet (%d rows)", len(df))


if __name__ == "__main__":
    main()
