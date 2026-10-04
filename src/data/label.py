"""title -> specific role + family via ordered regex rules.

Also scores the rules two ways against Naukri's own 2019 labels (reports/label_quality.json):
  * family agreement against Naukri `Functional Area` (strict one-to-one, IT-grouped, and 'plausible' many-to-many)
  * role agreement against Naukri `Role` (recruiter-chosen, independent of our title rules)
"""
from __future__ import annotations

import json
import re

import pandas as pd

from src.data.common import ROOT, load_params, load_yaml

INTERIM = ROOT / "data/interim"
PROCESSED = ROOT / "data/processed"
TRAIN_SOURCES = [2025, 2022, "internshala"]  # label rules on all; 2019 is the reference / validation snapshot


def compile_rules(cfg: dict | None = None):
    cfg = cfg or load_yaml("configs/role_rules.yaml")
    return [(re.compile(r["pattern"]), r["role"], r["family"]) for r in cfg["rules"]]


def label_title(title: str, rules) -> tuple[str | None, str | None]:
    t = str(title).lower()
    for rx, role, family in rules:
        if rx.search(t):
            return role, family
    return None, None


def label_frame(df: pd.DataFrame, rules) -> pd.DataFrame:
    res = df["title"].map(lambda t: label_title(t, rules))
    df = df.copy()
    df["role"] = res.map(lambda r: r[0])
    df["family"] = res.map(lambda r: r[1])
    return df


def naukri_family(area: str, mapping: dict) -> str | None:
    a = str(area).lower()
    for kw, fam in mapping.items():
        if kw in a:
            return fam
    return None


def plausible_families(area: str, mapping: dict) -> list[str] | None:
    a = str(area).lower()
    for kw, fams in mapping.items():
        if kw in a:
            return fams
    return None


def score_against_naukri(d19_all: pd.DataFrame, cfg: dict) -> dict:
    """d19_all: ALL labelled 2019 postings (before the role-size filter)."""
    grp = set(cfg["it_group"])
    fold = lambda f: "IT" if f in grp else f  # noqa: E731
    lab = d19_all.dropna(subset=["family"]).copy()
    lab["nf"] = lab["naukri_area"].map(lambda a: naukri_family(a, cfg["naukri_functional_area"]))
    lab["pl"] = lab["naukri_area"].map(lambda a: plausible_families(a, cfg["naukri_plausible"]))
    strict = lab.dropna(subset=["nf"])
    plaus = lab.dropna(subset=["pl"])
    rmap = load_yaml("configs/naukri_role_map.yaml")["map"]
    rl = lab[lab["naukri_role"].isin(rmap)]
    return {
        "coverage_2019": float(d19_all["role"].notna().mean()),
        "family_agreement_2019": float((strict["family"] == strict["nf"]).mean()),
        "family_agreement_2019_it_grouped": float((strict["family"].map(fold) == strict["nf"].map(fold)).mean()),
        "family_agreement_2019_plausible": float(plaus.apply(lambda r: r["family"] in r["pl"], axis=1).mean()),
        "role_agreement_vs_naukri_role": float(rl.apply(lambda r: r["role"] in rmap[r["naukri_role"]], axis=1).mean()) if len(rl) else 0.0,
        "agreement_n": int(len(plaus)),
        "role_agreement_n": int(len(rl)),
    }


def main() -> None:
    cfg = load_yaml("configs/role_rules.yaml")
    rules = compile_rules(cfg)
    min_size = load_params()["label"]["min_role_size"]
    PROCESSED.mkdir(parents=True, exist_ok=True)

    d19 = label_frame(pd.read_parquet(INTERIM / "postings_2019.parquet"), rules)
    quality = score_against_naukri(d19, cfg)

    frames = {}
    for src in TRAIN_SOURCES:
        f = label_frame(pd.read_parquet(INTERIM / f"postings_{src}.parquet"), rules)
        f["source"] = str(src)
        frames[src] = f
    pool = pd.concat([frames[2025], frames[2022]])
    counts = pool["role"].value_counts()
    keep = set(counts[counts >= min_size].index)
    d19_kept = d19[d19["role"].isin(keep)].reset_index(drop=True)
    for src, f in frames.items():
        frames[src] = f[f["role"].isin(keep)].reset_index(drop=True)
        frames[src].to_parquet(PROCESSED / f"labelled_{src}.parquet")
    d19_kept.to_parquet(PROCESSED / "labelled_2019.parquet")
    d19.drop(columns=["family"]).assign(role_all=d19["role"]).to_parquet(PROCESSED / "all_2019.parquet")

    quality.update({"n_roles": len(keep), **{f"n_labelled_{s}": int(len(f)) for s, f in frames.items()}})
    print(json.dumps(quality, indent=2))
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/label_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    print(frames[2025]["role"].value_counts().to_string())


if __name__ == "__main__":
    main()
