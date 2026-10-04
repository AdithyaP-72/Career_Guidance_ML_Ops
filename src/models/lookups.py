"""Lookup tables (pandas, no model training):
  * education prior   - which role families employers ask each degree for (Naukri 2015-17)
  * course links      - skill -> free NPTEL/SWAYAM course, Coursera as fallback
"""
from __future__ import annotations

import json
import re

import joblib
import numpy as np
import pandas as pd

from src.data.common import ROOT, load_params, load_yaml
from src.data.label import compile_rules, label_frame

RAW = ROOT / "data/raw"
OUT = ROOT / "models"
MIN_COVER = 0.4
GENERIC = {"analytical", "development", "technical", "technology", "operations", "research", "analysis", "business", "professional", "computer", "compliance"}


def _lift_table(share_by_group: dict[str, pd.Series], base: pd.Series, eps: float = 0.002) -> dict:
    return {g: {f: round(float((sh.get(f, 0) + eps) / (base[f] + eps)), 3) for f in base.index} for g, sh in share_by_group.items()}


def naukri_prior(min_postings: int) -> tuple[dict, dict]:
    """What employers ask each degree for: share of labelled postings per family among postings listing the degree."""
    df = pd.read_csv(RAW / "naukri_2017/naukri_com-job_sample.csv", usecols=["jobtitle", "education"]).dropna()
    lab = label_frame(df.rename(columns={"jobtitle": "title"}), compile_rules()).dropna(subset=["family"])
    base = lab["family"].value_counts(normalize=True)
    edu = lab["education"].str.lower()
    shares, sizes = {}, {}
    for name, cfg in load_yaml("configs/education.yaml").items():
        if "naukri" not in cfg:
            continue
        m = edu.str.contains(re.compile(cfg["naukri"]))
        sizes[name] = int(m.sum())
        if m.sum() >= min_postings:
            shares[name] = lab.loc[m, "family"].value_counts(normalize=True)
    return _lift_table(shares, base), sizes


def plfs_prior() -> tuple[dict, dict]:
    """What people with each education level actually do (PLFS 2023-24, survey-weighted, via NCO-2015 occupation)."""
    cfg = load_yaml("configs/family_nco.yaml")
    prefixes = sorted(cfg["prefixes"].items(), key=lambda kv: -len(str(kv[0])))
    df = pd.read_csv(RAW / "plfs/perv1_2023_24.csv", usecols=["b4q8_perv1", "b5pt1q6_perv1", "mult_perv1"], low_memory=False)
    df = df.dropna(subset=["b4q8_perv1", "b5pt1q6_perv1"])
    code = df["b5pt1q6_perv1"].astype(int).astype(str).str.zfill(3)

    def to_family(c: str):
        for pre, fam in prefixes:
            if c.startswith(str(pre)):
                return fam
        return None

    df["family"] = code.map(to_family)
    df = df.dropna(subset=["family"])
    w = df["mult_perv1"]
    base = w.groupby(df["family"]).sum()
    base = base / base.sum()
    shares, sizes = {}, {}
    for lvl in cfg["plfs_levels"]:
        m = df["b4q8_perv1"] == lvl
        g = w[m].groupby(df.loc[m, "family"]).sum()
        shares[lvl] = g / g.sum()
        sizes[lvl] = int(m.sum())
    return _lift_table(shares, base), sizes


def education_prior(min_postings: int) -> dict:
    nk, nk_sizes = naukri_prior(min_postings)
    pl, pl_sizes = plfs_prior()
    cfg = load_yaml("configs/education.yaml")
    families = list(next(iter(nk.values())).keys())
    final, used = {}, {}
    for name, c in cfg.items():
        parts = []
        if name in nk:
            parts.append(("naukri_2015_17", nk[name]))
        if c.get("plfs") in pl:
            parts.append(("plfs_2023_24", pl[c["plfs"]]))
        if not parts:
            continue
        used[name] = [p for p, _ in parts]
        # geometric mean of the available lifts; a family absent from a source stays neutral (1.0)
        final[name] = {f: round(float(np.exp(np.mean([np.log(t.get(f, 1.0)) for _, t in parts]))), 3) for f in families}
    return {"lift": final, "sources_used": used, "postings_per_degree_naukri": nk_sizes, "respondents_per_level_plfs": pl_sizes}


def course_links(vocab: list[str]) -> dict:
    nptel = pd.read_csv(RAW / "nptel/merged_courses.csv", usecols=["title", "provider", "url", "organisation"]).dropna(subset=["title", "url"])
    nptel["t"] = nptel["title"].str.lower()
    nptel = nptel.assign(len=nptel["t"].str.len()).sort_values(["provider", "len"], ascending=[False, True])  # NPTEL before SWAYAM, short titles first
    cour = pd.read_csv(RAW / "coursera/courses_en.csv", usecols=["url", "name", "skills"]).dropna(subset=["skills"])
    cour_idx: dict[str, tuple[str, str]] = {}
    for name, url, sk in zip(cour["name"], cour["url"], cour["skills"]):
        for s in str(sk).split(","):
            cour_idx.setdefault(s.strip().lower(), (name, url))
    links = {}
    for skill in vocab:
        if len(skill) < 3 or skill in GENERIC:
            continue
        rx = re.compile(r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])")
        hit = nptel[nptel["t"].str.contains(rx)]
        if len(hit):  # the skill must be most of the title ("GST" yes, "Analytical Chemistry" for "analytical" no)
            n_skill = len(skill.split())
            hit = hit.assign(cover=n_skill / hit["t"].str.split().map(len)).query("cover >= @MIN_COVER")
            hit = hit.sort_values(["cover", "provider"], ascending=[False, False]).head(1)
        if len(hit):
            r = hit.iloc[0]
            links[skill] = {"provider": f"{r['provider']} ({r['organisation']})" if pd.notna(r["organisation"]) else r["provider"], "title": r["title"], "url": r["url"]}
        elif skill in cour_idx:
            links[skill] = {"provider": "Coursera", "title": cour_idx[skill][0], "url": cour_idx[skill][1]}
    return links


def main() -> None:
    p = load_params()["lookups"]
    prior = education_prior(p["min_edu_postings"])
    (OUT / "education_prior.json").write_text(json.dumps(prior), encoding="utf-8")
    print("education degrees with prior:", list(prior["lift"]))
    pipe = joblib.load(OUT / "model.pkl")
    vocab = list(pipe["features"].named_transformers_["skills"].vocabulary_)
    links = course_links(vocab)
    (OUT / "course_links.json").write_text(json.dumps(links), encoding="utf-8")
    print(f"course links for {len(links)}/{len(vocab)} vocabulary skills")


if __name__ == "__main__":
    main()
