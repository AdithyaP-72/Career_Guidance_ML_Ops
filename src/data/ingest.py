"""raw -> tidy parquet: skills as normalised lists, experience as a number, duplicates dropped.

Sources: Naukri 2025 (main), Naukri 2019 (reference / label check), Naukri 2022 (extra training data),
Internshala 2025 (fresher / internship postings, extra training data).
"""
from __future__ import annotations

import html
import re

import pandas as pd

from src.data.common import ROOT, load_aliases, normalise_skills

RAW = ROOT / "data/raw"
OUT = ROOT / "data/interim"


def _title(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).map(html.unescape).str.replace(r"\s+", " ", regex=True).str.strip()


def _min_exp(s) -> float:
    m = re.match(r"\s*(\d+)", str(s))
    return float(m.group(1)) if m else float("nan")


def ingest_2025(aliases) -> pd.DataFrame:
    df = pd.read_excel(RAW / "naukri_2025/indian-job-market-dataset-2025.xlsx")
    return pd.DataFrame({
        "title": _title(df["title"]),
        "company": df["companyName"].fillna("").astype(str),
        "skills": df["tagsAndSkills"].map(lambda x: normalise_skills(x, ",", aliases)),
        "experience": df["minimumExperience"],
        "naukri_area": None,
    })


def ingest_2019(aliases) -> pd.DataFrame:
    df = pd.read_csv(RAW / "naukri_2019/naukri_2019.csv")
    return pd.DataFrame({
        "title": _title(df["Job Title"]),
        "company": "",
        "skills": df["Key Skills"].map(lambda x: normalise_skills(x, "|", aliases)),
        "experience": df["Job Experience Required"].map(_min_exp),
        "naukri_area": df["Functional Area"].astype(str).str.strip(),
        "naukri_role": df["Role"].astype(str).str.strip(),
    })


def ingest_2022(aliases) -> pd.DataFrame:
    df = pd.read_csv(RAW / "naukri_2022/Naukri Jobs Data.csv")
    return pd.DataFrame({
        "title": _title(df["job_post"]),
        "company": df["company"].fillna("").astype(str),
        "skills": df["required_skills"].map(lambda x: normalise_skills(str(x).replace("\n", ","), ",", aliases)),
        "experience": df["exp_required"].map(_min_exp),
        "naukri_area": None,
    })


def ingest_internshala(aliases) -> pd.DataFrame:
    df = pd.read_csv(RAW / "internshala/merged_internships_dataset.csv")
    return pd.DataFrame({
        "title": _title(df["profile"]),
        "company": df["company"].fillna("").astype(str),
        "skills": df["Skills"].map(lambda x: normalise_skills(x, ",", aliases)),
        "experience": 0.0,
        "naukri_area": None,
    })


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["skills"].map(len) > 0].copy()  # postings without skills are unusable
    key = df["title"].str.lower() + "|" + df["company"] + "|" + df["skills"].map(lambda s: ",".join(sorted(s)))
    df = df[~key.duplicated()].reset_index(drop=True)
    df["experience"] = df["experience"].fillna(0).clip(lower=0)
    return df


SOURCES = {2019: ingest_2019, 2022: ingest_2022, 2025: ingest_2025, "internshala": ingest_internshala}


def main() -> None:
    aliases = load_aliases()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in SOURCES.items():
        raw = fn(aliases)
        df = clean(raw)
        print(f"{name}: {len(raw)} raw -> {len(df)} usable postings")
        df.to_parquet(OUT / f"postings_{name}.parquet")


if __name__ == "__main__":
    main()
