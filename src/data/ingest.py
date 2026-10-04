"""Raw Naukri snapshots → one tidy table: data/interim/postings.parquet.

Every snapshot is mapped to the same columns:
    snapshot, posting_id, title, company, location, skills (list), n_skills,
    exp_min, exp_max, salary_min, salary_max,
    functional_area (Naukri's label, 2017/2019 only), naukri_role (2019 only),
    education_raw (2017 only)

Skills are split, normalised and alias-mapped (configs/skill_aliases.yaml).
Exact duplicate postings (same title, company, skills and experience) are dropped
per snapshot. Counts go to reports/ingest_summary.json.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import INTERIM, RAW, REPORTS, get_logger, load_aliases, parse_experience, parse_skills, write_json

log = get_logger("ingest")

COLUMNS = [
    "snapshot",
    "posting_id",
    "title",
    "company",
    "location",
    "skills",
    "n_skills",
    "exp_min",
    "exp_max",
    "salary_min",
    "salary_max",
    "functional_area",
    "naukri_role",
    "education_raw",
]


def _experience(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(series.map(parse_experience).tolist(), columns=["exp_min", "exp_max"], index=series.index)


def load_2025(aliases) -> pd.DataFrame:
    raw = pd.read_excel(RAW / "naukri_2025" / "naukri_2025.xlsx")
    df = pd.DataFrame(
        {
            "posting_id": raw["jobId"].astype(str),
            "title": raw["title"],
            "company": raw["companyName"],
            "location": raw["location"],
            "skills": raw["tagsAndSkills"].map(lambda s: parse_skills(s, r",", aliases)),
            "exp_min": raw["minimumExperience"],
            "exp_max": raw["maximumExperience"],
            # 0 means "Not disclosed" in this dataset
            "salary_min": raw["minimumSalary"].replace(0, np.nan),
            "salary_max": raw["maximumSalary"].replace(0, np.nan),
        }
    )
    return df.assign(snapshot=2025)


def load_2019(aliases) -> pd.DataFrame:
    raw = pd.read_csv(RAW / "naukri_2019" / "naukri_2019.csv")
    df = pd.DataFrame(
        {
            "posting_id": raw["Uniq Id"],
            "title": raw["Job Title"],
            "company": pd.NA,  # this snapshot has no company column
            "location": raw["Location"],
            "skills": raw["Key Skills"].map(lambda s: parse_skills(s, r"\|", aliases)),
            "functional_area": raw["Functional Area"],
            "naukri_role": raw["Role"],
        }
    )
    return df.join(_experience(raw["Job Experience Required"])).assign(snapshot=2019)


def load_2022(aliases) -> pd.DataFrame:
    raw = pd.read_csv(RAW / "naukri_2022" / "naukri_2022.csv")
    df = pd.DataFrame(
        {
            "posting_id": "2022-" + raw.index.astype(str),  # no id column in the source
            "title": raw["job_post"],
            "company": raw["company"],
            "location": raw["job_location"],
            "skills": raw["required_skills"].map(lambda s: parse_skills(s, r"[\n|]", aliases)),
        }
    )
    return df.join(_experience(raw["exp_required"])).assign(snapshot=2022)


def load_2017(aliases) -> pd.DataFrame:
    raw = pd.read_csv(RAW / "naukri_2017" / "naukri_2017.csv")
    df = pd.DataFrame(
        {
            "posting_id": raw["uniq_id"],
            "title": raw["jobtitle"],
            "company": raw["company"],
            "location": raw["joblocation_address"],
            # This snapshot has no skill tags: its `skills` column is really Naukri's functional area.
            "skills": [[] for _ in range(len(raw))],
            "functional_area": raw["skills"],
            "education_raw": raw["education"],
        }
    )
    return df.join(_experience(raw["experience"])).assign(snapshot=2017)


LOADERS = {2017: load_2017, 2019: load_2019, 2022: load_2022, 2025: load_2025}


def main() -> None:
    aliases = load_aliases()
    frames, summary = [], {}
    for year, loader in LOADERS.items():
        df = loader(aliases).reindex(columns=COLUMNS)
        df["title"] = df["title"].astype("string").str.strip()
        df = df[df["title"].notna() & (df["title"] != "")]
        df["n_skills"] = df["skills"].map(len)
        rows_raw = len(df)

        key = pd.DataFrame(
            {
                "t": df["title"].str.lower(),
                "c": df["company"].astype("string").str.lower(),
                "s": df["skills"].map(lambda s: "|".join(sorted(s))),
                "e1": df["exp_min"],
                "e2": df["exp_max"],
            }
        )
        df = df[~key.duplicated()]

        all_skills = df["skills"].explode().dropna()
        summary[str(year)] = {
            "rows_raw": rows_raw,
            "rows": len(df),
            "duplicates_removed": rows_raw - len(df),
            "share_no_skills": round(float((df["n_skills"] == 0).mean()), 4),
            "median_skills": float(df["n_skills"].median()),
            "unique_skills": int(all_skills.nunique()),
            "share_exp_missing": round(float(df["exp_min"].isna().mean()), 4),
        }
        log.info("%s: %s", year, summary[str(year)])
        frames.append(df)

    postings = pd.concat(frames, ignore_index=True)
    for col in ["posting_id", "company", "location", "functional_area", "naukri_role", "education_raw"]:
        postings[col] = postings[col].astype("string")
    postings["snapshot"] = postings["snapshot"].astype("int16")

    INTERIM.mkdir(parents=True, exist_ok=True)
    postings.to_parquet(INTERIM / "postings.parquet", index=False)
    write_json(summary, REPORTS / "ingest_summary.json")
    log.info("wrote %d postings -> data/interim/postings.parquet", len(postings))


if __name__ == "__main__":
    main()
