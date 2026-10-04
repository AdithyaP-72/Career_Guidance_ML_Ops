"""Profiling report per snapshot → reports/profiling/naukri_<year>.html.

Uses fg-data-profiling (`import data_profiling`), the continuation of ydata-profiling
(formerly pandas-profiling), by the same team. ydata-profiling 4.18 is deprecated, and
it breaks with current setuptools (it imports the removed `pkg_resources`).

The skills list column can't be profiled directly, so it is replaced by `n_skills`
and a `skills_text` string. Deeper, project-specific EDA lives in notebooks/01_eda.ipynb.
"""

from __future__ import annotations

import pandas as pd
from data_profiling import ProfileReport

from src.utils import INTERIM, REPORTS, get_logger, load_params

log = get_logger("profile")


def main() -> None:
    p = load_params()["profiling"]
    df = pd.read_parquet(INTERIM / "postings.parquet")
    out = REPORTS / "profiling"
    out.mkdir(parents=True, exist_ok=True)

    for snap in p["snapshots"]:
        g = df[df["snapshot"] == snap].drop(columns=["snapshot"])
        g = g.assign(skills_text=g["skills"].map("; ".join)).drop(columns=["skills"])
        g = g.dropna(axis=1, how="all")  # columns a snapshot doesn't have
        report = ProfileReport(g, title=f"Naukri {snap} postings", minimal=p["minimal"], progress_bar=False)
        report.to_file(out / f"naukri_{snap}.html")
        log.info("wrote reports/profiling/naukri_%s.html (%d rows)", snap, len(g))


if __name__ == "__main__":
    main()
