"""Decide which candidate roles become model classes by comparing every snapshot.

For each candidate role (from configs/role_rules.yaml) we count usable postings
(≥ data.min_skills skills; for 2017, which has no skills, any titled posting) in
2017, 2019, 2022 and 2025. We also record which Naukri 2019 `Role` label its postings
most often carry, as an independent sanity check of what the role really is.

Decision (thresholds in params.yaml → taxonomy):
    class / stable    ≥ min_train_postings in the train snapshot (2025) AND
                      ≥ min_reference_postings in the reference snapshot (2019).
                      These are usable for the 2019 → 2025 drift experiment.
    class / emerging  enough in 2025 but too rare in 2019 (new or growing roles).
                      The product recommends them; the 2019 model can't know them.
    excluded          too rare in 2025 to learn. Postings are dropped from training.

Outputs:
    data/processed/classes.json            the class list (+ status) used downstream
    reports/taxonomy/role_support.csv      the full comparison table, for review
    reports/taxonomy_summary.json          headline numbers (DVC metric)
"""

from __future__ import annotations

import pandas as pd

from src.utils import INTERIM, PROCESSED, REPORTS, get_logger, load_params, write_json

log = get_logger("taxonomy")


def usable(df: pd.DataFrame, min_skills: int) -> pd.Series:
    """Postings we can learn from. 2017 has no skill tags, so it only counts titles."""
    return (df["n_skills"] >= min_skills) | (df["snapshot"] == 2017)


def role_support(df: pd.DataFrame, min_skills: int) -> pd.DataFrame:
    lab = df[usable(df, min_skills) & df["role"].notna()]
    counts = lab.pivot_table(index=["role", "family"], columns="snapshot", values="title", aggfunc="size", fill_value=0)
    counts.columns = [f"n_{c}" for c in counts.columns]
    totals = lab.groupby("snapshot").size()
    for snap in totals.index:
        counts[f"share_{snap}"] = (counts[f"n_{snap}"] / totals[snap]).round(5)

    # What does Naukri itself call the 2019 postings we put in each role?
    nr = lab[(lab["snapshot"] == 2019) & lab["naukri_role"].notna()]
    top = nr.groupby("role")["naukri_role"].value_counts(normalize=True).groupby(level=0).head(1).reset_index()
    top.columns = ["role", "naukri_2019_top_role", "naukri_2019_top_role_share"]
    top["naukri_2019_top_role_share"] = top["naukri_2019_top_role_share"].round(3)
    return counts.reset_index().merge(top, on="role", how="left")


def main() -> None:
    p = load_params()
    t, min_skills = p["taxonomy"], p["data"]["min_skills"]
    train, ref = t["train_snapshot"], t["reference_snapshot"]

    df = pd.read_parquet(INTERIM / "labelled.parquet")
    sup = role_support(df, min_skills)

    is_class = sup[f"n_{train}"] >= t["min_train_postings"]
    in_ref = sup[f"n_{ref}"] >= t["min_reference_postings"]
    sup["status"] = "excluded"
    sup.loc[is_class & in_ref, "status"] = "stable"
    sup.loc[is_class & ~in_ref, "status"] = "emerging"
    ref_share = sup[f"share_{ref}"].where(sup[f"share_{ref}"] > 0)  # avoid dividing by 0
    sup["share_change_ref_to_train"] = (sup[f"share_{train}"] / ref_share).round(2)
    sup["status"] = pd.Categorical(sup["status"], ["stable", "emerging", "excluded"], ordered=True)
    sup = sup.sort_values(["status", f"n_{train}"], ascending=[True, False])

    out = REPORTS / "taxonomy"
    out.mkdir(parents=True, exist_ok=True)
    sup.to_csv(out / "role_support.csv", index=False)

    classes = sup[sup["status"] != "excluded"]
    write_json(
        {
            "train_snapshot": train,
            "reference_snapshot": ref,
            "classes": [{"role": r.role, "family": r.family, "status": r.status} for r in classes.itertuples()],
        },
        PROCESSED / "classes.json",
    )

    train_usable = df[(df["snapshot"] == train) & usable(df, min_skills)]
    summary = {
        "candidate_roles": len(sup),
        "classes": len(classes),
        "stable": int((sup["status"] == "stable").sum()),
        "emerging": int((sup["status"] == "emerging").sum()),
        "excluded": int((sup["status"] == "excluded").sum()),
        "families_covered": int(classes["family"].nunique()),
        f"share_{train}_usable_postings_in_classes": round(float(train_usable["role"].isin(classes["role"]).mean()), 4),
        f"share_{ref}_usable_postings_in_stable_classes": round(
            float(
                df[(df["snapshot"] == ref) & usable(df, min_skills)]["role"]
                .isin(classes.loc[classes["status"] == "stable", "role"])
                .mean()
            ),
            4,
        ),
    }
    write_json(summary, REPORTS / "taxonomy_summary.json")
    log.info("%s", summary)


if __name__ == "__main__":
    main()
