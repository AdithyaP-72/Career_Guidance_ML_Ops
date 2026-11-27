"""Phase 3 – Task 4: Drift / Degradation Experiment
====================================================
Formally measures temporal drift by evaluating the champion model on:
  1. The 2025 test set (current distribution)
  2. The 2019 reference set (older distribution)

Then computes per-role accuracy changes and flags roles with large drops.

Writes:
    reports/drift/drift_report.json         – summary with drift scores & alerts
    reports/drift/per_role_drift.csv        – per-role accuracy change (2025 vs 2019)
    reports/drift/skill_distribution_drift.json  – skill prevalence shift across snapshots

Usage:
    uv run python -m src.monitoring.drift_experiment
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import top_k_accuracy_score

from src.models.evaluate import inputs, read_split, score
from src.utils import MODELS, PROCESSED, REPORTS, ROOT, get_logger, read_json, write_json

log = get_logger("drift_experiment")

DRIFT_DIR = REPORTS / "drift"
INK, MUTED_INK, ACCENT = "#1f1f1f", "#5c5c5c", "#3987e5"


# ── helpers ────────────────────────────────────────────────────────────────────


def per_role_accuracy(pipe, df: pd.DataFrame, skills_col: str = "skills_eval") -> pd.Series:
    """Top-1 accuracy per role (fraction of test rows where top-1 == true role)."""
    from src.features.build import EXPERIENCE, SKILLS
    X = pd.DataFrame({"skills": df[skills_col].map(list).values,
                      "experience": df["experience"].values if "experience" in df else [(None, None)] * len(df)})
    # Use the same input format as model expects
    from src.models.evaluate import inputs as _inputs
    proba = pipe.predict_proba(_inputs(df, skills_col))
    classes = pipe.classes_
    pred = classes[proba.argmax(axis=1)]
    y = df["role"].to_numpy()
    result = {}
    for role in np.unique(y):
        mask = y == role
        result[role] = float((pred[mask] == role).mean())
    return pd.Series(result, name="accuracy")


def skill_prevalence(df: pd.DataFrame) -> pd.Series:
    """Fraction of postings mentioning each skill."""
    from src.features.build import SKILLS
    counter: dict[str, int] = {}
    n = len(df)
    for skills in df[SKILLS].map(list):
        for s in skills:
            counter[s] = counter.get(s, 0) + 1
    return pd.Series({s: c / n for s, c in counter.items()}, name="prevalence")


# ── main experiment ────────────────────────────────────────────────────────────


def main() -> None:
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    # Load champion model
    model_path = MODELS / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Champion model not found at {model_path}. Run 'uv run dvc repro select' first.")
    pipe = joblib.load(model_path)
    classes = pipe.classes_
    log.info("Loaded champion model: %d classes", len(classes))

    # Load test and reference splits
    test = read_split("test")
    ref = read_split("reference_2019")
    family_of = {r: f for r, f in zip(test["role"], test["family"], strict=False)}
    family_of.update({r: f for r, f in zip(ref["role"], ref["family"], strict=False)})
    log.info("Test set: %d rows | Ref 2019: %d rows", len(test), len(ref))

    # Score on both splits
    m_test, _ = score(pipe, test, "skills_eval", family_of)
    m_ref, _ = score(pipe, ref, "skills_eval", family_of)
    log.info("Test: top3=%.4f | Ref2019: top3=%.4f", m_test["top3"], m_ref["top3"])

    # Overall drift metrics
    drift_metrics = {
        "test_2025": {f"top{k}": m_test[f"top{k}"] for k in (1, 3, 5)},
        "ref_2019": {f"top{k}": m_ref[f"top{k}"] for k in (1, 3, 5)},
        "drift_top1": round(m_ref["top1"] - m_test["top1"], 4),
        "drift_top3": round(m_ref["top3"] - m_test["top3"], 4),
        "drift_top5": round(m_ref["top5"] - m_test["top5"], 4),
        "pct_drop_top3": round((m_test["top3"] - m_ref["top3"]) / max(m_test["top3"], 1e-9) * 100, 2),
    }

    # Drift alert
    pct_drop = drift_metrics["pct_drop_top3"]
    if pct_drop > 20:
        alert = f"CRITICAL – top-3 accuracy dropped {pct_drop:.1f}% from 2025 test to 2019 reference"
    elif pct_drop > 10:
        alert = f"WARNING – top-3 accuracy dropped {pct_drop:.1f}% from 2025 test to 2019 reference"
    else:
        alert = f"OK – top-3 drift is {pct_drop:.1f}% (within acceptable range)"
    drift_metrics["drift_alert"] = alert
    log.info("Drift alert: %s", alert)

    # Per-role accuracy comparison
    # top-1 accuracy for each role in test
    test_proba = pipe.predict_proba(inputs(test, "skills_eval"))
    test_pred = classes[test_proba.argmax(axis=1)]
    test_y = test["role"].to_numpy()

    ref_proba = pipe.predict_proba(inputs(ref, "skills_eval"))
    ref_pred = classes[ref_proba.argmax(axis=1)]
    ref_y = ref["role"].to_numpy()

    shared_roles = sorted(set(test_y) & set(ref_y))
    per_role_rows = []
    for role in shared_roles:
        t_mask = test_y == role
        r_mask = ref_y == role
        t_acc = float((test_pred[t_mask] == role).mean()) if t_mask.sum() > 0 else np.nan
        r_acc = float((ref_pred[r_mask] == role).mean()) if r_mask.sum() > 0 else np.nan
        delta = round(r_acc - t_acc, 4) if not np.isnan(t_acc) and not np.isnan(r_acc) else np.nan
        per_role_rows.append({
            "role": role,
            "acc_2025_test": round(t_acc, 4) if not np.isnan(t_acc) else None,
            "acc_2019_ref": round(r_acc, 4) if not np.isnan(r_acc) else None,
            "delta_2019_minus_2025": delta,
            "n_test": int(t_mask.sum()),
            "n_ref": int(r_mask.sum()),
        })

    per_role_df = pd.DataFrame(per_role_rows).sort_values("delta_2019_minus_2025", ascending=True)
    per_role_df.to_csv(DRIFT_DIR / "per_role_drift.csv", index=False)
    log.info("Written per_role_drift.csv (%d shared roles)", len(per_role_df))

    # Worst degraded roles
    worst = per_role_df.dropna(subset=["delta_2019_minus_2025"]).head(10)
    drift_metrics["worst_degraded_roles"] = [
        {"role": r["role"], "delta": r["delta_2019_minus_2025"]}
        for _, r in worst.iterrows()
    ]

    # Skill distribution drift: fraction of postings containing each skill
    from src.features.build import SKILLS
    test_prev = skill_prevalence(test)
    ref_prev = skill_prevalence(ref)
    shared_skills = sorted(set(test_prev.index) & set(ref_prev.index))
    skill_drift = {
        s: round(float(ref_prev[s]) - float(test_prev[s]), 5)
        for s in shared_skills
    }
    top_emerging = dict(sorted(skill_drift.items(), key=lambda x: x[1], reverse=True)[:20])
    top_declining = dict(sorted(skill_drift.items(), key=lambda x: x[1])[:20])
    write_json(
        {
            "top_20_emerging_skills_since_2019": top_emerging,
            "top_20_declining_skills_since_2019": top_declining,
            "total_skills_in_test": len(test_prev),
            "total_skills_in_ref": len(ref_prev),
            "new_skills_not_in_2019": len(set(test_prev.index) - set(ref_prev.index)),
            "skills_gone_since_2019": len(set(ref_prev.index) - set(test_prev.index)),
        },
        DRIFT_DIR / "skill_distribution_drift.json",
    )
    log.info("Written skill_distribution_drift.json")

    # --- Plot: per-role drift bar chart (bottom 20 most degraded) ---
    plot_df = per_role_df.dropna(subset=["delta_2019_minus_2025"]).head(20)
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(9, max(4, len(plot_df) * 0.42)), dpi=120)
        fig.patch.set_facecolor("white")
        colors = ["#e05" if d < -0.2 else "#f90" if d < 0 else "#2e7" for d in plot_df["delta_2019_minus_2025"]]
        ax.barh(plot_df["role"], plot_df["delta_2019_minus_2025"], color=colors, edgecolor="none", height=0.7)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_xlabel("Accuracy change (2019 ref − 2025 test)", color=MUTED_INK, fontsize=9)
        ax.set_title("Per-Role Accuracy Drift (worst 20)", loc="left", fontsize=11, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0, labelsize=7, colors=INK)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(DRIFT_DIR / "per_role_drift.png", facecolor="white", bbox_inches="tight", dpi=130)
        plt.close(fig)
        log.info("Written per_role_drift.png")

    write_json(drift_metrics, DRIFT_DIR / "drift_report.json")
    log.info("Drift report → reports/drift/drift_report.json  |  %s", alert)


if __name__ == "__main__":
    main()

