"""Scoring a trained pipeline. Used by train.py (every baseline) and drift_check.py.

Headline metric: top3_5skills, i.e. top-3 accuracy when each test posting is cut to 5
skills (`skills_eval`), because that's what users actually type. Also reported:
top-1/top-5, macro-F1, the family implied by the top role, the same on full postings,
and the same on the 2019 reference set.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (DVC stages, CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, f1_score, top_k_accuracy_score

from src.features.build import EXPERIENCE, SKILLS
from src.utils import PROCESSED

TOP_K = (1, 3, 5)

# Single-hue sequential ramp (light → dark blue) for magnitude; near-zero recedes to the surface.
SEQUENTIAL = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED_INK = "#1f1f1f", "#5c5c5c"


def read_split(name: str) -> pd.DataFrame:
    """data/processed/<name>.parquet with list columns as Python lists."""
    df = pd.read_parquet(PROCESSED / f"{name}.parquet")
    for col in (SKILLS, "skills_eval"):
        if col in df:
            df[col] = df[col].map(list)
    return df


def inputs(df: pd.DataFrame, skills_col: str = SKILLS) -> pd.DataFrame:
    """The two columns the model reads, optionally swapping in another skills column."""
    return pd.DataFrame({SKILLS: df[skills_col].values, EXPERIENCE: df[EXPERIENCE].values})


def score(pipe, df: pd.DataFrame, skills_col: str, family_of: dict[str, str]):
    proba = pipe.predict_proba(inputs(df, skills_col))
    classes = pipe.classes_
    pred = classes[proba.argmax(axis=1)]
    y = df["role"].to_numpy()
    out = {f"top{k}": float(top_k_accuracy_score(y, proba, k=k, labels=classes)) for k in TOP_K}
    out["macro_f1"] = float(f1_score(y, pred, average="macro", labels=np.unique(y), zero_division=0))
    out["family_top1"] = float(np.mean([family_of[p] == f for p, f in zip(pred, df["family"], strict=True)]))
    return out, pred


def latency_ms_per_1k(pipe, df: pd.DataFrame, skills_col: str = "skills_eval", repeats: int = 3) -> float:
    batch = inputs(df.head(1000), skills_col)
    best = min(_timed(pipe.predict_proba, batch) for _ in range(repeats))
    return round(best * 1000 * 1000 / len(batch), 2)


def _timed(fn, arg) -> float:
    t = time.perf_counter()
    fn(arg)
    return time.perf_counter() - t


def evaluate(pipe, test: pd.DataFrame, ref: pd.DataFrame | None, family_of: dict[str, str], out_dir: Path) -> dict:
    """All metrics as a flat dict + review files in out_dir (per-role report, confusions, chart)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    m5, pred5 = score(pipe, test, "skills_eval", family_of)
    mfull, _ = score(pipe, test, SKILLS, family_of)
    metrics = {f"{k}_5skills": v for k, v in m5.items()} | {f"{k}_full": v for k, v in mfull.items()}
    if ref is not None and len(ref):
        mref, _ = score(pipe, ref, "skills_eval", family_of)
        metrics |= {f"ref2019_{k}_5skills": v for k, v in mref.items()}
    metrics["predict_ms_per_1k"] = latency_ms_per_1k(pipe, test)

    report = pd.DataFrame(classification_report(test["role"], pred5, output_dict=True, zero_division=0)).T
    report.index.name = "role"
    report.round(4).to_csv(out_dir / "per_role_5skills.csv")

    conf = pd.crosstab(test["role"].rename("true_role"), pd.Series(pred5, name="predicted_role", index=test.index))
    pairs = conf.stack().rename("n").reset_index()
    pairs = pairs[pairs["true_role"] != pairs["predicted_role"]].sort_values("n", ascending=False).head(40)
    pairs.to_csv(out_dir / "top_confusions_5skills.csv", index=False)

    pred_family = pd.Series([family_of[p] for p in pred5], index=test.index, name="predicted_family")
    fam = pd.crosstab(test["family"].rename("true_family"), pred_family, normalize="index")
    fam = fam.reindex(index=fam.index, columns=fam.index, fill_value=0.0)
    fam.round(4).to_csv(out_dir / "family_confusion_5skills.csv")
    plot_family_confusion(fam, out_dir / "family_confusion_5skills.png")
    return {k: round(v, 4) for k, v in metrics.items()}


def plot_family_confusion(fam: pd.DataFrame, path: Path) -> None:
    """Row-normalised family confusion matrix: where does each true family's postings go?"""
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL)
    n = len(fam)
    fig, ax = plt.subplots(figsize=(12, 10.5), dpi=130)
    fig.patch.set_facecolor("white")
    mesh = ax.pcolormesh(fam.to_numpy(), cmap=cmap, vmin=0, vmax=1, edgecolors="white", linewidth=2)
    for i in range(n):
        for j in range(n):
            v = fam.iat[i, j]
            if v >= 0.05:  # label only the cells worth reading
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v >= 0.45 else INK,
                )
    ax.set_xticks(np.arange(n) + 0.5, fam.columns, rotation=60, ha="right", fontsize=8, color=INK)
    ax.set_yticks(np.arange(n) + 0.5, fam.index, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted family (of the top-1 role)", color=MUTED_INK, fontsize=9)
    ax.set_ylabel("True family", color=MUTED_INK, fontsize=9)
    ax.set_title("Where each family's test postings land (5-skill inputs, row %)", loc="left", fontsize=11, color=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, colors=MUTED_INK, length=0)
    cbar.set_label("share of the true family's postings", color=MUTED_INK, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
