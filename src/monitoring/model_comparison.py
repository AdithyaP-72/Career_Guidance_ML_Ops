"""Phase 3 – Task 6: Extended Model Comparison
===============================================
Extends the existing select.py model_comparison.csv with additional columns
useful for governance and monitoring:
  • 2019 reference accuracy (temporal drift proxy)
  • latency and size
  • a "verdict" column (Champion / Strong / Acceptable / Weak)
  • rank on each metric

Also writes a refreshed Markdown summary table to reports/model_comparison.md,
ready for the presentation and the MODEL_CARD.

Reads:
    reports/metrics/*.json       (per-model metrics from DVC train stages)
    reports/model_comparison.csv (existing comparison from select.py)
    params.yaml → select.metric

Writes:
    reports/model_comparison_extended.csv   (full comparison with verdicts)
    reports/model_comparison.md             (Markdown table for README / presentation)

Usage:
    uv run python -m src.monitoring.model_comparison
"""

from __future__ import annotations

import pandas as pd

from src.utils import REPORTS, get_logger, load_params, read_json, write_json

log = get_logger("model_comparison")

COMPARISON_CSV = REPORTS / "model_comparison.csv"
EXT_CSV = REPORTS / "model_comparison_extended.csv"
COMP_MD = REPORTS / "model_comparison.md"


METRICS_DISPLAY = [
    ("top3_5skills",          "Top-3 (5 skills)",     ".1%"),
    ("top1_5skills",          "Top-1 (5 skills)",     ".1%"),
    ("top5_5skills",          "Top-5 (5 skills)",     ".1%"),
    ("macro_f1_5skills",      "Macro-F1 (5 skills)",  ".3f"),
    ("family_top1_5skills",   "Family Top-1",         ".1%"),
    ("ref2019_top3_5skills",  "Ref-2019 Top-3",       ".1%"),
    ("top3_full",             "Top-3 (full)",         ".1%"),
    ("fit_seconds",           "Fit (s)",              ".1f"),
    ("model_size_mb",         "Size (MB)",            ".2f"),
    ("predict_ms_per_1k",     "Latency (ms/1k)",      ".1f"),
]


def _verdict(row: pd.Series, champion: str, metric: str) -> str:
    if row.name == champion:
        return "🏆 Champion"
    v = row.get(metric, 0)
    if v >= 0.78:
        return "✅ Strong"
    elif v >= 0.70:
        return "🟡 Acceptable"
    else:
        return "🔴 Weak"


def build_extended_comparison() -> pd.DataFrame:
    p = load_params()
    models = p["train"]["models"]
    metric = p["select"]["metric"]

    rows = {}
    for m in models:
        path = REPORTS / "metrics" / f"{m}.json"
        if path.exists():
            rows[m] = read_json(path)
    if not rows:
        log.warning("No metrics found in reports/metrics/ — nothing to compare")
        return pd.DataFrame()

    df = pd.DataFrame(rows).T
    df.index.name = "model"

    # Read champion from metrics.json
    champion = "unknown"
    if (REPORTS / "metrics.json").exists():
        champion = read_json(REPORTS / "metrics.json").get("best_model", "unknown")

    df["verdict"] = df.apply(lambda row: _verdict(row, champion, metric), axis=1)
    df["drift_drop_top3_pct"] = ((df.get("top3_5skills", 0) - df.get("ref2019_top3_5skills", 0))
                                  / df.get("top3_5skills", 1).clip(1e-9) * 100).round(1)

    # Rank each metric
    for col, _, _ in METRICS_DISPLAY:
        if col in df.columns:
            ascending = col in ("fit_seconds", "model_size_mb", "predict_ms_per_1k")
            df[f"rank_{col}"] = df[col].rank(ascending=ascending, method="min").astype(int)

    df = df.sort_values(metric, ascending=False)
    return df


def build_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    display_cols = [(col, label, fmt) for col, label, fmt in METRICS_DISPLAY if col in df.columns]
    display_cols = [("verdict", "Verdict", "s")] + display_cols + [("drift_drop_top3_pct", "Drift drop (%)", ".1f")]

    header = "| Model | " + " | ".join(label for _, label, _ in display_cols) + " |"
    sep = "|---|" + "|".join("---" for _ in display_cols) + "|"
    rows = []
    for model, row in df.iterrows():
        cells = []
        for col, _, fmt in display_cols:
            val = row.get(col, "—")
            if isinstance(val, float):
                cells.append(format(val, fmt) if fmt != "s" else str(val))
            else:
                cells.append(str(val))
        rows.append(f"| **{model}** | " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def main() -> None:
    df = build_extended_comparison()
    if df.empty:
        log.warning("No data to compare — aborting")
        return

    # Write extended CSV
    df.to_csv(EXT_CSV)
    log.info("Extended comparison → %s", EXT_CSV)

    # Write Markdown
    p = load_params()
    champion = read_json(REPORTS / "metrics.json").get("best_model", "?") if (REPORTS / "metrics.json").exists() else "?"
    metric = p["select"]["metric"]
    md = f"""# Model Comparison Report

**Selection metric:** `{metric}` (top-3 accuracy on 5-skill test inputs — closest to real usage)
**Champion:** `{champion}`

## Baselines

{build_markdown_table(df)}

> **Drift drop (%)**: how much top-3 accuracy drops when evaluating on the 2019 reference set.
> A large drop means the model is less reliable for older job-posting patterns.
>
> Generated automatically by `src.monitoring.model_comparison`.
"""
    COMP_MD.write_text(md, encoding="utf-8")
    log.info("Markdown comparison → %s", COMP_MD)


if __name__ == "__main__":
    main()

