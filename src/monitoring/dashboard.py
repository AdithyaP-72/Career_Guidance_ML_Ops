"""Phase 3 – Task 3: Performance Dashboard
==========================================
Generates a self-contained HTML performance dashboard that shows:
  • Model comparison bar chart (all baselines on top-3 accuracy)
  • Confidence distribution histogram from prediction logs
  • Per-role accuracy heatmap (from the champion model's per-role CSV)
  • Live vs training distribution drift chart
  • Key metric summary table

Output:
    reports/performance_dashboard.html

Usage:
    uv run python -m src.monitoring.dashboard
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import REPORTS, ROOT, get_logger, read_json

log = get_logger("dashboard")

MONITOR_DIR = REPORTS / "monitoring"
DASHBOARD_HTML = REPORTS / "performance_dashboard.html"

# Colour palette (matches existing evaluate.py)
SEQUENTIAL = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED_INK, ACCENT = "#1f1f1f", "#5c5c5c", "#3987e5"


# ── chart helpers ──────────────────────────────────────────────────────────────


def _fig_to_b64(fig) -> str:
    """Convert a Matplotlib figure to a base64-encoded PNG data URI."""
    import base64
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def chart_model_comparison() -> str:
    """Bar chart: all baselines sorted by top-3 accuracy (5-skill inputs)."""
    comp_path = REPORTS / "model_comparison.csv"
    if not comp_path.exists():
        log.warning("model_comparison.csv not found — skipping model comparison chart")
        return ""
    df = pd.read_csv(comp_path, index_col="model").sort_values("top3_5skills", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.55)), dpi=120)
    fig.patch.set_facecolor("white")
    colors = [ACCENT if m == df["top3_5skills"].idxmax() else "#9ec5f4" for m in df.index]
    bars = ax.barh(df.index, df["top3_5skills"], color=colors, edgecolor="none", height=0.6)
    for bar, val in zip(bars, df["top3_5skills"]):
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=8, color=INK)
    ax.set_xlim(0, min(1.0, df["top3_5skills"].max() + 0.08))
    ax.set_xlabel("Top-3 accuracy (5 skills)", color=MUTED_INK, fontsize=9)
    ax.set_title("Model Comparison – Top-3 Accuracy", loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0, labelsize=8, colors=INK)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_confidence_distribution() -> str:
    """Histogram of top-1 confidence scores from prediction log."""
    pred_log = ROOT / "logs" / "predictions.jsonl"
    if not pred_log.exists():
        log.warning("predictions.jsonl not found — skipping confidence chart")
        return ""
    rows = []
    with open(pred_log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return ""
    conf = [r.get("top1_confidence", 0) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
    fig.patch.set_facecolor("white")
    ax.hist(conf, bins=20, range=(0, 1), color=ACCENT, edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(conf), color="#e05", linewidth=1.5, linestyle="--",
               label=f"Mean {np.mean(conf):.2f}")
    ax.axvline(0.3, color="#f90", linewidth=1, linestyle=":", label="Low-confidence threshold (0.30)")
    ax.set_xlabel("Top-1 confidence", color=MUTED_INK, fontsize=9)
    ax.set_ylabel("# requests", color=MUTED_INK, fontsize=9)
    ax.set_title(f"Confidence Distribution (n={len(conf)} requests)", loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, labelsize=8, colors=INK)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_drift() -> str:
    """Bar chart comparing training vs live prediction distribution (top-20 roles by delta)."""
    dist_path = MONITOR_DIR / "prediction_dist.csv"
    if not dist_path.exists():
        log.warning("prediction_dist.csv not found — skipping drift chart")
        return ""
    df = pd.read_csv(dist_path).head(20)  # already sorted by |delta|
    if df.empty:
        return ""
    fig, ax = plt.subplots(figsize=(9, max(3, len(df) * 0.5)), dpi=120)
    fig.patch.set_facecolor("white")
    y = np.arange(len(df))
    h = 0.35
    ax.barh(y - h / 2, df["train_share"], height=h, color="#9ec5f4", label="Training", edgecolor="none")
    ax.barh(y + h / 2, df["pred_share"], height=h, color=ACCENT, label="Live predictions", edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(df["role"], fontsize=7, color=INK)
    ax.set_xlabel("Share of predictions", color=MUTED_INK, fontsize=9)
    ax.set_title("Top 20 Roles: Training vs Live Prediction Distribution", loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0, labelsize=8, colors=INK)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    ax.legend(fontsize=8, frameon=False)
    ax.invert_yaxis()
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_per_role_accuracy() -> str:
    """Scatter/bar of per-role F1 from the champion model's classification report."""
    # Find the best model's per_role CSV
    metrics_json = REPORTS / "metrics.json"
    if not metrics_json.exists():
        log.warning("reports/metrics.json not found — skipping per-role chart")
        return ""
    best = read_json(metrics_json).get("best_model", "complement_nb")
    per_role_csv = REPORTS / "eval" / best / "per_role_5skills.csv"
    if not per_role_csv.exists():
        log.warning("per_role_5skills.csv not found for %s — skipping", best)
        return ""
    df = pd.read_csv(per_role_csv, index_col="role")
    df = df[df.index != "accuracy"].dropna(subset=["f1-score"])
    df = df.sort_values("f1-score", ascending=True).tail(30)  # top 30 roles
    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.32)), dpi=120)
    fig.patch.set_facecolor("white")
    colors = ["#e05" if v < 0.4 else ACCENT if v >= 0.7 else "#9ec5f4" for v in df["f1-score"]]
    ax.barh(df.index, df["f1-score"], color=colors, edgecolor="none", height=0.65)
    ax.set_xlabel("F1-score (5-skill inputs)", color=MUTED_INK, fontsize=9)
    ax.set_title(f"Per-Role F1 – Top 30 (champion: {best})", loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0, labelsize=7, colors=INK)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.5)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── HTML assembly ──────────────────────────────────────────────────────────────


def _img_section(title: str, b64: str, caption: str = "") -> str:
    if not b64:
        return f"<section><h2>{title}</h2><p class='warn'>⚠ No data available yet.</p></section>\n"
    return (
        f"<section><h2>{title}</h2>"
        f"<img src='{b64}' alt='{title}' style='max-width:100%;border-radius:6px;'>"
        f"{'<p class=caption>' + caption + '</p>' if caption else ''}"
        f"</section>\n"
    )


def build_summary_table() -> str:
    """HTML table of key metrics from reports/metrics.json."""
    metrics_json = REPORTS / "metrics.json"
    if not metrics_json.exists():
        return "<p class='warn'>reports/metrics.json not found.</p>"
    m = read_json(metrics_json)
    bm = m.get("best_model_metrics", {})
    rows = [
        ("Champion model", m.get("best_model", "—")),
        ("Top-3 accuracy (5 skills)", f"{bm.get('top3_5skills', 0):.1%}"),
        ("Top-1 accuracy (5 skills)", f"{bm.get('top1_5skills', 0):.1%}"),
        ("Macro F1 (5 skills)", f"{bm.get('macro_f1_5skills', 0):.3f}"),
        ("Ref-2019 Top-3 (5 skills)", f"{bm.get('ref2019_top3_5skills', 0):.1%}"),
        ("Drift (2025→2019 drop)", f"{bm.get('top3_5skills', 0) - bm.get('ref2019_top3_5skills', 0):.1%}"),
        ("Latency (ms / 1k)", f"{bm.get('predict_ms_per_1k', 0):.1f}"),
        ("Model size (MB)", f"{bm.get('model_size_mb', 0):.2f}"),
        ("Margin over profile matching", f"{m.get('margin_over_profile_matching', 0):.1%}"),
    ]
    # Add monitor stats if available
    monitor_json = MONITOR_DIR / "monitor_report.json"
    if monitor_json.exists():
        mr = read_json(monitor_json)
        rows.append(("Logged live requests", str(mr.get("n_logged_requests", 0))))
        js = mr.get("js_divergence_train_vs_live")
        if js is not None:
            rows.append(("JS divergence (train vs live)", f"{js:.4f}"))
        cs = mr.get("confidence_stats", {})
        if cs:
            rows.append(("Mean live confidence", f"{cs.get('mean', 0):.2%}"))
            rows.append(("Low-confidence requests", f"{cs.get('low_confidence_pct', 0):.1f}%"))

    html_rows = "".join(
        f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>" for k, v in rows
    )
    return f"<table class='summary'><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{html_rows}</tbody></table>"


def build_dashboard() -> None:
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)

    comp_b64 = chart_model_comparison()
    conf_b64 = chart_confidence_distribution()
    drift_b64 = chart_drift()
    role_b64 = chart_per_role_accuracy()
    summary_html = build_summary_table()

    # Read monitor alert if available
    alert_html = ""
    monitor_json = MONITOR_DIR / "monitor_report.json"
    if monitor_json.exists():
        mr = read_json(monitor_json)
        alert = mr.get("drift_alert", "")
        if alert:
            level = "critical" if "CRITICAL" in alert else ("warn" if "WARNING" in alert else "ok")
            alert_html = f"<div class='alert {level}'>🔔 {alert}</div>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Career Guidance – Performance Dashboard</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #f5f7fb; color: #1f1f1f; }}
  header {{ background: #184f95; color: #fff; padding: 20px 32px; }}
  header h1 {{ margin: 0; font-size: 1.5rem; font-weight: 700; }}
  header p  {{ margin: 4px 0 0; font-size: 0.85rem; opacity: .8; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 24px 20px; }}
  section {{ background: #fff; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;
             box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  h2 {{ font-size: 1rem; font-weight: 600; color: #184f95; margin: 0 0 12px; }}
  .caption {{ font-size: 0.75rem; color: #5c5c5c; margin: 6px 0 0; }}
  .warn {{ color: #c62828; font-size: 0.85rem; }}
  .alert {{ border-radius: 6px; padding: 10px 16px; margin-bottom: 18px; font-size: 0.9rem; font-weight: 500; }}
  .alert.critical {{ background: #fdecea; color: #c62828; border-left: 4px solid #c62828; }}
  .alert.warn     {{ background: #fff8e1; color: #e65100; border-left: 4px solid #e65100; }}
  .alert.ok       {{ background: #e8f5e9; color: #2e7d32; border-left: 4px solid #2e7d32; }}
  table.summary {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
  table.summary th {{ background: #f0f4fc; color: #184f95; padding: 8px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #cde2fb; }}
  table.summary td {{ padding: 7px 12px; border-bottom: 1px solid #eef0f5; }}
  table.summary tr:last-child td {{ border-bottom: none; }}
</style>
</head>
<body>
<header>
  <h1>🎯 Career Guidance MLOps – Performance Dashboard</h1>
  <p>Auto-generated by <code>src.monitoring.dashboard</code> · Tracks champion model health &amp; live usage</p>
</header>
<main>
  {alert_html}
  <section>
    <h2>📊 Key Metrics Summary</h2>
    {summary_html}
  </section>
  {_img_section("🏆 Model Comparison (Baselines)", comp_b64,
                "Top-3 accuracy on 5-skill test inputs. Blue = champion.")}
  {_img_section("📈 Confidence Distribution (Live Requests)", conf_b64,
                "Distribution of top-1 confidence scores from prediction logs. Requests below 0.30 are low-confidence.")}
  {_img_section("🔄 Training vs Live Prediction Distribution (Top 20 Roles)", drift_b64,
                "Roles where live predictions diverge most from training distribution.")}
  {_img_section("🎯 Per-Role F1 Score (Champion Model, Top 30 Roles)", role_b64,
                "F1 on 5-skill inputs. Red = F1 < 0.40, blue = ≥ 0.70.")}
  <section style="font-size:0.78rem;color:#5c5c5c;">
    <strong>How to refresh:</strong>
    <code>uv run python -m src.monitoring.logger</code> (seed logs) →
    <code>uv run python -m src.monitoring.monitor</code> (compute stats) →
    <code>uv run python -m src.monitoring.dashboard</code> (regenerate HTML)
  </section>
</main>
</body>
</html>
"""
    DASHBOARD_HTML.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_HTML.write_text(html, encoding="utf-8")
    log.info("Dashboard written → %s", DASHBOARD_HTML)


def main() -> None:
    build_dashboard()


if __name__ == "__main__":
    main()

