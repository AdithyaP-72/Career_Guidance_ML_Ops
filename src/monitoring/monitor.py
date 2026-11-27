"""Phase 3 – Task 1: Model Monitoring
======================================
Tracks the health of live predictions by computing distributional statistics
over prediction logs and comparing them against the training baseline.

Writes:
    reports/monitoring/monitor_report.json   – distributional stats & drift alerts
    reports/monitoring/prediction_dist.csv   – per-role prediction counts & shares

Usage (standalone):
    uv run python -m src.monitoring.monitor

The script reads from:
    logs/predictions.jsonl     (written by src.monitoring.logger)
    models/model.joblib        (the champion model)
    data/processed/train.parquet  (training distribution baseline)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils import MODELS, PROCESSED, REPORTS, ROOT, get_logger, write_json

log = get_logger("monitor")

LOGS_DIR = ROOT / "logs"
MONITOR_DIR = REPORTS / "monitoring"
PRED_LOG = LOGS_DIR / "predictions.jsonl"


# ── helpers ────────────────────────────────────────────────────────────────────


def load_prediction_log(path: Path = PRED_LOG) -> pd.DataFrame:
    """Read the JSONL prediction log into a DataFrame. Returns empty DF if no log yet."""
    if not path.exists() or path.stat().st_size == 0:
        log.warning("No prediction log found at %s — returning empty DataFrame", path)
        return pd.DataFrame(columns=["timestamp", "skills", "experience_min", "experience_max",
                                      "top1_role", "top1_confidence", "top3_roles"])
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def training_role_distribution(train_path: Path = PROCESSED / "train.parquet") -> dict[str, float]:
    """Return per-role share in the training set."""
    df = pd.read_parquet(train_path, columns=["role"])
    counts = df["role"].value_counts(normalize=True)
    return counts.to_dict()


def compute_prediction_distribution(log_df: pd.DataFrame) -> dict[str, float]:
    """Return per-role share from recent prediction logs."""
    if log_df.empty or "top1_role" not in log_df.columns:
        return {}
    counts = log_df["top1_role"].value_counts(normalize=True)
    return counts.to_dict()


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen–Shannon divergence between two role distributions (0 = identical, 1 = fully disjoint).

    Uses the union of all keys; missing entries are treated as 0.
    """
    all_keys = sorted(set(p) | set(q))
    p_vec = np.array([p.get(k, 0.0) for k in all_keys], dtype=float)
    q_vec = np.array([q.get(k, 0.0) for k in all_keys], dtype=float)
    # Avoid division-by-zero by adding tiny epsilon
    p_vec = np.where(p_vec == 0, 1e-12, p_vec)
    q_vec = np.where(q_vec == 0, 1e-12, q_vec)
    m = 0.5 * (p_vec + q_vec)
    js = 0.5 * np.sum(p_vec * np.log(p_vec / m)) + 0.5 * np.sum(q_vec * np.log(q_vec / m))
    return float(np.clip(js, 0.0, 1.0))


def confidence_stats(log_df: pd.DataFrame) -> dict:
    """Mean, median, min, max confidence of top-1 predictions."""
    if log_df.empty or "top1_confidence" not in log_df.columns:
        return {}
    conf = pd.to_numeric(log_df["top1_confidence"], errors="coerce").dropna()
    if conf.empty:
        return {}
    return {
        "mean": round(float(conf.mean()), 4),
        "median": round(float(conf.median()), 4),
        "min": round(float(conf.min()), 4),
        "max": round(float(conf.max()), 4),
        "low_confidence_pct": round(float((conf < 0.3).mean() * 100), 2),
    }


def skill_frequency(log_df: pd.DataFrame, top_n: int = 20) -> list[dict]:
    """Top-N skills seen in prediction requests."""
    if log_df.empty or "skills" not in log_df.columns:
        return []
    counter: Counter = Counter()
    for entry in log_df["skills"]:
        if isinstance(entry, list):
            counter.update(entry)
        elif isinstance(entry, str):
            counter.update(json.loads(entry))
    return [{"skill": s, "count": c} for s, c in counter.most_common(top_n)]


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)

    log_df = load_prediction_log()
    n_requests = len(log_df)
    log.info("Loaded %d prediction log entries", n_requests)

    train_dist = training_role_distribution()
    pred_dist = compute_prediction_distribution(log_df)

    js_div = js_divergence(train_dist, pred_dist) if pred_dist else None
    conf = confidence_stats(log_df)
    top_skills = skill_frequency(log_df)

    # Drift alert: JSD > 0.1 is a warning, > 0.2 is critical
    drift_alert = None
    if js_div is not None:
        if js_div > 0.2:
            drift_alert = "CRITICAL – prediction distribution has shifted significantly from training"
        elif js_div > 0.1:
            drift_alert = "WARNING – mild drift detected between prediction and training distributions"
        else:
            drift_alert = "OK – prediction distribution is similar to training"

    # Per-role distribution table
    all_roles = sorted(set(train_dist) | set(pred_dist))
    dist_rows = [
        {
            "role": r,
            "train_share": round(train_dist.get(r, 0.0), 4),
            "pred_share": round(pred_dist.get(r, 0.0), 4),
            "delta": round(pred_dist.get(r, 0.0) - train_dist.get(r, 0.0), 4),
        }
        for r in all_roles
    ]
    dist_df = pd.DataFrame(dist_rows).sort_values("delta", key=abs, ascending=False)
    dist_df.to_csv(MONITOR_DIR / "prediction_dist.csv", index=False)
    log.info("Written prediction_dist.csv (%d roles)", len(dist_df))

    report = {
        "n_logged_requests": n_requests,
        "js_divergence_train_vs_live": round(js_div, 6) if js_div is not None else None,
        "drift_alert": drift_alert,
        "confidence_stats": conf,
        "top_20_skills_in_requests": top_skills,
        "note": "JSD > 0.1 → mild drift warning; JSD > 0.2 → critical drift. Run uv run python -m src.monitoring.monitor after collecting requests.",
    }
    write_json(report, MONITOR_DIR / "monitor_report.json")
    log.info("Monitor report written → reports/monitoring/monitor_report.json")
    if drift_alert:
        log.info("Drift status: %s (JSD=%.4f)", drift_alert, js_div or 0)


if __name__ == "__main__":
    main()

