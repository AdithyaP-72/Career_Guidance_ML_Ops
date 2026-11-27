"""Unit tests for Phase 3 – Monitoring, Feedback & Governance.

Tests use only in-memory / tiny synthetic data — no real model files required.
Run with:  uv run --no-sync pytest tests/test_monitoring.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Task 1: Model Monitoring ───────────────────────────────────────────────────

from src.monitoring.monitor import (
    compute_prediction_distribution,
    confidence_stats,
    js_divergence,
    load_prediction_log,
    skill_frequency,
)


def _make_log_df(roles: list[str], confs: list[float], skill_lists: list[list[str]]) -> pd.DataFrame:
    return pd.DataFrame({
        "top1_role": roles,
        "top1_confidence": confs,
        "skills": skill_lists,
    })


def test_compute_prediction_distribution_basic():
    df = _make_log_df(["role_a", "role_a", "role_b"], [0.9, 0.8, 0.5], [[], [], []])
    dist = compute_prediction_distribution(df)
    assert pytest.approx(dist["role_a"], abs=1e-6) == 2 / 3
    assert pytest.approx(dist["role_b"], abs=1e-6) == 1 / 3


def test_compute_prediction_distribution_empty():
    dist = compute_prediction_distribution(pd.DataFrame())
    assert dist == {}


def test_js_divergence_identical():
    p = {"a": 0.5, "b": 0.5}
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-5)


def test_js_divergence_disjoint():
    p = {"a": 1.0}
    q = {"b": 1.0}
    jsd = js_divergence(p, q)
    assert 0.5 < jsd <= 1.0   # Should be high (ln(2) ≈ 0.693 / ln(2) normalised)


def test_js_divergence_partial_overlap():
    p = {"a": 0.6, "b": 0.4}
    q = {"a": 0.3, "b": 0.7}
    jsd = js_divergence(p, q)
    assert 0.0 < jsd < 0.5


def test_confidence_stats_basic():
    df = _make_log_df(["r"] * 5, [0.9, 0.8, 0.5, 0.2, 0.1], [[] for _ in range(5)])
    cs = confidence_stats(df)
    assert cs["min"] == pytest.approx(0.1)
    assert cs["max"] == pytest.approx(0.9)
    assert 0.0 <= cs["low_confidence_pct"] <= 100.0


def test_confidence_stats_all_low():
    df = _make_log_df(["r"] * 3, [0.1, 0.2, 0.15], [[] for _ in range(3)])
    cs = confidence_stats(df)
    assert cs["low_confidence_pct"] == pytest.approx(100.0)


def test_skill_frequency():
    df = _make_log_df(
        ["r", "r", "r"],
        [0.9, 0.8, 0.7],
        [["python", "sql"], ["python", "excel"], ["sql"]],
    )
    top = skill_frequency(df, top_n=5)
    skills = [entry["skill"] for entry in top]
    assert "python" in skills
    assert "sql" in skills
    # python and sql both appear twice
    counts = {e["skill"]: e["count"] for e in top}
    assert counts["python"] == 2
    assert counts["sql"] == 2


def test_load_prediction_log_missing(tmp_path):
    """Should return empty DF when file doesn't exist."""
    df = load_prediction_log(tmp_path / "nonexistent.jsonl")
    assert df.empty


def test_load_prediction_log_reads_jsonl(tmp_path):
    p = tmp_path / "preds.jsonl"
    record = {"timestamp": "2026-01-01T00:00:00", "top1_role": "Data Analyst",
              "top1_confidence": 0.75, "skills": ["python", "sql"]}
    p.write_text(json.dumps(record) + "\n", encoding="utf-8")
    df = load_prediction_log(p)
    assert len(df) == 1
    assert df.iloc[0]["top1_role"] == "Data Analyst"


# ── Task 2: Logger ──────────────────────────────────────────────────────────────

from src.monitoring.logger import _rotate_if_needed, log_prediction


def test_log_prediction_writes_jsonl(tmp_path, monkeypatch):
    """log_prediction should append a valid JSON record."""
    pred_log = tmp_path / "predictions.jsonl"
    # Monkeypatch LOGS_DIR and PRED_LOG in logger
    import src.monitoring.logger as logger_mod
    monkeypatch.setattr(logger_mod, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(logger_mod, "PRED_LOG", pred_log)
    monkeypatch.setattr(logger_mod, "_MODEL_VERSION", "abc12345")

    top3 = [
        {"role": "Data Analyst", "confidence": 0.82},
        {"role": "Business Analyst", "confidence": 0.11},
        {"role": "Data Scientist", "confidence": 0.07},
    ]
    rid = log_prediction(["python", "sql"], 2.0, 5.0, top3, source="test")

    assert pred_log.exists()
    records = [json.loads(line) for line in pred_log.read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 1
    r = records[0]
    assert r["top1_role"] == "Data Analyst"
    assert r["top1_confidence"] == pytest.approx(0.82)
    assert r["skills"] == ["python", "sql"]
    assert r["source"] == "test"
    assert r["request_id"] == rid


def test_log_prediction_multiple_appends(tmp_path, monkeypatch):
    pred_log = tmp_path / "predictions.jsonl"
    import src.monitoring.logger as logger_mod
    monkeypatch.setattr(logger_mod, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(logger_mod, "PRED_LOG", pred_log)
    monkeypatch.setattr(logger_mod, "_MODEL_VERSION", "abc12345")

    top3 = [{"role": "X", "confidence": 0.9}]
    for _ in range(5):
        log_prediction(["python"], None, None, top3, source="test")
    records = [json.loads(l) for l in pred_log.read_text(encoding="utf-8").splitlines() if l]
    assert len(records) == 5


def test_rotate_if_needed(tmp_path):
    """Small file should not be rotated; oversized file should be renamed."""
    p = tmp_path / "predictions.jsonl"
    p.write_bytes(b"x" * 100)
    _rotate_if_needed(p, max_bytes=10_000)  # 100 < 10k → no rotation
    assert p.exists()
    _rotate_if_needed(p, max_bytes=50)      # 100 > 50 → rotate
    assert not p.exists()
    rotated = list(tmp_path.glob("predictions_*T*.jsonl"))
    assert len(rotated) == 1


# ── Task 5: Retraining Trigger ─────────────────────────────────────────────────

from src.monitoring.retrain_trigger import run_checks, verdict


def _make_drift_report(tmp_path: Path, pct_drop: float) -> None:
    (tmp_path / "drift").mkdir(parents=True, exist_ok=True)
    write = {"pct_drop_top3": pct_drop, "drift_alert": "test"}
    (tmp_path / "drift" / "drift_report.json").write_text(
        json.dumps(write), encoding="utf-8"
    )


def _make_monitor_report(tmp_path: Path, jsd: float, low_conf: float, n_req: int) -> None:
    (tmp_path / "monitoring").mkdir(parents=True, exist_ok=True)
    r = {
        "n_logged_requests": n_req,
        "js_divergence_train_vs_live": jsd,
        "drift_alert": "test",
        "confidence_stats": {"low_confidence_pct": low_conf},
    }
    (tmp_path / "monitoring" / "monitor_report.json").write_text(
        json.dumps(r), encoding="utf-8"
    )


def test_verdict_all_pass():
    checks = [
        {"check": "a", "status": "PASS"},
        {"check": "b", "status": "PASS"},
    ]
    v, code = verdict(checks)
    assert code == 0
    assert "NO_RETRAIN" in v


def test_verdict_one_fail():
    checks = [
        {"check": "a", "status": "PASS"},
        {"check": "b", "status": "FAIL"},
    ]
    v, code = verdict(checks)
    assert code == 1
    assert "RECOMMENDED" in v


def test_verdict_two_fails():
    checks = [
        {"check": "a", "status": "FAIL"},
        {"check": "b", "status": "FAIL"},
    ]
    v, code = verdict(checks)
    assert code == 2
    assert "REQUIRED" in v


def test_verdict_skips_dont_count():
    checks = [
        {"check": "a", "status": "SKIP"},
        {"check": "b", "status": "PASS"},
    ]
    v, code = verdict(checks)
    assert code == 0


# ── Task 6: Extended Model Comparison ─────────────────────────────────────────

from src.monitoring.model_comparison import build_markdown_table


def test_build_markdown_table_empty():
    result = build_markdown_table(pd.DataFrame())
    assert "_No data._" in result


def test_build_markdown_table_has_rows():
    df = pd.DataFrame({
        "top3_5skills": [0.80, 0.75],
        "verdict": ["🏆 Champion", "✅ Strong"],
    }, index=pd.Index(["nb", "logreg"], name="model"))
    md = build_markdown_table(df)
    assert "nb" in md
    assert "logreg" in md
    assert "|" in md   # markdown table rows


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v"] + sys.argv[1:])

