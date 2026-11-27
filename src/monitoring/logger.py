"""Phase 3 – Task 2: Structured Prediction Logger
==================================================
Writes every live prediction request (skills, experience, top-3 results,
confidence) as a newline-delimited JSON record to logs/predictions.jsonl.

This file is both:
  - Imported by API/app code:  from src.monitoring.logger import log_prediction
  - Run directly to seed a demo log from test data: uv run python -m src.monitoring.logger

Log file location:  <project_root>/logs/predictions.jsonl
The log rotates automatically when it exceeds MAX_LOG_BYTES (default 10 MB).

Fields per record:
    timestamp          ISO-8601 UTC
    request_id         UUID4
    skills             list[str]
    experience_min     float | null
    experience_max     float | null
    top1_role          str
    top1_confidence    float  (0–1)
    top3_roles         list[{role, confidence}]
    model_version      str    (joblib md5 or "unknown")
    source             str    (e.g. "api", "demo", "test")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.utils import ROOT, get_logger

log = get_logger("logger")

LOGS_DIR = ROOT / "logs"
PRED_LOG = LOGS_DIR / "predictions.jsonl"
MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB before rotation


# ── log rotation ───────────────────────────────────────────────────────────────


def _rotate_if_needed(path: Path, max_bytes: int = MAX_LOG_BYTES) -> None:
    """Rotate the log by renaming it with a timestamp suffix when it exceeds max_bytes."""
    if path.exists() and path.stat().st_size > max_bytes:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rotated = path.with_name(f"{path.stem}_{ts}{path.suffix}")
        path.rename(rotated)
        log.info("Log rotated → %s", rotated.name)


# ── model fingerprint ──────────────────────────────────────────────────────────


def _model_md5(model_path: Path) -> str:
    if not model_path.exists():
        return "unknown"
    h = hashlib.md5()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]   # short fingerprint


_MODEL_VERSION: str | None = None  # cached so we don't re-hash on every request


def _get_model_version() -> str:
    global _MODEL_VERSION
    if _MODEL_VERSION is None:
        _MODEL_VERSION = _model_md5(ROOT / "models" / "model.joblib")
    return _MODEL_VERSION


# ── public API ─────────────────────────────────────────────────────────────────


def log_prediction(
    skills: list[str],
    experience_min: float | None,
    experience_max: float | None,
    top3: list[dict[str, Any]],
    source: str = "api",
    request_id: str | None = None,
) -> str:
    """Append one prediction record to the JSONL log.

    Args:
        skills:          normalised input skill list
        experience_min:  min years of experience (or None)
        experience_max:  max years of experience (or None)
        top3:            list of dicts with keys 'role' and 'confidence', length 1–3
        source:          where the request came from ('api', 'demo', 'test', etc.)
        request_id:      caller-supplied UUID; auto-generated if None

    Returns:
        The request_id used for this record.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _rotate_if_needed(PRED_LOG)

    if not top3:
        log.warning("log_prediction called with empty top3 — skipping")
        return request_id or str(uuid4())

    rid = request_id or str(uuid4())
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": rid,
        "skills": list(skills),
        "experience_min": experience_min,
        "experience_max": experience_max,
        "top1_role": top3[0]["role"],
        "top1_confidence": round(float(top3[0]["confidence"]), 4),
        "top3_roles": [{"role": r["role"], "confidence": round(float(r["confidence"]), 4)} for r in top3[:3]],
        "model_version": _get_model_version(),
        "source": source,
    }

    with open(PRED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return rid


# ── demo / seed ────────────────────────────────────────────────────────────────


def _seed_demo_log(n: int = 200) -> None:
    """Seed a demo prediction log from the test split, so monitor.py has data to read.

    Uses the champion model (models/model.joblib) and test data (data/processed/test.parquet).
    Writes n records to logs/predictions.jsonl.
    """
    import joblib
    import numpy as np
    import pandas as pd

    from src.features.build import EXPERIENCE, SKILLS
    from src.models.evaluate import inputs
    from src.utils import MODELS, PROCESSED

    log.info("Seeding demo log with %d records from test split …", n)
    model = joblib.load(MODELS / "model.joblib")
    df = pd.read_parquet(PROCESSED / "test.parquet").head(n)
    for col in (SKILLS, "skills_eval"):
        if col in df:
            df[col] = df[col].map(list)

    classes = model.classes_
    # Build the input DataFrame the same way evaluate.py does (uses 'skills' + 'exp_min' columns)
    X = inputs(df, "skills_eval")  # DataFrame with 'skills' and 'exp_min' columns
    proba_all = model.predict_proba(X)

    for i, (_, row) in enumerate(df.iterrows()):
        skills_in = list(row["skills_eval"])
        exp_min_val = float(row["exp_min"]) if pd.notna(row["exp_min"]) else None
        exp_max_val = float(row["exp_max"]) if pd.notna(row["exp_max"]) else None

        proba = proba_all[i]
        top_idx = np.argsort(proba)[::-1][:3]
        top3 = [{"role": classes[j], "confidence": float(proba[j])} for j in top_idx]
        log_prediction(skills_in, exp_min_val, exp_max_val, top3, source="demo")

    log.info("Demo log seeded → %s", PRED_LOG)


if __name__ == "__main__":
    _seed_demo_log()
    log.info("Done. Run:  uv run python -m src.monitoring.monitor")
