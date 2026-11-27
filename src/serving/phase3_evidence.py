"""Expose bounded metrics from read-only historical and experimental reports."""

import json
import math
import re
from datetime import datetime
from pathlib import Path

_SAFE = re.compile(r"[A-Za-z0-9_.@ /-]{1,100}\Z")


def _read(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.loads(stream.read(2_000_000))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def evidence_metrics(directory: Path) -> str:
    """Export fixed fields only. Missing reports omit their series rather than implying a healthy result."""
    rows = []

    def emit(name: str, value: float, **labels: str) -> None:
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            return
        if not all(isinstance(label, str) and _SAFE.fullmatch(label) for label in labels.values()):
            return
        suffix = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}" if labels else ""
        rows.append(f"career_phase3_{name}{suffix} {float(value)}\n")

    def timestamp(value: str | None, source: str) -> None:
        if value:
            try:
                emit("evidence_timestamp_seconds", datetime.fromisoformat(value).timestamp(), source=source)
            except (TypeError, ValueError):
                pass

    pipeline = _read(directory / "pipeline.json")
    if pipeline:
        timestamp(pipeline.get("generated_at"), "repository_snapshot")
        identity = {
            k: pipeline.get(k, "unknown") for k in ("algorithm", "git_commit", "run_id", "train_md5", "model_md5")
        }
        emit("model_lineage_info", 1, **identity, dataset=pipeline.get("dataset", "unknown"))
        for stage in pipeline.get("pipeline", [])[:32]:
            emit("pipeline_stage_recorded", 1, stage=stage, source="dvc_lock")
        for name, check in list(pipeline.get("validation", {}).items())[:32]:
            emit("quality_pass", int(check.get("passed") is True), check=name, source="committed_report")
        for alias, model in pipeline.get("registry", {}).items():
            emit(
                "registry_alias_info",
                1,
                alias=alias,
                version=model["version"],
                algorithm=model.get("algorithm") or "unknown",
            )

    training = _read(directory / "retraining" / "latest.json")
    if training:
        timestamp(training.get("completed_at") or training.get("updated_at"), "synthetic_retraining")
        emit(
            "retraining_info",
            1,
            status=training.get("status") or "unknown",
            decision=training.get("decision") or "pending",
            trigger=training.get("trigger") or "unknown",
        )
        for stage, state in list(training.get("stages", {}).items())[:32]:
            emit("retraining_stage_info", 1, stage=stage, status=state.get("status", "unknown"))
        for name in ("champion", "challenger"):
            model = training.get(name) or {}
            for split in ("synthetic_eval", "real_2025", "synthetic_holdout"):
                metrics = (model.get("metrics") or {}).get(split) or {}
                for metric in ("top1", "top3", "macro_f1", "ece"):
                    emit("comparison_score", metrics.get(metric), model=name, split=split, metric=metric)
            if model:
                emit(
                    "experimental_model_info",
                    1,
                    model=name,
                    run_id=model.get("run_id") or "unknown",
                    version=str(model.get("registry_version") or "pending"),
                    algorithm=model.get("algorithm") or "unknown",
                )
        for name, check in list((training.get("comparison_checks") or {}).items())[:32]:
            emit("quality_pass", int(check.get("passed") is True), check=name, source="synthetic_comparison")
        for i, action in enumerate((training.get("audit") or [])[-12:]):
            try:
                recorded_at = datetime.fromisoformat(action.get("at", "")).timestamp()
            except (TypeError, ValueError, OSError):
                continue
            emit(
                "audit_info",
                recorded_at * 1000,
                sequence=str(i + 1),
                action=action.get("action", "unknown"),
                approved_by=action.get("approved_by") or "none",
                from_version=str(action.get("from_version") or "none"),
                to_version=str(action.get("to_version") or "none"),
            )

    replay = _read(directory / "demo" / "latest.json")
    if replay:
        timestamp(replay.get("completed_at"), "labelled_api_replay")
        for window in ("real_2025_demo", "synthetic_2026_demo"):
            window_report = (replay.get("windows") or {}).get(window) or {}
            drift = window_report.get("drift") or {}
            emit("replay_score", drift.get("overall_psi"), window=window, metric="max_psi")
            for metric in ("top1_accepted", "top3_accepted", "top3_all_submitted", "p95_http_latency_ms"):
                emit("replay_score", window_report.get(metric), window=window, metric=metric)
    return "".join(rows)
