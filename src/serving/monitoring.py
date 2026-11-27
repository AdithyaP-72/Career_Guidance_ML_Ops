"""Bounded per-process telemetry for the single-worker inference service."""

from __future__ import annotations

import hmac
import os
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from fastapi import HTTPException, Request

from src.models.drift import Observation, compare_observations, vocabulary_hash
from src.utils import read_json

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5)
CONFIDENCE_BUCKETS = tuple(i / 10 for i in range(1, 11))
WINDOW_NAMES = ("live", "real_2025_demo", "synthetic_2026_demo")
STATUSES = ("unavailable", "insufficient", "normal", "warning", "critical")


@dataclass(frozen=True, slots=True)
class Event:
    at: float
    status: int
    latency: float
    observation: Observation | None


class Histogram:
    def __init__(self, boundaries: tuple):
        self.boundaries = boundaries
        self.buckets = [0] * len(boundaries)
        self.count = 0
        self.total = 0.0

    def record(self, value: float) -> None:
        for index, boundary in enumerate(self.boundaries):
            self.buckets[index] += value <= boundary
        self.count += 1
        self.total += value

    def exposition(self, name: str) -> list[str]:
        return [
            f"# TYPE {name} histogram",
            *(f'{name}_bucket{{le="{le}"}} {count}' for le, count in zip(self.boundaries, self.buckets, strict=True)),
            f'{name}_bucket{{le="+Inf"}} {self.count}',
            f"{name}_count {self.count}",
            f"{name}_sum {self.total}",
        ]


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Monitor:
    """Keep bounded, redacted events. Compute PSI at scrape/read time, not per request."""

    def __init__(self, reference: dict | None, version: str, algorithm: str, size: int, age: int, refresh: int):
        if not (1 <= size <= 10_000 and 1 <= age <= 86_400 and 1 <= refresh <= age):
            raise ValueError("invalid monitoring window or refresh interval")
        if reference and reference["min_samples"] > size:
            raise ValueError("the monitoring window cannot hold the configured minimum sample count")
        self.reference = reference
        self.version, self.algorithm = version, algorithm
        self.events: deque[Event] = deque(maxlen=size)
        self.age, self.refresh = age, refresh
        self.lock = Lock()
        self.analysis_lock = Lock()
        self.started_at = time.time()
        self.window_started_at = self.started_at
        self.generation = 0
        self.window = "live"
        self.cache = None
        self.status = "unavailable" if reference is None else "insufficient"
        self.status_since = self.started_at
        self.last_success = 0.0
        self.total = self.errors = self.successes = 0
        self.role_counts: Counter[str] = Counter()
        self.latency = Histogram(LATENCY_BUCKETS)
        self.confidence = Histogram(CONFIDENCE_BUCKETS)

    def current_generation(self) -> int:
        with self.lock:
            return self.generation

    def reset(self, window: str) -> dict:
        with self.lock:
            self.events.clear()
            self.cache = None
            self.last_success = 0.0
            self.status, self.status_since = ("insufficient" if self.reference else "unavailable"), time.time()
            self.generation += 1
            self.window_started_at = time.time()
            self.window = window
            return {"window": self.window, "window_started_at": self.window_started_at}

    def _expire(self, now: float) -> None:
        while self.events and self.events[0].at < now - self.age:
            self.events.popleft()

    def record(self, generation: int, status: int, latency: float, observation: Observation | None) -> None:
        with self.lock:
            now = time.time()
            self.total += 1
            self.errors += status >= 400
            self.successes += 200 <= status < 300
            self.latency.record(latency)
            if (
                200 <= status < 300
                and observation
                and observation.confidence is not None
                and observation.role is not None
            ):
                self.confidence.record(observation.confidence)
                self.role_counts[observation.role] += 1
            if generation == self.generation:
                self._expire(now)
                self.events.append(Event(now, status, latency, observation))

    def report(self) -> dict:
        # Serialise analyses, but release the event lock while comparing populations.
        with self.analysis_lock:
            while True:
                now = time.time()
                with self.lock:
                    if self.cache and now - self.cache["generated_at_seconds"] < self.refresh:
                        return self.cache
                    self._expire(now)
                    records = [
                        event.observation
                        for event in self.events
                        if 200 <= event.status < 300 and event.observation and event.observation.confidence is not None
                    ]
                    generation = self.generation
                    window = self.window
                    started = self.window_started_at
                if self.reference:
                    result = compare_observations(self.reference, records)
                else:
                    result = {
                        "status": "unavailable",
                        "overall_psi": None,
                        "sample_count": len(records),
                        "generated_at": datetime.fromtimestamp(now, UTC).isoformat(),
                        "generated_at_seconds": now,
                        "features": {},
                        "top_drifted": [],
                    }
                result = result | {"window": window, "window_started_at": started, "model_version": self.version}
                with self.lock:
                    if generation != self.generation:
                        continue
                    if result["status"] != self.status:
                        self.status = result["status"]
                        self.status_since = result["generated_at_seconds"]
                    result["status_since_seconds"] = self.status_since
                    self.cache = result
                    if result["overall_psi"] is not None:
                        self.last_success = result["generated_at_seconds"]
                    return result

    def metrics(self) -> str:
        report = self.report()
        with self.lock:
            self._expire(time.time())
            known, unknown = 0, 0
            for event in self.events:
                if event.observation is not None:
                    known += len(event.observation.skill_ids)
                    unknown += event.observation.unknown_count
            # Missing traffic is NaN, not zero confidence or zero unknown-skill rate.
            unknown_rate = unknown / (known + unknown) if known + unknown else "NaN"
            confidence_average = self.confidence.total / self.confidence.count if self.confidence.count else "NaN"
            lines = [
                "# TYPE career_prediction_requests_total counter",
                f"career_prediction_requests_total {self.total}",
                "# TYPE career_prediction_errors_total counter",
                f"career_prediction_errors_total {self.errors}",
                "# TYPE career_prediction_successes_total counter",
                f"career_prediction_successes_total {self.successes}",
                f"career_prediction_error_rate {self.errors / self.total if self.total else 'NaN'}",
                f"career_prediction_confidence_average {confidence_average}",
                f"career_unknown_skill_rate {unknown_rate}",
                "# TYPE career_predicted_role_total counter",
                *(f'career_predicted_role_total{{role="{_label(role)}"}} {n}' for role, n in self.role_counts.items()),
                *self.latency.exposition("career_prediction_latency_seconds"),
                *self.confidence.exposition("career_prediction_confidence"),
                f'career_model_info{{version="{_label(self.version)}",algorithm="{_label(self.algorithm)}"}} 1',
                f"career_process_started_timestamp_seconds {self.started_at}",
                f"career_drift_last_success_timestamp_seconds {self.last_success}",
            ]
        window = _label(report["window"])
        lines.extend(
            [
                f'career_drift_window_info{{window="{window}"}} 1',
                f"career_drift_samples {report['sample_count']}",
                f"career_drift_status_since_timestamp_seconds {report['status_since_seconds']}",
                f"career_drift_psi {report['overall_psi'] if report['overall_psi'] is not None else 'NaN'}",
                *(f'career_drift_status{{status="{name}"}} {int(report["status"] == name)}' for name in STATUSES),
                *(
                    f'career_drift_threshold{{severity="{severity}"}} {value}'
                    for severity, value in report.get("thresholds", {}).items()
                ),
            ]
        )
        for feature, values in report["features"].items():
            name = _label(feature)
            lines.append(f'career_drift_feature_psi{{feature="{name}"}} {values["psi"]}')
            for label, ref, current in zip(values["labels"], values["reference"], values["current"], strict=True):
                key = f'feature="{name}",bin="{_label(label)}"'
                lines.append(f'career_drift_distribution{{{key},window="reference"}} {ref}')
                lines.append(f'career_drift_distribution{{{key},window="current"}} {current}')
        return "\n".join(lines) + "\n"


class PredictionTelemetry:
    """ASGI middleware sees failures before request-body validation and from the handler."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/predict" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        monitor = getattr(scope["app"].state, "monitor", None)
        if monitor is None:
            return await self.app(scope, receive, send)
        generation = monitor.current_generation()
        status = 500
        started = time.perf_counter()

        async def capture(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture)
        finally:
            captured = scope.get("state", {}).get("phase3_observation")
            monitor.record(generation, status, time.perf_counter() - started, captured)


def configure_monitor(bundle: Path, recommender, min_known: int) -> tuple[str | None, Monitor | None]:
    path = os.environ.get("MONITORING_TOKEN_FILE")
    token = Path(path).read_text(encoding="utf-8").strip() if path else os.environ.get("MONITORING_TOKEN", "").strip()
    if not token:
        return None, None
    ref_path = Path(os.environ.get("MONITORING_REFERENCE_PATH", bundle / "drift_reference.json"))
    reference = read_json(ref_path) if ref_path.is_file() else None
    if reference and (
        reference["schema_version"] != 1
        or reference["min_known"] != min_known
        or reference["vocabulary_sha256"] != vocabulary_hash(recommender.vocab)
        or reference["roles"] != list(recommender.model.classes_)
        or reference["model_run_id"] != recommender.meta.get("run_id")
    ):
        raise ValueError("monitoring reference is incompatible with the serving model or minimum skills setting")
    return token, Monitor(
        reference,
        str(recommender.meta["version"]),
        str(recommender.meta.get("algorithm") or "unknown"),
        int(os.environ.get("MONITORING_WINDOW_SIZE", "1000")),
        int(os.environ.get("MONITORING_WINDOW_SECONDS", "900")),
        int(os.environ.get("MONITORING_REFRESH_SECONDS", "5")),
    )


def require_monitor(request: Request) -> Monitor:
    monitor = request.app.state.monitor
    if monitor is None:
        raise HTTPException(404)
    supplied = request.headers.get("Authorization", "").encode("utf-8")
    expected = f"Bearer {request.app.state.monitoring_token}".encode()
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(401, "Monitoring credentials required", headers={"WWW-Authenticate": "Bearer"})
    return monitor
