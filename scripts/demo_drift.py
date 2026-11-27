"""Replay labelled 2025 and synthetic 2026 batches against the same local API."""

import argparse
import math
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import requests

from src.utils import PROCESSED, REPORTS, ROOT, write_json

DRIFT_ALERTS = {"warning": "CareerDataDriftWarning", "critical": "CareerDataDriftCritical"}


def loopback(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise argparse.ArgumentTypeError("Use a local HTTP loopback address for the replay.")
    if parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("Supply the local service origin without credentials or a path.")
    return value.rstrip("/")


def predict(api: str, item: tuple) -> dict:
    skills, experience, label = item
    years = float(experience) if pd.notna(experience) else None
    if years is not None:
        years = min(60.0, max(0.0, years)) if math.isfinite(years) else None
    payload = {"skills": [str(s)[:100] for s in skills[:60]], "experience_years": years, "top_k": 3}
    started = time.perf_counter()
    response = requests.post(api + "/predict", json=payload, timeout=20, allow_redirects=False)
    latency = (time.perf_counter() - started) * 1000
    if response.status_code == 429:
        raise RuntimeError("API replay exceeded the rate limit. Wait one minute or reduce --samples.")
    if response.status_code == 422:
        return {"status": 422, "latency_ms": latency}
    response.raise_for_status()
    body = response.json()
    recs = body["recommendations"]
    return {
        "status": 200,
        "latency_ms": latency,
        "confidence": recs[0]["confidence"],
        "role": recs[0]["role"],
        "top1": recs[0]["role"] == label,
        "top3": label in [r["role"] for r in recs],
        "model": body["model"],
    }


def replay(api: str, source: Path, samples: int, workers: int) -> dict:
    frame = pd.read_parquet(source, columns=["skills", "exp_min", "role"])
    batch = frame.sample(n=min(samples, len(frame)), random_state=42)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda item: predict(api, item), batch.itertuples(index=False, name=None)))
    accepted = [r for r in rows if r["status"] == 200]
    if not accepted:
        raise RuntimeError(f"No predictions were accepted from {source.name}.")
    models = {tuple(sorted(r["model"].items())) for r in accepted}
    if len(models) != 1:
        raise RuntimeError("The prediction model changed within the observation window.")
    return {
        "submitted": len(rows),
        "accepted": len(accepted),
        "rejected_422": len(rows) - len(accepted),
        "model": dict(models.pop()),
        "top1_accepted": float(np.mean([r["top1"] for r in accepted])),
        "top3_accepted": float(np.mean([r["top3"] for r in accepted])),
        "top3_all_submitted": sum(r["top3"] for r in accepted) / len(rows),
        "mean_confidence": float(np.mean([r["confidence"] for r in accepted])),
        "mean_http_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
        "p95_http_latency_ms": float(np.percentile([r["latency_ms"] for r in rows], 95)),
        "top_role_counts": dict(Counter(r["role"] for r in accepted)),
    }


def get_json(url: str, **kwargs) -> dict:
    response = requests.get(url, timeout=10, allow_redirects=False, **kwargs)
    response.raise_for_status()
    return response.json()


def get_drift(api: str, headers: dict, accepted: int) -> dict:
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        report = get_json(api + "/drift", headers=headers)
        if report.get("sample_count", 0) >= accepted:
            return report
        time.sleep(1)
    raise RuntimeError("The monitoring window did not observe the completed replay. Check window size and time limits.")


def alerts(prometheus: str, status: str, wait: int) -> tuple[list[dict], bool]:
    expected = DRIFT_ALERTS.get(status)
    deadline = time.monotonic() + wait
    while True:
        found = get_json(prometheus + "/api/v1/alerts")["data"]["alerts"]
        drift_alerts = [a for a in found if a["labels"]["alertname"].startswith("CareerDataDrift")]
        firing = {a["labels"]["alertname"] for a in drift_alerts if a["state"] == "firing"}
        verified = expected in firing if expected else not drift_alerts
        if verified or time.monotonic() >= deadline:
            return drift_alerts, verified
        time.sleep(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", type=loopback, default="http://127.0.0.1:8000")
    parser.add_argument("--prometheus", type=loopback, default="http://127.0.0.1:9090")
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--alert-wait", type=int, default=45, help="Seconds to wait for evaluation of Prometheus rules")
    parser.add_argument("--api-only", action="store_true", help="Run replay without Prometheus alert verification")
    args = parser.parse_args()
    if not 1 <= args.samples <= 1500 or args.alert_wait < 0:
        parser.error("Use 1–1500 samples per batch and nonnegative alert waiting time.")
    synthetic = ROOT / "data" / "drift" / "synthetic_2026_eval.parquet"
    if not synthetic.exists():
        subprocess.run([sys.executable, "-m", "scripts.generate_2026_drift"], cwd=ROOT, check=True)
    token = os.environ.get("MONITORING_TOKEN") or (ROOT / ".phase3/secrets/monitoring_token").read_text().strip()
    headers = {"Authorization": f"Bearer {token}"}
    started = datetime.now(UTC)
    out = REPORTS / "phase3" / "demo"
    runs = {}
    for window, path in (("real_2025_demo", PROCESSED / "test.parquet"), ("synthetic_2026_demo", synthetic)):
        print(f"Replaying {window} ({args.samples} requested rows)", flush=True)
        reset = requests.post(args.api + "/drift/reset", params={"window": window}, headers=headers, timeout=15)
        reset.raise_for_status()
        result = replay(args.api, path, args.samples, args.workers)
        result["drift"] = get_drift(args.api, headers, result["accepted"])
        result["alerts"], result["alert_verified"] = (
            (None, None) if args.api_only else alerts(args.prometheus, result["drift"]["status"], args.alert_wait)
        )
        runs[window] = result
        print(
            f"{window}: {result['accepted']}/{result['submitted']} accepted, "
            f"top-3 {result['top3_accepted']:.2%}, PSI {result['drift'].get('overall_psi')}, "
            f"status {result['drift']['status']}",
            flush=True,
        )
    if runs["real_2025_demo"]["model"] != runs["synthetic_2026_demo"]["model"]:
        raise RuntimeError("Model identities differ between replay windows.")
    report = {
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "scenario": "synthetic_2026",
        "seed": 42,
        "notes": "Accuracy uses job-posting labels. Rejected requests are included in the all-submitted metric.",
        "windows": runs,
    }
    write_json(report, out / f"replay_{started:%Y%m%dT%H%M%S%fZ}.json")
    write_json(report, out / "latest.json")
    print("Evidence saved under reports/phase3/demo/.")
    if any(run["alert_verified"] is False for run in runs.values()):
        raise SystemExit(
            "Prometheus did not confirm the predicted alert state before the timeout. See the replay report."
        )


if __name__ == "__main__":
    main()
