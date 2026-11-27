"""Phase 3 – Task 5: Retraining Trigger
========================================
Checks drift metrics and confidence statistics against configured thresholds
and prints a clear GO / NO-GO verdict with reasons.

Reads:
    reports/drift/drift_report.json        (from drift_experiment.py)
    reports/monitoring/monitor_report.json (from monitor.py)
    params.yaml → monitoring section

Writes:
    reports/retraining_trigger.json  – verdict + all checks (DVC metric)

Exit code:
    0  – no retraining needed  (CI: green)
    1  – retraining recommended (CI: yellow / warning)
    2  – retraining REQUIRED   (CI: red / fail gate)

Usage:
    uv run python -m src.monitoring.retrain_trigger
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.utils import REPORTS, get_logger, load_params, read_json, write_json

log = get_logger("retrain_trigger")

RETRAIN_REPORT = REPORTS / "retraining_trigger.json"


# ── thresholds (read from params.yaml → monitoring) ───────────────────────────

DEFAULT_THRESHOLDS = {
    "max_pct_drop_top3": 15.0,       # % drop in top-3 accuracy before GO
    "max_js_divergence": 0.15,       # JSD threshold for live-vs-training drift
    "max_low_confidence_pct": 25.0,  # % of requests below 0.30 confidence
    "min_logged_requests": 50,       # need at least this many logs to evaluate live drift
}


def load_thresholds() -> dict:
    p = load_params()
    return {**DEFAULT_THRESHOLDS, **p.get("monitoring", {}).get("thresholds", {})}


# ── checks ─────────────────────────────────────────────────────────────────────


def run_checks(thresholds: dict) -> list[dict]:
    checks = []

    # 1. Temporal drift (2025 test vs 2019 reference)
    drift_path = REPORTS / "drift" / "drift_report.json"
    if drift_path.exists():
        dr = read_json(drift_path)
        pct_drop = dr.get("pct_drop_top3", 0.0)
        max_drop = thresholds["max_pct_drop_top3"]
        status = "FAIL" if pct_drop > max_drop else "PASS"
        checks.append({
            "check": "temporal_drift_top3",
            "value": round(pct_drop, 2),
            "threshold": max_drop,
            "unit": "% accuracy drop (2025→2019)",
            "status": status,
            "detail": dr.get("drift_alert", ""),
        })
    else:
        checks.append({
            "check": "temporal_drift_top3",
            "status": "SKIP",
            "detail": "reports/drift/drift_report.json not found — run drift_experiment first",
        })

    # 2. Live distribution drift (JSD)
    monitor_path = REPORTS / "monitoring" / "monitor_report.json"
    if monitor_path.exists():
        mr = read_json(monitor_path)
        n_req = mr.get("n_logged_requests", 0)
        min_req = thresholds["min_logged_requests"]
        if n_req < min_req:
            checks.append({
                "check": "live_distribution_drift",
                "status": "SKIP",
                "detail": f"Only {n_req} logged requests (need ≥ {min_req} to evaluate)",
            })
        else:
            jsd = mr.get("js_divergence_train_vs_live")
            if jsd is None:
                checks.append({
                    "check": "live_distribution_drift",
                    "status": "SKIP",
                    "detail": "JSD not computed (empty prediction log)",
                })
            else:
                max_jsd = thresholds["max_js_divergence"]
                status = "FAIL" if jsd > max_jsd else "PASS"
                checks.append({
                    "check": "live_distribution_drift",
                    "value": round(jsd, 6),
                    "threshold": max_jsd,
                    "unit": "Jensen–Shannon divergence",
                    "status": status,
                    "detail": mr.get("drift_alert", ""),
                })

        # 3. Low-confidence rate
        cs = mr.get("confidence_stats", {})
        low_conf = cs.get("low_confidence_pct", 0.0)
        max_low = thresholds["max_low_confidence_pct"]
        status = "FAIL" if low_conf > max_low else "PASS"
        checks.append({
            "check": "low_confidence_rate",
            "value": round(low_conf, 2),
            "threshold": max_low,
            "unit": "% requests with confidence < 0.30",
            "status": status,
            "detail": f"{low_conf:.1f}% of requests had top-1 confidence < 0.30",
        })
    else:
        checks.append({
            "check": "live_distribution_drift",
            "status": "SKIP",
            "detail": "reports/monitoring/monitor_report.json not found — run monitor first",
        })
        checks.append({
            "check": "low_confidence_rate",
            "status": "SKIP",
            "detail": "monitor_report.json not found",
        })

    return checks


def verdict(checks: list[dict]) -> tuple[str, int]:
    """Return (verdict_str, exit_code): 0=OK, 1=RECOMMENDED, 2=REQUIRED."""
    statuses = [c["status"] for c in checks]
    fail_count = statuses.count("FAIL")
    if fail_count == 0:
        return "NO_RETRAIN – all checks passed", 0
    elif fail_count == 1:
        return "RETRAIN_RECOMMENDED – 1 check failed; consider scheduling a retrain", 1
    else:
        return f"RETRAIN_REQUIRED – {fail_count} checks failed; retrain immediately", 2


def main() -> None:
    thresholds = load_thresholds()
    log.info("Retraining trigger – thresholds: %s", thresholds)

    checks = run_checks(thresholds)
    v, code = verdict(checks)

    report = {
        "verdict": v,
        "exit_code": code,
        "thresholds": thresholds,
        "checks": checks,
        "retrain_command": "uv run dvc repro train select" if code == 2 else None,
    }
    write_json(report, RETRAIN_REPORT)

    for c in checks:
        status_sym = "✅" if c["status"] == "PASS" else ("⚠️" if c["status"] == "SKIP" else "❌")
        val = f"  value={c['value']}" if "value" in c else ""
        log.info("%s  %-35s %s%s", status_sym, c["check"], c["status"], val)

    log.info("\n🚦 Verdict: %s (exit code %d)", v, code)
    log.info("Report → %s", RETRAIN_REPORT)
    sys.exit(code)


if __name__ == "__main__":
    main()

