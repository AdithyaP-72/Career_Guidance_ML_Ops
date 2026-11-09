"""Validate the selected candidate; write review reports without promoting a registry alias."""

from __future__ import annotations

import math

from src.utils import REPORTS, get_logger, load_params, read_json, write_json

log = get_logger("validate")


def quality_checks(metrics: dict, thresholds: dict) -> dict:
    """Fail closed on missing, nonnumeric or nonfinite measurements."""
    checks = {}
    for name, config_key, direction, lower, upper in (
        ("top3_5skills", "min_top3_5skills", "minimum", 0, 1),
        ("ece_5skills", "max_ece_5skills", "maximum", 0, 1),
        ("predict_ms_per_request", "max_predict_ms_per_request", "maximum", 0, math.inf),
    ):
        value, threshold = metrics.get(name), thresholds[config_key]
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        valid = valid and math.isfinite(value) and lower <= value <= upper
        passed = valid and (value >= threshold if direction == "minimum" else value <= threshold)
        checks[name] = {"value": value, "threshold": threshold, "direction": direction, "passed": bool(passed)}
    return checks


def main() -> None:
    selected = read_json(REPORTS / "metrics.json")
    checks = quality_checks(selected["best_model_metrics"], load_params()["validate"])
    passed = all(check["passed"] for check in checks.values())
    status = "PASSED" if passed else "FAILED"
    write_json({"best_model": selected["best_model"], "passed": passed, "checks": checks}, REPORTS / "validation.json")
    rows = [
        f"| {name} | {check['value']} | {check['direction']} {check['threshold']} | "
        f"{'PASS' if check['passed'] else 'FAIL'} |"
        for name, check in checks.items()
    ]
    card = [
        "# Selected model card",
        "",
        f"Model: `{selected['best_model']}`",
        "",
        f"Quality gate: {status}",
        "",
        "| Metric | Measured | Required | Result |",
        "|---|---|---|---|",
        *rows,
        "",
        "Evaluation uses the held-out 2025 test set with five-skill inputs.",
        "ECE uses the top-ranked role's confidence in 10 uniform bins, including confidence 1.",
        "Latency is the median of 20 warmed single-posting predictions, including feature preprocessing.",
        "",
        "Calibration CV runs on augmented training rows; variants of the same source posting may span folds.",
        "Held-out test postings stay separate from training and calibration.",
        "",
        f"MLflow run: `{selected.get('mlflow_run_id')}`",
        f"MLflow model URI: `{selected.get('mlflow_model_uri')}`",
        "",
        "The candidate is not registered or promoted by this stage. Registry promotion follows PR merge.",
        "",
    ]
    (REPORTS / "model_card.md").write_text("\n".join(card), encoding="utf-8")
    log.info("Quality gate: %s", status)
    if not passed:
        raise SystemExit("Selected candidate failed the quality gate; see reports/validation.json")


if __name__ == "__main__":
    main()
