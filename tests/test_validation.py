import numpy as np
import pytest

from src.models import validate
from src.models.evaluate import expected_calibration_error
from src.models.validate import quality_checks
from src.utils import read_json, write_json

THRESHOLDS = {"min_top3_5skills": 0.75, "max_ece_5skills": 0.10, "max_predict_ms_per_request": 100}
PASSING = {"top3_5skills": 0.802, "ece_5skills": 0.072, "predict_ms_per_request": 20}


def test_ece_includes_certainty_and_uses_bin_accuracy():
    # Two certain predictions (one wrong) and two 0.5 predictions (one wrong).
    # The certainty bin contributes .5 * abs(.5 - 1) = .25.
    proba = np.array([[1, 0], [1, 0], [0.5, 0.5], [0.5, 0.5]])
    assert expected_calibration_error(["a", "b", "a", "b"], proba, ["a", "b"]) == pytest.approx(0.25)


def test_ece_detects_underconfidence():
    proba = np.array([[0.6, 0.4], [0.4, 0.6]])
    assert expected_calibration_error(["a", "b"], proba, ["a", "b"]) == pytest.approx(0.4)
    assert expected_calibration_error(["a", "b"], np.eye(2), ["a", "b"]) == 0


def test_gate_passes_inclusive_limits():
    metrics = {"top3_5skills": 0.75, "ece_5skills": 0.10, "predict_ms_per_request": 100}
    assert all(check["passed"] for check in quality_checks(metrics, THRESHOLDS).values())


@pytest.mark.parametrize(
    "metric,value",
    [
        ("top3_5skills", 0.749),
        ("ece_5skills", 0.101),
        ("predict_ms_per_request", 100.1),
        ("top3_5skills", float("nan")),
        ("ece_5skills", float("inf")),
        ("predict_ms_per_request", -1),
        ("top3_5skills", 1.1),
        ("ece_5skills", "0.05"),
        ("top3_5skills", True),
    ],
)
def test_gate_rejects_failed_or_invalid_measurements(metric, value):
    metrics = PASSING | {metric: value}
    assert not quality_checks(metrics, THRESHOLDS)[metric]["passed"]


def test_gate_rejects_missing_measurements():
    assert not any(check["passed"] for check in quality_checks({}, THRESHOLDS).values())


@pytest.mark.parametrize("passed", [True, False])
def test_validation_writes_card_and_blocks_failed_candidate(tmp_path, monkeypatch, passed):
    monkeypatch.setattr(validate, "REPORTS", tmp_path)
    monkeypatch.setattr(validate, "load_params", lambda: {"validate": THRESHOLDS})
    metrics = PASSING if passed else PASSING | {"ece_5skills": 0.43}
    write_json({"best_model": "complement_nb_cal", "best_model_metrics": metrics}, tmp_path / "metrics.json")
    if passed:
        validate.main()
    else:
        with pytest.raises(SystemExit, match="failed the quality gate"):
            validate.main()
    assert read_json(tmp_path / "validation.json")["passed"] is passed
    expected = "Quality gate: PASSED" if passed else "Quality gate: FAILED"
    assert expected in (tmp_path / "model_card.md").read_text(encoding="utf-8")
