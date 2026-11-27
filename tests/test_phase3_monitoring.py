import math

import pytest
from fastapi.testclient import TestClient

from src.models.drift import build_reference, compare_frame, population_stability_index, severity
from src.models.train import build_pipeline
from src.serving.app import app
from src.utils import load_params, write_json
from tests.toy_bundle import make_toy_bundle, toy_postings


def test_psi_known_result_and_proportional_distributions():
    assert population_stability_index([50, 50], [30, 70]) == pytest.approx(0.2 * math.log(7 / 3))
    assert population_stability_index([5, 5], [30, 30]) == pytest.approx(0.0)
    assert population_stability_index([0, 1], [0, 100]) == pytest.approx(0.0)
    assert math.isfinite(population_stability_index([100, 0], [0, 100]))
    assert population_stability_index([100, 0], [0, 100]) > 0.25
    with pytest.raises(ValueError, match="no observations"):
        population_stability_index([0, 0], [1, 2])


@pytest.mark.parametrize("psi,expected", [(0.099, "normal"), (0.1, "warning"), (0.25, "critical")])
def test_psi_threshold_boundaries(psi, expected):
    assert severity(psi, warning=0.1, critical=0.25) == expected


def test_reference_reused_for_single_feature_shift_and_small_sample():
    frame = toy_postings()
    pipe = build_pipeline("logreg", load_params()).fit(frame[["skills", "exp_min"]], frame["role"])
    ref = build_reference(frame, pipe, min_samples=5, min_known=3)
    assert compare_frame(ref, frame, pipe)["overall_psi"] == pytest.approx(0.0)
    shift = compare_frame(ref, frame.assign(exp_min=25), pipe)
    assert shift["status"] == "critical"
    assert shift["features"]["experience"]["psi"] > 0.25
    assert shift["features"]["experience"]["current"][-2] == 1
    assert all(sum(feature["reference"]) == pytest.approx(1) for feature in shift["features"].values())
    assert compare_frame(ref, frame.iloc[:2], pipe)["status"] == "insufficient"


def test_private_prediction_counters_and_demo_reset(tmp_path, monkeypatch, caplog):
    bundle = make_toy_bundle(tmp_path / "bundle")
    frame = toy_postings()
    pipe = build_pipeline("logreg", load_params()).fit(frame[["skills", "exp_min"]], frame["role"])
    reference = build_reference(frame, pipe, min_samples=2)
    write_json(reference, bundle / "drift_reference.json")
    monkeypatch.setenv("BUNDLE_DIR", str(bundle))
    token = tmp_path / "token"
    token.write_text("testing-token", encoding="utf-8")
    monkeypatch.setenv("MONITORING_TOKEN_FILE", str(token))
    monkeypatch.setenv("MONITORING_WINDOW_SIZE", "2")
    monkeypatch.setenv("PHASE3_DEMO", "true")
    sentinel = "private-input@example.test"
    headers = {"Authorization": "Bearer testing-token"}
    known = ["accounting", "tally", "gst", "tds", "excel"]

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 401
        assert client.get("/drift").status_code == 401
        assert client.post("/drift/reset").status_code == 401
        assert client.get("/drift", headers=headers).json()["status"] == "insufficient"
        assert client.post("/drift/reset?window=real_2025_demo", headers=headers).status_code == 200

        # Invalid input and body-size failures are captured outside the route handler.
        assert client.post("/predict", json={"skills": [sentinel]}).status_code == 422
        assert client.post("/predict", json={"skills": "not a list"}).status_code == 422
        assert client.post("/predict", json={"skills": ["x" * 70_000]}).status_code == 413
        for _ in range(3):
            body = {"skills": [*known, sentinel], "experience_years": 40}
            result = client.post("/predict", json=body)
            assert result.status_code == 200
            assert result.json()["unknown_skills"] == [sentinel]

        assert len(app.state.monitor.events) == 2
        assert sentinel not in repr(app.state.monitor.events)
        assert sentinel not in caplog.text
        current = client.get("/drift", headers=headers).json()
        assert current["window"] == "real_2025_demo"
        assert current["sample_count"] == 2
        assert current["status"] == "critical"
        assert current["features"]["experience"]["current"][-2] == 1
        assert current["status_since_seconds"] >= current["window_started_at"]
        assert 'career_drift_distribution{feature="experience"' in client.get("/metrics", headers=headers).text

        # Reset invalidates the empty-window cache. A previous in-flight request cannot join the next window.
        previous_generation = app.state.monitor.current_generation()
        old_observation = app.state.monitor.events[-1].observation
        assert client.post("/drift/reset?window=synthetic_2026_demo", headers=headers).status_code == 200
        app.state.monitor.record(previous_generation, 200, 0.001, old_observation)
        assert client.get("/drift", headers=headers).json()["sample_count"] == 0
        metrics = client.get("/metrics", headers=headers).text
        assert "career_prediction_requests_total 7" in metrics
        assert "career_prediction_errors_total 3" in metrics
        assert "career_prediction_successes_total 4" in metrics
        assert "career_prediction_latency_seconds_count 7" in metrics
        assert 'career_drift_status{status="insufficient"} 1' in metrics
        assert sentinel not in metrics
        monkeypatch.setenv("PHASE3_DEMO", "false")
        assert client.post("/drift/reset", headers=headers).status_code == 404
