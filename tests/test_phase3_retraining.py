"""Test the decision and alias boundaries of the isolated Phase 3 experiment."""

from unittest.mock import Mock, call

import numpy as np
import pytest
from mlflow import MlflowClient

from scripts.retrain_phase3 import (
    EXPERIMENT,
    REGISTRY,
    comparison_checks,
    registry_state,
    set_aliases,
    skip_decision,
)
from src.models.register import switch_champion_alias
from src.models.tracking import isolated_experiment
from src.models.validate import paired_gain


def test_retraining_trigger_respects_drift_threshold_and_minimum_sample():
    assert skip_decision(None, 0.25) == "skipped_insufficient"
    assert skip_decision(0.249, 0.25) == "skipped_no_drift"
    assert skip_decision(0.25, 0.25) is None


def test_paired_gain_uses_matched_examples():
    champion = np.array([True, False, False, True])
    challenger = np.array([True, True, False, False])
    gain, interval = paired_gain(champion, challenger)
    assert gain == 0
    assert interval[0] < 0 < interval[1]
    assert paired_gain(np.zeros(100, dtype=bool), np.ones(100, dtype=bool)) == (1.0, (1.0, 1.0))


def test_large_shifted_gain_does_not_waive_real_regression():
    champion = {
        "real_2025": {"top3": 0.80, "macro_f1": 0.71, "ece": 0.06},
        "synthetic_eval": {"top3": 0.60, "macro_f1": 0.50, "ece": 0.06},
    }
    challenger = {
        "real_2025": {"top3": 0.77, "macro_f1": 0.70, "ece": 0.06},
        "synthetic_eval": {"top3": 0.90, "macro_f1": 0.80, "ece": 0.06},
    }
    a = np.r_[np.ones(60, dtype=bool), np.zeros(40, dtype=bool)]
    b = np.r_[np.ones(90, dtype=bool), np.zeros(10, dtype=bool)]
    checks = comparison_checks(champion, challenger, (a, b))
    assert checks["shifted_top3_gain"]["passed"]
    assert checks["shifted_gain_ci_lower"]["passed"]
    assert not checks["real_top3_loss"]["passed"]


def test_alias_transition_restores_previous_alias_when_champion_update_fails():
    client = Mock(spec=MlflowClient)
    client.set_registered_model_alias.side_effect = [None, RuntimeError("update failed"), None]
    with pytest.raises(RuntimeError, match="update failed"):
        switch_champion_alias(client, REGISTRY, "2", "3", "1")
    assert client.set_registered_model_alias.call_args_list == [
        call(REGISTRY, "previous-champion", "2"),
        call(REGISTRY, "champion", "3"),
        call(REGISTRY, "previous-champion", "1"),
    ]


def make_versions(client: MlflowClient, tmp_path):
    client.create_registered_model(REGISTRY)
    for source, tags in (
        ("reference", {"model_md5": "reference"}),
        ("candidate", {"model_md5": "candidate", "phase3_experiment": "test", "validation_status": "passed"}),
    ):
        artifact = tmp_path / source
        artifact.mkdir()
        client.create_model_version(REGISTRY, artifact.as_uri(), tags=tags)
    client.set_registered_model_alias(REGISTRY, "champion", "1")
    client.set_registered_model_alias(REGISTRY, "challenger", "2")
    return {
        "decision": "eligible",
        "run_key": "test",
        "audit": [],
        "quality_gate": {"passed": True},
        "comparison_checks": {"holdout_top3_gain": {"passed": True}},
        "champion": {"registry_version": "1"},
        "challenger": {"registry_version": "2", "model_md5": "candidate"},
    }


def test_controlled_promotion_and_rollback_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.invalid/tracking")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "https://example.invalid/registry")
    with isolated_experiment(tmp_path, EXPERIMENT) as (client, _):
        report = make_versions(client, tmp_path)
        set_aliases(client, report, rollback=False, approved_by="Reviewer A")
        assert report["decision"] == "promoted"
        assert registry_state(client)["aliases"] == {"champion": "2", "challenger": "2", "previous-champion": "1"}
        assert report["audit"][-1]["approved_by"] == "Reviewer A"
        set_aliases(client, report, rollback=True, approved_by="Reviewer B")
        assert report["decision"] == "rolled_back"
        assert registry_state(client)["aliases"]["champion"] == "1"
        assert report["audit"][-1]["from_version"] == "2"


def test_rejected_or_changed_champion_cannot_be_promoted(tmp_path):
    with isolated_experiment(tmp_path, EXPERIMENT) as (client, _):
        report = make_versions(client, tmp_path)
        report["decision"] = "rejected"
        with pytest.raises(ValueError, match="requires eligibility"):
            set_aliases(client, report, rollback=False, approved_by="reviewer")
        assert registry_state(client)["aliases"]["champion"] == "1"
        report["decision"] = "eligible"
        report["champion"]["registry_version"] = "3"
        with pytest.raises(ValueError, match="same champion"):
            set_aliases(client, report, rollback=False, approved_by="reviewer")
        assert registry_state(client)["aliases"]["champion"] == "1"
