"""Quality-gate tests on a toy model: the measurements, the gate logic and the full run."""

import numpy as np
import pandas as pd
import pytest

from src.models.train import build_pipeline
from src.models.validate import BOOLEAN_CHECKS, THRESHOLD_CHECKS, behaviour, ece_top1, gate, invariance, run
from src.utils import load_params
from tests.toy_bundle import ROLES, toy_postings

PARAMS = load_params()


@pytest.fixture(scope="module")
def pipe():
    df = toy_postings()
    return build_pipeline("logreg", PARAMS).fit(df[["skills", "exp_min"]], df["role"])


def test_ece_is_zero_when_confidence_matches_accuracy():
    classes = np.array(["a", "b"])
    proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ece_top1(proba, classes, np.array(["a", "b"])) == 0.0


def test_ece_catches_overconfidence():
    classes = np.array(["a", "b"])
    proba = np.tile([0.95, 0.05], (10, 1))  # always 95% sure of "a"...
    y = np.array(["a"] * 5 + ["b"] * 5)  # ...but right only half the time
    assert ece_top1(proba, classes, y) == pytest.approx(0.45)


def passing_values():
    v = {name: 1.0 for name, _, rule in THRESHOLD_CHECKS if rule == ">="}
    v |= {name: 0.0 for name, _, rule in THRESHOLD_CHECKS if rule == "<="}
    return v | {name: True for name in BOOLEAN_CHECKS}


def test_gate_passes_when_every_value_is_on_the_right_side():
    checks = gate(passing_values(), PARAMS["validate"])
    assert all(c["passed"] for c in checks.values())
    assert set(checks) == {n for n, _, _ in THRESHOLD_CHECKS} | set(BOOLEAN_CHECKS)


@pytest.mark.parametrize(
    "name,value",
    [("top3_5skills", 0.5), ("ece_top1_5skills", 0.4), ("model_size_mb", 500), ("invariant_to_skill_order", False)],
)
def test_gate_fails_the_offending_check_only(name, value):
    checks = gate(passing_values() | {name: value}, PARAMS["validate"])
    assert [k for k, c in checks.items() if not c["passed"]] == [name]


def test_behaviour_rejects_cases_with_skills_the_model_does_not_know(pipe):
    with pytest.raises(ValueError, match="doesn't know"):
        behaviour(pipe, [{"expected": "Accountant", "skills": ["quantum knitting"], "experience": 1}])


def test_bag_of_skills_model_is_invariant(pipe):
    df = toy_postings(seed=1)
    assert all(invariance(pipe, df[["skills", "exp_min"]]).values())


def test_full_run_produces_a_report(pipe):
    test = toy_postings(seed=2).assign(skills_eval=lambda d: d["skills"])
    selected = {
        "best_model": "logreg",
        "margin_over_profile_matching": 0.2,
        "best_model_metrics": {"top3_5skills": 1.0, "ref2019_top3_5skills": 1.0, "model_size_mb": 0.1},
    }
    cases = [{"expected": role, "skills": skills[:4], "experience": 2} for role, (_, skills) in ROLES.items()]
    cfg = PARAMS["validate"] | {"latency_requests": 5}
    report = run(pipe, test, selected, cases, cfg)
    assert report["model"] == "logreg"
    assert report["checks"]["behaviour_pass_rate"]["value"] == 1.0
    assert report["passed"] == (not report["failed_checks"])
    assert isinstance(pd.Series(report["context"]["worst_roles"]), pd.Series)
