"""Registry promotion logic, end to end against a throwaway local MLflow (SQLite), never DagsHub."""

import mlflow
import pytest
from mlflow import MlflowClient

from src.models.register import CHALLENGER, CHAMPION, PREVIOUS, decide, register_candidate
from src.models.train import build_pipeline, log_model
from src.utils import load_params
from tests.toy_bundle import toy_postings

NAME = "career-role-classifier-test"


@pytest.mark.parametrize(
    "passed,candidate,champion,expected",
    [
        (True, 0.80, None, CHAMPION),  # first model ever
        (True, 0.80, 0.79, CHAMPION),  # better
        (True, 0.79, 0.79, CHALLENGER),  # a tie does not replace the champion
        (True, 0.70, 0.79, CHALLENGER),  # worse
        (False, 0.99, 0.79, "rejected"),  # failed the quality gate: never promoted, however accurate
    ],
)
def test_decide(passed, candidate, champion, expected):
    assert decide(passed, candidate, champion, min_improvement=0.0) == expected


def test_min_improvement_is_required_to_promote():
    assert decide(True, 0.795, 0.79, min_improvement=0.01) == CHALLENGER
    assert decide(True, 0.801, 0.79, min_improvement=0.01) == CHAMPION


@pytest.fixture
def registry(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_registry_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    df = toy_postings()
    pipe = build_pipeline("logreg", load_params()).fit(df[["skills", "exp_min"]], df["role"])
    with mlflow.start_run():
        uri = log_model(pipe, df[["skills", "exp_min"]].head(2))
    yield MlflowClient(), uri
    mlflow.set_tracking_uri(None)


def candidate(client, uri, md5, score, passed=True):
    tags = {"model_md5": md5, "top3_5skills": str(score)}
    return register_candidate(client, NAME, uri, tags, "top3_5skills", passed, 0.0)


def aliases(client):
    return {
        a: str(client.get_model_version_by_alias(NAME, a).version) for a in client.get_registered_model(NAME).aliases
    }


def test_promotion_flow(registry):
    client, uri = registry
    assert candidate(client, uri, "md5-a", 0.79) == ("1", CHAMPION)
    assert aliases(client) == {CHAMPION: "1"}

    assert candidate(client, uri, "md5-b", 0.80) == ("2", CHAMPION)  # better → promoted, old kept for rollback
    assert aliases(client) == {CHAMPION: "2", PREVIOUS: "1"}

    assert candidate(client, uri, "md5-c", 0.75) == ("3", CHALLENGER)
    assert aliases(client) == {CHAMPION: "2", PREVIOUS: "1", CHALLENGER: "3"}

    assert candidate(client, uri, "md5-d", 0.99, passed=False) == ("4", "rejected")
    assert client.get_model_version(NAME, "4").tags["decision"] == "rejected"
    assert CHAMPION in aliases(client) and aliases(client)[CHAMPION] == "2"


def test_same_model_file_is_not_registered_twice(registry):
    client, uri = registry
    candidate(client, uri, "md5-a", 0.79)
    assert candidate(client, uri, "md5-a", 0.79) == ("1", "already champion")
    assert len(client.search_model_versions(f"name = '{NAME}'")) == 1


def test_dry_run_changes_nothing(registry):
    client, uri = registry
    tags = {"model_md5": "md5-a", "top3_5skills": "0.8"}
    assert register_candidate(client, NAME, uri, tags, "top3_5skills", True, 0.0, dry_run=True) == (None, CHAMPION)
    assert client.search_registered_models() == []
