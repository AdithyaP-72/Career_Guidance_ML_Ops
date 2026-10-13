import pickle

import mlflow
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import skops.io

from src.models.estimators import ProfileMatcher, build_estimator
from src.models.evaluate import score
from src.models.tracking import flatten
from src.models.train import TRUSTED_TYPES, build_pipeline, log_model
from src.utils import load_params

PARAMS = load_params()

# Three tiny "roles" with clearly different skills (5 rows each so MLPClassifier's
# early_stopping validation split has ≥ 2 samples).
TOY = pd.DataFrame(
    {
        "skills": [
            ["tally", "gst"],
            ["tally", "tds"],
            ["gst", "tds", "excel"],
            ["tally", "excel"],
            ["gst", "tally", "tds"],
            ["react.js", "css"],
            ["css", "html"],
            ["react.js", "html", "javascript"],
            ["javascript", "css"],
            ["html", "react.js", "css"],
            ["nursing", "gnm"],
            ["nursing", "icu"],
            ["gnm", "icu", "patient care"],
            ["patient care", "nursing"],
            ["icu", "gnm", "nursing"],
        ],
        "exp_min": [1, 2, 3, 4, 2, 0, 1, 4, 3, 2, 2, 5, 3, 1, 4],
        "role": ["Accountant"] * 5 + ["Frontend Developer"] * 5 + ["Staff Nurse"] * 5,
        "family": ["Accounting & Finance"] * 5 + ["Software Development"] * 5 + ["Healthcare & Pharma"] * 5,
    }
)


def test_profile_matcher_ranks_the_obvious_role_first():
    X = sp.csr_matrix(np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float))
    m = ProfileMatcher().fit(X, ["a", "a", "b", "b"])
    proba = m.predict_proba(sp.csr_matrix([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=float))
    assert np.allclose(proba.sum(axis=1), 1)
    assert list(m.predict(sp.csr_matrix([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=float))) == ["a", "b"]


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError):
        build_estimator("xgboost", {})


@pytest.mark.parametrize("name", PARAMS["train"]["models"])
def test_every_configured_model_trains_predicts_and_pickles(name):
    pipe = build_pipeline(name, PARAMS).fit(TOY[["skills", "exp_min"]], TOY["role"])
    proba = pipe.predict_proba(TOY[["skills", "exp_min"]])
    assert proba.shape == (len(TOY), 3)
    assert np.allclose(proba.sum(axis=1), 1)
    restored = pickle.loads(pickle.dumps(pipe))  # serving loads a pickled pipeline
    assert np.allclose(restored.predict_proba(TOY[["skills", "exp_min"]]), proba)


@pytest.mark.parametrize("name", PARAMS["train"]["models"])
def test_every_model_reloads_with_skops_and_our_trusted_types(name):
    # MLflow stores models in skops format; loading fails on any type missing from TRUSTED_TYPES.
    pipe = build_pipeline(name, PARAMS).fit(TOY[["skills", "exp_min"]], TOY["role"])
    restored = skops.io.loads(skops.io.dumps(pipe), trusted=TRUSTED_TYPES)
    assert np.allclose(
        restored.predict_proba(TOY[["skills", "exp_min"]]), pipe.predict_proba(TOY[["skills", "exp_min"]])
    )


def test_log_model_uploads_a_loadable_model(tmp_path):
    # Same call select.py makes for the best baseline, against a throwaway MLflow database.
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    pipe = build_pipeline("logreg", PARAMS).fit(TOY[["skills", "exp_min"]], TOY["role"])
    with mlflow.start_run():
        uri = log_model(pipe, TOY[["skills", "exp_min"]].head(2))
    loaded = mlflow.sklearn.load_model(uri)
    assert np.allclose(loaded.predict_proba(TOY[["skills", "exp_min"]]), pipe.predict_proba(TOY[["skills", "exp_min"]]))


def test_score_reports_topk_and_family():
    pipe = build_pipeline("logreg", PARAMS).fit(TOY[["skills", "exp_min"]], TOY["role"])
    family_of = dict(zip(TOY["role"], TOY["family"], strict=True))
    metrics, pred = score(pipe, TOY, "skills", family_of)
    assert metrics["top3"] == 1.0  # only 3 classes
    assert 0 <= metrics["top1"] <= metrics["top3"]
    assert set(metrics) == {"top1", "top3", "top5", "macro_f1", "family_top1"}


def test_flatten_nested_params():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}
