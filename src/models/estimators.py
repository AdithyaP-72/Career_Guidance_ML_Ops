"""The baseline models, built from params.yaml → train.<name>.

    dummy          always predicts by class frequency (the floor)
    profile        nearest role profile by cosine similarity (no learning beyond averaging)
    logreg         multinomial logistic regression
    random_forest  bagged decision trees
    lightgbm       gradient-boosted trees

Every model sits behind the same feature pipeline (src/features/build.py), so they all
see exactly the same inputs and can be compared fairly.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize


class ProfileMatcher(ClassifierMixin, BaseEstimator):
    """Score each role by cosine similarity between the input and the role's average posting.

    A role's centroid is the share of its training postings that list each skill, the
    same thing models/skill_profiles.json describes. This is what a non-ML "match your
    skills against each role's typical skills" approach would do. If the ML models can't
    beat it, they aren't adding anything.
    """

    def __init__(self, temperature: float = 0.05):
        self.temperature = temperature

    def fit(self, X, y):
        X = sp.csr_matrix(X)
        self.classes_, idx = np.unique(np.asarray(y), return_inverse=True)
        membership = sp.csr_matrix(
            (np.ones(len(idx)), (idx, np.arange(len(idx)))), shape=(len(self.classes_), X.shape[0])
        )
        centroids = np.asarray((membership @ X).todense()) / np.bincount(idx)[:, None]
        self.centroids_ = normalize(centroids)
        return self

    def predict_proba(self, X):
        sims = np.asarray(normalize(sp.csr_matrix(X)) @ self.centroids_.T)
        z = sims / self.temperature
        z -= z.max(axis=1, keepdims=True)  # softmax, numerically stable
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


MODELS = {
    "dummy": DummyClassifier,
    "profile": ProfileMatcher,
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "lightgbm": LGBMClassifier,
    "mlp": MLPClassifier,
    "sgd": SGDClassifier,
    "complement_nb": ComplementNB,
}


def build_estimator(name: str, cfg: dict | None):
    """Instantiate a baseline from its params.yaml block (every key is a constructor argument)."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**(cfg or {}))
