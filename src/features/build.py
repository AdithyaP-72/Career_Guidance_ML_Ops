"""Feature pipeline: skills -> multi-hot over the top-K vocabulary, experience -> clipped & scaled.

The vocabulary is learned on the training split only and is the same list the search box offers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer


def identity(x):  # top-level so the pipeline can be pickled
    return x


def clip_scale(x, clip: int = 20):
    return np.clip(np.asarray(x, dtype=float), 0, clip) / clip


def build_features(top_k: int, exp_clip: int) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("skills", CountVectorizer(analyzer=identity, binary=True, max_features=top_k), "skills"),
            (
                "exp",
                FunctionTransformer(clip_scale, kw_args={"clip": exp_clip}, validate=False),
                ["experience"],
            ),
        ]
    )


def skill_dropout(df: pd.DataFrame, copies: int, lo: int, hi: int, seed: int) -> pd.DataFrame:
    """Add `copies` reduced versions (lo..hi random skills) of every posting. Training only."""
    rng = np.random.default_rng(seed)
    parts = [df]
    for _ in range(copies):
        d = df.copy()

        def cut(skills):
            n = int(rng.integers(lo, hi + 1))
            if len(skills) <= n:
                return list(skills)
            return [skills[i] for i in sorted(rng.choice(len(skills), n, replace=False))]

        d["skills"] = d["skills"].map(cut)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def sample_n_skills(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Fixed n-skill version of each posting that has at least n skills (the headline test input)."""
    rng = np.random.default_rng(seed)
    d = df[df["skills"].map(len) >= n].copy()
    d["skills"] = d["skills"].map(lambda s: [s[i] for i in sorted(rng.choice(len(s), n, replace=False))])
    return d


def sample_n_vocab_skills(df: pd.DataFrame, n: int, seed: int, vocab: set[str]) -> pd.DataFrame:
    """Like sample_n_skills but only draws skills from the vocabulary, i.e. what the search box can actually offer."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["skills"] = d["skills"].map(lambda s: [x for x in s if x in vocab])
    d = d[d["skills"].map(len) >= n].copy()
    d["skills"] = d["skills"].map(lambda s: [s[i] for i in sorted(rng.choice(len(s), n, replace=False))])
    return d
