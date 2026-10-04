"""Feature pipeline: skills list + years of experience → model input matrix.

    make_preprocessor(top_k, exp_clip)  sklearn ColumnTransformer, fit on the train split only
    skill_dropout(df, ...)              training-time augmentation (cut-down copies of postings)
    sample_skills(skills, n, rng)       fixed n-skill sample, used for the 5-skill test inputs

The fitted vocabulary (top_k most common training skills) is also the list the app's
skill search box offers, so what users can type always matches what the model knows.
Everything here must be importable at serving time, which is why `identity` and
`clip_experience` are top-level functions (lambdas can't be pickled).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

SKILLS, EXPERIENCE = "skills", "exp_min"


def identity(tokens):
    """Skills are already a list of tokens, so the vectorizer just takes them as-is."""
    return tokens


def clip_experience(x, upper: float):
    return np.clip(x, 0, upper)


def make_preprocessor(top_k: int, exp_clip: float) -> ColumnTransformer:
    skills = CountVectorizer(analyzer=identity, binary=True, max_features=top_k, lowercase=False)
    experience = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("clip", FunctionTransformer(clip_experience, kw_args={"upper": exp_clip})),
            ("scale", MinMaxScaler()),
        ]
    )
    return ColumnTransformer(
        [("skills", skills, SKILLS), ("experience", experience, [EXPERIENCE])],
        sparse_threshold=1.0,  # keep the output sparse: ~1,500 mostly-zero columns
    )


def sample_skills(skills, n: int, rng: np.random.Generator) -> list[str]:
    skills = list(skills)
    if len(skills) <= n:
        return skills
    return [skills[i] for i in sorted(rng.choice(len(skills), size=n, replace=False))]


def skill_dropout(df: pd.DataFrame, copies: int, min_k: int, max_k: int, seed: int) -> pd.DataFrame:
    """Return df plus `copies` extra copies whose skill lists are cut to min_k..max_k random skills.

    Apply to the TRAINING split only. The test split keeps its own fixed `skills_eval`.
    """
    rng = np.random.default_rng(seed)
    parts = [df.assign(augmented=False)]
    for _ in range(copies):
        ks = rng.integers(min_k, max_k + 1, size=len(df))
        cut = [sample_skills(s, int(k), rng) for s, k in zip(df[SKILLS], ks, strict=True)]
        parts.append(df.assign(**{SKILLS: cut, "augmented": True}))
    return pd.concat(parts, ignore_index=True)
