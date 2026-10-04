import pickle

import numpy as np
import pandas as pd

from src.features.build import make_preprocessor, sample_skills, skill_dropout

DF = pd.DataFrame(
    {
        "skills": [["python", "sql", "excel"], ["tally", "gst"], ["react.js", "css", "html", "javascript"]],
        "exp_min": [2.0, np.nan, 35.0],
    }
)


def test_preprocessor_shapes_and_pickles():
    pre = make_preprocessor(top_k=1500, exp_clip=20).fit(DF)
    X = pre.transform(DF)
    n_vocab = len(pre.named_transformers_["skills"].vocabulary_)
    assert X.shape == (3, n_vocab + 1)
    exp = X[:, -1].toarray().ravel()
    assert exp.min() >= 0 and exp.max() <= 1  # imputed, clipped at 20, scaled
    restored = pickle.loads(pickle.dumps(pre))  # must survive serving
    assert (restored.transform(DF) != X).nnz == 0


def test_unknown_skills_are_ignored_at_serving_time():
    pre = make_preprocessor(top_k=1500, exp_clip=20).fit(DF)
    X = pre.transform(pd.DataFrame({"skills": [["python", "never-seen-skill"]], "exp_min": [1.0]}))
    assert X[:, :-1].sum() == 1


def test_sample_skills_is_deterministic_and_bounded():
    s = ["a", "b", "c", "d", "e", "f", "g"]
    a = sample_skills(s, 5, np.random.default_rng(0))
    b = sample_skills(s, 5, np.random.default_rng(0))
    assert a == b and len(a) == 5 and set(a) <= set(s)
    assert sample_skills(["x", "y"], 5, np.random.default_rng(0)) == ["x", "y"]


def test_skill_dropout_adds_cut_down_copies():
    out = skill_dropout(DF, copies=2, min_k=1, max_k=2, seed=0)
    assert len(out) == 3 * len(DF)
    aug = out[out["augmented"]]
    assert aug["skills"].map(len).between(1, 2).all()
    originals = list(DF["skills"]) * 2  # copy 1 then copy 2, same row order
    assert all(set(a) <= set(o) for a, o in zip(aug["skills"], originals, strict=True))
