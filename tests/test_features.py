import pandas as pd

from src.data.common import normalise_skills
from src.features.build import build_features, sample_n_skills, skill_dropout


def df():
    return pd.DataFrame(
        {"skills": [["a", "b", "c", "d", "e", "f", "g"], ["a", "b", "x", "y", "z", "w"]], "experience": [1.0, 40.0], "role": ["r1", "r2"]}
    )


def test_normalise_skills_aliases_and_dedupe():
    out = normalise_skills(" React.js,reactjs, Excel ,", ",", {"react.js": "react", "reactjs": "react"})
    assert out == ["react", "excel"]


def test_dropout_only_adds_reduced_copies():
    out = skill_dropout(df(), copies=2, lo=2, hi=4, seed=0)
    assert len(out) == 6
    assert out["skills"].iloc[2:].map(len).between(2, 4).all()


def test_sample_n_skills_fixed_size_and_subset():
    out = sample_n_skills(df(), 5, seed=0)
    assert out["skills"].map(len).eq(5).all()


def test_pipeline_pickles_and_clips_experience():
    import pickle

    ct = build_features(top_k=10, exp_clip=20)
    X = ct.fit_transform(df()[["skills", "experience"]])
    assert X[:, -1].max() == 1.0  # 40 clipped to 20 -> 1.0
    pickle.loads(pickle.dumps(ct))


def test_drift_binned_is_stable_for_same_distribution_and_detects_shift():
    import numpy as np

    from src.monitoring import drift

    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(300) * 0.3)
    ref = p
    same = np.bincount(rng.choice(300, 700, p=p), minlength=300) / 700
    shifted = np.zeros(300)
    shifted[:5] = 0.2
    assert drift.js_distance(*drift.binned(ref, same)) < 0.2
    assert drift.js_distance(*drift.binned(ref, shifted)) > 0.35


def test_processed_data_schema():
    import pytest

    from src.data.common import ROOT

    f = ROOT / "data/processed/train.parquet"
    if not f.exists():
        pytest.skip("run the pipeline first")
    df = pd.read_parquet(f)
    assert {"title", "skills", "experience", "role", "family"} <= set(df.columns)
    assert df["skills"].map(len).min() >= 1 and df["experience"].min() >= 0
    assert df["role"].nunique() >= 90 and df["family"].nunique() <= 22
    assert not df.duplicated(subset=["title", "company"]).all()
