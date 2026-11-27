"""Synthetic data provenance and isolation tests; no production artifacts or network needed."""

import json

import numpy as np
import pandas as pd
import pytest

from scripts.generate_2026_drift import ORIGIN, generate, sha256_file, source_group_id, validate_split

ROLES = {
    "ML / AI Engineer": "Data & Analytics",
    "Cloud Engineer": "IT Infra & Support",
    "ServiceNow Developer": "ERP / Enterprise Apps",
}


def test_source_grouping_ignores_skill_order_duplicates_and_noisy_labels():
    first = source_group_id("Accountant", "Finance", ["sql", "excel", "sql"], 2.0)
    second = source_group_id("Different label", "Other family", ["excel", "sql"], 2)
    assert first == second
    assert source_group_id("Accountant", "Finance", ["sql", "excel"], 3) != first


@pytest.fixture
def inputs(tmp_path):
    def frame(prefix, offset, count=24):
        rows = []
        for role, family in ROLES.items():
            for i in range(count):
                rows.append(
                    {
                        "posting_id": f"{prefix}-{role}-{i}",
                        "role": role,
                        "family": family,
                        "skills": ["python", "sql", "linux", "cloud computing", "testing", f"{role}-{i}"],
                        "exp_min": offset + i / 100,
                    }
                )
        return pd.DataFrame(rows)

    train, evaluation = frame("train", 1), frame("eval", 4)
    # Identical features with a different ID must still be removed from the evaluation source pool.
    duplicate = train.iloc[[0]].assign(posting_id="duplicate-id")
    evaluation = pd.concat([evaluation, duplicate], ignore_index=True)
    source, eval_source = tmp_path / "train.parquet", tmp_path / "eval.parquet"
    train.to_parquet(source, index=False)
    evaluation.to_parquet(eval_source, index=False)
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps(["python", "sql", "linux", "cloud computing", "testing"]))
    return source, eval_source, vocab


def test_generation_is_reproducible_and_preserves_inputs(tmp_path, inputs):
    source, evaluation, vocab = inputs
    original_hashes = sha256_file(source), sha256_file(evaluation)
    one = generate(source, evaluation, tmp_path / "one", rows=60, vocabulary_path=vocab)
    two = generate(source, evaluation, tmp_path / "two", rows=60, vocabulary_path=vocab)
    assert original_hashes == (sha256_file(source), sha256_file(evaluation))
    assert one["seed"] == 42 and one["dataset_origin"] == ORIGIN
    assert one["evaluation_overlap_groups_removed"] == 1
    for split in ("train", "eval", "holdout"):
        assert one["splits"][split]["sha256"] == two["splits"][split]["sha256"]
        assert one["splits"][split]["distributions"] == two["splits"][split]["distributions"]
        frame = pd.read_parquet(one["splits"][split]["path"])
        validate_split(frame, ROLES, split)
        assert set(frame.role) == set(ROLES)
        assert one["splits"][split]["sha256"] == sha256_file(one["splits"][split]["path"])


def test_source_groups_and_final_features_never_cross_partitions(tmp_path, inputs):
    source, evaluation, vocab = inputs
    manifest = generate(source, evaluation, tmp_path / "out", rows=60, vocabulary_path=vocab)
    seen_sources, seen_features = set(), set()
    real_train = pd.read_parquet(source)
    train_sources = {source_group_id(r.role, r.family, r.skills, r.exp_min) for r in real_train.itertuples()}
    for split in ("train", "eval", "holdout"):
        frame = pd.read_parquet(manifest["splits"][split]["path"])
        groups = set(frame.source_group_id)
        assert not seen_sources & groups
        assert groups == set(manifest["splits"][split]["source_group_ids"])
        if split == "train":
            assert groups <= train_sources
        else:
            assert not groups & train_sources
        fingerprints = {source_group_id(r.role, r.family, r.skills, r.exp_min) for r in frame.itertuples()}
        assert len(fingerprints) == len(frame) and not seen_features & fingerprints
        seen_features |= fingerprints
        seen_sources |= groups


def test_synthetic_rows_are_role_conditioned_and_usable_by_api(tmp_path, inputs):
    source, evaluation, vocab = inputs
    manifest = generate(source, evaluation, tmp_path / "out", rows=60, vocabulary_path=vocab)
    frame = pd.read_parquet(manifest["splits"]["eval"]["path"])
    known = set(json.loads(vocab.read_text()))
    assert frame.skills.map(lambda s: len(set(s) & known) >= 5).all()
    assert frame.exp_min.between(0, 20).all()
    assert frame.skills.map(lambda s: any(skill not in known for skill in s)).any()
    assert frame.data_origin.eq("synthetic_2026").all()
    # A new scenario seed must change generated values, not just identifiers.
    other = generate(source, evaluation, tmp_path / "other", rows=60, seed=43, vocabulary_path=vocab)
    assert manifest["splits"]["eval"]["sha256"] != other["splits"]["eval"]["sha256"]


@pytest.mark.parametrize("mutation", ["label", "family", "marker", "experience", "sample", "duplicate"])
def test_schema_rejects_invalid_generated_records(tmp_path, inputs, mutation):
    source, evaluation, vocab = inputs
    manifest = generate(source, evaluation, tmp_path / mutation, rows=60, vocabulary_path=vocab)
    frame = pd.read_parquet(manifest["splits"]["eval"]["path"])
    if mutation == "label":
        frame.loc[0, "role"] = "Unsupported futurist"
    elif mutation == "family":
        frame.loc[0, "family"] = "Wrong family"
    elif mutation == "marker":
        frame.loc[0, "data_origin"] = "real_2026"
    elif mutation == "experience":
        frame.loc[0, "exp_min"] = np.inf
    elif mutation == "sample":
        frame.at[0, "skills_eval"] = ["not in skills"] * 5
    else:
        frame.loc[1, "source_group_id"] = frame.loc[0, "source_group_id"]
    with pytest.raises(ValueError):
        validate_split(frame, ROLES, "eval")


def test_generator_refuses_inadequate_independent_sources(tmp_path, inputs):
    source, evaluation, vocab = inputs
    overlapping = pd.read_parquet(source).assign(posting_id=lambda d: "copy-" + d.posting_id)
    overlapping.to_parquet(evaluation, index=False)
    with pytest.raises(ValueError, match="unseen evaluation groups"):
        generate(source, evaluation, tmp_path / "bad", rows=60, vocabulary_path=vocab)
