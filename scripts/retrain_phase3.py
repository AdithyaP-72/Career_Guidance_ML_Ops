"""Check for drift and evaluate an experimental challenger in a local MLflow registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from filelock import FileLock
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from scripts.generate_2026_drift import source_group_id, validate_split
from src.features.build import EXPERIENCE, SKILLS, skill_dropout
from src.models.drift import build_reference, compare_frame
from src.models.evaluate import inputs, read_split
from src.models.register import CHALLENGER, CHAMPION, PREVIOUS, decide, existing_version, switch_champion_alias
from src.models.tracking import flatten, isolated_experiment, md5, provenance_tags
from src.models.train import build_pipeline, log_model
from src.models.validate import evaluate_labelled, p95_latency_ms, paired_gain
from src.models.validate import run as validate_model
from src.utils import CONFIGS, MODELS, REPORTS, ROOT, get_logger, load_params, load_yaml, read_json, write_json

REGISTRY = "career-role-classifier-phase3-synthetic-2026"
EXPERIMENT = "phase3-synthetic-2026"
ALGORITHM = "complement_nb_cal"
DEFAULT_OUT = REPORTS / "phase3" / "retraining"
DEFAULT_DATA = ROOT / "data" / "drift"
POLICY = {
    "minimum_shifted_gain": 0.01,
    "maximum_real_top3_loss": 0.01,
    "maximum_real_macro_f1_loss": 0.02,
    "maximum_real_calibration_increase": 0.015,
    "maximum_shifted_ece": 0.10,
    "maximum_shifted_macro_f1_loss": 0.02,
}
log = get_logger("phase3.retrain")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(data: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    write_json(data, tmp)
    os.replace(tmp, path)


def save(report: dict, output: Path) -> None:
    report["updated_at"] = now()
    if report.get("run_key"):
        atomic_json(report, output / "runs" / report["run_key"] / "report.json")
    atomic_json(report, output / "latest.json")


def stage(report: dict, output: Path, name: str, status: str) -> None:
    report["stages"][name] = {"status": status, "timestamp": now()}
    save(report, output)
    log.info("%s: %s", name, status)


def audit(report: dict, action: str, **details) -> None:
    report["audit"].append({"action": action, "at": now(), **details})


def read_manifest(data_dir: Path) -> dict:
    manifest = read_json(data_dir / "synthetic_2026_manifest.json")
    if manifest["schema_version"] != 1 or manifest["dataset_origin"] != "synthetic_2026":
        raise ValueError("The manifest must describe the synthetic 2026 dataset.")
    sources = {
        "train": ROOT / "data" / "processed" / "train.parquet",
        "evaluation": ROOT / "data" / "processed" / "test.parquet",
    }
    for source, path in sources.items():
        if manifest["inputs"][source]["sha256"] != digest(path):
            raise ValueError(f"Synthetic {source} source differs from the current DVC artifact.")
    if manifest["generator_sha256"] != digest(ROOT / "scripts" / "generate_2026_drift.py"):
        raise ValueError("Synthetic generator changed. Regenerate the manifest and its partitions.")
    groups = [set(manifest["splits"][s]["source_group_ids"]) for s in ("train", "eval", "holdout")]
    if any(not split for split in groups) or sum(map(len, groups)) != len(set.union(*groups)):
        raise ValueError("The synthetic manifest must contain three disjoint, nonempty source-group sets.")
    return manifest


def verify_synthetic(data_dir: Path, manifest: dict, split: str) -> Path:
    path = data_dir / f"synthetic_2026_{split}.parquet"
    if digest(path) != manifest["splits"][split]["sha256"]:
        raise ValueError(f"Synthetic {split} checksum differs from its manifest.")
    return path


def read_synthetic(data_dir: Path, manifest: dict, split: str, roles: dict[str, str]) -> pd.DataFrame:
    entry = manifest["splits"][split]
    frame = pd.read_parquet(verify_synthetic(data_dir, manifest, split))
    validate_split(frame, roles, split)
    if len(frame) != entry["rows"] or set(frame["source_group_id"]) != set(entry["source_group_ids"]):
        raise ValueError(f"Synthetic {split} row or source group count differs from its manifest.")
    for column in (SKILLS, "skills_eval"):
        frame[column] = frame[column].map(list)
    return frame


def locked_artifacts() -> dict[str, str]:
    """Bind the benchmark to the DVC checkout, including the frozen champion."""
    lock = load_yaml(ROOT / "dvc.lock")["stages"]
    paths = (
        "data/processed/train.parquet",
        "data/processed/test.parquet",
        "data/processed/reference_2019.parquet",
        "models/model.joblib",
        "models/baselines/profile.joblib",
    )
    hashes = {}
    for path in paths:
        wanted = {artifact["md5"] for s in lock.values() for artifact in s.get("outs", []) if artifact["path"] == path}
        actual = md5(ROOT / path)
        if wanted != {actual}:
            raise ValueError(f"{path} is not the artifact pinned in dvc.lock. Run dvc pull before retraining.")
        hashes[path] = actual
    return hashes


def fit_challenger(real: pd.DataFrame, synthetic: pd.DataFrame, params: dict) -> tuple[Pipeline, int, float]:
    """Keep all copies of a source posting together in each calibration fold."""
    real = real[[SKILLS, EXPERIENCE, "role", "family"]].copy()
    real["source_group_id"] = [
        source_group_id(role, family, skills, exp)
        for skills, exp, role, family in real.itertuples(index=False, name=None)
    ]
    base = pd.concat([real, synthetic[real.columns]], ignore_index=True)
    train = skill_dropout(base, **params["features"]["dropout"]) if params["train"]["skill_dropout"] else base
    features, target = train[[SKILLS, EXPERIENCE]], train["role"]
    pipe = build_pipeline(ALGORITHM, params)
    calibrated = pipe.named_steps["model"]
    folds = calibrated.cv
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    calibrated.cv = list(splitter.split(features, target, groups=train["source_group_id"]))
    start = time.perf_counter()
    pipe.fit(features, target)
    # Predict uses the fitted calibrator, not the fold indices. Keep the portable training setting.
    calibrated.cv = folds
    return pipe, len(train), round(time.perf_counter() - start, 2)


def comparison_checks(champion: dict, challenger: dict, shifted_hits: tuple[np.ndarray, np.ndarray]) -> dict:
    def minimum(value: float, limit: float) -> dict:
        return {"value": value, "rule": ">=", "threshold": limit, "passed": math.isfinite(value) and value >= limit}

    def maximum(value: float, limit: float) -> dict:
        return {"value": value, "rule": "<=", "threshold": limit, "passed": math.isfinite(value) and value <= limit}

    shift_a, shift_b = champion["synthetic_eval"], challenger["synthetic_eval"]
    real_a, real_b = champion["real_2025"], challenger["real_2025"]
    gain, interval = paired_gain(*shifted_hits)
    return {
        "shifted_top3_gain": minimum(gain, POLICY["minimum_shifted_gain"]) | {"paired_95_ci": interval},
        "shifted_gain_ci_lower": minimum(interval[0], 0.0),
        "shifted_macro_f1_loss": maximum(
            shift_a["macro_f1"] - shift_b["macro_f1"], POLICY["maximum_shifted_macro_f1_loss"]
        ),
        "shifted_ece": maximum(shift_b["ece"], POLICY["maximum_shifted_ece"]),
        "real_top3_loss": maximum(real_a["top3"] - real_b["top3"], POLICY["maximum_real_top3_loss"]),
        "real_macro_f1_loss": maximum(real_a["macro_f1"] - real_b["macro_f1"], POLICY["maximum_real_macro_f1_loss"]),
        "real_ece_increase": maximum(real_b["ece"] - real_a["ece"], POLICY["maximum_real_calibration_increase"]),
    }


def registry_state(client: MlflowClient) -> dict:
    try:
        aliases = client.get_registered_model(REGISTRY).aliases
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        aliases = {}
    return {
        "name": REGISTRY,
        "aliases": {alias: str(version) for alias, version in aliases.items()},
        "scope": "local_sqlite",
    }


def register_local(
    client: MlflowClient,
    experiment_id: str,
    pipe: Pipeline,
    sample: pd.DataFrame,
    details: dict,
    tags: dict,
    run_dir: Path,
) -> dict:
    version = existing_version(client, REGISTRY, details["model_md5"])
    if version and version.tags.get("phase3_experiment") == tags["phase3_experiment"]:
        return details | {"run_id": version.run_id, "registry_version": str(version.version)}
    with mlflow.start_run(experiment_id=experiment_id, run_name=tags["purpose"]) as run:
        mlflow.set_tags(tags | {"model_md5": details["model_md5"], "data_origin": "synthetic_2026"})
        scalars = {
            f"{split}.{metric}": value
            for split, metrics in details["metrics"].items()
            for metric, value in metrics.items()
            if isinstance(value, (int, float))
        }
        mlflow.log_metrics(
            scalars | {"p95_latency_ms": details["p95_latency_ms"], "model_size_mb": details["model_size_mb"]}
        )
        mlflow.log_params(flatten(details["training_config"]))
        model_uri = log_model(pipe, sample.head(3))
        mlflow.log_artifacts(str(run_dir / "evidence"), artifact_path="phase3")
    version = mlflow.register_model(
        model_uri,
        REGISTRY,
        tags=tags | {"model_md5": details["model_md5"], "data_origin": "synthetic_2026"},
    )
    return details | {
        "run_id": run.info.run_id,
        "registry_version": str(version.version),
        "logged_model_uri": model_uri,
    }


def set_aliases(client: MlflowClient, report: dict, *, rollback: bool, approved_by: str) -> None:
    """Operate only on validated versions from this local experiment."""
    aliases = registry_state(client)["aliases"]
    current = aliases.get(CHAMPION)
    if rollback:
        previous = aliases.get(PREVIOUS)
        challenger = (report.get("challenger") or {}).get("registry_version")
        expected_previous = (report.get("champion") or {}).get("registry_version")
        if report["decision"] != "promoted" or current != challenger or previous != expected_previous or not previous:
            raise ValueError("Rollback requires this experiment's promoted champion and recorded previous champion.")
        destination = previous
        decision = "rolled_back"
    else:
        challenger = (report.get("challenger") or {}).get("registry_version")
        expected_current = (report.get("champion") or {}).get("registry_version")
        passed_gate = report.get("quality_gate", {}).get("passed") is True
        checks = report.get("comparison_checks", {})
        if (
            report["decision"] != "eligible"
            or current != expected_current
            or not challenger
            or not passed_gate
            or not checks
            or any(check.get("passed") is not True for check in checks.values())
        ):
            raise ValueError("Promotion requires eligibility and the same champion that was evaluated.")
        version = client.get_model_version(REGISTRY, challenger)
        if (
            version.tags.get("phase3_experiment") != report["run_key"]
            or version.tags.get("model_md5") != report["challenger"]["model_md5"]
            or version.tags.get("validation_status") != "passed"
        ):
            raise ValueError("The challenger registry record does not match the validated experiment.")
        destination, decision = challenger, "promoted"
    if current is None:
        raise ValueError("An isolated champion must exist before an alias change.")
    switch_champion_alias(client, REGISTRY, current, destination, aliases.get(PREVIOUS))
    audit(report, decision, approved_by=approved_by, from_version=current, to_version=destination)
    report["registry"] = registry_state(client)
    report["decision"] = decision


def run_key(hashes: dict, manifest: dict, config: dict) -> str:
    data = {
        "dvc": hashes,
        "synthetic_splits": {s: row["sha256"] for s, row in manifest["splits"].items()},
        "config": config,
        "implementations": {
            path: digest(ROOT / path)
            for path in (
                "scripts/retrain_phase3.py",
                "src/models/drift.py",
                "src/models/train.py",
                "src/models/estimators.py",
                "src/models/validate.py",
                "src/models/register.py",
                "src/models/tracking.py",
                "src/features/build.py",
                "configs/behaviour_cases.yaml",
                "uv.lock",
            )
        },
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def check_science_sources(manifest: dict, train: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    """Reject drift evaluations whose source features reoccur in real model fitting data."""
    columns = ["role", "family", SKILLS, EXPERIENCE]
    train_groups = {source_group_id(*row) for row in train[columns].itertuples(index=False, name=None)}
    evaluation_groups = {source_group_id(*row) for row in evaluation[columns].itertuples(index=False, name=None)}
    synthetic_training_groups = set(manifest["splits"]["train"]["source_group_ids"])
    excluded_groups = set(manifest["splits"]["eval"]["source_group_ids"]) | set(
        manifest["splits"]["holdout"]["source_group_ids"]
    )
    if not synthetic_training_groups <= train_groups or excluded_groups & train_groups:
        raise ValueError("Synthetic partitions contain sources inconsistent with the 2025 training split.")
    if not excluded_groups <= evaluation_groups:
        raise ValueError("Synthetic evaluation or holdout origins are outside the held-out 2025 split.")


def skip_decision(overall_psi: float | None, threshold: float) -> str | None:
    if overall_psi is None:
        return "skipped_insufficient"
    return "skipped_no_drift" if overall_psi < threshold else None


def model_card(report: dict) -> str:
    c, b = report["champion"], report["challenger"]
    return (
        "# Experimental challenger model card\n\n"
        f"- Experiment `{report['run_key']}` used real 2025 training and controlled synthetic 2026 training.\n"
        f"- {b['algorithm']} used {b['model_size_mb']:.2f} MB with model p95 of {b['p95_latency_ms']:.2f} ms.\n"
        f"- Synthetic evaluation top-3 changed {c['metrics']['synthetic_eval']['top3']:.2%} → "
        f"{b['metrics']['synthetic_eval']['top3']:.2%}.\n"
        f"- Real 2025 top-3 changed {c['metrics']['real_2025']['top3']:.2%} → "
        f"{b['metrics']['real_2025']['top3']:.2%}.\n"
        f"- Synthetic holdout top-3 changed {c['metrics']['synthetic_holdout']['top3']:.2%} → "
        f"{b['metrics']['synthetic_holdout']['top3']:.2%}.\n"
        f"- The quality gate {'passed' if report['quality_gate']['passed'] else 'failed'}. "
        f"The registry decision was `{report['decision']}`.\n\n"
        "This model supports offline career guidance drift research using simulated 2026 behavior. "
        "The experiment measures performance on job-role labels. "
        "Production release requires independently collected labelled data and explicit approval.\n"
    )


def evaluate_workflow(data_dir: Path, output: Path, threshold: float, min_samples: int, trigger: str) -> dict:
    params, manifest = load_params(), read_manifest(data_dir)
    hashes = locked_artifacts()
    config = {
        "features": params["features"],
        "model": params["train"][ALGORITHM],
        "skill_dropout": params["train"]["skill_dropout"],
        "quality_gate": params["validate"],
        "comparison": POLICY,
        "drift_threshold": threshold,
        "min_samples": min_samples,
        "calibration": "stratified groups by source posting; seed 42",
    }
    key = run_key(hashes, manifest, config)
    job = output / "runs" / key
    if (job / "report.json").exists():
        prior = read_json(job / "report.json")
        if prior["status"] in {"completed", "skipped"}:
            verified_splits = ("train", "eval", "holdout") if prior["status"] == "completed" else ("eval",)
            for split in verified_splits:
                verify_synthetic(data_dir, manifest, split)
            audit(prior, "reused_evidence", trigger=trigger)
            save(prior, output)
            return prior
    real, test, history = read_split("train"), read_split("test"), read_split("reference_2019")
    check_science_sources(manifest, real, test)
    roles = dict(zip(real["role"], real["family"], strict=True))
    shifted = read_synthetic(data_dir, manifest, "eval", roles)
    previous = read_json(REPORTS / "metrics.json")
    champion = joblib.load(MODELS / "model.joblib")
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    report = {
        "schema_version": 1,
        "mode": "synthetic_2026_experiment",
        "status": "running",
        "decision": None,
        "run_key": key,
        "started_at": now(),
        "trigger": trigger,
        "stages": {},
        "audit": [],
        "lineage": provenance_tags(["train", "test", "reference_2019"])
        | {
            "code_dirty": bool(git_status),
            "dvc_artifact_md5": hashes,
            "synthetic_sha256": {s: row["sha256"] for s, row in manifest["splits"].items()},
            "synthetic_manifest_sha256": digest(data_dir / "synthetic_2026_manifest.json"),
            "training_configuration": config,
            "production_promotion_allowed": False,
        },
    }
    stage(report, output, "retrieve_and_validate_data", "passed")
    reference = build_reference(
        real,
        champion,
        skills_column=SKILLS,
        min_samples=min_samples,
        model_run_id=previous["mlflow_run_id"],
        source_sha256=manifest["inputs"]["train"]["sha256"],
    )
    drift = compare_frame(reference, shifted, champion, skills_column=SKILLS)
    report["drift"] = drift
    stage(report, output, "detect_drift", drift["status"])
    if decision := skip_decision(drift["overall_psi"], threshold):
        report["decision"] = decision
        report["status"], report["completed_at"] = "skipped", now()
        audit(report, report["decision"])
        stage(report, output, "train_challenger", "skipped")
        return report
    champ_shifted, hits_a = evaluate_labelled(champion, shifted)
    champ_real, _ = evaluate_labelled(champion, test)
    champ_latency = p95_latency_ms(champion, inputs(test, "skills_eval"), params["validate"]["latency_requests"])
    stage(report, output, "evaluate_champion", "passed")

    training = read_synthetic(data_dir, manifest, "train", roles)
    candidate_path, fit_path = job / "challenger.joblib", job / "fit.json"
    if (
        fit_path.exists()
        and candidate_path.exists()
        and (fit_record := read_json(fit_path))["md5"] == md5(candidate_path)
    ):
        challenger = joblib.load(candidate_path)
    else:
        challenger, rows, elapsed = fit_challenger(real, training, params)
        joblib.dump(challenger, candidate_path.with_suffix(".tmp"), compress=3)
        os.replace(candidate_path.with_suffix(".tmp"), candidate_path)
        fit_record = {"md5": md5(candidate_path), "train_rows": rows, "fit_seconds": elapsed, "trained_at": now()}
        atomic_json(fit_record, fit_path)
    stage(report, output, "train_challenger", "passed")
    challenger_real, _ = evaluate_labelled(challenger, test)
    challenger_shifted, hits_b = evaluate_labelled(challenger, shifted)
    challenger_history, _ = evaluate_labelled(challenger, history)
    profile, _ = evaluate_labelled(joblib.load(MODELS / "baselines" / "profile.joblib"), test)
    size = candidate_path.stat().st_size / 1e6
    gate_input = {
        "best_model": ALGORITHM,
        "margin_over_profile_matching": challenger_real["top3"] - profile["top3"],
        "best_model_metrics": {
            "top3_5skills": challenger_real["top3"],
            "ref2019_top3_5skills": challenger_history["top3"],
            "model_size_mb": size,
        },
    }
    cases = load_yaml(CONFIGS / "behaviour_cases.yaml")["cases"]
    quality = validate_model(challenger, test, gate_input, cases, params["validate"])
    checks = comparison_checks(
        {"real_2025": champ_real, "synthetic_eval": champ_shifted},
        {"real_2025": challenger_real, "synthetic_eval": challenger_shifted},
        (hits_a, hits_b),
    )
    report["quality_gate"], report["comparison_checks"] = quality, checks
    provisional = decide(
        quality["passed"] and all(c["passed"] for c in checks.values()),
        challenger_shifted["top3"],
        champ_shifted["top3"],
        0.0,
    )
    report["selection_before_holdout"] = "eligible" if provisional == CHAMPION else "rejected"
    stage(report, output, "select_on_evaluation", report["selection_before_holdout"])

    holdout = read_synthetic(data_dir, manifest, "holdout", roles)
    a_holdout, _ = evaluate_labelled(champion, holdout)
    b_holdout, _ = evaluate_labelled(challenger, holdout)
    # Predefined one-time holdout veto. Holdout is never used to refit or select a different model.
    holdout_gain = b_holdout["top3"] - a_holdout["top3"]
    checks["holdout_top3_gain"] = {"value": holdout_gain, "rule": ">=", "threshold": 0.0, "passed": holdout_gain >= 0}
    eligible = provisional == CHAMPION and checks["holdout_top3_gain"]["passed"]
    report["decision"] = "eligible" if eligible else "rejected"
    report["decision_reason"] = (
        "The required gates, evaluation comparisons and holdout veto passed."
        if eligible
        else ", ".join(quality["failed_checks"] + [name for name, check in checks.items() if not check["passed"]])
    )
    report["champion"] = {
        "algorithm": previous["best_model"],
        "source_run_id": previous["mlflow_run_id"],
        "model_md5": hashes["models/model.joblib"],
        "p95_latency_ms": champ_latency,
        "model_size_mb": (MODELS / "model.joblib").stat().st_size / 1e6,
        "training_config": {"origin": "pinned_real_2025_model"},
        "metrics": {"real_2025": champ_real, "synthetic_eval": champ_shifted, "synthetic_holdout": a_holdout},
    }
    report["challenger"] = {
        "algorithm": ALGORITHM,
        "model_md5": fit_record["md5"],
        "fitted_at": fit_record["trained_at"],
        "fit_seconds": fit_record["fit_seconds"],
        "train_rows": fit_record["train_rows"],
        "p95_latency_ms": quality["checks"]["p95_latency_ms"]["value"],
        "model_size_mb": size,
        "training_config": config,
        "metrics": {
            "real_2025": challenger_real,
            "synthetic_eval": challenger_shifted,
            "synthetic_holdout": b_holdout,
            "real_2019": challenger_history,
        },
    }
    stage(report, output, "evaluate_holdout", "passed")
    atomic_json(quality, job / "evidence" / "quality_gate.json")
    atomic_json(manifest, job / "evidence" / "dataset_manifest.json")
    atomic_json(report, job / "evidence" / "comparison.json")
    (job / "evidence" / "model_card.md").write_text(model_card(report), encoding="utf-8", newline="\n")
    tags = {
        "phase3_experiment": key,
        "git_commit": report["lineage"]["git_commit"],
        "production_promotion_allowed": "false",
        "synthetic_manifest_sha256": report["lineage"]["synthetic_manifest_sha256"],
    }
    with isolated_experiment(output, EXPERIMENT) as (client, experiment):
        sample = inputs(shifted, "skills_eval")
        found = existing_version(client, REGISTRY, report["champion"]["model_md5"])
        if found:
            report["champion"] |= {"registry_version": str(found.version), "run_id": found.run_id}
        else:
            report["champion"] = register_local(
                client,
                experiment,
                champion,
                sample,
                report["champion"],
                tags | {"purpose": "reference-2025", "validation_status": "approved_in_source_pipeline"},
                job,
            )
        if not registry_state(client)["aliases"].get(CHAMPION):
            client.set_registered_model_alias(REGISTRY, CHAMPION, report["champion"]["registry_version"])
        report["challenger"] = register_local(
            client,
            experiment,
            challenger,
            sample,
            report["challenger"],
            tags | {"purpose": "challenger-synthetic-2026", "validation_status": "passed" if eligible else "rejected"},
            job,
        )
        if eligible:
            client.set_registered_model_alias(REGISTRY, CHALLENGER, report["challenger"]["registry_version"])
        report["registry"] = registry_state(client)
    report["status"], report["completed_at"] = "completed", now()
    audit(report, report["decision"], reason=report["decision_reason"])
    stage(report, output, "register_and_record_decision", report["decision"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threshold", type=float, default=0.25, help="Minimum overall PSI triggering a candidate fit.")
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--trigger", choices=("local", "schedule", "workflow_dispatch"), default="local")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--promote", action="store_true", help="Approve an eligible candidate in the local registry.")
    actions.add_argument("--rollback", action="store_true", help="Restore the prior experimental champion.")
    parser.add_argument("--approved-by", help="Name or other local reviewer identifier to record in the audit history.")
    args = parser.parse_args()
    changes_alias = args.promote or args.rollback
    if changes_alias and not (args.approved_by or "").strip():
        parser.error("--promote and --rollback require --approved-by.")
    if not math.isfinite(args.threshold) or args.threshold <= 0 or args.min_samples <= 0:
        parser.error("Threshold and minimum sample count must be positive.")
    output = args.output.resolve()
    if output in {ROOT, REPORTS, MODELS, DEFAULT_DATA} or not output.name:
        parser.error("Choose a dedicated experiment output directory.")
    output.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(output / ".write.lock", timeout=0):
            if changes_alias:
                report = read_json(output / "latest.json")
                with isolated_experiment(output, EXPERIMENT) as (client, _):
                    set_aliases(client, report, rollback=args.rollback, approved_by=args.approved_by.strip())
                save(report, output)
            else:
                latest = output / "latest.json"
                prior = latest.read_bytes() if latest.exists() else None
                try:
                    report = evaluate_workflow(
                        args.data_dir.resolve(), output, args.threshold, args.min_samples, args.trigger
                    )
                except Exception as exc:
                    failure = {"status": "error", "decision": None, "at": now(), "error": str(exc)}
                    atomic_json(failure, output / "last_error.json")
                    if latest.exists() and latest.read_bytes() != prior:
                        report = read_json(latest)
                        report.update(failure)
                    else:
                        report = {
                            "schema_version": 1,
                            "mode": "synthetic_2026_experiment",
                            "trigger": args.trigger,
                            "stages": {},
                            "audit": [],
                        } | failure
                    audit(report, "execution_error", reason=str(exc))
                    save(report, output)
                    raise
            print(f"{report['decision']}: {output / 'latest.json'}")
    except Exception as exc:
        log.error("Phase 3 operation failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
