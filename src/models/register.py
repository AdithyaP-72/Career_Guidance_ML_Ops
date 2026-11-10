"""Register the selected model in the MLflow Model Registry:   uv run python -m src.models.register [--dry-run]

Run by hand AFTER `dvc repro` (select + validate). Not a DVC stage on purpose: it changes shared
state on the team's MLflow server, so a person decides when it happens.

What it does:
    1. reads reports/metrics.json (which model select picked, its MLflow run + logged model)
       and reports/validation.json (did it pass the quality gate)
    2. attaches what the API needs next to the model in that MLflow run:
           serving/   the lookup files in src/serving/bundle.py → LOOKUPS (profiles, co-occurrence, vocab,
                      courses, keyness): stored with the model so a deployed version always gets its own copy
           validation/  validation.json, model_card.md
    3. registers the logged model as a new version of params.yaml → register.model_name
       (or reuses the existing version if this exact model file was registered before; see model_md5)
    4. moves aliases, which is what deployment reads:
           @champion            the version that is served (CD builds the image from it)
           @previous-champion   the champion it replaced, for one-command rollback
           @challenger          passed the gate but did not beat the champion
       A version that fails the gate gets no alias, only validation_status=failed.

Aliases, not "stages": MLflow deprecated Staging/Production stages in 2.9; aliases are the replacement.
"""

from __future__ import annotations

import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.models.tracking import configure, md5
from src.serving.bundle import LOOKUPS
from src.utils import MODELS, REPORTS, get_logger, load_params, read_json

log = get_logger("register")

CHAMPION, PREVIOUS, CHALLENGER = "champion", "previous-champion", "challenger"
SERVING_FILES = LOOKUPS  # what bundle.py downloads again from serving/


def decide(passed: bool, candidate: float, champion: float | None, min_improvement: float) -> str:
    """'champion' | 'challenger' | 'rejected'. Pure function (unit-tested)."""
    if not passed:
        return "rejected"
    if champion is None or candidate > champion + min_improvement:
        return CHAMPION
    return CHALLENGER


def current_champion(client: MlflowClient, name: str):
    try:
        return client.get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        return None  # no registered model yet, or nothing has the alias


def existing_version(client: MlflowClient, name: str, model_md5: str):
    """A version already registered from this exact model file (same md5 as in dvc.lock)."""
    try:
        versions = client.search_model_versions(f"name = '{name}'")
    except MlflowException:
        return None
    return next((v for v in versions if v.tags.get("model_md5") == model_md5), None)


def register_candidate(
    client: MlflowClient,
    name: str,
    model_uri: str,
    tags: dict[str, str],
    metric: str,
    passed: bool,
    min_improvement: float,
    dry_run: bool = False,
) -> tuple[str | None, str]:
    """Register (or reuse) a version and move aliases. Returns (version, decision)."""
    champ = current_champion(client, name)
    champ_score = float(champ.tags[metric]) if champ and metric in champ.tags else None
    version = existing_version(client, name, tags["model_md5"])

    if version is not None and champ is not None and str(version.version) == str(champ.version):
        return str(version.version), "already champion"
    decision = decide(passed, float(tags[metric]), champ_score, min_improvement)
    log.info(
        "candidate %s=%s vs champion %s → %s",
        metric,
        tags[metric],
        f"v{champ.version} ({champ_score})" if champ else "none",
        decision,
    )
    if dry_run:
        return (str(version.version) if version else None), decision

    if version is None:
        version = mlflow.register_model(model_uri, name, tags=tags)
        log.info("registered %s version %s from %s", name, version.version, model_uri)
    client.set_model_version_tag(name, version.version, "decision", decision)

    if decision == CHAMPION:
        if champ is not None:
            client.set_registered_model_alias(name, PREVIOUS, champ.version)
        client.set_registered_model_alias(name, CHAMPION, version.version)
    elif decision == CHALLENGER:
        client.set_registered_model_alias(name, CHALLENGER, version.version)
    return str(version.version), decision


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    p = load_params()
    cfg, metric = p["register"], p["register"]["metric"]
    selected = read_json(REPORTS / "metrics.json")
    report = read_json(REPORTS / "validation.json")
    best = selected["best_model"]
    run = read_json(REPORTS / "runs" / f"{best}.json")

    if report["model"] != best:
        raise SystemExit(f"validation.json is for {report['model']!r} but select picked {best!r}: run `dvc repro`")
    if run["tracking"] == "local" or not selected.get("mlflow_model_uri"):
        raise SystemExit(
            f"{best}'s MLflow run is not on the team server (tracking={run['tracking']!r}, "
            f"model_uri={selected.get('mlflow_model_uri')}). Fill in .env, then rerun its training and "
            f"selection so they log to DagsHub:  uv run dvc repro -f -s train@{best} && uv run dvc repro"
        )

    model_md5 = md5(MODELS / "model.joblib")
    m = selected["best_model_metrics"]
    tags = {
        "model": best,
        "model_md5": model_md5,  # same hash as models/model.joblib in dvc.lock → registry ↔ DVC link
        "validation_status": "passed" if report["passed"] else "failed",
        "failed_checks": ",".join(report["failed_checks"]) or "none",
        metric: str(m[metric]),
        "ece_top1_5skills": str(report["checks"]["ece_top1_5skills"]["value"]),
        "ref2019_top3_5skills": str(m["ref2019_top3_5skills"]),
    }

    configure(run["experiment"])
    client = MlflowClient()
    if not dry_run:
        with mlflow.start_run(run_id=run["run_id"]):
            for f in SERVING_FILES:
                mlflow.log_artifact(str(MODELS / f), artifact_path="serving")
            mlflow.log_artifact(str(REPORTS / "validation.json"), artifact_path="validation")
            mlflow.log_artifact(str(REPORTS / "model_card.md"), artifact_path="validation")
            mlflow.set_tag("validation_status", tags["validation_status"])

    version, decision = register_candidate(
        client, cfg["model_name"], selected["mlflow_model_uri"], tags, metric, report["passed"],
        cfg["min_improvement"], dry_run,
    )  # fmt: skip
    log.info("%s%s v%s: %s", "[dry run] " if dry_run else "", cfg["model_name"], version or "(new)", decision)


if __name__ == "__main__":
    main()
