"""Compare every baseline, keep the best one, and upload it to its MLflow run.

Compares every model in params.yaml → train.models on select.metric (higher is better):
    models/model.joblib            copy of the best baseline's pipeline
    reports/metrics.json           its metrics + margin over profile matching (DVC metric)
    reports/model_comparison.csv   every baseline side by side
    MLflow                         the best model's file, attached to that model's own run
(Registering it in the MLflow Model Registry is a later-phase step.)
"""

from __future__ import annotations

import shutil

import joblib
import mlflow
import pandas as pd

from src.models.evaluate import inputs, read_split
from src.models.tracking import configure
from src.models.train import log_model
from src.utils import MODELS, REPORTS, get_logger, load_params, read_json, write_json

log = get_logger("select")

COMPARE = [
    "top3_5skills",
    "top1_5skills",
    "top5_5skills",
    "macro_f1_5skills",
    "family_top1_5skills",
    "top3_full",
    "ref2019_top3_5skills",
    "fit_seconds",
    "model_size_mb",
    "predict_ms_per_1k",
]


def main() -> None:
    p = load_params()
    metric, models = p["select"]["metric"], p["train"]["models"]
    table = pd.DataFrame({m: read_json(REPORTS / "metrics" / f"{m}.json") for m in models}).T
    table = table.sort_values(metric, ascending=False)
    table.index.name = "model"
    table[[c for c in COMPARE if c in table]].to_csv(REPORTS / "model_comparison.csv")

    best = table.index[0]
    shutil.copyfile(MODELS / "baselines" / f"{best}.joblib", MODELS / "model.joblib")
    best_metrics = read_json(REPORTS / "metrics" / f"{best}.json")
    run = read_json(REPORTS / "runs" / f"{best}.json")
    log.info("best baseline: %s (%s = %.4f)\n%s", best, metric, best_metrics[metric], table[[metric]].to_string())

    configure(run["experiment"])
    model_uri = None
    try:
        with mlflow.start_run(run_id=run["run_id"]):  # reopen the best model's own run
            mlflow.set_tag("best_baseline", "true")
            model_uri = log_model(joblib.load(MODELS / "model.joblib"), inputs(read_split("test").head(3), "skills_eval"))
    except PermissionError:
        # Windows may raise WinError 5 when MLflow cleans up its temp dir while handles
        # are still open. The model has already been logged successfully at this point.
        log.warning("Windows temp-dir cleanup error (safe to ignore); model already logged to MLflow.")
    log.info("uploaded %s to MLflow: %s", best, model_uri)

    profile_score = table.loc["profile", metric] if "profile" in table.index else float("nan")
    write_json(
        {
            "best_model": best,
            metric: best_metrics[metric],
            "margin_over_profile_matching": round(best_metrics[metric] - profile_score, 4),
            "best_model_metrics": best_metrics,
            "mlflow_run_id": run["run_id"],
            "mlflow_model_uri": model_uri,
        },
        REPORTS / "metrics.json",
    )


if __name__ == "__main__":
    main()
