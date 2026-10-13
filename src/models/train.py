"""Train, evaluate and log ONE baseline:  uv run python -m src.models.train <model>

<model> is one of params.yaml → train.models. dvc.yaml runs this once per model
(the `train@<model>` stages), so changing one model's settings retrains only that model.

    models/baselines/<model>.joblib     the fitted pipeline (features + model), DVC-tracked
    reports/metrics/<model>.json        its metrics (DVC metric; compare with `dvc metrics show`)
    reports/eval/<model>/               per-role report, top confusions, family confusion chart
    reports/runs/<model>.json           which MLflow run it became (used by select.py)
    + one MLflow run with params, metrics and the review files. The model file itself is
      uploaded to MLflow only for the best baseline, by select.py (the random forest alone
      is ~375 MB in MLflow's format; every model file is versioned in DVC anyway).
"""

from __future__ import annotations

import sys
import time

import joblib
import mlflow
from sklearn.pipeline import Pipeline

from src.features.build import EXPERIENCE, SKILLS, make_preprocessor, skill_dropout
from src.models.estimators import build_estimator
from src.models.evaluate import evaluate, read_split
from src.models.tracking import configure, flatten, provenance_tags
from src.utils import MODELS, REPORTS, ROOT, get_logger, load_params, write_json

log = get_logger("train")

# MLflow saves scikit-learn models with skops, which refuses to load any type it hasn't been
# told to trust (safer than pickle, which runs arbitrary code). These are the extra types our
# five pipelines contain: our own feature functions/estimator plus a few library internals.
# tests/test_models.py fails if a model needs a type that isn't listed here.
TRUSTED_TYPES = [
    "src.features.build.identity",
    "src.features.build.clip_experience",
    "src.models.estimators.ProfileMatcher",
    "numpy.dtype",
    "collections.OrderedDict",
    "sklearn.tree._tree.Tree",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "sklearn.neural_network._multilayer_perceptron.MLPClassifier",
    "sklearn.neural_network._stochastic_optimizers.AdamOptimizer",
    "sklearn.linear_model._stochastic_gradient.SGDClassifier",
    "sklearn.naive_bayes.ComplementNB",
]


def build_pipeline(name: str, p: dict) -> Pipeline:
    f = p["features"]
    return Pipeline(
        [
            ("features", make_preprocessor(f["top_k_skills"], f["exp_clip"])),
            ("model", build_estimator(name, p["train"].get(name))),
        ]
    )


def fit(pipe: Pipeline, train, p: dict):
    """Fit on the training split, with skill-dropout copies if enabled. Returns (pipe, rows, seconds)."""
    data = skill_dropout(train, **p["features"]["dropout"]) if p["train"]["skill_dropout"] else train
    t = time.perf_counter()
    pipe.fit(data[[SKILLS, EXPERIENCE]], data["role"])
    return pipe, len(data), round(time.perf_counter() - t, 1)


def log_model(pipe: Pipeline, example) -> str:
    """Upload a fitted pipeline to the active MLflow run (skops format + our src/ code). Returns its URI."""
    info = mlflow.sklearn.log_model(
        pipe, name="model", code_paths=[str(ROOT / "src")], input_example=example, skops_trusted_types=TRUSTED_TYPES
    )
    return info.model_uri


def main() -> None:
    name = sys.argv[1]
    p = load_params()
    if name not in p["train"]["models"]:
        raise SystemExit(f"{name!r} is not in params.yaml → train.models")

    train, test, ref = read_split("train"), read_split("test"), read_split("reference_2019")
    family_of = dict(zip(train["role"], train["family"], strict=True))

    pipe, n_rows, fit_s = fit(build_pipeline(name, p), train, p)
    log.info("%s: fitted on %d rows in %.1fs", name, n_rows, fit_s)

    eval_dir = REPORTS / "eval" / name
    metrics = evaluate(pipe, test, ref, family_of, eval_dir)
    model_path = MODELS / "baselines" / f"{name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path, compress=3)
    metrics |= {"fit_seconds": fit_s, "model_size_mb": round(model_path.stat().st_size / 1e6, 2)}
    write_json(metrics, REPORTS / "metrics" / f"{name}.json")
    log.info("%s: %s", name, {k: metrics[k] for k in ("top3_5skills", "top1_5skills", "macro_f1_5skills")})

    uri = configure(p["train"]["experiment"])
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags(
            provenance_tags(["train", "test", "reference_2019"])
            | {"model": name, "dvc_stage": f"train@{name}", "kind": "baseline"}
        )
        mlflow.log_params(
            flatten(
                {
                    "model": name,
                    name: p["train"].get(name) or {},
                    "features": p["features"],
                    "skill_dropout": p["train"]["skill_dropout"],
                    "data.min_skills": p["data"]["min_skills"],
                    "taxonomy": p["taxonomy"],
                    "n_train_rows": n_rows,
                    "n_classes": len(pipe.classes_),
                }
            )
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(eval_dir), artifact_path="eval")
    write_json(
        {
            "model": name,
            "run_id": run.info.run_id,
            "experiment": p["train"]["experiment"],
            "tracking": "local" if uri.startswith("sqlite") else uri,
        },
        REPORTS / "runs" / f"{name}.json",
    )
    log.info("%s: logged MLflow run %s", name, run.info.run_id)


if __name__ == "__main__":
    main()
