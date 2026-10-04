"""Train the role classifier and track it in MLflow.  `python -m src.models.train [--model rf|lgbm|logreg]`"""
from __future__ import annotations

import argparse
import json
import subprocess

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.common import ROOT, load_params
from src.features.build import build_features, skill_dropout

P = ROOT / "data/processed"
MODELS = ROOT / "models"
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"


def make_estimator(tp: dict, name: str):
    if name == "logreg":
        return LogisticRegression(C=tp["C"], class_weight=tp["class_weight"], max_iter=tp["max_iter"])
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight=tp["class_weight"], n_jobs=-1, random_state=42
        )
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=120, learning_rate=0.08, num_leaves=15, min_child_samples=10, colsample_bytree=0.2,
            reg_lambda=1.0, class_weight=tp["class_weight"], random_state=42, n_jobs=-1, verbose=-1,
        )
    raise ValueError(name)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "uncommitted"


def data_hash() -> str:
    lock = ROOT / "dvc.lock"
    if lock.exists():
        import hashlib
        return hashlib.md5(lock.read_bytes()).hexdigest()[:12]
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-save", action="store_true", help="track in MLflow only (experiment run)")
    args = ap.parse_args()

    params = load_params()
    fp, tp = params["features"], dict(params["train"])
    name = args.model or tp["model"]

    train = pd.read_parquet(P / "train.parquet")
    aug = skill_dropout(train, fp["dropout_copies"], fp["dropout_min"], fp["dropout_max"], seed=params["split"]["seed"])
    pipe = Pipeline(
        [("features", build_features(fp["top_k_skills"], fp["exp_clip"])), ("clf", make_estimator(tp, name))]
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("career-baselines")
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"git_commit": git_commit(), "data_md5": data_hash(), "model": name})
        mlflow.log_params({**{f"features.{k}": v for k, v in fp.items()}, "model": name,
                           "train_rows": len(aug), "n_roles": train["role"].nunique()})
        pipe.fit(aug[["skills", "experience"]], aug["role"])
        mlflow.sklearn.log_model(
            pipe, name="model",
            skops_trusted_types=["src.features.build.clip_scale", "src.features.build.identity"],
        )
        MODELS.mkdir(exist_ok=True)
        if not args.no_save:
            joblib.dump(pipe, MODELS / "model.pkl")
            (MODELS / "mlflow_run.json").write_text(json.dumps({"run_id": run.info.run_id, "model": name}), encoding="utf-8")
        print(f"trained {name}: vocab={len(pipe['features'].named_transformers_['skills'].vocabulary_)} run={run.info.run_id}")


if __name__ == "__main__":
    main()
