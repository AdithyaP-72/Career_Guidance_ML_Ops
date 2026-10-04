"""Train + evaluate several model types and log them side by side in MLflow (does not touch models/model.pkl)."""
from __future__ import annotations

import json

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.common import ROOT, load_params
from src.features.build import build_features, sample_n_skills, skill_dropout
from src.models.evaluate import evaluate_model
from src.models.train import TRACKING_URI, data_hash, git_commit, make_estimator

P = ROOT / "data/processed"


def main(models=("logreg", "rf", "lgbm")) -> None:
    params = load_params()
    fp, tp = params["features"], params["train"]
    train = pd.read_parquet(P / "train.parquet")
    test5 = sample_n_skills(pd.read_parquet(P / "test.parquet"), fp["eval_n_skills"], params["split"]["seed"])
    aug = skill_dropout(train, fp["dropout_copies"], fp["dropout_min"], fp["dropout_max"], params["split"]["seed"])
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("career-compare")
    rows = []
    for name in models:
        pipe = Pipeline([("features", build_features(fp["top_k_skills"], fp["exp_clip"])), ("clf", make_estimator(tp, name))])
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"git_commit": git_commit(), "data_md5": data_hash(), "model": name})
            mlflow.log_params({**{f"features.{k}": v for k, v in fp.items()}, "model": name})
            pipe.fit(aug[["skills", "experience"]], aug["role"])
            m = evaluate_model(pipe, test5)
            mlflow.log_metrics({"top3_5skills": m["top3"], "top1_5skills": m["top1"], "macro_f1": m["macro_f1"], "family_top3": m["family_top3"]})
            rows.append({"model": name, "top3": round(m["top3"], 4), "top1": round(m["top1"], 4), "macro_f1": round(m["macro_f1"], 4), "family_top3": round(m["family_top3"], 4)})
            print(f"{name:7s} top3={m['top3']:.3f} top1={m['top1']:.3f} macroF1={m['macro_f1']:.3f} familyTop3={m['family_top3']:.3f}")

    (ROOT / "reports/model_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
