"""Champion / challenger gate.

After `dvc repro` (or `make retrain`) the freshly trained model in models/ is the *challenger*. It is scored on the
same fixed test inputs as the serving *champion* (models/champion/). It is promoted only if its top-3 accuracy on
5-skill inputs is at least as good (within TOLERANCE). The decision is logged to MLflow (registry aliases) and to
models/promotion_log.json.
"""
from __future__ import annotations

import json
import shutil
import time

import joblib
import mlflow
import pandas as pd

from src.data.common import ROOT, load_params
from src.features.build import sample_n_skills
from src.models.evaluate import evaluate_model
from src.models.recommend import CHAMPION, MODELS
from src.models.train import TRACKING_URI

TOLERANCE = 0.0
REGISTERED = "career-role-classifier"
FILES = ["model.pkl", "skill_profiles.json", "education_prior.json", "course_links.json"]
REPORTS = ["metrics.json", "label_quality.json", "drift_2019_2025.json", "model_comparison.json", "per_role.csv"]


def score(pipe, test5) -> dict:
    m = evaluate_model(pipe, test5)
    return {"top3": m["top3"], "top1": m["top1"], "macro_f1": m["macro_f1"]}


def main(force: bool = False) -> dict:
    params = load_params()
    test = pd.read_parquet(ROOT / "data/processed/test.parquet")
    test5 = sample_n_skills(test, params["features"]["eval_n_skills"], params["split"]["seed"])
    challenger = joblib.load(MODELS / "model.pkl")
    c_score = score(challenger, test5)

    champ_score, decision = None, "promoted (no champion yet)"
    promote = True
    if (CHAMPION / "model.pkl").exists() and not force:
        champ = joblib.load(CHAMPION / "model.pkl")
        # a champion trained on other roles cannot be compared fairly on this test set -> restrict to shared roles
        shared = set(champ.classes_) & set(challenger.classes_)
        t5 = test5[test5["role"].isin(shared)]
        champ_score = score(champ, t5)
        c_score = score(challenger, t5)
        promote = c_score["top3"] >= champ_score["top3"] - TOLERANCE
        decision = "promoted" if promote else "rejected (challenger worse than champion)"
    elif force:
        decision = "promoted (forced)"

    if promote:
        CHAMPION.mkdir(exist_ok=True)
        for f in FILES:
            shutil.copy2(MODELS / f, CHAMPION / f)
        (CHAMPION / "reports").mkdir(exist_ok=True)
        for f in REPORTS:  # so a fresh clone can show the Insights tab without re-running the pipeline
            if (ROOT / "reports" / f).exists():
                shutil.copy2(ROOT / "reports" / f, CHAMPION / "reports" / f)

    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "decision": decision, "challenger": c_score, "champion": champ_score}
    log_f = MODELS / "promotion_log.json"
    log = json.loads(log_f.read_text(encoding="utf-8")) if log_f.exists() else []
    log.append(rec)
    log_f.write_text(json.dumps(log[-20:], indent=2), encoding="utf-8")

    try:  # registry aliases in the local MLflow store
        run_file = MODELS / "mlflow_run.json"
        if run_file.exists():
            mlflow.set_tracking_uri(TRACKING_URI)
            run_id = json.loads(run_file.read_text(encoding="utf-8"))["run_id"]
            mv = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED)
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(REGISTERED, "challenger", mv.version)
            if promote:
                client.set_registered_model_alias(REGISTERED, "champion", mv.version)
    except Exception as e:  # registry is a nice-to-have; never block serving on it
        print("registry skipped:", e)
    print(json.dumps(rec, indent=2))
    return rec


if __name__ == "__main__":
    import sys

    main(force="--force" in sys.argv)
