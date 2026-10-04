"""Real drift evidence: 2019 postings = reference, 2025 = 'live'.

Writes reports/drift_2019_2025.json:
  * skill-distribution drift (JS distance, PSI, biggest risers/fallers)
  * a model trained on 2019 only, scored on held-out 2019 vs on 2025 (the performance drop)
"""
from __future__ import annotations

import json

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.common import ROOT, load_params
from src.features.build import build_features, sample_n_skills, skill_dropout
from src.models.evaluate import evaluate_model
from src.models.train import TRACKING_URI, make_estimator
from src.monitoring import drift

P = ROOT / "data/processed"
MIN_2019 = 30


def main() -> None:
    params = load_params()
    fp, tp, seed = params["features"], params["train"], params["split"]["seed"]
    d19 = pd.read_parquet(P / "labelled_2019.parquet")
    test25 = pd.read_parquet(P / "test.parquet")

    counts = d19["role"].value_counts()
    roles = counts[counts >= MIN_2019].index
    d19 = d19[d19["role"].isin(roles)].reset_index(drop=True)
    tr19, te19 = train_test_split(d19, test_size=0.2, random_state=seed, stratify=d19["role"])
    te25 = test25[test25["role"].isin(roles)]

    aug = skill_dropout(tr19, fp["dropout_copies"], fp["dropout_min"], fp["dropout_max"], seed)
    pipe = Pipeline([("features", build_features(fp["top_k_skills"], fp["exp_clip"])), ("clf", make_estimator(tp, "logreg"))])
    pipe.fit(aug[["skills", "experience"]], aug["role"])

    n = fp["eval_n_skills"]
    m19 = evaluate_model(pipe, sample_n_skills(te19, n, seed))
    m25 = evaluate_model(pipe, sample_n_skills(te25, n, seed))

    vocab = list(pipe["features"].named_transformers_["skills"].vocabulary_)
    ref = drift.skill_frequencies(d19["skills"].tolist(), vocab)
    live = drift.skill_frequencies(pd.read_parquet(P / "labelled_2025.parquet")["skills"].tolist(), vocab)
    delta = pd.Series(live - ref, index=vocab)
    mover = lambda s: [{"skill": k, "share_2019": round(float(ref[vocab.index(k)]), 4), "share_2025": round(float(live[vocab.index(k)]), 4)} for k in s.index]  # noqa: E731
    js = drift.js_distance(ref, live)
    res = {
        "n_roles_compared": int(len(roles)),
        "js_distance": round(js, 4), "psi": round(drift.psi(ref, live), 4), "status": drift.status(js),
        "top3_on_2019": round(m19["top3"], 4), "top3_on_2025": round(m25["top3"], 4),
        "performance_drop": round(m19["top3"] - m25["top3"], 4),
        "risers": mover(delta.nlargest(8)), "fallers": mover(delta.nsmallest(8)),
    }
    (ROOT / "reports/drift_2019_2025.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("risers", "fallers")}, indent=2))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("career-drift")
    with mlflow.start_run(run_name="2019-model-on-2025"):
        mlflow.log_metrics({"top3_2019": res["top3_on_2019"], "top3_2025": res["top3_on_2025"],
                            "performance_drop": res["performance_drop"], "js_distance": js})


if __name__ == "__main__":
    main()
