"""Evaluate the trained model against baselines; write reports/metrics.json and log to MLflow."""
from __future__ import annotations

import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from src.data.common import ROOT, load_params  # noqa: E402
from src.features.build import sample_n_skills, sample_n_vocab_skills  # noqa: E402
from src.models.common import topk_accuracy  # noqa: E402
from src.models.train import TRACKING_URI  # noqa: E402

P = ROOT / "data/processed"
R = ROOT / "reports"


def profile_matching_proba(train: pd.DataFrame, vocab_pipe, test: pd.DataFrame, classes) -> np.ndarray:
    """Baseline with no training: cosine similarity between user skills and each role's mean skill vector."""
    cv = vocab_pipe.named_transformers_["skills"]
    Xtr = cv.transform(train["skills"]).astype(float)
    y = train["role"].to_numpy()
    prof = np.vstack([np.asarray(Xtr[y == c].mean(axis=0)).ravel() for c in classes])
    prof /= np.linalg.norm(prof, axis=1, keepdims=True) + 1e-9
    Xte = cv.transform(test["skills"]).astype(float).toarray()
    Xte /= np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9
    return Xte @ prof.T


def evaluate_model(pipe, df: pd.DataFrame) -> dict:
    proba = pipe.predict_proba(df[["skills", "experience"]])
    classes = pipe.classes_
    pred = classes[proba.argmax(1)]
    fam_of = dict(zip(df["role"], df["family"]))
    top3 = classes[np.argsort(-proba, axis=1)[:, :3]]
    fam_hit = np.mean([df["family"].iloc[i] in {fam_of.get(r) for r in top3[i]} for i in range(len(df))])
    return {
        "top1": topk_accuracy(proba, classes, df["role"], 1),
        "top3": topk_accuracy(proba, classes, df["role"], 3),
        "top5": topk_accuracy(proba, classes, df["role"], 5),
        "macro_f1": float(f1_score(df["role"], pred, average="macro")),
        "family_top3": float(fam_hit),
    }


def independent_checks(pipe, params) -> dict:
    """Validation that does not depend on our title rules.

    (1) Model trained on rule labels from 2025, scored on Naukri 2019 postings against the RECRUITER-chosen `Role`.
    (2) Hand-labelled gold sample, if data/gold/gold_labelled.csv exists.
    """
    import yaml

    out = {}
    rmap = yaml.safe_load((ROOT / "configs/naukri_role_map.yaml").read_text(encoding="utf-8"))["map"]
    d19 = pd.read_parquet(P / "all_2019.parquet")
    d19 = d19[d19["naukri_role"].isin(rmap)]
    if len(d19):
        proba = pipe.predict_proba(d19[["skills", "experience"]])
        top3 = pipe.classes_[np.argsort(-proba, axis=1)[:, :3]]
        hit3 = np.mean([any(r in rmap[nr] for r in row) for row, nr in zip(top3, d19["naukri_role"])])
        hit1 = np.mean([row[0] in rmap[nr] for row, nr in zip(top3, d19["naukri_role"])])
        out["naukri_2019_recruiter_role"] = {"top3_hit": float(hit3), "top1_hit": float(hit1), "n": int(len(d19)),
                                             "note": "model trained on 2025 rule labels, scored on 2019 postings vs recruiter-chosen role"}
    gold = ROOT / "data/gold/gold_labelled.csv"
    if gold.exists():
        g = pd.read_csv(gold)
        g = g[g["true_role"].notna() & (g["true_role"] != "NONE") & g["true_role"].isin(pipe.classes_)]
        if len(g):
            out["hand_labelled_gold"] = {"rule_accuracy": float((g["rule_role"] == g["true_role"]).mean()), "n": int(len(g))}
    return out


def main() -> None:
    params = load_params()
    n_eval = params["features"]["eval_n_skills"]
    pipe = joblib.load(ROOT / "models/model.pkl")
    train = pd.read_parquet(P / "train.parquet")
    test = pd.read_parquet(P / "test.parquet")
    test5 = sample_n_skills(test, n_eval, seed=params["split"]["seed"])

    metrics = {"full_postings": evaluate_model(pipe, test), f"{n_eval}_skills": evaluate_model(pipe, test5)}

    # baselines on the 5-skill headline input
    classes = pipe.classes_
    top3_major = train["role"].value_counts().head(3).index
    metrics["baseline_majority_top3"] = float(test5["role"].isin(top3_major).mean())
    pm = profile_matching_proba(train, pipe["features"], test5, classes)
    metrics["baseline_profile_matching"] = {
        "top1": topk_accuracy(pm, classes, test5["role"], 1),
        "top3": topk_accuracy(pm, classes, test5["role"], 3),
    }
    # accuracy vs number of skills entered
    metrics["by_n_skills"] = {}
    for k in (3, 5, 8):
        sub = sample_n_skills(test, k, seed=params["split"]["seed"])
        metrics["by_n_skills"][str(k)] = {"top3": evaluate_model(pipe, sub)["top3"], "n": int(len(sub))}
    # realistic UI input: skills drawn only from the vocabulary the search box offers
    vocab = set(pipe["features"].named_transformers_["skills"].vocabulary_)
    vsub = sample_n_vocab_skills(test, n_eval, params["split"]["seed"], vocab)
    metrics["in_vocabulary_5_skills"] = {k: v for k, v in evaluate_model(pipe, vsub).items() if k in ("top1", "top3", "top5")}
    metrics["independent_checks"] = independent_checks(pipe, params)
    metrics["headline_top3_5_skills"] = metrics[f"{n_eval}_skills"]["top3"]
    metrics["n_test"] = int(len(test))
    metrics["n_test_5_skills"] = int(len(test5))

    # per-role report + family-level confusion matrix
    proba = pipe.predict_proba(test5[["skills", "experience"]])
    pred = classes[proba.argmax(1)]
    fam = dict(zip(train["role"], train["family"]))
    rows = []
    for c in classes:
        m = test5["role"] == c
        rows.append({"role": c, "n": int(m.sum()), "top1_recall": float((pred[m.values] == c).mean()) if m.any() else None})
    R.mkdir(exist_ok=True)
    pd.DataFrame(rows).sort_values("top1_recall").to_csv(R / "per_role.csv", index=False)

    fams = sorted(set(fam.values()))
    cm = confusion_matrix(test5["family"], [fam[p] for p in pred], labels=fams, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(fams)), fams, rotation=90, fontsize=7)
    ax.set_yticks(range(len(fams)), fams, fontsize=7)
    ax.set_title("Family-level confusion (5-skill inputs, row-normalised)")
    fig.tight_layout()
    fig.savefig(R / "confusion_matrix.png", dpi=110)

    (R / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    run_file = ROOT / "models/mlflow_run.json"
    if run_file.exists():
        mlflow.set_tracking_uri(TRACKING_URI)
        with mlflow.start_run(run_id=json.loads(run_file.read_text(encoding="utf-8"))["run_id"]):
            flat = {"top3_5skills": metrics["headline_top3_5_skills"],
                    "top3_full": metrics["full_postings"]["top3"],
                    "top1_5skills": metrics[f"{n_eval}_skills"]["top1"],
                    "macro_f1": metrics[f"{n_eval}_skills"]["macro_f1"],
                    "profile_matching_top3": metrics["baseline_profile_matching"]["top3"]}
            lq = ROOT / "reports/label_quality.json"
            if lq.exists():
                flat["label_agreement_2019"] = json.loads(lq.read_text(encoding="utf-8"))["family_agreement_2019_it_grouped"]
            mlflow.log_metrics(flat)
            mlflow.log_artifact(str(R / "confusion_matrix.png"))


if __name__ == "__main__":
    sys.exit(main())
