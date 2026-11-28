"""Prepare the isolated Compose build context, reference and read-only dashboard evidence."""

import argparse
import os
import secrets
import subprocess
import sys
from datetime import UTC, datetime

from src.serving.bundle import from_local
from src.utils import REPORTS, ROOT, load_yaml, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-evidence", action="store_true", help="Read aliases from configured MLflow")
    args = parser.parse_args()
    runtime = ROOT / ".phase3"
    secret_dir = runtime / "secrets"
    secret_dir.mkdir(parents=True, exist_ok=True)
    for name in ("monitoring_token", "grafana_password"):
        path = secret_dir / name
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(secrets.token_urlsafe(32))
    bundle = runtime / "bundle"
    from_local(bundle)
    subprocess.run(
        [sys.executable, "-m", "src.models.drift", "baseline", "--output", str(bundle / "drift_reference.json")],
        cwd=ROOT,
        check=True,
    )
    selected = read_json(REPORTS / "metrics.json")
    validation = read_json(REPORTS / "validation.json")
    lock = load_yaml(ROOT / "dvc.lock")["stages"]
    model_out = next(o for o in lock["select"]["outs"] if o["path"] == "models/model.joblib")
    training = next(o for o in lock["split"]["outs"] if o["path"] == "data/processed/train.parquet")
    evidence = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "run_id": selected["mlflow_run_id"],
        "algorithm": selected["best_model"],
        "model_md5": model_out["md5"],
        "train_md5": training["md5"],
        "validation": validation["checks"],
        "pipeline": list(lock),
        "dataset": "Naukri 2025",
        "limitations": ["Rule-derived labels", "Career exploration only", "Synthetic drift is a controlled simulation"],
    }
    if args.registry_evidence:
        from dotenv import load_dotenv
        from mlflow import MlflowClient

        load_dotenv(ROOT / ".env")
        uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if not uri.startswith("https://dagshub.com/"):
            raise SystemExit("Set the project's DagsHub MLflow tracking URI for registry evidence.")
        client = MlflowClient(tracking_uri=uri)
        name = "career-role-classifier"
        aliases = client.get_registered_model(name).aliases
        evidence["registry"] = {
            alias: {"version": str(version), "algorithm": client.get_model_version(name, version).tags.get("model")}
            for alias, version in aliases.items()
            if alias in {"champion", "previous-champion", "challenger"}
        }
    write_json(evidence, REPORTS / "phase3" / "pipeline.json")
    print("Compose bundle, drift reference and dashboard evidence are ready.")
    print("Local Grafana login: admin. Password file: .phase3/secrets/grafana_password")


if __name__ == "__main__":
    main()
