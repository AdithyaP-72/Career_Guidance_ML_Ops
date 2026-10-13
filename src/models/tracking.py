"""MLflow setup + provenance tags, shared by every training/selection script.

Where runs go:
    MLFLOW_TRACKING_URI set (e.g. in .env)  → that server, normally our DagsHub MLflow:
        MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
        MLFLOW_TRACKING_USERNAME=<dagshub user>
        MLFLOW_TRACKING_PASSWORD=<dagshub token>
    not set                                 → a local SQLite file, mlflow.db (gitignored),
                                              browse with `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`

Every run is tagged with the git commit and the md5 of the exact data files used.
That's what makes it traceable: any run → which code (git) + which data (DVC hashes).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from src.utils import PROCESSED, ROOT


def configure(experiment: str) -> str:
    load_dotenv(ROOT / ".env")
    uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance_tags(data_files: list[str]) -> dict[str, str]:
    """git commit, whether code/config had uncommitted edits, and md5 of each data file (same as DVC's)."""
    code_paths = ["src", "configs", "params.yaml", "dvc.yaml"]
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", *code_paths)
    tags = {"git_commit": _git("rev-parse", "HEAD"), "code_dirty": str(bool(dirty) and dirty != "unknown")}
    for name in data_files:
        tags[f"data.{name}_md5"] = md5(PROCESSED / f"{name}.parquet")
    return tags


def flatten(d: dict, prefix: str = "") -> dict:
    """{'dropout': {'copies': 2}} → {'dropout.copies': 2} for mlflow.log_params."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out |= flatten(v, f"{key}.") if isinstance(v, dict) else {key: v}
    return out
