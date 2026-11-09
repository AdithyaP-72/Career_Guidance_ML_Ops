"""Build the model bundle: the ONE folder the API reads, and the only model input to the Docker image.

    uv run python -m src.serving.bundle --from-registry [--alias champion]   # what CD does
    uv run python -m src.serving.bundle --from-local                          # dev: models/ from dvc pull

bundle/
    model.skops            the fitted pipeline in skops format (safe to load: no arbitrary code, unlike pickle)
    trusted_types.json     the non-sklearn types that file needs; checked against src/models/train.py
    skill_profiles.json    per-role core skills + growth  (readiness, skill gap, learning path)
    skill_cooccurrence.npz + skill_vocab.json              (learning path order)
    meta.json              which model/version/run this is, for /model and the logs

--from-registry downloads `models:/<name>@<alias>` plus the `serving/` files that register.py stored
in the same MLflow run, so the lookups always match the model they were registered with.
This module needs the `pipeline` dependency group (MLflow); the API itself does not.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import skops.io

from src.models.train import TRUSTED_TYPES
from src.utils import MODELS, REPORTS, ROOT, get_logger, load_params, read_json, write_json

log = get_logger("bundle")

LOOKUPS = ["skill_profiles.json", "skill_cooccurrence.npz", "skill_vocab.json"]


def finalise(out: Path, meta: dict) -> dict:
    """Check model.skops only needs types we trust, record them, smoke-test it, write meta.json."""
    needed = skops.io.get_untrusted_types(file=out / "model.skops")
    unexpected = sorted(set(needed) - set(TRUSTED_TYPES))
    if unexpected:
        raise SystemExit(f"model.skops contains types not in TRUSTED_TYPES (src/models/train.py): {unexpected}")
    write_json(sorted(needed), out / "trusted_types.json")
    pipe = skops.io.load(out / "model.skops", trusted=needed)
    missing = [f for f in LOOKUPS if not (out / f).exists()]
    if missing:
        raise SystemExit(f"bundle is missing {missing}")
    meta |= {"n_classes": len(pipe.classes_), "built_at": datetime.now(UTC).isoformat(timespec="seconds")}
    write_json(meta, out / "meta.json")
    log.info("bundle ready in %s: %s", out, {k: meta[k] for k in ("model_name", "version", "algorithm")})
    return meta


def write_local(out: Path, pipe, lookups_dir: Path, meta: dict) -> dict:
    """Bundle an in-memory pipeline + lookup files (used by --from-local and the tests)."""
    out.mkdir(parents=True, exist_ok=True)
    skops.io.dump(pipe, out / "model.skops")
    for f in LOOKUPS:
        shutil.copyfile(lookups_dir / f, out / f)
    return finalise(out, meta)


def from_local(out: Path) -> dict:
    selected = read_json(REPORTS / "metrics.json")
    meta = {
        "source": "local",
        "model_name": load_params()["register"]["model_name"],
        "version": "local",
        "alias": None,
        "algorithm": selected["best_model"],
        "run_id": selected.get("mlflow_run_id"),
        "top3_5skills": selected["best_model_metrics"]["top3_5skills"],
    }
    return write_local(out, joblib.load(MODELS / "model.joblib"), MODELS, meta)


def from_registry(out: Path, alias: str) -> dict:
    import mlflow  # pipeline group only
    from mlflow import MlflowClient

    from src.models.tracking import configure

    p = load_params()
    name = p["register"]["model_name"]
    configure(p["train"]["experiment"])  # reads .env / MLFLOW_* environment variables
    mv = MlflowClient().get_model_version_by_alias(name, alias)
    log.info("%s@%s is version %s (run %s)", name, alias, mv.version, mv.run_id)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(mlflow.artifacts.download_artifacts(f"models:/{name}@{alias}", dst_path=f"{tmp}/model"))
        shutil.copyfile(model_dir / "model.skops", out / "model.skops")
        serving = Path(mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="serving", dst_path=tmp))
        for f in LOOKUPS:
            shutil.copyfile(serving / f, out / f)
    meta = {
        "source": "registry",
        "model_name": name,
        "version": str(mv.version),
        "alias": alias,
        "algorithm": mv.tags.get("model"),
        "run_id": mv.run_id,
        "model_md5": mv.tags.get("model_md5"),
        "top3_5skills": float(mv.tags["top3_5skills"]) if "top3_5skills" in mv.tags else None,
        "validation_status": mv.tags.get("validation_status"),
    }
    return finalise(out, meta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-registry", action="store_true", help="download a registered version (CD)")
    src.add_argument("--from-local", action="store_true", help="use models/ from dvc pull (development)")
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--out", type=Path, default=ROOT / "bundle")
    args = ap.parse_args()
    if args.out.exists():
        shutil.rmtree(args.out)
    meta = from_registry(args.out, args.alias) if args.from_registry else from_local(args.out)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
