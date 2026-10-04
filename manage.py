"""Cross-platform task runner (Windows / macOS / Linux). Usage: uv run python manage.py <command>

  serve      start the app on http://127.0.0.1:8000
  download   fetch the raw datasets (only needed to rebuild the model)
  pipeline   rebuild everything: data -> labels -> model -> metrics -> profiles -> lookups -> drift, then promote
  retrain    same as pipeline (use after new data or rule edits); the new model is promoted only if not worse
  compare    logistic regression vs random forest vs LightGBM
  test       run the test-suite
  simulate   send sample traffic to a running app (add --shift to simulate drift)
"""
import subprocess
import sys

PY = [sys.executable]
STEPS = {
    "serve": [PY + ["run.py"]],
    "download": [PY + ["scripts/download.py"]],
    "pipeline": [PY + ["-m", "dvc", "repro"], PY + ["-m", "src.models.promote"]],
    "retrain": [PY + ["-m", "dvc", "repro"], PY + ["-m", "src.models.promote"]],
    "compare": [PY + ["-m", "src.models.compare"]],
    "test": [PY + ["-m", "pytest", "-q"]],
    "simulate": [PY + ["scripts/simulate_traffic.py"]],
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in STEPS:
        print(__doc__)
        sys.exit(1)
    for step in STEPS[cmd]:
        code = subprocess.call(step + (sys.argv[2:] if step is STEPS[cmd][-1] else []))
        if code:
            sys.exit(code)
