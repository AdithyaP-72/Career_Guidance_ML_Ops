# Everything runs locally. Requires only `uv` (https://docs.astral.sh/uv/).
.PHONY: setup pipeline serve test lint compare retrain

setup:      ## install pinned dependencies
	uv sync

pipeline:   ## reproduce data -> labels -> model -> metrics -> profiles -> lookups -> drift report
	uv run dvc repro
	uv run python -m src.models.promote

serve:      ## start the app on http://127.0.0.1:8000
	uv run python run.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests run.py

compare:    ## logreg vs random forest vs lightgbm (needs libomp on macOS: brew install libomp)
	uv run python -m src.models.compare

retrain:    ## rebuild with current data/rules, gate the new model against the champion
	uv run dvc repro
	uv run python -m src.models.promote
