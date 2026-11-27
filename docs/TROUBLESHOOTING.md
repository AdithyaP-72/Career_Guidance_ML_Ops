# Phase 3 troubleshooting

| Symptom | Check or command |
|---|---|
| Compose says the engine is unavailable | `docker info` and `docker desktop start --detach`. For Podman, `podman machine list` and `podman machine start` |
| Missing model or Parquet files | `uv run dvc pull`. Use the configured DagsHub remote with your authorized credentials |
| Docker build cannot copy `.phase3/bundle` | `uv run python -m scripts.prepare_phase3` before `docker compose up --build -d` |
| Port is occupied | Set `PHASE3_API_PORT`, `PHASE3_PROMETHEUS_PORT` or `PHASE3_GRAFANA_PORT` before starting Compose. Pass corresponding `--api` or `--prometheus` origins to the demo |
| API is unavailable | `docker compose ps` and `docker compose logs api` to inspect startup, model loading and memory |
| `/metrics` or `/drift` returns 401 | Use the local monitoring bearer token. Check the same file is readable by Prometheus and API containers |
| Prometheus shows the target down | `docker compose logs prometheus`. Check `/targets`, the service name `api:8000` and authorization file |
| Monitoring reports insufficient samples | Run a labelled replay with enough accepted predictions. Check the rolling window and that traffic reached the instrumented API |
| Alert does not immediately fire | Allow for the scrape and evaluation intervals plus the rule's sustained `for` duration. Check the actual PSI score and the rules on `/alerts` |
| Grafana login fails | Initial password resides in `.phase3/secrets/grafana_password`. The Grafana named volume preserves its first configured password and later admin changes |
| Registry or retraining panels are empty | Export optional registry evidence with preparation or run the local retraining command to generate the corresponding report |
| Candidate was rejected | Inspect `decision_reason` and gate values in the saved report. Rejection preserves the existing alias |
| GitHub does not show a Run workflow button | Run locally from the feature branch. The workflow definition must exist on the default branch before dispatch |
| An experiment is already running | Let the process holding the local file lock finish, or inspect it before retrying. Each reports directory has its own state |

To verify alert syntax and logic with the pinned Prometheus image, run from the repository root in PowerShell:

```powershell
docker run --rm --entrypoint /bin/promtool --mount "type=bind,source=$((Get-Location).Path)/monitoring,target=/work,readonly" --workdir /work prom/prometheus:v3.5.1 test rules alerts.test.yml
```
