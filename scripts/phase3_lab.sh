#!/usr/bin/env bash
# Phase 3 local lab on Linux/macOS: one entry point for the drift demo.
#
#   scripts/phase3_lab.sh prepare   # synthetic 2026 data + Compose bundle + drift reference + secrets
#   scripts/phase3_lab.sh up        # build and start API, Prometheus, Grafana (runs prepare if needed)
#   scripts/phase3_lab.sh demo      # replay real 2025 then synthetic 2026 traffic, wait for the alert
#   scripts/phase3_lab.sh retrain   # drift check -> challenger fit -> isolated registry (~2 min)
#   scripts/phase3_lab.sh promote|rollback "Reviewer name"
#   scripts/phase3_lab.sh registry  # local MLflow UI for the isolated registry on :5002
#   scripts/phase3_lab.sh real-drift  # real 2019 vs 2025 degradation report (no containers)
#   scripts/phase3_lab.sh test-alerts # promtool unit tests for monitoring/alerts.yml
#   scripts/phase3_lab.sh status | logs | down | urls
#
# Uses Podman if present (rootless socket is started automatically), else Docker.
# Ports can be moved with PHASE3_API_PORT / PHASE3_PROMETHEUS_PORT / PHASE3_GRAFANA_PORT.
set -euo pipefail
cd "$(dirname "$0")/.."

API_PORT="${PHASE3_API_PORT:-8000}"
PROM_PORT="${PHASE3_PROMETHEUS_PORT:-9090}"
GRAFANA_PORT="${PHASE3_GRAFANA_PORT:-3000}"
export PHASE3_API_PORT="$API_PORT" PHASE3_PROMETHEUS_PORT="$PROM_PORT" PHASE3_GRAFANA_PORT="$GRAFANA_PORT"
PHASE3_GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo local-phase3)"
export PHASE3_GIT_SHA
PY=(uv run --no-sync python)

engine() {
  if command -v podman >/dev/null 2>&1; then
    # podman compose delegates to docker-compose, which talks to the Podman API socket.
    if [[ -z "${DOCKER_HOST:-}" && "$(uname)" == Linux ]]; then
      systemctl --user start podman.socket 2>/dev/null || true
      export DOCKER_HOST="unix:///run/user/$(id -u)/podman/podman.sock"
    fi
    echo podman
  elif command -v docker >/dev/null 2>&1; then
    echo docker
  else
    echo "Neither podman nor docker is installed." >&2
    exit 1
  fi
}

compose() { local e; e="$(engine)"; "$e" compose "$@"; }

need_data() {
  local missing=0
  for f in data/processed/train.parquet data/processed/test.parquet data/processed/reference_2019.parquet \
           models/model.joblib models/skill_vocab.json reports/metrics.json reports/validation.json; do
    [[ -e "$f" ]] || { echo "missing: $f" >&2; missing=1; }
  done
  if (( missing )); then
    echo "Run 'uv run dvc pull' (DagsHub credentials) and retry." >&2
    exit 1
  fi
}

prepare() {
  need_data
  [[ -f data/drift/synthetic_2026_manifest.json ]] || "${PY[@]}" -m scripts.generate_2026_drift
  "${PY[@]}" -m scripts.prepare_phase3 "$@"
}

urls() {
  cat <<EOF

  API health   http://localhost:${API_PORT}/health      (try requests: http://localhost:${API_PORT}/docs)
  Prometheus   http://localhost:${PROM_PORT}/alerts
  Grafana      http://localhost:${GRAFANA_PORT}/d/career-phase3/
               login: admin / $(cat .phase3/secrets/grafana_password 2>/dev/null || echo '<run prepare>')
  /metrics and /drift need: Authorization: Bearer \$(cat .phase3/secrets/monitoring_token)
EOF
}

wait_healthy() {
  for _ in $(seq 1 60); do
    curl -fsS "http://localhost:${API_PORT}/health" >/dev/null 2>&1 \
      && curl -fsS "http://localhost:${GRAFANA_PORT}/api/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "Stack did not become healthy; check: $0 logs" >&2
  return 1
}

cmd="${1:-help}"
shift || true
case "$cmd" in
  prepare) prepare "$@" ;;
  up)
    [[ -f .phase3/bundle/drift_reference.json && -f reports/phase3/pipeline.json ]] || prepare
    compose up --build -d
    wait_healthy
    compose ps
    urls
    ;;
  demo)
    "${PY[@]}" -m scripts.demo_drift --api "http://127.0.0.1:${API_PORT}" \
      --prometheus "http://127.0.0.1:${PROM_PORT}" "$@"
    ;;
  retrain) "${PY[@]}" -m scripts.retrain_phase3 "$@" ;;
  promote | rollback)
    [[ $# -ge 1 ]] || { echo "usage: $0 $cmd \"Reviewer name\"" >&2; exit 2; }
    "${PY[@]}" -m scripts.retrain_phase3 "--$cmd" --approved-by "$1"
    ;;
  registry)
    "${PY[@]}" -m mlflow server --backend-store-uri sqlite:///reports/phase3/retraining/registry.db \
      --host 127.0.0.1 --port 5002 --workers 1
    ;;
  real-drift) "${PY[@]}" -m scripts.real_drift_2019 ;;
  test-alerts)
    e="$(engine)"
    "$e" run --rm --entrypoint /bin/promtool -v "$PWD/monitoring:/work:ro,Z" -w /work \
      docker.io/prom/prometheus:v3.5.1 test rules alerts.test.yml
    ;;
  status) compose ps ;;
  logs) compose logs --tail=100 "$@" ;;
  down) compose down "$@" ;;
  urls) urls ;;
  *) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
