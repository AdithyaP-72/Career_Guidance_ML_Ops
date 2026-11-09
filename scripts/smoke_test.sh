#!/usr/bin/env bash
# Smoke-test an API image: start it, wait until healthy, make one real prediction, stop it.
#
#   scripts/smoke_test.sh <image> [expected top role]
#   scripts/smoke_test.sh career-api:dev Accountant
#
# Used by CI (toy model) and CD (the real champion, BEFORE it is pushed or deployed). Uses podman,
# or docker if podman isn't installed. Exits non-zero on any failure, which fails the workflow.
set -euo pipefail

IMAGE="$1"
EXPECTED="${2:-}"
ENGINE="$(command -v podman || command -v docker)"
NAME="smoke-$$"
PORT=18080

cleanup() {
  echo "── container logs ──"
  "$ENGINE" logs "$NAME" 2>&1 | tail -20 || true
  "$ENGINE" rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"$ENGINE" run -d --name "$NAME" -p "$PORT:8000" "$IMAGE" >/dev/null
echo "started $IMAGE with $ENGINE"

for i in $(seq 1 60); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null; then break; fi
  [ "$i" = 60 ] && { echo "FAIL: /health never answered"; exit 1; }
  sleep 2
done
echo "health: $(curl -s "http://localhost:$PORT/health")"

# The image's own HEALTHCHECK (only present when built with --format docker under podman).
if [ "$(basename "$ENGINE")" = podman ]; then
  "$ENGINE" healthcheck run "$NAME" && echo "HEALTHCHECK: healthy"
fi

BODY='{"skills": ["accounting", "tally", "gst", "tds", "excel"], "experience_years": 2}'
RESPONSE="$(curl -sf -X POST "http://localhost:$PORT/predict" -H 'Content-Type: application/json' -d "$BODY")"
TOP="$(echo "$RESPONSE" | python3 -c 'import json, sys; print(json.load(sys.stdin)["recommendations"][0]["role"])')"
echo "predict: top role = $TOP"
if [ -n "$EXPECTED" ] && [ "$TOP" != "$EXPECTED" ]; then
  echo "FAIL: expected '$EXPECTED'"
  exit 1
fi
echo "PASS"
