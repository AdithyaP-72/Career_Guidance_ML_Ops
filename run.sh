#!/usr/bin/env bash
# macOS / Linux:  ./run.sh   (first time: chmod +x run.sh)
set -e
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (one-time)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync
uv run python run.py
