# syntax=docker/dockerfile:1
# Career Guidance API image (works with Podman and Docker: both build OCI-standard images).
#
#   Build:  the model bundle must exist first (it is baked into the image, so image tag = model version):
#             uv run python -m src.serving.bundle --from-registry      # real champion (needs MLflow creds)
#             uv run python -m tests.toy_bundle bundle                  # or: toy model, for a quick try
#           podman build --format docker -t career-api .      (or: docker build -t career-api .)
#   Run:    podman run --rm -p 8000:8000 career-api           → http://localhost:8000/docs
#
# Podman note: `--format docker` matters. Podman's default OCI image format has no HEALTHCHECK field,
# so without it the healthcheck below is silently dropped. Image names are fully qualified
# (docker.io/..., ghcr.io/...) so Podman never has to ask which registry to pull from.
#
# Two stages: `build` installs dependencies with uv from the lockfile (exact versions), `runtime`
# copies only the virtualenv + code + bundle. No compilers, no uv, no DVC/MLflow in the final image.

ARG PYTHON_VERSION=3.12

# ── build: dependencies only (cached until pyproject.toml / uv.lock change) ──
FROM docker.io/library/python:${PYTHON_VERSION}-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
# Base dependencies + the `serve` group only (no pipeline/dev groups), exactly as pinned in uv.lock.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-default-groups --group serve

# ── runtime ──
FROM docker.io/library/python:${PYTHON_VERSION}-slim AS runtime
# LightGBM needs the OpenMP runtime (only used if the champion is a LightGBM model, but cheap to keep).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Non-root user: if the app were ever compromised, the attacker is not root inside the container.
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY src/ src/
COPY configs/skill_aliases.yaml configs/skill_aliases.yaml
COPY configs/skill_equivalents.yaml configs/skill_equivalents.yaml
COPY bundle/ bundle/

# Which commit this image was built from; /health reports it so CD can confirm what is live.
ARG GIT_SHA=dev
ENV PATH="/app/.venv/bin:$PATH" \
    GIT_SHA=${GIT_SHA} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BUNDLE_DIR=/app/bundle \
    PORT=8000
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
# `sh -c` so $PORT is expanded (Render injects its own PORT; elsewhere it stays 8000), and
# `exec` so uvicorn replaces the shell and receives stop signals directly (clean shutdown).
CMD ["sh", "-c", "exec uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
