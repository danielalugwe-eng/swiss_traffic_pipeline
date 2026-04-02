# =============================================================================
# Dockerfile — Swiss Traffic MLOps Pipeline
# =============================================================================
#
# WHAT THIS DOES IN ONE SENTENCE:
# Builds a self-contained Linux box that has everything needed to run the
# full 7-stage pipeline — Python, Bruin CLI, all packages, and the project code.
#
# HOW TO BUILD:
#   docker build -t swiss-traffic .
#
# HOW TO RUN (use docker-compose instead — see docker-compose.yml):
#   docker compose up
#
# LAYER STRATEGY (each RUN = one cached layer):
# Docker builds images in layers. If a layer hasn't changed, it's reused from
# cache. We order layers from "least likely to change" → "most likely to change"
# so that editing your Python code doesn't re-download the internet.
# =============================================================================

# ── Base Image ──────────────────────────────────────────────────────────────
# python:3.11-slim = Debian Linux with Python 3.11 pre-installed, minimal size.
# "slim" removes docs and test suites — saves ~400MB vs the full image.
# We pin to 3.11 (not "latest") so the image is reproducible.
FROM python:3.11-slim

# ── System Dependencies ──────────────────────────────────────────────────────
# curl    : needed to download the Bruin CLI installer
# git     : Bruin may need it for some internal operations
# ca-certs: SSL certificates for HTTPS downloads
# We clean up apt caches in the same RUN command to keep the layer small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Install Bruin CLI ────────────────────────────────────────────────────────
# Bruin provides an official install script. This drops the `bruin` binary
# into /usr/local/bin so it's on PATH for all subsequent steps.
# We install a pinned version for reproducibility.
RUN curl -LsSf https://raw.githubusercontent.com/bruin-data/bruin/main/install.sh \
    | sh -s -- -b /usr/local/bin

# ── Install uv (fast Python package manager) ─────────────────────────────────
# uv is 10-100x faster than pip for installing packages.
# Bruin uses uv under the hood; we also use it directly.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx

# ── Set Working Directory ─────────────────────────────────────────────────────
# All subsequent commands run from /app inside the container.
# This maps to the project root on your host (via volume mount in compose).
WORKDIR /app

# ── Install Python Dependencies ───────────────────────────────────────────────
# Copy ONLY pyproject.toml first. Docker will cache this layer.
# If you only change pipeline code (not dependencies), this layer is reused
# and you skip the ~2 minute install step on rebuilds.
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml 2>/dev/null \
    || pip install --no-cache-dir \
        "pandas>=2.1.0" \
        "numpy>=1.26.0" \
        "duckdb>=0.10.0" \
        "scikit-learn>=1.4.0" \
        "mlflow>=2.11.0" \
        "joblib>=1.3.0" \
        "matplotlib>=3.8.0" \
        "seaborn>=0.13.0" \
        "jinja2>=3.1.0"

# ── Copy Project Code ─────────────────────────────────────────────────────────
# Copy everything else AFTER dependencies.
# This means editing a Python file only invalidates this layer (fast rebuild).
# .dockerignore excludes .venv, __pycache__, mlruns, traffic.duckdb, etc.
COPY . .

# ── Create Output Directories ──────────────────────────────────────────────────
# Ensure reports/ and models/ exist even if the volume isn't mounted yet.
RUN mkdir -p reports models mlruns

# ── Environment Variables ─────────────────────────────────────────────────────
# Tell matplotlib to use the non-interactive "Agg" backend.
# Without this, matplotlib would crash in a headless container (no display).
ENV MPLBACKEND=Agg

# Tell Python not to buffer stdout/stderr so logs appear immediately in
# `docker compose logs`. Without this, you'd see nothing until the script finishes.
ENV PYTHONUNBUFFERED=1

# MLflow tracking directory (inside the container, mapped to host via volume)
ENV MLFLOW_TRACKING_URI=/app/mlruns

# ── Default Command ───────────────────────────────────────────────────────────
# What runs when you do `docker compose up pipeline`.
# Runs the full 7-stage pipeline end to end.
CMD ["python", "run_pipeline.py"]
