# syntax=docker/dockerfile:1
# -----------------------------------------------------------------------------
# Stephanie — full-system image
#
# Builds the Stephanie reasoning engine and the SIS web application. The
# companion services (PostgreSQL+pgvector, NATS, SearXNG, Ollama) are defined in
# docker-compose.yml; this image is the application tier.
#
# See docs/docker.md for the full guide.
# -----------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_ENABLE_HF_TRANSFER=0

# System libraries required to build and run the dependency set
# (psycopg2, lxml, unstructured, pdfminer, tokenizers, ...).
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libpq-dev \
        libmagic1 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so source edits don't invalidate this layer.
COPY requirements.txt requirements_sis.txt pyproject.toml setup.py MANIFEST.in README.md ./
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r requirements.txt \
 && python -m pip install -r requirements_sis.txt

# Copy the source and install the project (console entry points + package data).
COPY . .
RUN python -m pip install -e . --no-deps

# SIS listens here (see config/sis.yaml and docker-compose.yml).
EXPOSE 8000

# Default process: the SIS web application. Override for pipelines, e.g.
#   docker compose run --rm app python -m stephanie.main goal.goal_text="..."
CMD ["python", "-m", "sis"]
