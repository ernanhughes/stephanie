# -----------------------------------------------------------------------------
# Stephanie Dev Container
# -----------------------------------------------------------------------------
# Base: Microsoft Dev Containers Python image with Python 3.13 on Debian Bookworm.
FROM mcr.microsoft.com/devcontainers/python:3.13-bookworm

# -----------------------------------------------------------------------------
# Install system dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        git \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Create dedicated virtual environment for Stephanie
# -----------------------------------------------------------------------------
RUN python -m venv /venv \
    && /venv/bin/pip install --upgrade pip setuptools wheel

# Add venv to PATH
ENV PATH="/venv/bin:$PATH"

# -----------------------------------------------------------------------------
# Preinstall Stephanie + dependencies in editable mode
# -----------------------------------------------------------------------------
# Copy only metadata files first (for caching layers)
WORKDIR /workspace
COPY pyproject.toml README.md ./

RUN pip install -e .

# Copy the full source last (so code changes don’t break cached install)
COPY . /workspace

# -----------------------------------------------------------------------------
# Default command (keep container alive in dev mode)
# -----------------------------------------------------------------------------
CMD ["sleep", "infinity"]
