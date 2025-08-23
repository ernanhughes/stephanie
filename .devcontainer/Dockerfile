# -----------------------------------------------------------------------------
# Stephanie Dev Container
# -----------------------------------------------------------------------------
# Base: Microsoft Dev Containers (Ubuntu 24.04 with zsh, sudo, etc.)
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# -----------------------------------------------------------------------------
# Install system dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        build-essential \
        libpq-dev \
        git \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Create dedicated virtual environment for Stephanie
# -----------------------------------------------------------------------------
RUN python3 -m venv /venv \
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
