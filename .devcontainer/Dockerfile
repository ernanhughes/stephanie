# Start from Microsoft devcontainers base (gives you zsh/sudo/dev UX)
FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Install Python + build deps for Stephanie
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        build-essential \
        libpq-dev \
        git \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Working directory matches devcontainer mount
WORKDIR /workspace

# Default command (keep alive in dev)
CMD ["sleep", "infinity"]
