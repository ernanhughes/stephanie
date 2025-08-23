# Stephanie Dev Container 🐳

This project ships with a **VS Code Dev Container** setup for running Stephanie with **Postgres + pgvector**.
It gives you a fully reproducible local dev environment: Python, database, embeddings, and all dependencies.

---

## 🚀 Quick Start

### 1. Build & Start the Dev Container

From the project root, you can do either:

**Linux / Mac (bash):**

```bash
docker compose -f .devcontainer/docker-compose.yml up --build -d
```

Or do a full clean rebuild if you hit errors:

```bash
docker compose -f .devcontainer/docker-compose.yml down -v
docker compose -f .devcontainer/docker-compose.yml build --no-cache
docker compose -f .devcontainer/docker-compose.yml up -d
```

**Windows (PowerShell):**

```powershell
docker compose -f .devcontainer\docker-compose.yml up --build -d
```

Or clean rebuild:

```powershell
docker compose -f .devcontainer\docker-compose.yml down -v
docker compose -f .devcontainer\docker-compose.yml build --no-cache
docker compose -f .devcontainer\docker-compose.yml up -d
```

This will start:

* **db** → Postgres with `pgvector`
* **app** → Stephanie workspace container (mounted at `/workspace`)

---

### 2. Attach VS Code

* Open **VS Code**
* Run `Ctrl+Shift+P` → **Dev Containers: Attach to Running Container**
* Pick `stephanie-app-1` (or whatever name `docker ps` shows)

Alternatively, from your terminal:

```bash
docker exec -it stephanie-app-1 bash
```

---

### 3. Setup Python Environment

Inside the container:

```bash
cd /workspace
python3 -m venv venv
source venv/bin/activate
pip install -e . --break-system-packages
```

On Windows (PowerShell inside the container):

```powershell
cd /workspace
python3 -m venv venv
.\venv\Scripts\activate
pip install -e . --break-system-packages
```

---

### 4. Initialize Database

Run the DB init script:

```bash
./scripts/init_db.sh
```

---

### 5. Run Stephanie

Once the database is ready, you can run:

```bash
python -m stephanie.main
```

Or launch with a specific agent (example: proximity):

```bash
python -m stephanie.main +agents=proximity
```

---

### 6. Verify

* Logs will show pipeline stages starting:

  ```
  🖇️⏩ [PipelineStageStart] {'stage': 'proximity'}
  🤖 [AgentInitialized] {'agent_key': 'proximity'}
  ```

* You can connect to the database:

```bash
psql -h localhost -U postgres -d stephanie
```

---

## 🛠️ Tips

* Rebuild container if dependencies change:

  ```bash
  docker compose -f .devcontainer/docker-compose.yml build --no-cache
  ```

* Stop everything:

  ```bash
  docker compose -f .devcontainer/docker-compose.yml down
  ```

* Logs:

  ```bash
  docker compose -f .devcontainer/docker-compose.yml logs -f
  ```

---

## ❗ Troubleshooting

Here are some common issues you may hit:

### 1. `invalid file request .venv/bin/python` during build

This happens if your local `.venv/` folder gets copied into the Docker build context.
**Fix:** Add `.venv/` to `.dockerignore` (already included in this repo).

### 2. `unable to get image ... error during connect ... pipe/dockerDesktopLinuxEngine`

Docker Desktop for Windows wasn’t running, or WSL integration isn’t enabled.
**Fix:**

* Start Docker Desktop manually.
* In Docker Desktop → **Settings** → **Resources → WSL Integration**, ensure your distro is enabled.

### 3. `E: Unable to locate package python3.11`

Ubuntu 24.04 ships with **Python 3.12**, so `python3.11` won’t install.
**Fix:** Our Dockerfile already uses `python3` and `python3.12`. Rebuild with:

```bash
docker compose -f .devcontainer/docker-compose.yml build --no-cache
```

### 4. VS Code can’t attach to the container

Make sure the container is running:

```bash
docker ps
```

If you don’t see `stephanie-app-1`, start it again:

```bash
docker compose -f .devcontainer/docker-compose.yml up -d
```

Then retry **Dev Containers: Attach to Running Container** in VS Code.

### 5. `externally-managed-environment` pip error

Debian/Ubuntu now block global `pip install`.
**Fix:** Always use a venv inside the container:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e . --break-system-packages
```

---

💡 If you hit an issue not listed here, run:

```bash
docker compose -f .devcontainer/docker-compose.yml logs app
```

…and check the error output.

