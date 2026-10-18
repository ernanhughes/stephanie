# 🛠️ Installation Guide

This guide walks you through setting up and running the `stephanie` framework locally.

---

## 📦 Requirements

- Python **3.9+** (the version declared in `pyproject.toml`)
- PostgreSQL with **pgvector** extension
- [Ollama](https://ollama.com) (for local LLM inference)
- [Poetry](https://python-poetry.org/) **OR** standard `pip` + `venv`
- Optional: Docker (for running PostgreSQL locally)

---

## 🔧 1. Clone the Repository

```bash
git clone https://github.com/ernanhughes/stephanie.git
cd stephanie
````

---

## 🐍 2. Create a Virtual Environment

**Using `venv` (PowerShell):**

```bash
py -3.13 -m venv venv
venv\Scripts\Activate.ps1
```

**Or with `poetry`:**

```bash
poetry install
poetry shell
```

---

## 📚 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For the local SIS web interface, also install its dependencies:

```bash
pip install -r requirements_sis.txt
```

To install the `stephanie` package itself (including the `stephanie` and `sis`
console entry points) in editable mode:

```bash
pip install -e .
```

For development tests, install the optional test extra after installing the
project:

```bash
pip install -e ".[test]"
```

OR using `pyproject.toml` directly:

```bash
pip install .
```

---

## 🧠 4. Configure PostgreSQL + pgvector

**Option A: Using Docker**

```bash
docker run --name coai-db -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=coai \
  ankane/pgvector
```

**Option B: Local Installation**

1. Install PostgreSQL

2. Install `pgvector`:

   ```bash
   CREATE EXTENSION vector;
   ```

3. Configure the database in `config/db/postgres.yaml` (or supply an explicit
   Hydra override). SQLAlchemy stores and MemoryTool use this same composed
   database configuration. Supply credentials through environment variables,
   for example `$env:STEPHANIE_DB_PASSWORD = "<password>"` in PowerShell; do not
   commit them to YAML or logs.

4. Ensure the configured database is already provisioned with the Stephanie
   schema and pgvector extension. This repository does not currently have a
   versioned migration chain; do not use `create_all()` as a substitute for a
   reviewed schema migration.

---

## 🤖 5. Install & Run Ollama

```bash
ollama run qwen:latest
```

Or for smaller models:

```bash
ollama run llama2
```

Make sure `Ollama` is running on `http://localhost:11434`.

---

## ⚙️ 6. Run a research pipeline

```bash
python -m stephanie.main goal.goal_text="The USA is on the verge of defaulting on its debt"
```

Or with CLI args:

```bash
python -m stephanie.main --config-name=config goal.goal_text="My research goal here"
```

Start SIS locally (loopback only by default):

```powershell
$env:STEPHANIE_DB_PASSWORD = "<database password>"
$env:SIS_AUTH_USERNAME = "stephanie"
$env:SIS_AUTH_PASSWORD_HASH = (& venv\Scripts\python.exe -m sis.security --hash-password)
$env:SIS_SESSION_SECRET = (& venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(48))")
python -m sis
```

The password-hash command prompts without echoing the password. Keep the hash
and session secret in the process environment or an approved secret manager;
never commit them. If authentication is not configured, private SIS routes fail
closed and the login page shows the setup requirement.

SIS requires a valid configured account for all private application routes.
The default host is `127.0.0.1`; remote hosts are rejected unless explicitly
allowlisted, and remote HTTP is refused (use HTTPS). When serving HTTPS behind a
trusted TLS terminator, configure proxy handling and set `SIS_COOKIE_SECURE=true`.
Use a single SIS worker because its revocable session registry is process-local.

Interactive chat remains disabled unless `chat_runtime.enabled=true` is
explicitly configured **and** `chat_runtime.readiness.model_verified` and
`chat_runtime.readiness.live_persistence_verified` are set after their real
acceptance checks. Registration/health of the model service alone is not proof
of inference. See `docs/stephanie-next/stn02-security-design.md` before enabling
the feature.

Check `/healthz` on the local SIS instance for separate authentication,
database, model-configuration, model-verification, persistence-verification,
and feature states. It contains no credentials. Do not enable chat if any gate
is false.

---

## 📝 Notes

* Logs are stored in `logs/` as structured JSONL.
* Prompts are saved in `prompts/` and tracked in the database.
* You can inspect all configuration using Hydra or customize each agent via `config/`.

---

## ✅ You’re Ready!

You now have a full pipeline for running research-style hypothesis generation, evaluation, and prompt tuning.

