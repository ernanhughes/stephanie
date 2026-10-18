# 🐳 Docker — Full System Stack

`docker compose up` starts the **whole Stephanie system**, not only web search.
The application tier is built from this repository; the backing services
(database, message bus, search, model runtime) are pinned companion images.

This supersedes the earlier root `docker-compose.yml`, which started a single
SearXNG container and nothing else.

---

## What the stack runs

| Service | Image | Purpose | Default bind | Config |
|---|---|---|---|---|
| `app` | built from `Dockerfile` | Stephanie engine + SIS web app | `127.0.0.1:8000` | `config/sis.yaml`, `config/config.yaml` |
| `db` | `pgvector/pgvector:pg16` | PostgreSQL + `pgvector` for MemoryTool | `127.0.0.1:5432` | `config/db/postgres.yaml` |
| `db-init` | `pgvector/pgvector:pg16` | One-shot schema loader (`schema_raw.sql`) | — | — |
| `nats` | `nats:2.10-alpine` | JetStream message bus | `127.0.0.1:4222` (monitor `8222`) | `config/bus/nats.yaml` |
| `searxng` | `searxng/searxng:latest` | Web-search backend | `127.0.0.1:8080` | `web_search` node |
| `ollama` | `ollama/ollama:latest` | Local LLM + embedding runtime | `127.0.0.1:11434` | `chat_runtime.model`, `embeddings` |

Startup order is enforced with health/completion gates:
`db` (healthy) and `nats` (healthy) and `db-init` (completed) → `app`.

Every host port binds to `127.0.0.1` by default to match Stephanie's local-first
security posture. Set the matching `*_BIND` variable to expose a port elsewhere,
and only behind TLS.

---

## Quick start

```bash
# 1. Configure secrets and ports
cp .env.docker.example .env              # PowerShell: Copy-Item .env.docker.example .env
cp .env.sis.example .env.sis             # SIS auth secrets (kept separate, see below)
# edit .env      — at minimum STEPHANIE_DB_PASSWORD
# edit .env.sis  — SIS_AUTH_PASSWORD_HASH and SIS_SESSION_SECRET

# 2. Build the app image and start the stack
docker compose up -d --build

# 3. Pull the models the default config expects
docker compose exec ollama ollama pull qwen3
docker compose exec ollama ollama pull mxbai-embed-large

# 4. Watch the app come up
docker compose logs -f app
```

Open `http://localhost:8000` and sign in with `SIS_AUTH_USERNAME` and the
password whose hash you put in `SIS_AUTH_PASSWORD_HASH`.

Generate the two SIS secrets (run either in the containerised app or in a local
`venv`):

```bash
docker compose run --rm app python -m sis.security --hash-password
docker compose run --rm app python -c "import secrets; print(secrets.token_urlsafe(48))"
```

> **Why `.env.sis` and not `.env`?** Docker Compose interpolates `$` inside
> `.env` **and** inside `env_file` values; a PBKDF2 hash
> (`pbkdf2_sha256$...$...$...`) would be silently truncated. `.env.sis` is
> loaded as an `env_file`, so **double every `$` as `$$`**. Generate the escaped
> value directly:
>
> ```bash
> python -c "from sis.security import hash_password; print(hash_password('YOUR-PASSWORD').replace('$','$$'))"
> ```
>
> The file is optional — without it SIS fails closed.

> The first build installs the full Python dependency set (including PyTorch and
> sentence-transformers) and can take several minutes and multiple GB. Later
> builds reuse cached layers.

---

## Configuration

Docker Compose reads `.env` for `${VAR}` substitution in `docker-compose.yml`.
The container environment variables are then consumed by Stephanie through the
Hydra config, so no values are hardcoded:

| Environment variable | Used by | Config key |
|---|---|---|
| `STEPHANIE_DB_HOST` / `_PORT` / `_NAME` / `_USER` / `_PASSWORD` | app, db-init | `db.*` (`config/db/postgres.yaml`) |
| `STEPHANIE_NATS_SERVERS` | app | `bus.servers` (`config/bus/nats.yaml`) |
| `STEPHANIE_OLLAMA_URL` | app | `chat_runtime.model.api_base`, `embeddings.endpoint` |
| `STEPHANIE_SEARXNG_URL` | app | `web_search.instance_url` |
| `SIS_HOST` / `SIS_PORT` | app | SIS bind address |
| `SIS_AUTH_USERNAME` / `SIS_COOKIE_SECURE` | app | SIS authentication (`.env`) |
| `SIS_AUTH_PASSWORD_HASH` / `SIS_SESSION_SECRET` | app | SIS credentials (`.env.sis`; double every `$` as `$$`) |

Each base config keeps its original default, so the same files work for local
runs (`localhost`) and inside Compose (service names like `db`, `nats`,
`ollama`, `searxng`). Verify the resolved values:

```bash
docker compose run --rm app python -c \
  "from hydra import compose, initialize_config_dir; \
   initialize_config_dir(version_base=None, config_dir='/app/config').__enter__(); \
   c=compose(config_name='sis'); print(c.db.host, c.bus.servers, c.chat_runtime.model.api_base)"
```

---

## Database schema

Stephanie has no versioned migration chain, so the stack applies the reviewed
schema dump committed at `schema_raw.sql` **exactly once**:

1. `db-init` waits for `db` to report healthy.
2. It ensures the `vector` extension exists.
3. If `public.chat_conversations` is absent it loads `schema_raw.sql`; otherwise
   it prints `Stephanie schema already present; skipping`.

The application image does **not** create schema implicitly (`init_db(...)` is
called without `create_schema`, and `create_all()` is not a migration path). See
`docs/installation.md` for the reasoning.

To provision the schema yourself instead, remove `db-init` from the app's
`depends_on` or start with `docker compose up -d db app` after applying your own
DDL.

---

## Running pipelines and one-off commands

The `app` service defaults to the SIS web app. Run the reasoning engine, tests,
or any CLI through the same image:

```bash
# A research pipeline
docker compose run --rm app python -m stephanie.main \
  goal.goal_text="The USA is on the verge of defaulting on its debt"

# Interactive shell in the app container
docker compose run --rm app bash

# Deterministic test suite
docker compose run --rm app python -m pytest tests/test_stn06_preregistration.py -q
```

Inside the compose network the config automatically points at `db`, `nats`,
`ollama` and `searxng`, so no overrides are needed.

---

## Models

The default config expects `ollama/qwen3` (chat) and `mxbai-embed-large`
(embeddings). They are not bundled; pull them after first start:

```bash
docker compose exec ollama ollama pull qwen3
docker compose exec ollama ollama pull mxbai-embed-large
docker compose exec ollama ollama list
```

For NVIDIA GPUs, add an override file (e.g. `docker-compose.gpu.yml`):

```yaml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
```

and start with `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`.

---

## Security notes

- All published ports bind to `127.0.0.1`. Changing a `*_BIND` to `0.0.0.0`
  exposes the service; do it only behind a trusted TLS terminator.
- SIS fails closed without a configured account. Set `SIS_AUTH_PASSWORD_HASH`
  and `SIS_SESSION_SECRET`, and keep them in `.env` (git-ignored) or a secret
  manager — never in versioned YAML.
- Change `STEPHANIE_DB_PASSWORD` and `STEPHANIE_SEARXNG_SECRET` from the example
  defaults before any non-local use.
- Interactive chat stays disabled until `chat_runtime.enabled=true` **and** the
  readiness flags are set after real acceptance checks (`config/sis.yaml`).
- The Worker-backed coding executor is a **separate loopback service** and is
  not part of this stack; nothing here enables coding execution.

---

## Troubleshooting

| Symptom | Check |
|---|---|
| `bind: Only one usage of each socket address` on `docker compose up` | A local service already owns that host port (common with a host PostgreSQL on 5432 or Ollama on 11434). Set the matching `*_FORWARD_PORT` / `*_PORT` in `.env` to a free port; container-internal ports are unchanged. |
| SIS login always fails | The password hash may have been truncated by `$` interpolation — ensure **every `$` in `.env.sis` is doubled as `$$`**, and that the hash is in `.env.sis`, not `.env`. |
| `app` restarts with a database error | `docker compose ps` — is `db` healthy? Is `db-init` `Exited (0)`? |
| `app` logs `KnowledgeBus` connect failures | `docker compose logs nats`; confirm `STEPHANIE_NATS_SERVERS=nats://nats:4222`. |
| `/healthz` reports auth or model not ready | Set `SIS_AUTH_*`; pull the Ollama models. |
| Schema errors mentioning missing tables | Re-run `docker compose up db-init` after checking `schema_raw.sql` loaded. |
| Build runs out of memory | Build with more Docker memory or `DOCKER_BUILDKIT=1 docker build --memory=8g`. |
| SearXNG returns no results | `docker compose logs searxng`; check `SEARXNG_SECRET` is set. |

Reset everything (destroys data):

```bash
docker compose down -v
```

---

## Files

| File | Role |
|---|---|
| `docker-compose.yml` | Full stack definition |
| `Dockerfile` | Application image (engine + SIS) |
| `.dockerignore` | Build-context exclusions |
| `.env.docker.example` | Compose environment template (ports, db, username) |
| `.env.sis.example` | SIS auth secret template (loaded verbatim via `env_file`) |
| `schema_raw.sql` | Reviewed schema loaded by `db-init` |
| `.devcontainer/` | Optional VS Code dev container (separate, unchanged) |
