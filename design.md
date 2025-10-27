I # Stephanie Foundations — Paste-In Primer

## What Stephanie is

Stephanie is a **self-improving system**. You give it a task + config, set it running (overnight, on a drive, on a robot, in a repo), and it:

* learns from logs and prior work,
* generates/implements solutions,
* keeps improving autonomously.

The long-term vision is a **live, purposeful AI** that can be deployed with a small footprint and work continuously, locally if needed.

---

## The Four Core Components (present everywhere)

### 1) Configuration

* **Role:** The single source of truth. Defines **goal**, **data/storage backends**, **services**, **event bus**, model choices, and feature flags.
* **Shape:** Hydra-style YAML (or Python DictConfig). Always the first thing we load.
* **Contract:** All components accept `cfg` and **derive their own sub-config** (no globals).
* **Example:**

  ```yaml
  self_play:
    enabled: true
    verification_threshold: 0.85
  storage:
    driver: postgres   # future: sqlite/duckdb/fs fallback
    dsn: ${env:DB_URL}
  bus:
    enabled: true
    driver: nats       # future: inproc fallback
  goals:
    task: "summarize conversations and propose next actions"
  ```

### 2) Log (Data Log)

* **Role:** Not just “logging”—this is **data exhaust for learning**. Every batch job (scoring, embeddings, imports, etc.) writes **JSONL** logs that downstream steps **consume**.
* **Analogy:** Like your OpenAI **chat history export**—small, complete, and everything you need to rebuild state or train new components.
* **Contract:** One JSON object per line; **stable keys**; **append-only**; JSON-safe (NumPy, datetimes, etc. sanitized).

### 3) Memory Tool

* **Role:** Stephanie’s memory. A unified interface over **databases + filesystem**, plus the **event bus**.
* **Today:** SQLAlchemy stores (tables wrapped in base classes). Backends: Postgres now; **SQLite/DuckDB/file** when local-first.
* **Also:** Contains the **bus** (usually NATS; in-process fallback) and a simple **retriever** utility for domain/entity lookups.
* **Contract:** Stores expose consistent verbs (`insert/create/upsert/list_recent/annotation_progress/...`). Bus exposes `publish/subscribe` (or no-op if disabled).

### 4) Container

* **Role:** A service container (managed map of services) so code stays thin and configurable.
* **What lives here (now):** Scorers, prompts, risk/monitoring, model router, bus handle, metrics—grows as needed.
* **Why:** Anywhere in the app we can resolve services **via config**, not imports. Easier testing, swapping, and local-first.

---

## Design Principles (how we build)

* **Local-first trajectory:** Works great with Postgres/NATS **today**, but target **SQLite/DuckDB + in-proc bus + filesystem** so anyone can run in a shell/notebook.
* **Config-in, outputs-as-logs:** Every job takes `cfg`, emits JSONL; downstream steps read logs to learn/improve.
* **Graceful degradation:** If a backend/tool isn’t present, we **fallback** (no crashes).
* **Idempotent & resumable:** Reruns don’t corrupt state; outputs are versioned.
* **Single-responsibility modules:** Each component/file does one thing well.
* **Traceability:** Every meaningful action emits a **PlanTrace** to logs + memory (for SIS UI, audits, and training).

---

## Quick Lifecycle (how to use it)

1. **Write config** (goals, storage, bus, models/features).
2. **Boot MemoryTool + Container** from config.
3. **Run jobs** (e.g., import chats → embed → score → index). Each step writes **data logs**.
4. **Run SSP/Jitter** to learn from logs and generate new work.
5. **Inspect in SIS UI** (dashboards over plan traces, jobs, and results).

---

## What “good” looks like (acceptance checks)

* Every module accepts `cfg`; no hidden global state.
* Logs are JSON-safe, append-only, and usable by other modules.
* Memory Tool can **switch backends** without code changes.
* Container provides services by **name**; swappable via config.
* Background loops (SSP/Jitter) can be started/stopped and always **report status** via PlanTraces + HTTP.
* All errors fail **soft** (fallbacks + traces recorded).

---

## Minimal Boot Pseudocode

```python
cfg = load_cfg("config.yaml")            # hydra-style dict
memory = MemoryTool(cfg)                 # DB + bus + stores
container = build_container(cfg, memory) # scorers/models/etc.

# Job: import & embed
logs = run_import_chats(cfg, memory)
run_embed(cfg, memory, logs)

# Train/Improve
ssp = SspComponent(cfg)
ssp.start(background=True)               # emits PlanTraces
```

---

If you paste this at the top of a conversation, we’ll both align on:

* **What Stephanie is** (self-improving, long-running, local-first target),
* **The four universals** (Configuration, Log, Memory Tool, Container),
* **How we design** (config-in, logs-out, graceful fallbacks),
* **How we judge quality** (traceable, swappable, resumable, safe).

