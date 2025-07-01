# ⚙️ Configuration

`stephanie` uses **Hydra** for flexible and hierarchical configuration. Each agent, tool, and subsystem loads settings from a structured YAML config. This allows you to dynamically switch models, pipelines, memory backends, and more.

---

## 🗂️ Configuration Structure

The root config lives in `config/config.yaml` and includes nested paths for:

```yaml
defaults:
  - db: default
  - logging: default
  - report: default
  - agents:
      - generation
      - ranking
      - review
      - reflection
      - evolution
      - proximity
      - meta_review
````

---

## 🧠 Memory Configuration (`db.yaml`)

Example for a PostgreSQL database with `pgvector`:

```yaml
host: localhost
port: 5432
database: stephanie
user: postgres
password: yourpassword

# Optional store overrides
extra_stores: []
```

---

## 📜 Logging (`logging.yaml`)

```yaml
logger:
  log_path: logs/
  log_file: ""
```

* `log_path`: Base directory for logs
* `log_file`: Specific log file; if empty, will auto-generate using timestamp and run ID

---

## 📊 Reporting (`report.yaml`)

```yaml
generate_report: true
format: yaml
```

---

## 🧪 Agent Configuration

Each agent has its own file under `config/agents/`. For example:

### ✨ Generation Agent

```yaml
name: generation
prompt_type: file
prompt_file: generation_goal_aligned.txt
top_k: 5
```

### 🧠 Reflection Agent

```yaml
name: reflection
prompt_type: file
prompt_file: reflect_consistency.txt
preferences:
  - goal_consistency
  - factual
  - reliable_source
  - simplicity
```

---

## 🧬 Prompt Tuning Agent

```yaml
name: prompt_tuning
model: qwen2.5
num_prompts: 10
```

---

## 📁 Prompt Directory Override

You can override the prompt directory from the command line or within the context:

```yaml
PROMPT_DIR: "prompts"
```

---

## 🧪 Example Run with Overrides

You can override any value from the CLI:

```bash
python stephanie/main.py goal="Can AI assist scientific discovery?" logging.logger.log_path=run_logs
```

Or define a full run config:

```yaml
goal: "Evaluate LLMs for literature review"
run_id: "review_run_01"
```

---

## 📘 Tips

* Every config is passed to its agent as a flat `dict`
* Use `.get()` for optional config keys
* Log config values using `OmegaConf.to_yaml(cfg)`

---

## 🔁 Runtime Mutation

Agents can mutate config or context, but best practice is to treat config as static and context as dynamic.

---
