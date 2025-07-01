# 🧰 Tools

`stephanie` ships with a suite of modular tools that agents can access via injection. These tools wrap shared functionality such as embeddings, prompt loading, logging, search, and evaluation.

---

## 🧠 Memory Tool

The `MemoryTool` manages pluggable stores, each handling a specific data type:

### Built-in Stores

| Store           | Description                                      |
|----------------|--------------------------------------------------|
| `EmbeddingStore` | Caches and retrieves embeddings (via `pgvector`) |
| `HypothesisStore` | Stores and retrieves hypotheses per goal        |
| `ContextStore`    | Persists intermediate context state             |
| `PromptLogger`    | Logs and versions prompts                       |
| `ReportLogger`    | Saves YAML reports per run                      |

```python
memory.hypotheses.insert(goal, text, confidence, review, features)
memory.context.save(run_id, stage, context_dict)
````

You can access any store via:

```python
memory.get("hypotheses")
```

---

## 🔎 Web Search Tool

The `WebSearchTool` supports simple search using DuckDuckGo or a local instance of SearxNG.

### Example

```python
results = await WebSearchTool().search2("US debt ceiling history")
```

Each result is returned as a formatted string with title, snippet, and URL.

> ⚠️ DuckDuckGo is rate-limited. SearxNG is recommended for production use.

---

## 🧪 Prompt Loader

The `PromptLoader` handles loading prompts using four modes:

| Mode       | Source                                   |
| ---------- | ---------------------------------------- |
| `file`     | From local prompt templates (`.txt`)     |
| `static`   | Hardcoded in YAML                        |
| `tuning`   | Best version from memory tuning          |
| `template` | Jinja2 templating with context injection |

```python
prompt = prompt_loader.load_prompt(cfg, context)
```

Prompts are formatted with context values, e.g. `{goal}`.

---

## ⚖️ Evaluation

The `OllamaEvaluator` scores refinements using an LLM (e.g. `qwen2.5`) running locally:

```python
evaluation = evaluator.judge(original, proposal)
print(evaluation.score, evaluation.reason)
```

Used for tuning prompts and comparing generated text quality.

---

## 📚 Template Utilities

Prompt templates live under `prompts/<agent>/filename.txt`.

You can render them using:

```python
from jinja2 import Template

Template(template_text).render(**context)
```

---

## 🔁 Embedding Tool

The `get_embedding(text, cfg)` helper uses the configured embedding model and caches results in the database.

---

## 📝 JSON Logger

Structured logging for every event in the pipeline:

```python
logger.log("HypothesisStored", {"goal": goal, "text": text[:100]})
```

Each log is saved as a `.jsonl` file per run.

---

## 🔧 Pluggable Stores

You can add custom stores via config:

```yaml
extra_stores:
  - stephanie.memory.MyCustomStore
```

Register them via:

```python
memory.register_store(MyCustomStore(...))
```

---

## 🧩 Adding Your Own Tools

You can pass any tool to agents by extending the agent constructor and updating the supervisor.

---

