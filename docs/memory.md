# 🧠 Memory System

The `MemoryTool` in `stephanie` serves as the shared memory backbone for the entire pipeline. It manages embeddings, hypotheses, reviews, prompt versions, logs, and more—all stored in a PostgreSQL database with pgvector support.

---

## 📦 Core Design

The memory system consists of modular "stores", each implementing a specific aspect of persistent storage. These stores all inherit from a `BaseStore` and are registered into a central `MemoryTool`.

### MemoryTool Overview

```python
memory = MemoryTool(cfg=cfg.db, logger=logger)
````

It registers multiple internal stores automatically:

```text
- embeddings: vector storage and similarity search
- hypotheses: hypothesis versions, links, metadata
- context: stores pipeline-level state/context
- prompts: prompt text, responses, evaluations
- reports: final YAML or HTML reports
```

---

## 📚 Store Breakdown

### 📐 EmbeddingStore

Stores vector embeddings of hypotheses and enables similarity search via `pgvector`.

**Key Methods:**

* `get_or_create(text)`
* `similar(text, top_k=5)`

---

### 💡 HypothesisStore

Stores hypotheses generated throughout the pipeline, along with evaluations and links to the prompt that created them.

**Schema includes:**

* `text`
* `goal`
* `confidence`
* `features` (JSON)
* `embedding` (vector)
* `prompt_id` (foreign key)

---

### 💾 PromptStore

Tracks prompt versions, tuning attempts, evaluations, and strategies used during generation.

**Key Features:**

* Save new prompts with `save()`
* Log evaluation results with `store_evaluation()`
* Retrieve recent prompts via `get_latest_prompts(n)`

---

### 📥 ContextStore

Stores and retrieves pipeline context snapshots to allow state recovery or debugging between stages.

**Stored As:**

* `yaml`
* Timestamped entries
* One entry per run ID

---

### 📜 ReportLogger

Stores the final pipeline result YAML in the `reports` table and allows for post-run analysis or export.

---

## 🧩 Custom Stores

You can define and plug in your own store like so:

```python
class MyCustomStore(BaseStore):
    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger
        self.name = "my_custom_store"
```

Then register it:

```python
memory.register_store(MyCustomStore(db, logger))
```

---

## 🗄️ Database Schema (Simplified)

Here are a few tables used:

```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    text TEXT UNIQUE,
    embedding VECTOR(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

```sql
CREATE TABLE hypotheses (
    id SERIAL PRIMARY KEY,
    text TEXT,
    goal TEXT,
    confidence FLOAT,
    prompt_id INT,
    embedding VECTOR(1024),
    ...
);
```

```sql
CREATE TABLE prompts (
    id SERIAL PRIMARY KEY,
    agent_name TEXT,
    prompt_text TEXT,
    version INT,
    is_current BOOLEAN,
    ...
);
```

---

## 🛠 Config Example

Hydra config fragment:

```yaml
db:
  host: localhost
  port: 5432
  database: stephanie
  user: postgres
  password: yourpassword
```

---

## ✅ Benefits

* Consistent logging and traceability
* Unified access to evolving data
* Easy to extend with new stores
* Enables historical comparisons and tuning

