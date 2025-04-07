# 🧠 Stephanie — The Freestyle Memory Engine

Stephanie is your AI-powered **cognitive memory engine** — a personal knowledge system that grows with you.  
It lets you import conversations, reflect on patterns, and run hybrid searches across all your interactions with AI.

Built for the Freestyle Cognition project, Stephanie is more than a database. It’s your **long-term AI memory**, companion, and co-thinker.

---

## 🚀 What Can You Do With Stephanie?

- 📥 Import ChatGPT conversations (JSON or HTML export)
- 🧠 Embed memory using [Ollama](https://ollama.com) + pgvector
- 🔎 Search memories by meaning (vector), keywords (text), or both (hybrid)
- ✏️ Edit summaries and tags directly in the UI
- 📤 Export conversations to Markdown + JSON
- 🧭 Filter by persona or user identity (e.g. "Ernan:Mobile", "Work", "Reflection")
- 🧰 Run locally with FastAPI + jQuery (no cloud required)

---

## 🛠️ Tech Stack

- Python 3.11+
- FastAPI + Hydra for config
- Ollama (e.g. `mxbai-embed-large`) for embeddings
- PostgreSQL + `pgvector` for hybrid search
- jQuery UI (lightweight, fast)
- Qt desktop version deprecated (replaced by web UI)

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/stephanie.git
cd stephanie
# 🧠 Stephanie — The Freestyle Memory Engine

Stephanie is your AI-powered **cognitive memory engine** — a personal knowledge system that grows with you.  
It lets you import conversations, reflect on patterns, and run hybrid searches across all your interactions with AI.

Built for the Freestyle Cognition project, Stephanie is more than a database. It’s your **long-term AI memory**, companion, and co-thinker.

---

## 🚀 What Can You Do With Stephanie?

- 📥 Import ChatGPT conversations (JSON or HTML export)
- 🧠 Embed memory using [Ollama](https://ollama.com) + pgvector
- 🔎 Search memories by meaning (vector), keywords (text), or both (hybrid)
- ✏️ Edit summaries and tags directly in the UI
- 📤 Export conversations to Markdown + JSON
- 🧭 Filter by persona or user identity (e.g. "Ernan:Mobile", "Work", "Reflection")
- 🧰 Run locally with FastAPI + jQuery (no cloud required)

---

## 🛠️ Tech Stack

- Python 3.11+
- FastAPI + Hydra for config
- Ollama (e.g. `mxbai-embed-large`) for embeddings
- PostgreSQL + `pgvector` for hybrid search
- jQuery UI (lightweight, fast)
- Qt desktop version deprecated (replaced by web UI)

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/stephanie.git
cd stephanie
# 🧠 Stephanie — The Freestyle Memory Engine

Stephanie is your AI-powered **cognitive memory engine** — a personal knowledge system that grows with you.  
It lets you import conversations, reflect on patterns, and run hybrid searches across all your interactions with AI.

Built for the Freestyle Cognition project, Stephanie is more than a database. It’s your **long-term AI memory**, companion, and co-thinker.

---

## 🚀 What Can You Do With Stephanie?

- 📥 Import ChatGPT conversations (JSON or HTML export)
- 🧠 Embed memory using [Ollama](https://ollama.com) + pgvector
- 🔎 Search memories by meaning (vector), keywords (text), or both (hybrid)
- ✏️ Edit summaries and tags directly in the UI
- 📤 Export conversations to Markdown + JSON
- 🧭 Filter by persona or user identity (e.g. "Ernan:Mobile", "Work", "Reflection")
- 🧰 Run locally with FastAPI + jQuery (no cloud required)

---

## 🛠️ Tech Stack

- Python 3.11+
- FastAPI + Hydra for config
- Ollama (e.g. `mxbai-embed-large`) for embeddings
- PostgreSQL + `pgvector` for hybrid search
- jQuery UI (lightweight, fast)
- Qt desktop version deprecated (replaced by web UI)

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/stephanie.git
cd stephanie
```

## 2. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate         # On Windows
pip install -r requirements.txt
```

## 3. Set up PostgreSQL with pgvector

```sql
CREATE DATABASE stephanie;
\c stephanie
CREATE EXTENSION vector;

-- Optional table creation:
CREATE TABLE memory (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    user_text TEXT,
    ai_text TEXT,
    embedding VECTOR(1536),
    tags TEXT[],
    summary TEXT,
    source TEXT,
    length INT,
    importance INT,
    archived BOOLEAN,
    openai_url TEXT,
    user_identity TEXT
);
```

## 4. Configure Hydra

Edit config/stephanie.yaml to set your DB and Ollama details.


## 💻 Running the App

```bash
uvicorn app.main:app --reload
```
Then open:

```bash
http://127.0.0.1:8000/ui/
```

Use the search bar to explore your memory, preview content, edit tags/summary, and export as Markdown.

---

## 📥 Import Your ChatGPT Data

### JSON
```bash
python app/import_openai_json.py "C:\path\to\conversations.json" "ernan:mobile"
```

## 🔍 Hybrid Search

    Text mode = fast keyword matches using Postgres full-text search

    Vector mode = meaning-based semantic similarity (Ollama)

    Hybrid mode = combines both, with configurable scoring weights

Search results show title, summary, tags, and a full preview.


## 🛠 Dev Tools
```bash
# Rebuild your DB
python scripts/reset_db.py

# Batch import
import_openai_json.bat "my_file.json"
```

## 🧪 What’s Next

    Weekly summary AI pipeline

    Auto-tagging + insights

    Pattern recognition over time

    Multi-persona reflections



## 🪪 License

MIT License
© 2024–2025 Ernan Hughes
Contributions welcome



## 🤝 Contributing

Stephanie is designed to be modular, composable, and extendable.

Feel free to:

    Add new data sources (Twitter, voice, notes)

    Build new interfaces (mobile, Electron)

    Fork and repurpose for creative cognition apps

    Stephanie is part of the Freestyle Cognition Project — guiding users to develop long-term collaboration with AI


