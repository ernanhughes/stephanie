# 🛠️ Installation Guide

This guide walks you through setting up and running the `stephanie` framework locally.

---

## 📦 Requirements

- Python **3.8+**
- PostgreSQL with **pgvector** extension
- [Ollama](https://ollama.com) (for local LLM inference)
- [Poetry](https://python-poetry.org/) **OR** standard `pip` + `venv`
- Optional: Docker (for running PostgreSQL locally)

---

## 🔧 1. Clone the Repository

```bash
git clone https://github.com/ernanhughes/co-ai.git
cd co-ai
````

---

## 🐍 2. Create a Virtual Environment

**Using `venv`:**

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
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

OR using `pyproject.toml`:

```bash
pip install .
```

---

## 🧠 4. Set Up PostgreSQL + pgvector

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

3. Create the database:

   ```bash
   createdb coai
   ```

4. Run the schema:

   ```bash
   psql -U postgres -d coai -f schema.sql
   ```

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

## ⚙️ 6. Run the App

```bash
python stephanie/main.py goal="The USA is on the verge of defaulting on its debt"
```

Or with CLI args:

```bash
python stephanie/main.py --config-name=config goal="My research goal here"
```

---

## 📝 Notes

* Logs are stored in `logs/` as structured JSONL.
* Prompts are saved in `prompts/` and tracked in the database.
* You can inspect all configuration using Hydra or customize each agent via `config/`.

---

## ✅ You’re Ready!

You now have a full pipeline for running research-style hypothesis generation, evaluation, and prompt tuning.

