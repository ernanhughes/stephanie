# 🧠 System Profile: CO-AI

## 📌 System Overview

The **Co.AI framework** is a modular, self-improving AI system designed to:

* Evaluate its outputs via internal scoring agents (MR.Q, SVM, DPO-style tuning)
* Learn from examples using **in-context learning (ICL)** and store them as reusable cartridges
* Integrate symbolic reasoning into workflows using a **StepCompilerAgent**
* Operate entirely on **local models (e.g., via Ollama)** and **structured memory backends (Postgres + PGVector)**

---

## 🎯 Primary Goals

* Build a self-improving AI pipeline using local resources and no cloud dependencies
* Automatically refine and rerun its pipelines based on scoring feedback
* Evolve "worldviews" and belief systems by integrating multi-modal and multi-source research
* Store, score, and compress research into **KnowledgeCartridges** and shareable artifacts

---

## 🛠️ Core Architecture

### 🔧 Agents

| Agent Name                 | Function                                                                   |
| -------------------------- | -------------------------------------------------------------------------- |
| `DocumentProfilerAgent`    | Parses, classifies, and structures raw documents                           |
| `KnowledgeLoaderAgent`     | Loads domain-specific insights from documents into memory                  |
| `ICLAgent`                 | Synthesizes in-context examples to enhance prompt-based reasoning          |
| `PromptEditorAgent`        | Explores prompt variations, scores them, selects the best                  |
| `StepCompilerAgent`        | Converts procedural documents into symbolic, executable traces             |
| `UsefulInfoExtractorAgent` | Extracts high-value insights from research based on similarity and scoring |
| `PaperScoreAgent`          | Assigns multidimensional scores to papers (e.g., novelty, coherence)       |

### 🧠 Memory / Storage

* **Postgres**: Main backend
* **PGVector**: Stores embeddings for similarity search
* **SQLAlchemy ORM**: Unified data access across events, documents, prompts, etc.

### 🧪 Evaluation

* **MR.Q / DPO-style scoring**
* **SVM-based local scoring**
* **Regression tuner for score alignment**
* **Multi-dimensional scoring (e.g., alignment, novelty, completeness)**

---

## 📦 Core Concepts

### 📁 Knowledge Cartridge

A portable, markdown-backed unit containing:

* Core thesis
* Supporting evidence / contradictions
* In-context learning examples
* Score metadata (e.g., completeness, actionability)

### 📚 ICL Strategy

* Stores prompt/response pairs with task type + scores
* Generates new examples and includes them in future tasks
* Prompt construction dynamically chooses examples via `ICLHelper`

---

## 🔍 Desired Extensions

### 🔎 Targeted Search Goals

The system actively looks for:

* Code or methods for **self-editing**, **reflection**, or **dynamic prompt optimization**
* New strategies for **symbolic reasoning**, **workflow abstraction**, or **reasoning trace compression**
* Innovations in **in-context learning compression**, pruning, or **example selection**
* Scoring models that outperform MRQ/SVM/DPO in **low-data local environments**

### 📥 Input Sources

* Daily ArXiv feeds
* Papers with Code
* GitHub repos tagged `LLM`, `prompt-engineering`, `symbolic AI`, `DPO`, `in-context learning`

---

## 🧠 Belief + Worldview Formation

### 🌐 Worldview Engine

* Given a goal, the system spawns agents to explore documents, generate hypothesis, and form an evolving worldview
* This worldview is expressed as a `KnowledgeCartridge` or final **blog-post-style research report**
* The system scores its own output and **tunes itself** based on cartridge quality

---

## ✅ Usage

This system profile can be embedded into:

* `UsefulInfoExtractorAgent` as semantic context
* `ResearchFinderAgent` for guided discovery
* `ICLAgent` to enrich prompt context
* Downstream pipelines to validate new inputs against system relevance
