# 🧠 SAGE Paper vs `stephanie` Implementation Checkpoint

This page summarizes how the `stephanie` implementation maps directly to the concepts and structure proposed in the SAGE paper ([arXiv:2502.18864](https://arxiv.org/abs/2502.18864)).

## ✅ Goal

Automate and assist scientific hypothesis generation, ranking, refinement, and evaluation using LLMs and a modular agent framework.

---

## 🔹 1. Pipeline-Oriented Architecture

**SAGE Paper:** Modular pipeline of agents: literature, generation, ranking, reflection, evolution.
**co\_ai:**

* Modular `Supervisor` class runs a Hydra-configured pipeline
* Agents derived from `BaseAgent`
* Context is passed through pipeline stages
* Agents dynamically enabled/disabled via config

---

## 🔹 2. Literature-Driven Hypothesis Generation

**SAGE Paper:** Hypotheses are seeded from retrieved literature.
**co\_ai:**

* `LiteratureAgent` supports DuckDuckGo or SearxNG (optional)
* Results are embedded into `VectorMemory`
* Used to ground hypothesis generation and ranking

---

## 🔹 3. Hypothesis Generation

**SAGE Paper:** LLMs generate hypotheses based on scientific goals.
**co\_ai:**

* `GenerationAgent` uses configurable prompts
* Prompt strategies: static, file, template, tuning
* Integrates prompt tracking and logging

---

## 🔹 4. Ranking & Evaluation

**SAGE Paper:** Hypotheses evaluated via pairwise ranking.
**co\_ai:**

* `RankingAgent` implements tournament-style ranking
* Stores scores and feedback in memory
* Supports re-ranking and evaluation review

---

## 🔹 5. Reflection & Meta-Review

**SAGE Paper:** Reflections assess hypothesis quality and alignment.
**co\_ai:**

* `ReflectionAgent` uses preferences and goal alignment
* Outputs markdown-formatted reflections
* Logs reflections and stores review

---

## 🔹 6. Evolution Agent

**SAGE Paper:** Refines hypotheses through transformation.
**co\_ai:**

* `EvolutionAgent` includes grafting of similar hypotheses
* Performs simplification and clarification
* Logs transformation path and stores result

---

## 🔹 7. Memory & Storage

**SAGE Paper:** Persistent storage of hypotheses and evaluations.
**co\_ai:**

* `MemoryTool` wraps `pgvector`-backed PostgreSQL stores
* Separate stores for hypotheses, prompts, context, evaluations
* Agents log activity to structured JSONL logs

---

## 🔹 8. Prompt Tuning Loop

**SAGE Paper:** Prompts evolve using feedback from prior runs.
**co\_ai:**

* `PromptTuningAgent` uses `OllamaEvaluator`
* Compares modified prompts against originals
* Saves evaluation results in `prompt_evaluations` table

---

## 🔹 9. Logging & Traceability

**SAGE Paper:** Transparent agent operation and outputs.
**co\_ai:**

* Emoji-annotated structured logs (JSONL)
* YAML dumps of context at each stage
* Unique log files per run for reproducibility

---

## 🔹 10. Extensibility & Modularity

**SAGE Paper:** Agents and stages should be reusable/extensible.
**co\_ai:**

* Fully pluggable agent framework
* Easily extend pipeline with new custom stages
* Prompts, configs, memory, and evaluators are all replaceable

---

## 🎉 Summary

The `stephanie` project faithfully implements the key components of the SAGE framework and enhances it with:

* Fine-grained logging
* Prompt tuning loops
* Modular memory system
* CLI-based pipeline control

This makes it suitable for scientific automation, reproducible research workflows, and ongoing improvement through prompt evolution.
