# stephanie: Collaborative AI Hypothesis Engine

Welcome to the documentation for **stephanie**, a modular LLM-powered framework designed to assist in scientific hypothesis generation, evaluation, and refinement. This project is inspired by the SAGE architecture proposed in [arXiv:2502.18864](https://arxiv.org/abs/2502.18864) and aims to simulate a collaborative AI research team.

---

## 🔍 What is `stephanie`?

`stephanie` is an extensible agent-based pipeline framework built around a central Supervisor and a suite of intelligent agents. Each agent performs a distinct role — such as generating hypotheses, ranking them, reflecting on their quality, or evolving better ones — all while sharing state through a common memory and logging system.

The system is designed to:

- **Generate high-quality hypotheses** using goal-driven prompts
- **Evaluate and refine outputs** using ranked feedback and few-shot learning
- **Tune itself** over time using embedded prompt evaluations
- **Persist context and decisions** for future runs

---

## 🧠 Key Features

- 🧩 **Modular agent architecture** (Generation, Ranking, Reflection, Evolution)
- 🧠 **Vector memory store** powered by PostgreSQL + pgvector
- 📂 **Context preservation** across agents via memory tools
- 📜 **Prompt tuning** via DSPy or Ollama-based evaluations
- ⚙️ **Hydra configuration system** for flexible runtime setups
- 📈 **Logging** with structured JSONL + emoji-tagged stages

---

## 🚀 Example Use Case

You define a research goal (e.g., *"The USA is on the verge of defaulting on its debt"*). `stephanie` spins up a pipeline to:

1. Generate multiple hypotheses
2. Reflect on their quality
3. Rank and evolve them using internal feedback
4. Store results, logs, prompts, and evaluations
5. Optionally tune the prompts used in the process for the next iteration

Everything is modular and can be extended with custom agents, tools, and storage plugins.

---

## 📦 Project Structure

```bash
stephanie/
├── agents/           # Agent classes (generation, reflection, etc.)
├── memory/           # Memory and store definitions
├── logs/             # Structured logging system
├── tuning/           # Prompt tuning tools
├── tools/            # External API utilities (e.g., web search)
├── main.py           # Entry point
└── supervisor.py     # Pipeline orchestration
config/
prompts/


````

---

## 🔗 Resources

* [GitHub Repository](https://github.com/ernanhughes/stephanie)
* [The SAGE Paper (arXiv)](https://arxiv.org/abs/2502.18864)
* [Prompt Tuning Overview](prompt_tuning.md)
* [Configuration Guide](configuration.md)

---

## 👨‍🔬 Why Use This?

`stephanie` isn’t just another LLM wrapper — it’s a framework designed to **amplify human creativity and reasoning** through a configurable, extensible AI assistant team. Whether you're testing theories, validating hypotheses, or generating structured research output, `stephanie` turns prompts into pipelines, and pipelines into progress.

```


---

## Stephanie Next

Current direction, architecture, migration, and milestone documentation lives in [Stephanie Next](stephanie-next/index.md).
