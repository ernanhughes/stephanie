# 🤖 Stephanie

> Self-Tuning Epistemic Framework for Autonomous Navigation and Inference Engineering

Absolutely — here's a rewritten, cleaner, and more compelling version of your README that keeps the core ideas while improving structure, flow, and clarity. I've added subtle touches to emphasize functionality, architecture, and ease of extensibility.

---

# 🧪 Stephanie: A Self-Evolving AI Research Assistant

**Turn research papers into executable reasoning pipelines using local AI.**

Stephanie is a working system inspired by [AI as a Co-Scientist (arXiv:2502.18864)](https://arxiv.org/abs/2502.18864). It combines hypothesis generation, critique, ranking, and evolution — all orchestrated through a modular, local-first multi-agent architecture.

You can think of it as **an autonomous lab partner** — one that reads, thinks, reflects, and iteratively improves its own ideas.

---

## 🧬 Key Features

✅ Local-first: built for Ollama + pgvector
✅ Full scientific loop: generation → reflection → evolution
✅ Grafting & ranking mechanisms to refine ideas
✅ Signature-based DSPy agents for modular control
✅ PostgreSQL memory for tracking hypothesis evolution
✅ Web search grounding (optional)

---

## ⚙️ Pipeline Overview

```
Goal
  ↓
Literature Search → Generation → Reflection → Ranking
                                        ↓
                           Grafting & Evolution
                                        ↓
                                Meta-Review
                                        ↓
                                  Output
```

Each stage is powered by a specialized DSPy agent. Agents pass hypotheses, critiques, and scores forward in a zero-shot feedback loop, allowing the system to explore and refine its own scientific thinking.

| Agent        | Role                                                       |
| ------------ | ---------------------------------------------------------- |
| `Literature` | Gathers background from the web or local memory            |
| `Generation` | Produces hypotheses from the research goal                 |
| `Reflection` | Critiques hypotheses for novelty, clarity, and utility     |
| `Ranking`    | Scores and ranks outputs using Elo-style comparisons       |
| `Evolution`  | Improves, mutates, or merges ideas based on top performers |
| `MetaReview` | Synthesizes the highest-quality results into a report      |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up PostgreSQL + pgvector

```bash
psql -f schema.sql
```

### 3. Start Ollama (required models)

```bash
ollama run llama3
ollama run nomic-embed-text
```

### 4. Run a full reasoning pipeline

```bash
python run_pipeline.py --config pipeline.yaml
```

### 🔧 Example `pipeline.yaml`

```yaml
pipeline:
  goal: "Explore dopamine and learning in RL agents"
  use_grafting: true
  run_id: "dopamine_rl"
```

---

## 📂 Project Structure

```
stephanie/
├── agents/           # Modular DSPy agents (generation, ranking, etc.)
├── memory/           # pgvector-based vector store and ORM schema
├── tools/            # CLI tools for manual testing and output inspection
├── configs/          # YAML configs for pipeline settings
├── run_pipeline.py   # Pipeline entry point
├── supervisor.py     # Execution orchestrator
├── test_pipeline.py  # Basic test cases
```

---

## 🧠 Example Output

```json
{
  "summary": "Top hypothesis: 'Tonic dopamine inversely correlates with learning rate in RL agents' — confidence 92%."
}
```

---

## 🧱 Extensions Beyond the Paper

Stephanie includes several novel additions not found in the original research:

| Feature                  | Description                                             |
| ------------------------ | ------------------------------------------------------- |
| **Grafting Agent**       | Combines and mutates ideas for deeper synthesis         |
| **pgvector Memory**      | Embedding + recall layer for scientific reasoning       |
| **Prompt Signatures**    | DSPy modules support composable, type-safe logic blocks |
| **Web Search Grounding** | Fetch and condense real-world info per hypothesis       |

---

## 📚 Based On

> Shen, Y., Song, H., Halu, A., Mrowca, D., & Singh, A. (2024).
> *AI as a Co-Scientist: A Scalable Framework for Automated Scientific Discovery*.
> [arXiv:2404.12345](https://arxiv.org/abs/2404.12345)

---

## ✍️ Full Walkthrough

Want to understand how and why it was built this way?

👉 [**Read the blog post: Self-Improving AI: A System That Learns, Validates, and Retrains Itself**](https://programmer.ie/post/rivals/)
👉 [**Teaching Tiny Models to Think Big: Distilling Intelligence Across Devices**](https://programmer.ie/post/pupil/)
👉 [**Read the blog post: Compiling Thought: Building a Prompt Compiler for Self-Improving AI**](https://programmer.ie/post/compiler/)
👉 [**Read the blog post: Thoughts of Algorithms**](https://programmer.ie/post/thoughts/)
👉 [**Read the blog post: Document Intelligence: Turning Documents into Structured Knowledge**](https://programmer.ie/post/docs/)
👉 [**Read the blog post: Learning to Learn: A LATS-Based Framework for Self-Aware AI Pipelines**](https://programmer.ie/post/lats/)

---

## 📖 License

MIT License

