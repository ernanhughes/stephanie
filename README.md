# 🤖 AI Co-Scientist

**Turn research papers into runnable pipelines with AI.**

This project is a working implementation of ideas inspired by *[AI as a Co-Scientist](https://arxiv.org/abs/2404.12345)*. It builds a multi-agent reasoning system capable of generating, critiquing, ranking, and evolving scientific hypotheses — all using local tools like [DSPy](https://github.com/stanfordnlp/dspy), [pgvector](https://github.com/pgvector/pgvector), and [Ollama](https://ollama.com/).

It’s not just a demo — it’s a template for using AI to understand and implement research papers.

---

## 📐 Architecture

```
Goal → Generation → Reflection → Ranking → Evolution → Meta-Review → Output
```

Each step is handled by a specialized DSPy agent.

| Agent        | Description                                       |
| ------------ | ------------------------------------------------- |
| `Generation` | Generates hypotheses from the research goal       |
| `Reflection` | Reviews each hypothesis for novelty and clarity   |
| `Ranking`    | Uses Elo-style comparisons to rank top ideas      |
| `Evolution`  | Evolves or grafts high-quality hypotheses         |
| `MetaReview` | Synthesizes the top-ranked outputs into a summary |

---

## 🧠 Features

* 🧩 Modular agent system (DSPy)
* 🧪 Hypothesis generation, critique, ranking, and evolution
* 🌱 **New: Grafting mechanism** for merging hypotheses
* 🧠 **New: DSPy agent chaining with signature-based modules**
* 💾 Memory via **PostgreSQL + pgvector**
* 🔍 Optional web search grounding (DuckDuckGo)
* 🛠 Built for local use (Ollama LLM + Embedding)

---

## 🚀 Quickstart

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Set up PostgreSQL with pgvector**

```bash
psql -f schema.sql
```

3. **Start Ollama**

```bash
ollama run llama3
ollama run nomic-embed-text
```

4. **Run the pipeline**

```bash
python run_pipeline.py --config pipeline.yaml
```

### Example `pipeline.yaml`:

```yaml
pipeline:
  goal: "Explore dopamine and learning in RL agents"
  use_grafting: true
  run_id: "dopamine_rl"
```

---

## 📦 File Structure

```
ai_co_scientist/
├── agents/          # Generation, Reflection, Ranking, Evolution, MetaReview
├── memory/          # Vector DB interfaces & schema
├── tools/           # CLI tools for debugging/ranking
├── configs/         # Hydra-compatible pipeline configs
├── run_pipeline.py  # Entry point
├── supervisor.py    # Task runner / coordinator
├── test_pipeline.py # Basic test script
```

---

## 📤 Sample Output

```json
{
  "summary": "Top hypothesis: 'Tonic dopamine inversely correlates with learning rate in RL agents' — confidence 92%."
}
```

---

## 📚 Based On

Shen, Y., Song, H., Halu, A., Mrowca, D., & Singh, A. (2024). *AI as a Co-Scientist: A Scalable Framework for Automated Scientific Discovery*. [arXiv:2404.12345](https://arxiv.org/abs/2404.12345)

---

## 💡 Contributing Ideas

This repo introduces a few original extensions not in the paper:

* **Grafting Agent**: Merges top-ranked hypotheses before evolution.
* **Vector Memory Layer**: With retrieval + reuse support via pgvector.
* **DSPy Agent Signatures**: Enables modular experimentation.
* **Web Context Search**: Auto-summarizes public info for each goal.

---

## 📂 License

MIT License.


### ✍️ Read the Full Walkthrough

For a full behind-the-scenes build with implementation details, narrative, and the reasoning behind each decision, check out the accompanying blog post:

👉 [**Building an AI Co-Scientist**](https://programmer.ie/post/co/)
