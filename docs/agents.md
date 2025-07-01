# 🧠 Agents Overview

`stephanie` is a modular, agent-based framework inspired by the SAGE architecture. Each agent is a specialized module that performs a distinct role in the scientific hypothesis lifecycle.

---

## 🗂️ Architecture Overview

Each agent inherits from a `BaseAgent` class and is configured via Hydra. Agents are loaded dynamically based on your pipeline configuration.

```yaml
pipeline:
  - cls: stephanie.agents.generation.GenerationAgent
    ...
  - cls: stephanie.agents.review.ReviewAgent
    ...
````

---

## 🔍 Agent Descriptions

### 🧪 GenerationAgent

**Role:** Produces an initial batch of hypotheses from a given research goal.

**Inputs:**

* `goal`

**Outputs:**

* `hypotheses` (list of hypothesis strings)

**Configurable:**

* Prompt template
* Number of hypotheses
* Strategy (`goal-aligned`, `wild`, etc.)

---

### 🧑‍⚖️ ReviewAgent

**Role:** Reviews and rates hypotheses using criteria like novelty, feasibility, and correctness.

**Inputs:**

* `hypotheses`
* `preferences` (e.g., `["factual", "simple", "reliable_source"]`)

**Outputs:**

* `reviews` (dict of hypothesis → review)

---

### 🏆 RankingAgent

**Role:** Ranks hypotheses using a tournament-style evaluation process.

**Inputs:**

* `reviews`
* `hypotheses`

**Outputs:**

* `ranked` (list of `(hypothesis, score)`)

---

### 🤔 ReflectionAgent

**Role:** Reflects on hypotheses in the context of the goal. Identifies gaps or misalignments.

**Inputs:**

* `hypotheses`
* `goal`
* `preferences`

**Outputs:**

* `reflections` (markdown summaries)

---

### 🧬 EvolutionAgent

**Role:** Refines, combines, or mutates top-ranked hypotheses to improve clarity, novelty, and testability.

**Inputs:**

* `ranked`
* `goal`

**Outputs:**

* `evolved` (list of new hypotheses)

---

### 🗺️ ProximityAgent

**Role:** Computes similarity between current hypotheses and past ones to detect clustering or redundancy.

**Inputs:**

* `hypotheses`
* `goal`

**Outputs:**

* `proximity_graph`, `graft_candidates`, `clusters`

---

### 📈 MetaReviewAgent *(optional)*

**Role:** Provides a final high-level summary and critique of hypothesis quality and coverage.

---

### 🧪 PromptTuningAgent *(experimental)*

**Role:** Uses generated hypotheses + reviews to tune and evaluate prompt templates post-run.

**Inputs:**

* `goal`
* `prompts` (retrieved from DB)

**Outputs:**

* Updated prompt evaluations

---

## ⚙️ Custom Agents

You can add your own agent by inheriting from `BaseAgent`:

```python
class MyCustomAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        ...
        return context
```

Add your agent to `config/pipeline.yaml` and you're good to go.

---

## 📚 Notes

* Each agent automatically receives `logger`, `memory`, and `cfg`.
* Logs are structured per stage and can be filtered or visualized later.
* Agents can store output to database (hypotheses, reviews, prompt logs, etc.)

---

