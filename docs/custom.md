# Building a Custom AI Pipeline with Minimal Configuration

How to build a custom scientific hypothesis generation pipeline using the `stephanie` framework with the **minimum required configuration**. This chapter assumes you’re familiar with Python and basic YAML syntax.

---

## 🎯 What You Will Build

By the end of this chapter, you'll have a pipeline that:

- Accepts a scientific goal
- Generates hypotheses using a custom agent
- Logs and stores the results
- Is fully driven by a Hydra-based config

---

## 🧱 Required Files

At a minimum, your project must include:

```

co\_ai/
└── main.py                  # Entry point
└── agents/
└── my\_agent.py        # Your custom agent
config/
└── config.yaml              # Hydra config for the run
└── agent\_
prompts/
└── my\_agent/
└── default.txt        # Prompt template (if used)

````

---

## 🔧 1. Minimal `config.yaml`

This is the root Hydra config. It sets the logging path and the pipeline to use:

```yaml
defaults:
  - pipeline: pipeline
  - _self_

logging:
  logger:
    log_path: logs/

db:
  host: localhost
  port: 5432
  user: postgres
  password: postgres
  database: stephanie

stages:
  - name: generate
    cls: stephanie.agents.my_agent.MyGenerationAgent
    enabled: true
    strategy: default
    prompt_file: default
    prompt_type: file
```

Each stage requires:

* `name`: a unique identifier
* `cls`: the full import path of your agent
* `enabled`: whether the stage should run
* any agent-specific config you want to inject (e.g., `prompt_type`)

---

## 🧠 3. `MyGenerationAgent`: Your Custom Agent

Every agent should inherit from `BaseAgent` and implement a `run()` method.

```python
# stephanie/agents/my_generation.py

from stephanie.agents.base_agent import BaseAgent

class MyGenerationAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        prompt = self.prompt_loader.load_prompt(self.cfg, context)
        result = self.call_llm(prompt, context)
        context["hello"] = "World"
        return context
```

This agent:

* Loads a prompt
* Calls the language model
* Extracts a list of hypotheses
* Stores them in the shared `context`

---

## 📜 4. Prompt Template

Place a file at `prompts/my_agent/default.txt` like:

```txt
Generate 3 hypotheses for the goal: {{ goal }}
```

---

## 🚀 5. Run the Pipeline

Use the `main.py` launcher provided in the framework:

```bash
python stephanie/main.py goal="The USA may default on its debt"
```

This will:

* Create a timestamped log file
* Initialize memory and logging
* Execute the pipeline as defined

---

## 🧩 Optional Additions

You can later add:

* More stages (e.g. ranking, reflection)
* Prompt tuning
* Web search tools
* Report generation

Each addition is just a YAML node and a Python class away.

---

## ✅ Recap

| Component         | Purpose                          |
| ----------------- | -------------------------------- |
| `config.yaml`     | Global settings, DB, logging     |
| `pipeline.yaml`   | Defines agent stages             |
| `Agent class`     | Implements agent logic           |
| `Prompt template` | Guides model output              |
| `main.py`         | Launches the configured pipeline |

By following this chapter, you've built a reusable, testable AI pipeline using `stephanie` with the bare essentials. From here, you can scale out to review, reflect, evolve, and rank your hypotheses in modular stages.
