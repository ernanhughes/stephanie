# User guide

From `C:\Projects\stephanie`, with the Python 3.13 environment active:

1. Freeze once: `python -m stephanie.improvement se2-freeze`
2. Inspect the manifest and its 24-call fixed plan.
3. Execute once: `python -m stephanie.improvement se2-run`
4. Evaluate: `python -m stephanie.improvement se2-evaluate`

Storage is isolated at `data/stephanie_evolution02/`; an existing generation-results file stops reruns. Calls use local Ollama only; no external writes or paid calls are configured. Inspect `generation-results.json` and `evaluation-results.json` directly. The evaluator is a development checklist, not a production acceptance gate.
