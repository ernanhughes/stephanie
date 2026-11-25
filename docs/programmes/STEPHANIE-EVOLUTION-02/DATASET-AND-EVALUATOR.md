# Dataset and evaluator

Eight prompts use sanitized E1/Harness mechanisms: unknown cost; malformed structured response; retrieval relevance; compute confounding; multiple plausible causes; mechanism versus effectiveness; unsupported intervention; evaluator regression. These are development tasks based on known artifacts, not protected holdouts; exact answers may be inferable from history.

The protected checklist is `stephanie/improvement/evolution02_rubric.json`. It checks output structure, valid evidence references, memory identity availability, task-specific required concepts and abstention. It is deterministic and capped at 8 points. It verifies basic consistency only; it cannot reliably score causal nuance, evidence interpretation quality, or verbosity-independent reasoning. No model judge was used. No false-positive/negative rate claim is made; verifier examples are covered by focused unit controls, not a calibrated benchmark.
