# STEPHANIE-EVOLUTION-03 — proposed protocol (not authorized to run)

SE3 remains gated; do not run it from this document. Keep the memory-transfer research question and A/B/C conditions, but use **new tasks** because the original eight SE2 prompts have now been exposed in model diagnostics. Freeze the task set, prior-memory corpus, evaluator, model, effective generation route, output contract, retrieval selection, sample size, stop rules, missing-output treatment, and context/compute accounting before any protected inference.

## Gate sequence

1. Deterministic infrastructure tests for protected data separation, retrieval identity, frozen hashes, schema validity, and result preservation.
2. A distinct development readiness set demonstrates the chosen execution route can produce three separately evaluable outputs. The direct-AI local JSON Schema route passed this narrow format/citation gate. Worker adaptive on this model is not qualified yet; do not mix it into A/B/C until separately qualified.
3. Freeze new independently controlled tasks and evaluator. Require at least one independent verifier blind to arm identity.
4. Only after explicit authorization, execute each preregistered arm once under a shared model/config. No answer-driven harness tuning.
5. Score all responses, including unusable and empty outputs, under frozen rules. Report effectiveness separately from execution reliability.
6. Independent review and bounded interpretation; no promotion or operational integration.

Preferred initial route: shared `ai.inference` with native Ollama JSON Schema and raw-response/trace persistence, since that path passed 3/3 development structural/citation checks. Use Worker mediation only if that exact model/route meets its own three-task readiness gate; a single 1,024-token success is insufficient. Record provider-reported tokens, `done_reason`, latency, and unknown cost. Add a context-budget-matched neutral-history arm if the study aims to attribute memory relevance rather than added information volume.
