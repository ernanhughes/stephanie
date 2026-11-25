# Praxis Comparison Evidence

Stephanie called the existing Praxis `compare_effectiveness` function with actual saved Worker outcomes and the independent evaluator observations. Praxis default minimum cohort is three runs per arm.

- C1: `retain`; candidate exact-answer score 3/3, baseline 0/3. Both use one Worker strategy invocation per task.
- C2: `reject` (no-benefit control).
- C3: `reject` (injected regression control).
- C4: `insufficient_evidence` (one record per arm).
- C5: This separate resource-confound control compares `structured.v1` with `corrective.v1` (not the frozen adaptive candidate). Praxis raw result is `retain`, but Stephanie withholds a recommendation as `WITHHOLD_RESOURCE_CONFOUNDED`: corrective used two strategy invocations per task versus one for baseline. Cost is unknown. This guard supplements Praxis rather than replacing its evaluator.
- Live local-model pair: `insufficient_evidence`, one comparison per arm. Both exact-output checks pass and invocation count is equal.

The evaluator is independent of the Worker request and reports output digest evidence without embedding the expected value in the evaluation record. Praxis outcomes are comparative recommendations, not Mission acceptance. The corrected deterministic set is under `data/stephanie_convergence03_corrected/praxis-c*.json` and `integrated-results.json`. The earlier `data/stephanie_convergence03_final/` set remains preserved as historical and C5 there is unqualified due to the frozen-limit mismatch.
