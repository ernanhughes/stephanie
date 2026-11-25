# S4 — Controlled experiment and evaluation

The deterministic experiment compares an intentionally defective baseline (`execution completed means verified`) with the candidate against a frozen independent rule (`PASS` and explicit `ACCEPTED`). Four supplied records are all pending acceptance or unevaluated. Baseline agreement is 0/4; candidate agreement is 4/4. Result: `PASS_FIXTURE_ONLY`, not task-effectiveness evidence.

Baseline and candidate arms, protocol/candidate digests, fixture results, and unknown resource fields are retained using Stephanie `ExperimentStore`. Additional compute without a compute-matched control forces `INCONCLUSIVE`. The evaluator digest is protected from candidate alteration. No operational executor is connected.
