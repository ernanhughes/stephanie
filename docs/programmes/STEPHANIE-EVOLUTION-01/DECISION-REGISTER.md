# Decision register

1. Use the existing AI `ai.inference` request, budget, provider, result and structured-validation APIs; no shared AI source change was needed.
2. Keep opportunity policy, candidate identity, experiment orchestration and Harness mapping in Stephanie.
3. Keep STN06 separately identified and frozen; its source and preregistration were not changed.
4. Persist experiment records with Stephanie `ExperimentStore` and retained cases with `CaseBookStore`; isolated SQLite is supported through PostgreSQL JSONB variants.
5. Treat execution completion, independent evaluation and human acceptance as separate states. `PASS` alone is not `VERIFIED`; unknown billing remains null.
6. Offline fixture success grants no operational promotion authority. S8/S9 remain out of scope.
