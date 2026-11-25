# AI Shared Foundation Assessment

## Scope and repository snapshot

This is a source and packaging assessment for STEPHANIE-EVOLUTION-01's AI
architecture amendment. It is not a full S0-S7 implementation report.

| Repository | Snapshot | Working tree at inspection |
|---|---|---|
| `C:\Projects\ai` | `cf43eddc1f8109ea2f575ac290b6c1af849f3459` (`main`) | clean |
| `C:\Projects\stephanie` | `98388a4b06374f515ae4e84c3e87f0466e8e0fb7` (`main`) | dependency files already modified before this task; those edits were preserved except the Hydra/OmegaConf incompatibility described below |

AI is an installable Python package (`ai`, version `0.2.0`, Python `>=3.11`,
base dependencies NumPy and Pydantic). It contains implemented embedding,
retrieval, memory, consolidation, experience, inference, experiment, scoring,
research and neural-memory packages. The public root API exports embedding;
other capabilities expose package-level APIs such as `ai.inference` and
`ai.experiments`.

The strongest immediate fit for Stephanie is `ai.inference`: provider-neutral
request/result contracts, provider adapters, explicit budgets, usage and
identity reporting, fail-closed errors, optional trace recording, and
Pydantic-validated structured output. The deterministic fake provider allows
offline integration checks. This should support Stephanie's model-assisted
investigation without transferring Stephanie's opportunity policy or Worker
execution authority into AI.

AI also has `ai.experiments` CORE-10 research-to-experiment contracts and a
service for frozen plans, authorization, bounded execution, independent
evaluation and findings. Its implemented executors and evaluators currently
target retrieval and document research; this is not yet a demonstrated
executor for Stephanie's Worker coding experiments. `ai.memory` and
`ai.experience` have namespace-scoped APIs, but their persistence path needs
packaging review before adoption.

## Compatibility and packaging findings

- Stephanie now declares Python `>=3.13`; its recreated local environment is
  Python 3.13.15. AI's `>=3.11` floor is compatible.
- Stephanie's Pydantic 2 requirement and NumPy 2 requirement satisfy AI's
  declared lower bounds.
- AI is not declared in Stephanie's package dependencies. There is no
  established package-index or release artifact in this workspace from which
  Stephanie can reproducibly install it. The local 3.13 environment has AI
  installed editable from `C:\Projects\ai` for development qualification; this
  is not a production distribution solution.
- Existing uncommitted dependency updates requested OmegaConf 2.4 while Hydra
  1.3.8 requires OmegaConf below 2.4. The compatible constraint is now
  `omegaconf>=2.3,<2.4`; the rest of those dependency edits remain intact.
- AI's `ai.experiments` and database-backed memory/experience code import SQL
  persistence components, but the base and optional dependency lists in
  `C:\Projects\ai\pyproject.toml` do not declare SQLAlchemy and a PostgreSQL
  driver. Do not assume those packages are installable from a clean AI-only
  install until this is corrected or the required extra is documented.

## Conclusion

AI is a real reusable implementation foundation. Reuse `ai.inference` first;
do not create another generic model caller or structured-output validator in
Stephanie. No new AI primitive is justified by the evidence reviewed so far.
Before promoting a durable Stephanie-to-AI package dependency, establish a
reproducible package distribution and decide how AI's database-backed extras
are declared. Keep lifecycle policy and Worker execution specific to their
application owners.

This assessment does not claim that Stephanie currently calls AI from its
application code, that CORE-10 runs Worker coding experiments, or that any
model inference occurred.
