# AI Dependency Map

## Intended direction

```text
Stephanie ─┐
Worker ────┤
Praxis ────┼──> AI shared mechanisms
Mission ───┤
Writer ────┘

AI ──X──> Stephanie / Worker / Praxis / Mission / Writer
```

The source review found no application-level imports from those repositories
inside `C:\Projects\ai\src\ai`. AI declares only generic runtime dependencies
in its base project metadata. Preserve this direction; do not move application
policy into AI.

## Current source consumers found

| Consumer | Language/runtime | Evidence in source | Status |
|---|---|---|---|
| Stephanie | Python | No `from ai...` or `import ai...` in `stephanie/` or `sis/`; no declared AI dependency before this assessment | Not yet an application consumer. Local Python 3.13 dev environment now has editable AI package installed. |
| Worker | TypeScript/Bun | No Python AI package import in Worker source; Worker exposes its own HTTP execution API | Not a direct package consumer. Use versioned schemas/fixtures if a shared cross-language contract is needed. |
| Praxis | Python | `examples/praxis16_lineage.py` imports `ai.scoring.technical.sha256_text` | One example-level consumer; no package dependency declared in Praxis `pyproject.toml`. |
| Mission Control | Python | `scripts/expo02/run_semantic.py` imports `ai.inference` DTO/provider; `scripts/expo02/repair_candidate.py` imports `ReasoningRequest` | Research/experiment scripts only; Mission's main package has no declared AI dependency in `pyproject.toml`. |
| Writer | Python | `scripts/expo03a/*` and `scripts/expo03b/opencode_runner.py` use AI inference DTOs/provider | Research/experiment scripts only; Writer package metadata does not declare AI. |
| Workbench | Python | No AI imports found in inspected application code | No direct consumer found. |

These are source-import findings, not proof that no scripts, generated artifacts,
or deployment environments use AI dynamically.

## Python package edges and constraints

- AI: Python `>=3.11`; base dependencies NumPy and Pydantic.
- Stephanie: now Python `>=3.13`; NumPy and Pydantic requirements are compatible
  with AI's lower bounds.
- Worker: TypeScript/Bun; no direct Python runtime edge is appropriate for small
  shared value contracts.
- Praxis: Python `>=3.12`; compatible at interpreter level.
- Mission: Python package does not currently declare AI; inspect its exact
  runtime environment before enabling a package dependency.
- Writer: Python `>=3.10`; AI's `>=3.11` excludes some currently declared
  Writer interpreters, so adoption needs a deliberate floor change or an
  equivalent schema/fixture contract.
- AI experiment and memory packages use database-backed components whose
  dependencies are not declared by AI's current metadata; do not add these
  edges until packaging is corrected.

## Dependency actions

1. Package and release AI from an immutable, reproducible source before adding
   it to application dependency declarations. The current local editable
   install is development-only.
2. Use `ai.inference` as Stephanie's first direct integration because it is
   Python-compatible and does not require moving Worker or Mission logic.
3. Treat Worker interoperability as a versioned schema/fixture problem, not a
   Python runtime-service dependency.
4. Do not retrofit Writer, Praxis, Mission or Workbench during this commission.
