# AI Consumer Compatibility

## Compatibility snapshot

| Consumer | Current AI usage | Interpreter/package compatibility | Risk and disposition |
|---|---|---|---|
| Stephanie | No source imports/dependency before local integration setup | Python now `>=3.13`; AI `>=3.11`; Pydantic 2 and NumPy 2 bounds compatible | Local editable install succeeded. A reproducible distributable/dependency declaration is still needed. Database-backed AI modules have undeclared dependencies. |
| Worker | No Python AI package import; TypeScript/Bun harness | Cross-language runtime mismatch | Keep Worker read-only. Share schemas and conformance fixtures if a concrete contract is needed. |
| Praxis | One example imports `ai.scoring.technical` | Python `>=3.12`; AI floor compatible | Example does not establish a production package dependency. No change. |
| Mission Control | Two experiment scripts use AI inference DTOs/provider | Main package has no declared AI dependency | Existing scripts are not evidence of deployed Mission runtime compatibility. No change. |
| Writer | Several experiment scripts use AI inference | Writer declares Python `>=3.10`; AI requires `>=3.11` | AI cannot be declared for all supported Writer interpreters without changing that contract. No change to Writer. |
| Workbench | No AI imports found | Python `>=3.10`; AI requires `>=3.11` | No current direct consumer. No change. |

## Compatibility evidence gathered

- `ai` built and installed editable into the new Stephanie Python 3.13.15
  environment without changing AI source.
- `import ai`, `import ai.inference`, `InferenceService` and
  `ReasoningProvider` resolved from `C:\Projects\ai\src`.
- Stephanie environment reports Python 3.13.15 and `pip check` reports no
  broken requirements after installation of available runtime dependencies.
- Hydra 1.3.8 and OmegaConf 2.3.1 are installed together. The existing
  uncommitted `omegaconf>=2.4` change was incompatible with Hydra 1.3.8 and was
  narrowed to `>=2.3,<2.4` in the package and requirements metadata.
- No AI test suite was run as part of the Python upgrade; no code was changed
  in AI. AI's experiment/memory behavior and every other consumer remain
  unqualified by this environment check.

## Protected compatibility boundary

Only Stephanie and AI are authorized write scopes in the architecture
amendment. Worker, Praxis, Mission, Writer and Workbench source were inspected
read-only and remain unmodified. This assessment makes no claim that their
runtime deployments have been upgraded or migrated.
