# AI Reuse and Extension Decisions

| Candidate capability | Evidence | Placement decision | Reason / condition |
|---|---|---|---|
| Provider-neutral model invocation | `ai.inference.ReasoningRequest`, `ReasoningResult`, `ReasoningProvider`, `InferenceService` and fake/OpenCode/Ollama adapters exist | Reuse AI | It is application-independent and already records route, status, usage and fingerprints. Stephanie retains purpose, prompt and authorization policy. |
| Structured output parsing and validation | `InferenceService.generate_structured` validates a Pydantic schema and reports malformed output | Reuse AI | No duplicate generic parser. Opportunity/candidate DTOs remain Stephanie-owned. |
| Provider capability descriptions | Providers expose `describe()`, but no cross-provider capability schema was found | Do not add a new abstraction yet | First prove a consumer requires normalized capability fields; current `describe()` may suffice. |
| Model/configuration identity | `InferenceProfile`, result route/model/revision, fingerprints and trace fields exist | Reuse fields where present; record a Stephanie-level frozen digest as needed | Do not conflate provider route, model revision and Stephanie policy identity. |
| Evidence references and artifact identity | CORE-10 `SourceRefDTO`/`ArtifactDTO`, evidence refs, hashes; AI Memory `EvidenceDTO` | Reuse compatible primitives; keep Mission Work Outcome authoritative | Cross-system mappings need explicit provenance semantics; no automatic equivalence claim. |
| Unknown and missing values | Optional fields/status enums exist in inference, experiments and memory contracts | Preserve `None` and explicit statuses | Never convert unavailable token/cost/evidence fields to zero or success. |
| Failure classification | Inference distinguishes error/timeout/unavailable/malformed/refused; experiment executor has typed failures | Reuse and map losslessly | Do not flatten provider failures into task correctness. |
| Resource accounting | Inference budget and provider-reported usage; experiment `ResourceBudgetDTO` | Reuse generic budgets/observations; Worker remains runtime enforcer | Provider token/cost fields may be unknown; do not estimate unless a separate field explicitly identifies estimate. |
| Retrieval/memory | AI has namespace-scoped Retrieval and Memory services; Stephanie has existing CaseBook/MemCube | No migration or duplicate store | Validate authorization, data semantics, DB dependencies and history-retention fit before adapter work. |
| Stephanie experiment lifecycle | Stephanie STN05/STN06 includes CaseBook, `ExperimentStore`, frozen preregistration, fixtures and Worker dispatch; AI CORE-10 has a separate research experiment service | Keep Stephanie orchestration in Stephanie; selectively reuse AI contracts | CORE-10's current executor scope is retrieval/document research; it is not a drop-in Worker coding experiment engine. |
| New shared AI primitive | No missing primitive required by the reviewed Stephanie paths is established | None proposed | Existing inference API is sufficient for the first narrow integration. Do not add modules speculatively. |
| AI persistence packaging | AI database modules import SQLAlchemy/PostgreSQL components absent from declared AI extras | Follow-up AI packaging correction before memory/experiment dependency | This is a reproducible-install defect for those surfaces, not a reason to copy them into Stephanie. |

No source changes to AI were justified by this review. The immediate Stephanie
integration should consume existing inference APIs when its stage contract
specifies an actual model-assisted step; no model call is required to verify
the deterministic boundary.
