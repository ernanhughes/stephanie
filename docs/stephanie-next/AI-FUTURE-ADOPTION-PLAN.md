# AI Future Adoption Plan

| Consumer | Candidate adoption | Preconditions | Boundary |
|---|---|---|---|
| Stephanie | `ai.inference` for bounded, schema-validated opportunity/hypothesis assistance | Define the S-stage invocation points and authority; package AI reproducibly; retain deterministic fake-provider coverage | Stephanie chooses when/why to investigate; Worker executes only separately authorized plans |
| Stephanie | AI memory/experience or retrieval behind a narrow read adapter | Fix AI database dependency metadata; prove namespace mapping, retention, authorization, and ability to retrieve rejected/negative evidence; compare with existing CaseBook/MemCube/ExperimentStore | No bulk migration and no duplicate memory lifecycle |
| Stephanie | AI CORE-10 experiment contracts | Map task, fixture, implementation digest, evaluator, resource, and result contracts; qualify a Worker executor/evaluator integration separately | CORE-10's current built-in executor/evaluator scope is retrieval and document research; it does not replace Stephanie STN06 coding-task execution |
| Worker | Shared versioned schemas/fixtures where generic request/result values align | Identify a real cross-language contract gap and compatibility tests | Keep TypeScript runtime enforcement and strategy composition in Worker; no Python service call for trivial DTO reuse |
| Praxis | Shared evidence/resource DTOs if its evaluator requires them | Praxis-owned contract review and direct test evidence | Praxis retains effectiveness methodology and verdict |
| Mission Control | AI inference/evidence integration only where an authorized feature requires it | Mission-owned API review, policy and deployment qualification; distribution boundary | Mission owns operational policy, Work Outcome and acceptance; it remains read-only in this commission |
| Writer | Selective use of AI inference or retrieval | Writer interpreter floor and existing dependency strategy must accommodate AI's Python floor | Writer/ECK and writing-specific behavior remain Writer-owned |
| Workbench and other consumers | Reuse after capability-specific need is demonstrated | Inventory current implementation and deployment dependency before each adoption | Do not migrate consumers solely to satisfy a diagram |

## Recommended sequence

1. Complete the Stephanie stages against the approved original stage contract;
   prefer existing AI inference for any generic model call or structured
   parsing.
2. Add a reproducible AI packaging/release path before adding an install-time
   dependency to published Stephanie artifacts. The current editable install
   is specific to the local sibling workspaces.
3. Correct and document AI optional dependencies for SQL-backed packages before
   adopting memory or experiment persistence.
4. Build consumer-specific adapters and conformance fixtures only when a
   concrete cross-repository integration is authorized and tests can prove
   compatibility.
5. Offer Worker, Praxis and Mission adoption as separate future work. Do not
   change their repositories or production services in this commission.

## Current completion boundary

This plan is an adoption map, not evidence of adoption by Worker, Praxis,
Mission, Writer or Workbench. The local editable Stephanie-to-AI environment
proves the Python package boundary is importable under Python 3.13; it does not
prove the application workflows are integrated or production-ready.
