# Mission Control compatibility

## Current authority and callable surfaces

Mission owns operational objectives/governance, Work Outcome records, envelopes, corrective proposals and final acceptance. Current source includes:

- Work observation episodes, requirements, associations, findings and evaluations (`/api/work-episodes`, `/api/work-evaluation`, related observation services).
- Pre-execution Work Envelope registration and post-execution linking (`POST/GET /api/work-envelopes`, `POST /api/work-envelopes/{id}/executions`). Envelope identity binds episode, objective, requirements, repositories, scope and acceptance criteria.
- Corrective proposal lifecycle (`/api/work-corrective/proposals`, scope, transition, handoff). API routes require explicit actor/reason/confirmation inputs; values are not themselves a cryptographic human identity.
- H6 Harness read/catalogue/execution/artifact routes under `/api/harness`. `GET /api/harness/catalogue` was read-only observed as reachable and connected to Worker 0.2.0.

Mission live health reported version 0.1.0, schema 54, database OK. W9 programme status is conditional acceptance with closure recommended. H6 is accepted for isolated integration only. H6 final report says no operational activation and no real provider inference. The current Harness catalogue proves route reachability, not a safe authorized live execution.

## Current boundary

Stephanie has no Mission adapter. Its local `Opportunity`, `Hypothesis`, `Candidate`, `Evaluation` and recommendation are Stephanie-owned records; none is a Mission `Proposal`, `WorkEnvelope`, `EpisodeAssessment`, human acceptance, or verified outcome.

Future adapter must submit references and bounded typed fields only: observed episode/objective, requirement IDs, evidence/source digests, hypothesis/candidate/manifest IDs, expected acceptance criteria, compute envelope, and explicit human review requirement. Mission must resolve the objective and authorize the envelope before Worker dispatch. Do not accept caller-supplied objective text as authority, auto-apply a candidate, or infer human identity from actor text.

## Activation and blockers

The old H6A availability assumption needs revision: current Mission/Worker processes are live and catalogue reads work. The old **activation decision** is not superseded: H6 is still documented as isolated-only, Mission and Worker trees are dirty, live process/source identity is not bound, actual provider execution and termination behavior are unqualified, and H6 actor identity is operator-attested. This commission performed no activation, writes, migration or service restart.

H6/H7 implementation belongs to Mission; Work Outcome schema, acceptance and operational controls remain Mission-owned. Stephanie may build a read-only proposal/reference adapter only after Mission publishes a stable versioned schema and the operator identity/authorization contract is accepted.
