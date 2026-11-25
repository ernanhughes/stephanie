# Praxis compatibility

## Existing contracts

- `praxis.contracts.Goal`, `Strategy`, `Run`, `Evidence`, `Evaluation`, `Candidate` and `Experiment` are stdlib dataclasses. They can express candidate identity, per-run evaluation references and a recommendation (`promote` / `reject` / `abstain`) in the legacy in-memory harness contracts.
- The canonical newer surface is `EffectivenessRecord` (`praxis.effectiveness`), `EffectivenessComparison`, `compare_effectiveness`, and `work_evidence.project_execution`. It carries execution refs, strategy and capability versions, acceptance criteria/provenance, independent evaluation, and optional baseline comparison.
- Comparisons require adequate records; unknown measurements stay `None`, and `insufficient_evidence` is an explicit outcome. A single execution is not evidence of improvement. Praxis comparison is advisory and does not mutate Mission proposals.
- Mission bridge/source references exist in Praxis; there is no callable Stephanie-to-Praxis adapter or current Praxis HTTP service observed.

## Readiness categories

| Capability | State | Evidence / limitation |
|---|---|---|
| Experiment and candidate data structures | Available, tested | Source contracts and harness tests. Older `Experiment` dataclass defaults unknown score fields to zero; do not use it where missingness matters. Prefer the canonical effectiveness record. |
| Independent effectiveness comparison | Available, but not qualified for Stephanie | `compare_effectiveness` and PRAXIS-00..18 records. Praxis's current PRAXIS-13 gate reports `not_yet_evaluable`; no new accepted outcomes/cohort. |
| Mission execution projection | Available and previously demonstrated | `work_evidence.project_execution` and prior Praxis-Mission report; recheck current schema/source identities before use. |
| Stephanie integration | Missing | No Python dependency, adapter, protocol test, or operational linkage. |
| Recommendation / promotion authority | Mission/human-owned | Praxis may recommend `retain`, `reject`, or insufficient evidence; it cannot accept or promote a candidate in Mission/Worker. |

## Integration direction

At the first slice, Stephanie should store only Praxis `record_id`/comparison IDs and immutable candidate/manifest digests after an authorized call. Praxis should consume canonical Worker/Mission evidence references and independently score a frozen comparison. Do not serialize Stephanie's fixture-only evaluation as Praxis evidence, invent scores for missing data, copy a second evaluation engine, or treat a recommendation as acceptance.

Repository ownership: all Praxis changes belong in `C:\Projects\praxis`; this commission did not change that tree. A future integration should be adapter-only in Stephanie plus versioned conformance fixtures agreed with Praxis, not a new evaluation implementation.
