# First integrated vertical slice — proposed only

No part of this vertical slice was executed or deployed in this commission.

| Step | Reused mechanism | Current evidence | Required before slice |
|---|---|---|---|
| 1. Mission supplies observed problem | Work Outcome Episode, Objective, Requirements, Evidence/Findings | Mission APIs and live schema 54; H6 catalogue current. | Pin source/API version; identify a sanitized test episode and exact project scope. |
| 2. Stephanie receives bounded evidence | Stephanie `Evidence` with source identity and digest | Current evidence import and source-drift checks. | Add a read-only Mission evidence adapter; preserve UNKNOWN and omit protected evaluation fields. |
| 3. Retrieve experience | Stephanie CaseBook/ExperienceStore; existing retrieval | Existing CaseBook and ExperimentStore persistence, local fixture retrieval. | Use only eligible reviewed historical records; define reference identity and relevance label; don't treat retrieval as truth. |
| 4. Construct hypothesis | Direct shared AI via explicit local route | Structured development outputs passed 3/3; semantic checklist 1/3; direct probe 1/1. | Fix direct AI exception evidence loss; pin provider/config; fresh semantic evaluator and tasks. No automatic fallback. |
| 5. Construct immutable candidate | Stephanie candidate digest, protected evaluator digest | Candidate is evidence-bound and fixture tests reject mutation/drift. | Candidate must identify Worker source/strategy and cannot alter evaluators/policy. |
| 6. Run baseline and candidate | Mission Work Envelope → Worker `/v1/work`, explicit route/strategy | Mission-H6 isolated scripted run and Worker API exist; SE2Q Worker reliability 1/5. | Resolve Worker empty-response observability; actual local/provider qualification separately authorized; freeze common resource budgets and candidate artifact digest. |
| 7. Independent effectiveness assessment | Praxis canonical effectiveness record/comparison or existing independent evaluator | Praxis contracts and comparison functions; no eligible PRAXIS-13 cohort. | Agreement on a fresh-task evaluation schema, independent evaluator boundary, unknown handling and minimum cohort. No Stephanie fixture evaluator as external evidence. |
| 8. Mission presents recommendation | Work Outcome evaluation/findings + corrective Proposal | Current API contracts exist; W9 is conditional. | Proposal adapter, source lineage and actor/auth contract; distinguish comparison recommendation from Mission decision. |
| 9. Human decision | Mission approval/acceptance contract | Existing review/transition paths; no acceptance in H6. | Human reviews exact candidate, comparison evidence and limitations; acceptance remains explicit and durable. |
| 10. Retain experience | Stephanie stores reference to canonical Mission/Worker/Praxis IDs | CaseBook/ExperimentStore support isolated persistence. | Save references and digests; indicate “stored reference, source not independently verified” until rechecked. |

## Smallest implementation order

1. Stephanie: add reference-only Mission evidence/candidate model, direct-AI failure-result capture, and serialization/unknown-value tests. Keep routes explicit and no auto fallback.
2. Worker/Mission owners: resolve the documented Worker response attribution issue and publish an accepted clean H6/H7 source+API package; verify actual response retention and authenticated actor policy.
3. Stephanie: add a Worker adapter that retains full WorkResult/H3/artifact references and fails closed on ambiguous dispatch.
4. Praxis: accept/version the baseline/candidate input record and provide canonical comparison references. Do not implement this in Stephanie.
5. Mission: approve the proposal/Work Envelope mapping and human acceptance route.
6. Run only a synthetic, isolated, separately authorized vertical slice after all identities and policies are pinned. No activation or live model execution is authorized by this report.

## SE3 preservation

Keep `STEPHANIE-EVOLUTION-02/EXPERIMENT-PROTOCOL.md`, original 24 SE2 calls, SE2Q review, protected oracle boundaries and STN06's frozen identity separate. A future study may use Worker for controlled execution and Praxis for independent comparison, but it needs a new preregistration, fresh tasks/holdouts, and an explicit decision on whether it still tests experience transfer. Do not repurpose exposed SE2 tasks as holdouts or silently amend STN06.
