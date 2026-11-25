# Execution success semantics

This is an analytical mapping of observations to existing contracts. It does not add lifecycle states, database states or Work Outcome values.

| Level | Question | Existing contract/owner | SE2 / SE2Q observation |
|---|---|---|---|
| L0 Admission | Was request valid, authorized and within bounds? | Mission Work Envelope / Worker request validation | Original SE2 calls were submitted under the frozen local study; SE2Q Worker request was also blocked once before provider dispatch. Keep admission distinct from execution. |
| L1 Dispatch | Did the intended executor/provider receive it? | Worker execution/invocation ledger; AI provider result/trace | 7 of 8 initial/follow-up requests dispatched, 1 blocked; plus one direct exact-API call. Empty dispatched records do not establish whether provider inference ran. |
| L2 Response | Was a body returned and retained? | AI `ReasoningResult.content`; Worker WorkResult/artifact refs | Historical SE2 raw bodies not retained. SE2Q includes two Worker empty outputs; one Worker body truncated. Direct success bodies retained. |
| L3 Parseability | Is the body parseable? | AI/Worker parsers; Stephanie parser error field | All 24 SE2 failures were parser-observed invalid JSON; exact reason cannot be reconstructed. SE2Q one Worker body was incomplete JSON. |
| L4 Structure | Does it satisfy the declared schema? | AI Pydantic response model; Worker JSON Schema; Stephanie second validation | Direct AI 3/3 and exact API probe 1/1 passed structure for narrow development tasks. Worker one result passed in 5. Historical schema minima caveat remains. |
| L5 Evidence | Are evidence references supported by the task inputs? | Stephanie evidence gate; Mission source/evidence links | Direct AI and exact API probe cited task-visible IDs; Worker 1 valid output passed. Citation validity does not verify factual truth. |
| L6 Semantics | Does it correctly answer the task? | Independent domain evaluator / protected rubric | Direct semantic/concept checklist complete on 1/3 only; not qualified. Worker not qualified. |
| L7 Effectiveness | Did the intervention improve outcome against baseline? | Praxis comparative effectiveness + independent evaluation | No integrated Stephanie baseline/candidate comparison; SE2 is inconclusive for experience benefit. |
| L8 Acceptance | Was outcome accepted under formal governance? | Mission human decision and acceptance contract | No SE2Q candidate acceptance or promotion. Evaluation/recommendation cannot imply human acceptance. |

Names L0–L8 are explanatory labels only. Implement using existing Work request/result, Work Evaluation, Work Outcome episode, Praxis EffectivenessRecord and Mission acceptance records. Execution status, provider response, parse/schema/evidence correctness, semantic correctness, comparative effectiveness and formal acceptance remain distinct assertions.
