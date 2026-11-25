# Risk register

| Risk | Control / remaining limit |
|---|---|
| Model invents evidence | Structured output schema plus evidence-ID subset validation; the model result remains a hypothesis. |
| Execution is mislabeled as verified | Frozen evaluator requires independent PASS and explicit ACCEPTED; unknown stays unverified. |
| Evaluator weakened by a candidate | Candidate digest binds the frozen evaluator; mutation is rejected. |
| False effectiveness claim | Fixture-only status and recommendation explicitly prohibit operational promotion. |
| Extra compute drives apparent gain | Compute is declared; without a compute-matched control evaluation becomes inconclusive. |
| Incomplete or conflicting records | Explicit evaluation/acceptance states and abstention; no zero-cost default. |
| Persistence reaches production | CLI selects an isolated SQLite file under the selected root; no service/operational database adapter is provided. |
| Dependency distribution | AI is installed editable in this development workspace; a reproducible published AI package source remains unqualified. |
| Independent review | A separate local model review is recorded in `INDEPENDENT-REVIEW.md`; it supplements, not replaces, human acceptance. |
