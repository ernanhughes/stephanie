# Evidence register

| Evidence | Source | Use and limit |
|---|---|---|
| H3Q | `HARNESS-EVOLUTION-01/H3Q-NEGATIVE-EXECUTION-EVIDENCE.md`, SHA-256 `a7a92b928693e4279120b07aa413cb00f0014317bf38e7a0dc5d56bcf9ce57c7` | Synthetic baseline 4/4; acceptance pending; N1 failed and N2/N3 not evaluated. |
| H4 | `HARNESS-EVOLUTION-01/H4-EXPERIMENT-RESULTS.md`, SHA-256 `1e02671d151786285d8858af2b48f6ea7c67db29e266ab8a63f740dadbee855f` | Representation fixture comparison; one fixture is not effectiveness evidence. |
| H5 | `HARNESS-EVOLUTION-01/H5-EXPERIMENT-RESULTS.md`, SHA-256 `c9dad13127eb5601913f8ba10b11ff6d1fa271b649dd0a210f1fe54bec322bd5` | Local run passed pinned fixture; acceptance pending; single sample. |
| H6 | `HARNESS-EVOLUTION-01/H6-END-TO-END-EVIDENCE.md`, SHA-256 `0393485b1e5634596b2319b3ae1a4c54350ee39f1152ff33a2a1f42b25dee586` | Isolated scripted execution and Work Envelope correlation; no independent evaluation or acceptance. |
| AI calls | `data/stephanie_evolution01_final/ai-executions.json` | One eligible local Ollama `ornith-1.5:9b` structured call, hypothesis only; 11,791 ms; 1,419 input + 259 output tokens; cost null. Earlier 74.5-second invocation is retained but marked ineligible because its input contained a fixture-oracle label. |
| Fixture evaluation | `data/stephanie_evolution01_final/results.json` | P1 score 0.0 → 1.0 under frozen rule; fixture-label agreement only. |
| Independent review | `INDEPENDENT-REVIEW-RESULT.json` | Final review of evidence-digest-bound code by local `mistral:latest`; 4,741 ms; 2,051 input + 153 output tokens; conditional only on fixture limits. One malformed response attempt is recorded separately. |
