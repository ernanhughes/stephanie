# S2 — Evidence and opportunity discovery

The offline Harness adapter imports bounded sanitized summaries for H3Q, H4, H5 and H6. Each record includes source identity/digest, task/execution/strategy where known, evaluation and acceptance status, observed facts, limitations, and explicit unknown cost. Evidence identities are immutable; conflicting re-imports fail.

With four unresolved observations, discovery identifies a medium-confidence opportunity to separate execution, independent evaluation and acceptance. Fewer than two unresolved observations produce an abstention. Missing evaluation is never treated as pass; `PASS` is never treated as accepted.
