# Bounded experiment proposal

Local record: `experiment-proposal.json` created by `propose-worker-diagnostic`.

Propose one authorized synthetic lifecycle observation with zero model invocations and no retries. Capture stable request and invocation IDs, admission/dispatch, response presence, extraction, timeout/cancellation, termination and usage, persistence, and artifact digest. Limit to one run, 120 seconds, and zero provider cost. Independent review checks completeness and source identity. Reject if state is inferred from HTTP acceptance, identity is lost, retries occur, or raw response retention lacks approved data policy.

Potential regressions are sensitive-data retention, latency and incorrect dispatch attribution. This proposal is `NOT_SUBMITTED_NOT_AUTHORIZED`. It is not sent to Mission and cannot authorize Worker execution.
