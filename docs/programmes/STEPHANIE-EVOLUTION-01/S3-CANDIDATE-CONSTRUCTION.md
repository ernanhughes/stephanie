# S3 — Hypothesis and candidate construction

The latest schema-valid AI hypothesis is passed into candidate construction and remains explicitly uncertain and evidence-bound. Candidate `d80cd1cbb12e2f2e75107745` is content-addressed over the hypothesis, protected evaluator and normalized evidence-record digests. It changes only the status-label policy: require independent PASS and explicit ACCEPTED before `VERIFIED`. Edits behind an evidence ID invalidate the candidate. It does not modify strategy execution, evaluators, Worker, Harness or STN06.

Candidate records are immutable in the local ledger. Evidence outside the opportunity, evaluator-digest changes, missing compute declarations, and unapproved experiment invocation are rejected. A candidate is not a production patch or promotion request.
