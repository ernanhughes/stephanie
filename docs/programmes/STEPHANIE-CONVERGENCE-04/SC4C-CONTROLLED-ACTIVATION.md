# SC4C — controlled activation

**Verdict: BLOCKED; no activation performed.** On 2026-11-25, read-only checks against Mission `:8790` reported API health `ok`, version `0.1.0`, database `ok`, schema 54, AI/monitor enabled. `/api/improvement/integration` exists but reports Stephanie unavailable because no isolated run root is configured. `/api/improvement/records` exposes a different `isolated_sc3_artifacts` record with formal acceptance `UNKNOWN`; it is not the SC4B owner-review experiment. No write, migration, restart or dispatch was made.

The owner-review application on `:8791` remains a separate isolated process and is available at `/improvement`. Its experiment remains reported complete with the formal decision undecided. Do not treat either runtime observation as approval to dispatch from `:8790`.

SC4 negative controls remain incomplete: missing Worker, direct stale-version/revocation, multi-process concurrency, forced experience-retention failure. The activation package is not reproducibly isolated from concurrent dirty work. No operational database identity, qualified backup/restore, service rollback rehearsal, full compatibility check, or independent final review is available. The health endpoint does not prove source or data provenance.

Required artifacts are tracked in `SC4C-ACTIVATION-PACKAGE.md`, `BACKUP-AND-ROLLBACK-EVIDENCE.md`, and `INDEPENDENT-ACTIVATION-REVIEW.md`. All three gates remain blocked. Keep activation OFF and do not promote a candidate. Existing Mission functionality on `:8790` has not been disturbed.
