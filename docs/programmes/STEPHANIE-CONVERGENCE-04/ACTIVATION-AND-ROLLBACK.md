# SC4 activation and rollback

## Decision

No activation or rollback was attempted. Activation gate failed because the running Mission API is pre-SC4 (`GET /api/improvement/records` returns 404), shared worktrees are dirty with concurrent work, and a controlled package plus verified backup/rollback procedure is absent.

## Safe future sequence

1. Isolate and review the SC4 change set against the accepted source revision; preserve concurrent work.
2. Produce a versioned package and record its hashes.
3. Capture and verify the Mission database/configuration backup and document a tested rollback target.
4. Deploy only to an isolated environment, verify health and the improvement route, then run the authorized isolated workflow and review evidence.
5. Obtain the required acceptance before operational activation.

If any gate fails, keep the feature inactive and restore the captured package/configuration/database state using the verified rollback procedure. These steps are a future gate, not evidence that a deployment or rollback occurred.
