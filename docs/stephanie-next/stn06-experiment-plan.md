# STN06 — Experience Transfer Experiment Plan

## Research question

Does a relevant, human-approved Stephanie experience improve the same OpenCode/GPT-6 Luna agent's performance on unseen coding tasks, compared with the same agent running without retrieved experience?

No STN06 model call has been made. STN04 acceptance and STN05 experience approval are prerequisites.

## Precondition gates

1. A human explicitly accepts STN04 using the SIS review endpoint.
2. A human separately approves the deterministic STN05 lesson candidate.
3. The approved MemCube is active, source-linked, and retrievable using the exact approved-only policy.
4. The experience payload and retrieval policy are frozen and hashed before evaluation tasks are submitted.
5. Use STN05's stable SHA-256 `ExperimentStore` assignment and preserve the preregistered task-to-arm schedule.
6. The operator explicitly authorizes any paid GPT-6 Luna calls.

If any gate is missing, the experiment remains NOT RUN.

## Proposed pilot

- **Pilot size:** 24 paired task IDs (48 total model executions), six tasks in each of four held-out Python task families.
- **Families:** arithmetic edge cases; boundary conditions; list/string transformations; and small state/reporting logic.
- The original calculator task and its exact code/tests are excluded from the held-out set.
- Each pair uses two fresh, identical disposable workspaces and the same pre-registered objective, tool policy, resource limits, Docker image, OpenCode version, and `opencode-go/gpt-6-luna` model.
- Arm A receives the standard objective only.
- Arm B receives the exact same objective plus the bounded, source-attributed, approved experience context. No other prompt, system configuration, or model setting differs.
- Randomize which arm runs first using a frozen SHA-256 schedule over `(experiment_id, task_id, preregistered_salt)`. Do not use Python's built-in `hash()`.
- Keep each arm's workspace/output inaccessible to the other arm. Never continue a session across arms.

The 24-pair pilot is intended to reveal implementation issues and estimate variance; it is not sufficient to claim broad improvement. If the effect is small or uncertain, preregister a larger confirmatory sample based on pilot success rates before further paid runs.

## Task and evidence preparation

Before inference, prepare a manifest containing for every task:

- Stable task ID and family
- Objective text and hash
- Fixture file hashes
- Independent test command and expected result
- Expected in-scope modified paths
- Prohibited paths/effects
- Arm schedule and fresh workspace identities

Seal the independent test oracle from the model-facing task context. The agent may execute tests present in its workspace as part of ordinary work, but evaluation labels, expected diffs, and other arm outputs must not be supplied as hidden coaching context.

## Outcomes and analysis

### Primary outcome

Binary task success: the independent verifier passes and every modified path is within the preregistered scope.

### Secondary outcomes

- Test failures and failure category
- Unnecessary or out-of-scope changes
- Tool errors/retries
- Wall-clock duration
- Input/output token usage and cost only if the runtime genuinely reports them
- Retrieval IDs, scores, context length, and context hash

Report paired differences with confidence intervals and the complete task-level outcomes. Use a paired binary method (for example, exact McNemar) for pass/fail and paired robust summaries for duration. Treat missing usage/cost as unavailable; do not estimate it from response length. Report negative and harmful transfer, not just favorable examples.

No single success, model explanation, or test score establishes system-level improvement. Do not change acceptance criteria after observing results.

## Leakage and safety controls

- The experience is frozen before held-out task/solution disclosure.
- Only approved MemCube experiences enter Arm B.
- Retrieved text is bounded and labeled untrusted task context, never a system instruction.
- No automatic prompt mutation, skill promotion, model training, or autonomous policy change occurs.
- The Docker executor remains explicit and isolated; Worker remains a loopback-only bridge without authentication.
- External publication and Git writes remain prohibited.

## ExperimentStore integration

Use existing ExperimentStore for experiment, variants, trials, and metrics if its live schema/store path is verified. STN05 replaces the process-randomized `hash()` arm assignment with SHA-256 and tests it under multiple `PYTHONHASHSEED` values. Verify the live schema before use; if store integration is not reliable, persist the preregistration and immutable schedule through the existing experiment services and block the run rather than silently assigning arms.

## Approval

STN06 execution requires a separate operator approval after STN04 review, STN05 candidate approval, and fixture/power-plan review. This document is a plan, not evidence that an experiment ran or that experience improves performance.
