# Model routing evidence

## Deterministic routing

Focused suite: `tests/test_free_first_chat.py` (13 passed; marked `long` because module import/run exceeds the ordinary two-second loop). It covers preferred route, quota rotation/cooldown, authentication/unknown failures, capability and privacy gates, explicit pinning, local fallback, stable Go session header and external override rejection.

The active isolated config contains only the `subscription_included` OpenCode Go route, allowed for `approved_external`. Unknown/private/project turns are ineligible and use the configured local model. No metered paid model appears in the route list.

## Genuine qualification budget

Maximum six intentional requests were permitted. SA1 used six attempts: one OpenCode Free CLI request (429), one attempted local Ollama request which did not return usable response evidence, one initial Go request that failed before the required stable session header was added, one successful direct `LLMService.achat` synthetic response, and two successful synthetic SIS turns. No extra model call was made during web/API verification, persistence restart verification, page checks, or replay.

Observed successful Go usage:

| Invocation | Input tokens | Output tokens | Reported cost | Billing source |
|---|---:|---:|---:|---|
| Direct synthetic `LLMService.achat` | 50 | 45 | 0 | Go allowance |
| Persisted turn 1 | 129 | 114 | 0 | Go allowance |
| Persisted turn 2 | 177 | 69 | 0 | Go allowance |

The direct synthetic response was “Continuity is ready.” The persisted turns acknowledged the Alpha objective and correctly recalled “three study notes by Friday.” These examples establish basic request understanding and transcript continuity only, not broad model quality.
