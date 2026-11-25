# SE2Q-H — Diagnostic results

Frozen readiness protocol digest: `0efa0758849a7ebd278f2e9a40c3132923ba9f1fa77c7807e6022d680931a4ed`. A preflight YAML route-insertion defect was caught before inference; its invalid preflight manifest is preserved as `preflight-superseded-manifest.json`. The corrected manifest froze before any model request.

| Path | Tasks | Structured usable | Resource observations |
|---|---:|---:|---|
| Direct AI, 512 requested | 3 | 3 | 1,001 input / 394 output / 1,395 total tokens; 13.061 s; USD cost null |
| Worker adaptive, 512 | 3 | 0 | One body: 288 input / 512 output; one dispatched empty output with usage unknown; one blocked before dispatch |
| Worker adaptive, 1,024 follow-up | 2 | 1 | One body: 288 input / 533 output; one dispatched empty output with usage unknown |

Across the initial diagnostic and Worker budget follow-up, eight Work/inference attempts were authorized and recorded (three direct AI requests plus five Worker Work requests). Seven provider requests were dispatched; the third 512-token Worker request was blocked before dispatch. Five response bodies were returned (three direct, two Worker; one of the Worker bodies was malformed); inference execution and usage remain unknown for two empty-output dispatched requests. No correction invocation ran. No paid route was used; local compute cost is unmeasured. Worker reports configured-free estimate $0 with billed cost null.

One separately frozen exact-API parser probe then called `InferenceService.generate_structured()` on a development task and returned READY: 399 input / 129 output / 528 total tokens, 7.777 seconds, `done_reason=stop`, local cost null. This brings the total to nine diagnostic model invocations under the ten-call ceiling. It provides direct evidence that the exact shared structured API can succeed for this development task; it does not establish why the historical 24 outputs failed.

Direct response bodies are in `data/stephanie_evolution02q/readiness-results.json`; full Worker results and raw/transformed output are retained there and in the run-local H3 ledger. The 1,024-token results are in `worker-1024-results.json`; the exact API probe is in `structured-api-probe.json`; all rubric checks are summarized in `independent-readiness-evaluation.json`. The 512 vs 1,024 Worker follow-up changed the requested output budget only; it reused development tasks, not protected acceptance answers. One of two follow-up tasks passed, one did not. This small, development-only check makes no reliability or effectiveness claim. The already executed follow-up manifest lacks parent-manifest and prompt-source hash bindings, a gap identified by independent review and corrected in the runner for future freezes; its original manifest and results are preserved as executed.
