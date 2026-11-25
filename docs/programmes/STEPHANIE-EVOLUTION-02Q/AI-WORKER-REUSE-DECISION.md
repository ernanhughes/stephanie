# SE2Q-B/C — AI and Worker reuse decision

## Existing capabilities

Shared AI already provides `InferenceService`, JSON Schema request formatting for Ollama, Pydantic structured validation, a trace-sink interface, provider-reported token usage, and Ollama `done_reason`/duration metadata in `ReasoningResult.raw_metadata`. The missing SE2 evidence was a Stephanie caller persistence choice: it supplied no trace sink and discarded the result when `generate_structured()` raised. No AI repository change was justified.

Worker's existing `ModelExecutor` consumes `/v1/work` with explicit model, policy, `data_policy`, output schema, timeout, and output budget. `structured.v1` performs one structural validation; `corrective.v1` can make one schema-directed correction; `adaptive.v1` first applies only `json-fence.v1` to a single complete valid fenced JSON document, otherwise validates and can make at most one bounded structural correction. Its H3 strategy record retains raw invocation output, transformed output/digests, validation, decision, and provider token usage when available. The current OpenCode chat adapter forwards `max_tokens` but does not pass a native JSON Schema to Ollama; Worker policy is post-generation structural enforcement.

## Live compatibility and choice

`C:\Projects\stephanie\stephanie\improvement\worker_harness.py` is a narrow loopback Work client. It delegates all execution policy/recovery to Worker and does not vendor the strategy runtime. The isolated local Worker test established:

- Direct shared-AI JSON Schema route: 3/3 outputs structurally parsed, all citations were within the visible task evidence, provider reported 129–134 output tokens per answer, and Ollama reported `done_reason=stop`.
- Worker `adaptive.v1` at 512: one captured response used exactly 512 reported output tokens and ended mid-JSON; it could not be normalized and there was no output budget left for correction. A second dispatched request returned empty output (inference status unknown); a third was blocked before dispatch after Worker marked the provider unavailable.
- A single preregistered output-budget change to 1,024 (all other task/model/strategy inputs fixed) produced one valid Worker answer using 533 output tokens. The other task returned empty output again. No more calls were made.

**Decision: federated, explicitly qualified execution (Option C).** Keep direct AI as the qualified route for these bounded JSON hypotheses because it passed all three development format/citation checks and applies Ollama's native JSON Schema format. Keep Worker available as an opt-in adapter for workloads where its extra strategy/evidence features are needed, but mark this Worker/model route **not yet qualified for reliable hypothesis generation**. Do not claim Worker did better or worse scientifically from this readiness sample. Worker reuse was demonstrated; consistent readiness was not.

No Worker or AI source changed. No general inference, parsing, or recovery library was added to Stephanie.
