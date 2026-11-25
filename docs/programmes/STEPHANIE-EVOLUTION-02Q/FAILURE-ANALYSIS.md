# SE2Q-A — SE2 failure analysis

## What was actually retained

`data/stephanie_evolution02/generation-results.json` contains 24 arm/task rows with parser exception text, `prompt_chars`, memory IDs, and latency. It contains **no raw response body**, provider response fingerprint, token usage, finish/termination reason, model revision, or cost. The earlier summary's statement that the model replies were truncated at 240 tokens was too strong: truncation was only a plausible explanation from those records. The old artifact is unchanged.

Every row is a shared-AI `StructuredOutputError` reporting invalid JSON. The error-message breakdown is: 20 “Unterminated string”, one “Expecting property name enclosed in double quotes”, two “Expecting value”, and one “Expecting ',' delimiter”. This confirms the parser observed syntactically invalid JSON. It does not identify why those bytes were produced. Markdown fencing, exact response lengths, effective token cap, provider termination, and parser input bytes are unknown.

The old requests set `max_output_tokens=240`; `OllamaReasoningProvider` maps that to `options.num_predict=240` and adds native JSON Schema `format`. The caller used `generate_structured()` without a trace sink. On parse failure, AI marks the `ReasoningResult` malformed and raises `StructuredOutputError`; the Stephanie caller saved only the exception string. The provider had returned a nonempty body for the parser to inspect, but its bytes and `raw_metadata.done_reason` were lost at the application boundary.

## Evidence versus cause

| Cause | Evidence status |
|---|---|
| Malformed JSON reached the AI parser | Confirmed for all 24 by `JSONDecodeError` messages |
| Empty response / transport failure | Not indicated in the 24 caller errors; provider evidence was not retained |
| Truncation | Unknown for the 24; syntactic errors are consistent with truncation but not proof |
| Markdown fences | No evidence either way |
| Schema validation mismatch | No; failures occurred during JSON decoding before schema validation |
| Parser defect | No evidence; later direct and Worker parsers accepted valid responses |
| 240-token effective limit | Requested and sent by the AI adapter; provider enforcement/termination is unknown for the 24 |
| Schema/prompt complexity | Not attributable from the missing response bodies |

The prompt character totals retained by arm were A 9,433, B 13,762, C 34,361 (these count the runner's JSON prompt, not all provider-side framing/schema bytes). Token counts are unavailable. No historical response size or output-token distribution can be reconstructed.
