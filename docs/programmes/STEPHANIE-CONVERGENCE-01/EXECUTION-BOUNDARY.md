# Stephanie execution boundary

## Routes

| Route | Use | Explicit identity/evidence | Current limit |
|---|---|---|---|
| Direct shared AI | Simple structured inference and hypothesis construction. | AI `provider`, `model`, `route`, `model_revision`, request/trace IDs, fingerprints, usage, cost status, latency, `raw_metadata`; Stephanie's source evidence IDs/digests and hypothesis-only authority. | AI `generate_structured()` validates a Pydantic schema and returns a result plus model. On parse failure it raises `StructuredOutputError`; AI's optional trace contains status/error/fingerprints/usage/metadata, not raw content. Stephanie currently supplies no trace sink/session and its caller stores the result only after success. Thus parse-failure body and application failure record are lost. No general reliability claim. |
| Worker-mediated | Governed baseline/candidate execution, strategies, bounded recovery, tools and detailed execution evidence. | Work request/result IDs, actual backend/model/provider when reported, strategy/version/digest, policy/config, invocation IDs, raw/transformed artifact refs, H3 ledger, resource and uncertainty fields; Mission envelope/objective/episode refs when correlated. | `WorkerHarnessClient` is loopback-bounded and explicit, but current Stephanie path is diagnostic rather than a unified lifecycle adapter. Empty HTTP/transport outcomes can leave inference UNKNOWN; artifact/ledger references must be retained. Current model route reliability is not qualified. |

Route choice is explicit in the caller/manifest. No automatic switching, retry, or fallback between AI and Worker is allowed. A route failure remains that route's outcome. Do not build a generic router or a second strategy runtime in Stephanie.

## AI schema/provider behavior

The public `ai.inference` interface accepts `ReasoningRequest` (`prompt`, optional `system`, optional JSON `response_schema`, timeout, temperature, output budget and metadata) and Pydantic `BaseModel` response types for structured validation. `InferenceService.generate_structured` returns `(ReasoningResult, T)` and enforces budget/status. Invalid JSON, non-object roots and Pydantic failures raise `StructuredOutputError`; provider errors raise `InferenceUnavailable`.

Ollama receives `response_schema` as native `/api/generate` `format` and `max_output_tokens` as `options.num_predict`; Ollama reports eval counts and `done_reason` where returned. OpenCode provider composes the schema into the prompt, so this is not native schema enforcement. Provider identity/config is therefore route-specific. The direct local Ollama `generate_structured()` probe passed once; live Ollama was unavailable during this assessment. Provider acceptance, output schema validity and semantic correctness are different claims.

## Failure evidence requirements

For each attempt, capture what is returned by the selected interface: raw body if available, provider status/error, parser and Pydantic failure, termination reason, usage/cost, latency, provider/model/route/revision, request/trace/Worker IDs, effective strategy and configuration identity, input source identity/digest, and artifact/ledger refs. Unknown values remain `null`/`UNKNOWN`, never zero or success.

Known current gap: `ImprovementEngine.investigate_with_ai` stores `ReasoningResult` (including content/metadata) on success, but does not catch and persist `generate_structured` failures. The AI default has neither durable trace sink nor DB session. SE2Q's runner worked around this by calling `InferenceService.generate()` and manually validating, but that diagnostic runner is not the production improvement adapter. This needs a narrowly scoped Stephanie implementation before controlled semantic qualification. Do not change AI or Worker inside this convergence assessment.

Worker result payloads and ledgers preserve richer execution data on observed Work results. An empty client response does not prove provider execution or non-execution. Keep inference state, usage, termination and provider error unknown unless Worker/provider evidence supplies them.
