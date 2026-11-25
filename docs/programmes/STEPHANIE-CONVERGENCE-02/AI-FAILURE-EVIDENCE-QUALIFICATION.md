# AI failure evidence qualification

Shared AI remains the only inference implementation. `StructuredOutputError` carries `.result` and `failure_kind`; `InferenceUnavailable` carries `.result` for provider-returned error results. Pre-dispatch exceptions have no result. `generate_structured()` remains the public parser and validator.

Stephanie stores available result, raw response, provider/model/config, usage, termination metadata in `raw_metadata`, execution references, trace, request ID, exception category and UNKNOWN status where result is absent. Raw body is locally retained in `ai-executions.json`; ordinary `ai-evidence` output is metadata-only and raw display requires `--include-raw`. Shared trace payload remains body-free.

Deterministic focused tests passed for valid output, invalid JSON, schema failure, empty output, missing usage, provider unavailable, provider timeout result absence, and trace privacy. Citation-membership failures are handled in the same evidence-preservation boundary as parser failures. No paid or local model call was made.
