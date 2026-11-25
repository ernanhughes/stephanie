# Final report — STEPHANIE-ASSISTANT-01

## Verdict

**CONDITIONALLY ACCEPTED — SPECIFIC LIMITATIONS REMAIN**

## Release

- Interface: existing SIS at `http://127.0.0.1:8766/chats`.
- Start: `powershell -ExecutionPolicy Bypass -File C:\Projects\stephanie\scripts\start_stephanie_assistant_01.ps1`.
- Isolated transcript DB: `stephanie_assistant_01`; operational Stephanie database was not used for SA1 turns.
- External model: `opencode-go/deepseek-v4.1-flash`, only with per-turn approval and fully eligible history. It consumes existing Go subscription allowance; the three successful test calls reported cost 0 and `billing_source=go`.
- Local path: `ollama/mistral-small:latest` for unknown/private/Mission context. Its conversational quality is not qualified.

## Results

Real synthetic Go requests succeeded, the same conversation persisted and resumed after a fresh process, and the follow-up recalled its objective. A deterministic fixture test covers model switch continuity. The project picker and detail projection use read-only Mission GET requests, identify observation source/time, and remain local-only. Credential/protected-evaluation content is blocked before inference; old unclassified turns remain local. Failure state is retained without saving raw provider exception text.

OpenCode Free direct third-party access is not claimed. The CLI attempt returned 429, and current OpenCode documentation says Free chat models are available only inside OpenCode. `opencode/gpt-6-luna` is metered and unused. No paid fallback, new AI repository edits, Mission writes, or SC4C action occurred.

## Quality and limits

The two-turn sample demonstrates basic acknowledgement and recall only. The local path was attempted but yielded no usable response evidence. Global cross-conversation retrieval, attachment ingestion, stored preferences, streaming, and specialist delegation remain future work. Project context cannot route to Go in this release.

## Verification and review

See `TEST-EVIDENCE.md`, `MODEL-ROUTING-EVIDENCE.md`, and `INDEPENDENT-REVIEW.md`. The independent read-only review is complete. Its Mission projection-size finding was fixed and covered by a focused test. The conditional verdict remains because local conversational quality is not qualified and the successful external sample is small.
