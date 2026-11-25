# OpenCode application qualification

## Qualified route

- Requested/effective model: `opencode-go/deepseek-v4.1-flash`.
- App transport: LiteLLM OpenAI-compatible chat completions.
- Endpoint: `https://opencode.ai/inference/go/openai/v1`.
- Authentication: existing `OPENCODE_API_KEY`, never displayed or persisted.
- Go session continuity: stable `x-opencode-session` header derived from the same SIS conversation ID.
- Outcome: one standalone synthetic response and two synthetic SIS turns succeeded; observed provider/model/usage are stored in ChatStore. Requests reported zero metered cost and `billing_source=go`.

The user explicitly suggested this Go route for testing. It consumes the existing fixed OpenCode Go plan's allowance; it is not pay-as-you-go. No subscription was purchased and no paid fallback is enabled.

## Free routes

The installed OpenCode v2.3.2 CLI was called once with synthetic text and returned `429 Free usage limit reached` with a retry indication of about 6 hours. We stopped that route. Current OpenCode docs also state Free chat models are only available within OpenCode and third-party clients receive 403. SA1 does not claim direct LiteLLM compatibility for OpenCode Free.

`opencode/gpt-6-luna` was not used because Console lists it as metered pay-as-you-go. The Meta contributor route was not used. No unsupported provider or endpoint was guessed.

## Evidence and limits

The model catalogue GET returned HTTP 200 and listed `deepseek-v4.1-flash`. The actual model response came through Stephanie's `LLMService.achat`, not an OpenCode CLI probe. See `MODEL-ROUTING-EVIDENCE.md` for exact observed token counts.
