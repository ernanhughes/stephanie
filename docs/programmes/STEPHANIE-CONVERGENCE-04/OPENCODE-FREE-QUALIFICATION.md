# OpenCode Free qualification

## Observed provider invocations

The only two genuine free-model invocations authorized by this commission were already consumed by synthetic probes through the installed OpenCode CLI. No private source or conversation was sent, and no further provider calls were made:

| Route | Result | Provider-reported usage | Cost |
|---|---|---:|---:|
| `opencode/deepseek-v4.1-flash-free` | replied exactly `OPEN_CODE_FREE_ROUTE_OK` | 4,627 input / 9 output tokens | $0 |
| `opencode/hy4-flash-free` | replied exactly `OPENCODE_HY4_FREE_OK` | 2,885 input / 10 output tokens | $0 |

The installed CLI was observed as OpenCode v2.3.2; `opencode models` exposed provider-qualified IDs. A model list establishes catalogue visibility, not current quota, future availability, or API-route compatibility. The probe used the CLI from Stephanie checkout context, so it did not establish an empty/project-free working directory.

## Supported direct chat path

OpenCode's current Console model reference documents application SDK/API calls to `https://opencode.ai/inference/openai/v1/chat/completions`, model IDs `deepseek-v4.1-flash-free` and `hy4-flash-free`, and Bearer authentication with the OpenCode Console API key. The Stephanie router uses LiteLLM's existing OpenAI-compatible transport to that documented base URL. It does not call the CLI for each chat turn, so it does not create a parallel archived OpenCode conversation. Direct API compatibility is **not live-qualified** here because the two-call probe limit was already consumed; no result should imply it was tested.

Current OpenCode model documentation labels both selected routes free and states zero retention/no model training for these two providers. This policy is time-sensitive and must be rechecked before enabling live SIS chat. Free model pricing is temporary; it does not imply service-level availability or quota. [OpenCode Console model reference](https://opencode.ai/docs/console/models/)

Known distinctions: `opencode-go/deepseek-v4.1-flash`, GPT 6 Luna and Claude Haiku routes are not free-tier OpenCode Console routes; some are subscription-covered and some have retention. Do not silently treat them as free or add them to this policy. Never rotate credentials to evade limits. Error classes and retry-after are recorded by the local router, but provider quota/reset status has not been queried or inferred.
