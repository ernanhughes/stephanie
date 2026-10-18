# Worker Free OpenCode Models

Stephanie can route model calls through the local **Worker**
(`C:\Projects\worker`, loopback `http://127.0.0.1:7317`) to the free OpenCode
models it advertises, and **falls back to its normal provider (Ollama)** when no
free route is usable.

Implemented in `stephanie/services/worker_model_bridge.py`.

## Policy

Selection order (each tier is a hard stop before the next):

1. **Free route** — only if it *probes* as admissible. Discovery alone is never
   enough: free OpenCode routes are advertised but have historically been denied
   through Worker's API path, so a probe (`POST /v1/models/probe`) is mandatory.
2. **Cheapest paid `opencode-go` route under a hard price cap** — default cap
   **$0.40 / million tokens**, provider fixed to `opencode-go`. Chosen by price
   from Worker's own price data (`GET /v1/models/cli?max_price=0.40&sort=price`),
   not by a nominated model name.
3. **Ollama** — the normal provider, whenever neither of the above qualifies.

Notes:

- **Paid routes cannot be probe-verified.** Worker's probe endpoint rejects
  non-zero-priced models (`MODEL_NOT_ELIGIBLE`), so tier 2 is price-selected and
  falls back at call time if it fails.
- **Data policy.** Free routes report `data_policy: training_allowed`; prompts
  sent to them may be used for training. Paid `opencode-go` routes are
  `private`. The operator has accepted the free-tier policy; keep private data on
  Ollama or a paid/private route.
- Current prices (illustrative): `muse-spark-1.2/1.3-contributor` $0.20,
  `mimo-v2.5`/`mimo-v2.6-flash` $0.28, `deepseek-v4.1-flash` $0.60 (excluded by
  the $0.40 cap), `gpt-6-luna` $0.75 (excluded).

## How it works

| Step | Endpoint | Purpose |
|---|---|---|
| Discover (free) | `GET /v1/models/cli?free=true` | list free routes (id, provider, price 0) |
| Discover (paid) | `GET /v1/models/cli?max_price=0.40&sort=price` | cheapest `opencode-go` routes within the cap |
| Probe | `POST /v1/models/probe` | confirm a **free** route is callable (`request_admissible` ≠ `BLOCKED`) |
| Invoke | `POST /v1/work` | Worker has **no completion endpoint**; models run inside a Work task via its `CALL_MODEL` capability |

Selection is negative-cached (`cache_ttl_seconds`, default 300s) so a failed
probe is not repeated on every call. Discovery uses a longer timeout
(`discovery_timeout`, default 90s) because it shells out to the OpenCode CLI.

## Current live status

As of this writing the local Worker advertises **12 free routes**, and **every
one probed is not admissible**:

```
opencode-go/beam-free               status=FAILED http=503 admissible=BLOCKED
opencode-go/longcat-2.5-preview-free status=FAILED http=503 admissible=BLOCKED
opencode/space-bunny-free            status=FAILED http=503 admissible=BLOCKED
```

This matches Worker's own documentation: the OpenCode "Free" entitlement returns
`permission_denied`/unavailable through that API path ("free tier can only be
used from within OpenCode"). So today the bridge correctly falls back to Ollama;
it will start using a free route automatically if one ever probes as admissible.

## Check it

```powershell
venv\Scripts\python.exe scripts\worker_free_model_check.py --probe 3
```

Prints Worker health, the advertised free routes, probe results, and whether the
bridge would use a free route or fall back. Safe to run as a preflight.

## API

```python
from stephanie.services.worker_model_bridge import (
    WorkerModelBridge, WorkerModelProvider, prefer_free_route, FreeRouteUnavailable,
)

bridge = WorkerModelBridge()                 # default loopback base_url
route = prefer_free_route(bridge=bridge)     # FreeRoute or None (never raises)
if route is None:
    ...                                      # use Ollama
```

`WorkerModelProvider` is a `stephanie.models.provider.ModelProvider` that raises
`ProviderInvocationError` when no route is admissible, so a model runtime can
layer it ahead of the existing provider with the fallback built in.

## Not yet wired to the default call path

`LLMService` still resolves models exactly as before. Wiring the bridge in front
of `LLMService.complete/chat` (behind a `model_defaults.prefer_free` flag, so
that an inadmissible route transparently falls through to Ollama) is the
remaining step. It is deliberately separate because free routes are currently
unavailable, so enabling it changes nothing today.

## Tests

```powershell
venv\Scripts\python.exe -m pytest tests\test_worker_model_bridge.py -q
```

Covers route parsing, probe-result interpretation, admissible-route selection,
negative caching, silent fallback when Worker is unreachable, Work-result text
extraction, and the provider's `ProviderInvocationError` fallback contract.
