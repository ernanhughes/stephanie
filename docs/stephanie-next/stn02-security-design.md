# STN02 — SIS Security Design

**Decision:** Protect the whole SIS HTTP surface with one local account and same-origin CSRF controls. Keep remote access off by default and chat disabled in configuration.  
**Implementation:** `sis/security.py`, `sis/routes/auth.py`, `sis/main.py`, `sis/__main__.py`, `sis/templates/login.html`, and the shared `sis/templates/base.html`.

## Identity and account scope

- SIS currently supports one configured local account, not multi-user identity management. Credentials are supplied by `SIS_AUTH_USERNAME` and a PBKDF2-SHA256 verifier in `SIS_AUTH_PASSWORD_HASH`. The plaintext password is entered only into the local sign-in POST.
- Generate a verifier with `venv\Scripts\python.exe -m sis.security --hash-password`; the command prompts without echoing the password.
- `SIS_SESSION_SECRET` must contain at least 32 bytes. Generate it with Python's `secrets.token_urlsafe(48)` and keep it in the process environment/approved secret manager.
- No login credentials, password, session token or API key is in source, query strings, application diagnostics or the readiness endpoint.
- Conversations created by SIS include the authenticated username in existing `ChatConversationORM.meta`. Chat list/detail/score/star/progress and execution paths enforce the owner where present. Owner-less imported conversations remain available to the one configured local account; this is not a multi-user sharing model.

## Sessions

- A successful login creates a new 384-bit random opaque session ID; only its SHA-256 key is held in the process-local registry. A previous session cookie is revoked during re-login, mitigating fixation.
- Session lifetime is eight hours. Logout revokes the server-side session. Process restart invalidates all sessions (fail closed, require sign-in again).
- A revoked/expired session ID cannot be replayed because every request resolves it against the server-side registry. A stolen active bearer cookie can still be replayed until expiry; HttpOnly, SameSite, HTTPS/Secure cookies and short lifetime reduce that risk, but this local session design is not a device-bound credential.
- Cookie is host-only, `Path=/`, `HttpOnly`, `SameSite=Strict`; `Secure` is set whenever the request is HTTPS or `SIS_COOKIE_SECURE=true`.
- Sessions are process-local; SIS is a single-worker local application. Shared/multi-worker sessions and account administration are not implemented.
- Login failures are rate-limited to five attempts per client address per five-minute window (in-process).

## Whole-application access boundary

`SISAuthMiddleware` wraps every HTTP route and websocket upgrade. Only `/login`, minimal `/healthz`, and `/static/*` are public. FastAPI docs/OpenAPI are disabled. All other pages, JSON APIs, database/log/document/config views, pipeline and code execution actions, and state mutations require a valid session. Unauthenticated HTML navigation redirects to `/login`; API requests receive 401. If auth secrets are missing/invalid, private routes return 503 and the login page displays setup state.

The active application route inventory was audited: all mounted routers are behind the same middleware rather than selectively decorating just the chat route. The old `mind` websocket router is not mounted; the middleware still rejects unauthenticated websocket upgrades if one is added.

## CSRF and origin/host checks

- Every POST/PUT/PATCH/DELETE requires an allowed exact Origin (or same-origin Referer) and synchronizer CSRF value. Login uses a signed, expiring pre-auth token bound to a host-only cookie; authenticated sessions use a separate per-session CSRF token.
- The sign-in form includes the pre-auth token. `base.html` attaches the session token to same-origin fetch mutations and hidden fields in same-origin forms.
- Origin must both be allowlisted and exactly match the request's scheme/Host. Host is checked against an exact allowlist. This mitigates cross-origin form/fetch attacks and DNS-rebinding hostnames; wildcard Host/origin entries are not used.
- Unknown Host, missing/bad CSRF, and cross-origin mutations fail closed. Responses are non-cacheable and carry `nosniff`, `SAMEORIGIN` frame, and same-origin referrer headers; HTTPS responses add HSTS.
- Default host/origins are `127.0.0.1`, `localhost`, and `[::1]` on the configured port. Remote hostnames must be explicitly allowlisted and remote HTTP is rejected; use HTTPS. With a TLS terminator, configure trusted proxy handling and set `SIS_COOKIE_SECURE=true`.

## Feature and readiness gates

`chat_runtime.enabled` remains `false`. Separate readiness values are exposed at `/healthz`; service registration is not model verification. The endpoint reports only booleans/model-configured status and never returns credentials or the model endpoint.

Even if `chat_runtime.enabled=true`, chat create/send returns unavailable until both `readiness.model_verified` and `readiness.live_persistence_verified` are true, auth config is valid, the database identity check passes, and model name/base are configured. The two readiness assertions remain false in checked-in Hydra config. They should be set only after repeating a real provider call and authenticated persistence/readback procedure for the deployment.

## Threat boundary and remaining limits

- This design targets a trusted, single-user local SIS. It does not implement multi-user accounts, roles, remote SSO, shared sessions, reverse-proxy configuration, security-key recovery or a general audit-event ledger.
- Local HTTP is accepted only on loopback. For HTTPS, Secure cookies and HSTS are applied. Do not expose SIS remotely over plain HTTP.
- Static assets are public and contain frontend resources only. Sensitive database, logs, prompts, conversations, models, plans, traces, artifacts and execution actions are behind the session gate.
- Jinja autoescaping is retained; model responses are plain text, not trusted HTML. No new content execution, tool invocation, file access, or autonomous workflow endpoint was added.
- Existing legacy state-changing forms remain functional through the global same-origin CSRF injector; deterministic tests cover both form and JSON mutation checks and rejected origins/tokens.
- A restrictive CSP is not yet enabled because legacy SIS pages still use inline scripts and third-party CDN assets. Do not render model text as HTML; nonce-based CSP and local/pinned frontend assets remain a separate hardening opportunity.

See `stn02-implementation-results.md` for tests and the live acceptance evidence.
