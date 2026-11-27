# Stephanie contextual prompt preparation

## Available workflow

Open `/stephanie`, select a repository registered in Capsule, choose `latest` or a hexadecimal commit, review the displayed data classification, optionally acknowledge external handoff for internal/personal data, choose a destination and size profile, add task instructions, and select **Generate prompt**. Stephanie calls Capsule's `repository.review` process with `prepare-prompt`. Capsule gathers evidence, compiles the final handoff Capsule, renders the configured Jinja prompt, and stops before substantive review inference. The generated prompt and JSON provenance package can be inspected, copied as Markdown, or downloaded. Generation never dispatches to a model.

The resulting package includes process/version, destination, size profile, handoff stage, template and prompt digests, Capsule identity/digest and lineage, resolved subject revision, evidence references and classifications, requirements where available, limitations, warnings, context estimate and exclusions. It also stores the complete gathered Capsule as a local JSON artifact (default `%LOCALAPPDATA%\\Capsule\\handoff-capsules`, override with `CAPSULE_HANDOFF_DIR`); this artifact may contain the unredacted local evidence. The portable prompt package is separately redacted for external handoff. Provider-observed input tokens and provider context window are `null` unless measured by a later provider execution.

Hydra currently advertises seven destination profiles: Codex, OpenCode, Claude Code, ChatGPT web, Claude web, Worker structured handoff, and local model (capacity unverified). Size profiles are compact (8,000), standard (32,000), detailed (64,000), and custom (up to 1,000,000) token projection ceilings. The profile is a ceiling, not a promise to fill the budget. Local-workspace destinations can refer to repository files but do not embed all source; browser destinations are self-contained and may omit evidence. Those omissions and any withheld contributor content are reported in the generated artifact. Worker selection creates a structured handoff only; it does not enqueue work.

## Context handoff v1

Host applications may provide a reference using a same-window `stephanie.open` CustomEvent, an allowlisted parent `postMessage`, or the `context` page query parameter (JSON encoded). The event only fills the visible context; it never generates a prompt or performs inference.

```json
{
  "schema_version": "stephanie.context.v1",
  "event_type": "stephanie.open",
  "event_id": "mission-open-uuid",
  "correlation_id": "mission-view-uuid",
  "application": "mission",
  "context_type": "repository",
  "subject": "writer",
  "revision": "resolved-40-character-commit-sha",
  "view": "commit-details",
  "requested_mode": "generate_prompt"
}
```

Unknown fields are rejected. Identifiers and optional selection text are bounded; this is a context reference, not a mechanism for submitting whole documents or repositories. The host is responsible for sending canonical subject/revision identifiers. A request idempotency key returns the same response for retries and rejects reuse with changed input. Reopening the panel does not start a process.

Configure `STEPHANIE_CONTEXT_ORIGINS` as a comma-separated exact origin allowlist for cross-origin embedded panels. Same-window custom events are supported without this setting. `STEPHANIE_CAPSULE_ROOT` and `STEPHANIE_AI_ROOT` may override the local source roots. Capsule runs as a child process with a filtered environment; `CAPSULE_DSN` and model/provider credentials are not forwarded. Prompt preparation does not request storage.

## Data handling

Capsule subject classification is shown in the selector. Internal and personal subjects require an explicit handoff confirmation in both the interface request and backend validation. Stephanie also checks classifications attached to generated evidence and blocks packages containing `protected_evaluation`. Capsule's existing provider projection redaction is reused across the full handoff package, including requirement text and user instructions; the prompt digest is calculated over the redacted prompt that is actually handed off, and the source prompt digest is retained separately. The user's copy/download remains a deliberate manual action. Nothing is opened or submitted in ChatGPT, Claude, Codex, or another provider, and no provider submission is asserted.

The external confirmation records the user's decision for a local prompt package. It does not certify that a particular external provider is contractually or technically eligible for every included evidence item. Review the package's evidence classifications and the target provider's policy before copying.

## Current integration boundary

- **Mission Control:** its current frontend is React/TypeScript/Vite, and its APIs can resolve a project repository and exact commit. The context contract accepts a canonical repository plus exact revision and never silently substitutes the latest revision. Mission is read-only under this commission. A Stephanie panel prototype is present in Mission's existing dirty worktree, but it is outside this commission's authorized edit scope and must not be treated as accepted or operationally activated. The existing Mission backend and Stephanie service were not restarted.
- **Workbench:** its current interface is server-rendered HTML with JavaScript modules. `capabilities_panel.js` reads a bounded context object containing book/project path, chapter, persisted revision/hash, paragraph selection identity/hash/revision, and staleness. It listens for `wb-chapter-context` while the capabilities panel is open. The existing ZeroMQ outbox publishes durable, sequenced events with IDs and replay/deduplication; no Stephanie emitter or `stephanie.open` event exists. The context object does not carry selected paragraph text or unsaved composer drafts. Workbench remains untouched.
- **Writer books/documents:** document, chapter, section and selection references are part of the bounded handoff contract. Prompt preparation for these contexts returns a clear `422` because no Writer document evidence contributor is connected. No manuscript content is fetched or modified.
- **Sidecar:** reuse remains an integration option; this implementation does not add a global event bus.
- **Chat and execution:** the existing chat page remains available. This first Stephanie page implements Generate prompt only; direct chat remains the existing separate chat interface. Process preview/run, resume, and provider dispatch are not presented as supported features here.

## Qualification and limits

On 26 November 2026, deterministic Capsule preparation used the retained Writer revision `85a503203c76f98669733465cbed9f51fcb0d563` and made no model call. Codex/standard produced a 3,172-token rendered-prompt estimate (12,689 UTF-8 bytes), within its 32,000-token ceiling; the context was marked incomplete and full contents for `git_diff`, `inspect`, `requirements`, and `aitest` were withheld from that local-reference profile. ChatGPT web/compact produced a 5,002-token estimate (20,035 bytes), within its 8,000-token ceiling; `git_diff` and `inspect` were omitted. ChatGPT web/custom at the 1,000,000-token ceiling produced a 421,179-token rendered-prompt estimate (1,685,102 bytes), with no omitted contributors and `incomplete_context=false`. This confirms the larger projection path was not silently reduced. The complete gathered Capsule file existed locally for all three runs and had a digest. None was dispatched. These are estimates, not provider-observed token counts or capacity guarantees.

Focused verification: Stephanie context/security tests **15 passed**; Capsule process tests **11 passed, 1 deselected**, plus the directly targeted prompt-boundary test **1 passed, 11 deselected** (`long`, run because this change directly targets that boundary); Mission's existing side-panel test **3 passed** and frontend TypeScript/Vite build succeeded; Worker request, executor, strategy, and ledger tests **42 passed**, including an explicit 180-second strategy run retained by a fresh ledger reader. AI capsule contract tests passed **12** (AITEST reused prior verified evidence for an identical snapshot and plan).

No genuine inference, external submission, database write, migration, Worker activation/restart, or operational Mission/Workbench modification occurred. The independent review identified that an unauthenticated iframe would be redirected to a login page blocked by `X-Frame-Options: SAMEORIGIN`. Stephanie now sends the configured loopback-only CSP `frame-ancestors` policy on `/login` as well as the prompt page, while retaining `SAMEORIGIN` as a legacy fallback; the W3C CSP specification says an enforcing `frame-ancestors` policy overrides X-Frame-Options ([CSP Level 3](https://www.w3.org/TR/CSP/#frame-ancestors)). The focused security test confirms the headers and ordinary private pages' same-origin restriction. This is source/test qualification only; the active service was not restarted. The independent review also flagged Mission worktree changes as outside this commission's edit authorization; they remain untouched here.

There is no live integrated URL yet. Mission's Vite server is not running, no Stephanie SIS server is listening, and port `8000` is currently occupied by the separate ALIVE service on this host. A future isolated run must choose a free SIS port (for example `8011`) and configure the Mission host origin consistently. No server was started because the current scope specifically holds host integration/activation pending its separate authorization.

## Mission Control live acceptance addendum — 26 November 2026

The authorized Mission Control integration was exercised in Chrome at `http://127.0.0.1:5173/projects/writer`. The global Stephanie launcher opened the existing prompt page in an authenticated iframe. Mission attached Writer and the exact project HEAD `65a2683985904651824b1772eda11fd66974ac18`. Prompt preparation produced a compact Codex/local-workspace package estimated at 3,426 rendered prompt tokens (1,819 projection tokens; 14,309 Markdown characters) and a standard ChatGPT-web package estimated at 5,159 rendered prompt tokens (3,647 projection tokens; 21,230 Markdown characters). The estimates are below their 8,000 and 32,000 ceilings. Both packages reported incomplete context and enumerated withheld/stale evidence, including Inspect artifacts stale against the pinned revision and missing CoCoder review evidence. Both Markdown packages were copied to the local clipboard for acceptance; neither was pasted or submitted to a provider, and neither dispatched inference.

Historical commit `32c2bc83d297e17896c4dcbfc2383386ee1a45e5` was opened from Activity and attached to Stephanie. The revision field was read-only and retained that full SHA. Closing the panel and switching to Mission attached Mission at `35b19c7c7ffc0b549f0816ca30c969b3ce3b96ff`; the previous Writer prompt was cleared and the external-copy confirmation reset. Reopening Writer attached its current full SHA again. The `stephanie.context.v1` handoff remained bounded and exact.

The isolated Stephanie prompt-only host is healthy at `http://127.0.0.1:8766` (`/healthz`: `ready`, `prompt-only`, authentication enabled, database not required, chat disabled). It reuses Stephanie's existing authenticated UI and prompt route, but deliberately does not initialize the full SIS memory/chat stack. Full SIS startup was attempted and stopped because the installed environment lacks optional `hnswlib`; no packages were installed. Authentication is available at the local host with username `codex-local` and the development-only password provided in the delivery notes. Port `8000` remains untouched.

Mission's Vite frontend is running at `http://127.0.0.1:5173`. Its configured API proxy points to the already-running Mission backend at `127.0.0.1:8790`. That backend was not restarted; the isolated UI acceptance used only read endpoints, while Capsule's generated full evidence artifacts were directed to a dedicated temporary directory. This leaves the backend process shared with the existing Mission service, so the frontend/prompt host are isolated but the Mission API process/database are not. No Mission record was written. Workbench remains read-only and was not integrated.

The final independent review found no remaining actionable issues after the iframe-load repost, pinned-revision, consent-reset, and stale-response safeguards. Focused verification: Mission Stephanie panel tests **4 passed** (0.92 s); frontend TypeScript/Vite production build succeeded (existing large-chunk warning); Stephanie prompt-host, context-handoff, and security tests **17 passed** (2.65 s). No long suite was run during this acceptance. `git diff --check` passed in both repositories. Unrelated dirty worktree changes were preserved; nothing was committed or pushed.

### Local restart commands

Start the Mission development frontend (requires the existing backend at `127.0.0.1:8790`):

```powershell
Set-Location C:\Projects\mission\frontend
$env:VITE_STEPHANIE_URL = 'http://127.0.0.1:8766'
npm run dev -- --host 127.0.0.1
```

Start the authenticated, prompt-only Stephanie host in a separate PowerShell window. The password prompt below creates a fresh local password verifier each time; use the same password if you want to keep the current login, or choose a replacement and sign in with it. This host does not start chat, MemoryTool, Worker, or database services.

```powershell
Set-Location C:\Projects\stephanie
$env:PYTHONPATH = (Get-Location).Path
$env:SIS_HOST = '127.0.0.1'
$env:SIS_PORT = '8766'
$env:SIS_ALLOWED_HOSTS = '127.0.0.1,localhost'
$env:SIS_ALLOWED_ORIGINS = 'http://127.0.0.1:8766,http://localhost:8766'
$env:STEPHANIE_CONTEXT_ORIGINS = 'http://127.0.0.1:5173'
$env:SIS_AUTH_USERNAME = 'codex-local'
$env:SIS_COOKIE_SECURE = 'false'
$env:SIS_SESSION_SECRET = [Convert]::ToBase64String([Security.Cryptography.RandomNumberGenerator]::GetBytes(48))
$env:SIS_AUTH_PASSWORD_HASH = python -c "import getpass; from sis.security import hash_password; print(hash_password(getpass.getpass('Local Stephanie password: ')))"
$env:CAPSULE_HANDOFF_DIR = Join-Path $env:TEMP 'stephanie-sc03-20261126\Capsule-handoffs'
$env:STEPHANIE_CAPSULE_ROOT = 'C:\Projects\capsule'
$env:STEPHANIE_AI_ROOT = 'C:\Projects\ai'
python -m sis.prompt_host
```

The existing Mission backend on port `8790` is not part of these restart commands and should not be restarted as part of this local UI workflow.
