# Stephanie Assistant 02 — Final Report

**Status:** implemented and operationally verified in the isolated SIS chat service. No commit or push was made.

## User experience

- Added a persistent model dropdown and project selector to each live chat, with settings available from the same page. Changing a route updates conversation metadata; it does not replace the conversation or delete its transcript.
- Default preference is `opencode-go/deepseek-v4.1-flash`; automatic free-first remains selectable. Preferences live in an owner-local JSON file and take effect without editing YAML or restarting the service.
- Settings permit subscription-included inference (on by default), force metered inference off, and separately govern ordinary project metadata, repository locations, and publication status. Each external turn still needs its own approval. No project data categories are enabled by default.
- The dropdown is based on configured routes and the local Ollama catalogue. `unverified`, `worked`, and `blocked` are observations, not catalogue claims. Ollama cloud-tagged and embedding-only entries remain visible but unavailable for chat. The automatic option tells the owner when no free route is configured.
- OpenCode requests reuse Stephanie's existing `LLMService` and `FreeFirstChatRouter`/LiteLLM transport, with exact provider endpoint validation, the existing OpenCode Go session header, a 45-second ceiling, and a 512-token output cap. Explicit routes do not silently fall back. Metered and unknown-cost external routes are rejected. A cloud-tagged Ollama model cannot enter through either explicit selection or automatic local fallback.

## WorkBench pattern review

Read-only review of `C:\Projects\workbench\src\workbench\preferences.py` and the catalogue flow in `src\workbench\service.py` informed three choices:

1. Persist the owner's choice separately from configuration and conversation messages.
2. Treat catalogue presence as unverified; only a completed call marks a route worked, and an observed refusal marks it blocked.
3. Make preference-write failures distinct from best-effort observation-write failures.

Stephanie keeps its own preference storage because WorkBench's Pi session engine is not Stephanie's chat engine. The OpenCode transport is not duplicated; the existing shared Stephanie/AI inference path is used.

## Context compilation and Jev evidence

The compiler makes bounded read-only Mission GETs and whitelists fields. It does not open source code, manuscripts, private project notes, credentials, or protected evaluator content. The UI shows source paths, observation times, missing fields, category eligibility, and a bounded context-size estimate. The model prompt omits the inspector's duplicate projection; retrieved fields are sent as a separate untrusted reference message, with system instructions to treat their values as data.

Live Jev evidence retrieved on 2026-11-26:

- Mission project `jev` is active. Its description says it is a build-and-investigate book about typed, uncertain decisions. Repository and manuscript locations are exposed from Mission and the Writer pipeline registry.
- Writer's read-only pipeline projection reports 37 chapter files, 36 applicable chapters, and `_index.md` skipped as a non-chapter. It reports validation enabled and human approval required; no validation, build, audit, or publication was run to produce this view.
- Publication reports lifecycle `development`, matching manuscript fingerprint, and gate `mandatory evidence missing: evidence_completeness, prose_review, technical_verification`.
- Latest readiness assessment is `NOT_YET_ASSESSED` / `CURRENT`. Evidence says completeness has no assessment, prose review is current for 1 of 36 discovered chapters, and no technical claims have been inventoried. Build and referenced-asset gates passed; chapter structure has an advisory `parts_not_declared` warning. Author approval is not recorded.
- Mission has no recorded current milestone, next action, outstanding project tasks, linked Work Outcomes, or related projects in the project detail response. These absences are shown as missing information. Suggested next steps must be labelled as recommendations inferred from the publication evidence, not as a recorded Mission action.
- Sources: `/api/projects/jev`, `/api/pipeline/books/jev`, `/api/publication/publishing`, `/api/publication/assessments`. The live context response carried source observation timestamps; the assessment's stored record was created at `2026-11-25T22:55:37Z`.

No Jev project context was sent to an external provider during qualification because all three external project categories remain disabled by default. The owner can enable only the categories they intend to share and must also approve the project-context checkbox on the turn.

## OpenCode Go qualification

One intentional synthetic model invocation completed through the isolated Stephanie chat service:

- Requested and effective route: `opencode-go/deepseek-v4.1-flash`.
- Provider-reported tier/billing source: subscription-included / Go.
- Response: `ROUTE QUALIFIED`.
- Provider-reported usage: 122 prompt tokens, 28 completion tokens, 150 total. Reported cost was `$0`; provider metadata also included list/allowance accounting estimates, which are not a separate charge.
- No project or personal data was sent. Metered fallback was not used. One of the six permitted qualification calls was used.

The resulting synthetic qualification is visible as conversation #5. Conversation #3 remains available; no existing conversation was deleted or rewritten. Mid-conversation model switching and history retention are covered by the focused continuity test.

An earlier synthetic preflight used the word “project,” which correctly triggered project-context policy and was rejected before inference; that failed preflight is visible as conversation #4. It did not consume a model call or send content externally.

## Verification

- Python 3.13 compile check: passed for modified modules and tests.
- Short focused suites: `test_conversation_policy.py`, `test_chat_context.py`, `test_mission_project_context.py`, `test_model_preferences.py` — 16 passed.
- Targeted routing suite: `test_free_first_chat.py -m long` — 21 passed.
- Targeted chat acceptance cases: `test_sis_chat_execution.py -m long -k "external_history_rechecks_categories_and_selected_project or one_conversation_continues_after_model_change_without_losing_objective or sis_chat_persists_user_and_assistant_messages_and_turn or failed_inference_can_retry_same_request_without_duplicate_user_message or chat_templates_compile_with_interactive_forms"` — 4 passed, 28 deselected.
- `git diff --check`: passed.
- Independent gpt-6-luna review: passed after corrections for project-history clearing, exact provider endpoint revalidation, untrusted retrieved-content boundaries, and cloud/embedding fallback rejection.
- Full backend suite and unrelated long suites were intentionally skipped; no broad Mission, Writer, or frontend suite was run.

## Use it

Open [Stephanie chats](http://127.0.0.1:8766/chats). Open an existing conversation or create one; use the visible Model dropdown to choose OpenCode Go or Automatic — free first. Select Jev in Project context and open “Inspect loaded project context” to review the current evidence.

To send Jev evidence to an external model, visit Settings, enable only the categories you approve, return to the chat, select OpenCode Go, check both external-turn approval boxes, then send the message. Selecting Jev or OpenCode Go alone does not share project information.

No SC5/SE3 work, SC4C activation, commit, or push was performed.

## Remaining limits

- The only configured external route discovered is OpenCode Go; OpenCode Free is not configured. Automatic therefore has no free route to try today: it uses the subscription-included Go route only when that permission is on, otherwise it falls back to the local model.
- The first Jev conversation using external inference has not been run. The implementation deliberately leaves all project-category permissions off; the owner must enable the intended categories and approve the two per-turn checkboxes. The real provider qualification used only a synthetic prompt.
- Mission returned no linked Work Outcomes or related-project records for Jev. The compiler also does not read CoCoder, AITEST, or programme-ledger documents in this stage; it grounds current status in Mission, Writer's registry projection, and Publication's stored assessment. The inspector makes the missing source coverage visible rather than implying full ecosystem coverage.
- The full backend and unrelated long test corpora were not run. No SC5/SE3 work, SC4C activation, commit, or push was performed.
