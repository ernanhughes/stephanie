# SWF01 — First Stephanie → Writer Editorial Handoff

**Status: IMPLEMENTED and LIVE-VERIFIED** on a real Stephanie-generated draft.
The pipeline stage is wired and smoke-checked; it has not yet been exercised
inside a full pipeline run (that requires a paid run — see "Not yet verified").

## 1. What was inspected

Writer checkout: `C:\Projects\Writer` (also reachable as `C:\Projects\writer`).
CLI: `C:\Projects\writer\venv\Scripts\writer.exe`; `writer` on PATH is
`D:\tools\writer.bat`.

* `writer --help` — ~50 operational command groups (`validation`, `qc`, `improve`,
  `format`, `corpus`, `repair`, `apply`, `editorial`, `info`, …).
* `writer info search "editorial improvement of a markdown article draft"` — top
  hit `editorial` (0.192).
* `writer info describe editorial` / `writer info operations editorial` —
  **rejected**: V1.2 is *diagnose-only, read-only with respect to manuscript
  source*, and **0/6 operations are selectable** (`TRANSITIONAL`, "catalog status
  partial: capability incomplete"). Not a handoff target.

Existing Stephanie integration (`stephanie/integrations/writer_runtime.py`,
`docs/writer-integration.md`): `WriterRuntimeDiscovery` shells out for
`info search|describe|guide|operations|agent-context` only; the SIS routes are
explicitly read-only. **No operational (mutating or analysing) call existed** —
discovery was implemented, a real handoff was not.

## 2. The chosen operation

```
writer validation validate-file <file_path> --profile <profile> --json
```

`writer validation --help`: *"project-level status and one-off text/file checks"*.
It is a **supported public CLI command**, is **read-only with respect to the
draft**, and returns a machine-readable `ValidationReportDTO`:

```
id, target_type, target_id, source_name, source_hash, profile,
validation_version, status, decision, gate, coverage, total_findings,
critical_count, high_count, medium_count, low_count, info_count, findings,
metrics, readiness, grade, freeze_blocked, report_markdown,
document_artifact_id, document_id, project_id, book_id, meta
```

That covers every field the brief requires: Writer operation identity, artifact
identity/path, actual status, result reference, error details and review
requirements. `validate-text` exists too and is left for a later, non-file case.

## 3. Implementation

| File | Purpose |
| --- | --- |
| `stephanie/integrations/writer_handoff.py` | **New.** `WriterHandoffRequest` → `WriterDraftHandoff.submit()` → `WriterHandoffRecord`. Invokes the allow-listed CLI operation, hashes the artifact, maps Writer's verdict, decides `review_required`, and persists the record as `<draft-stem>.writer-handoff.json` next to the artifact. |
| `stephanie/components/information/agents/writer_handoff.py` | **New.** `WriterHandoffAgent` — pipeline stage; reads `context["paper_blog_artifact_path"]`, submits, writes `context["writer_handoff"]`. Never raises. |
| `stephanie/components/information/agents/paper_blog_generator.py` | Publishes `context["paper_blog_artifact_path"]` (the path `save_to_timestamped_file` already returned) so downstream stages reference the artifact instead of copying its content. |
| `config/information.yaml` | New final stage `writer_handoff` (`profile: fiction`, `timeout_seconds: 120`, `record_dir: null`, `project_root: null`). |
| `scripts/writer_handoff.py` | **New.** Operator CLI to submit one draft and print/inspect the record. Exit 0 = PASS, 2 = non-PASS verdict, 3 = handoff not performed. |
| `tests/test_writer_handoff.py` | **New.** 13 deterministic tests. |

Safety properties enforced in code:

* `ALLOWED_OPERATIONS = {("validation", "validate-file")}` — checked at call time;
  the bridge cannot be repurposed to run an arbitrary Writer command;
* discovery success is never treated as handoff success — a real operation must
  return;
* **Stephanie never acts on Writer's decision.** `review_required` and
  `human_review_reasons` are recorded; the draft is never rewritten and no
  policy/review gate is bypassed;
* a missing/failing Writer yields an explicit `unavailable`/`error` record (with
  `review_required = true`) — never a silent fallback to another engine.

## 4. Live result (real draft, no model spend)

```
python scripts/writer_handoff.py \
  C:\Projects\stephanie\runs\paper_blogs\10038\2506.21734_blog_20261018_190634.md \
  --paper 2506.21734 --run-id 10038 --profile fiction
```

```
artifact        : ...\runs\paper_blogs\10038\2506.21734_blog_20261018_190634.md   (89.3 KB)
artifact sha256 : f65caae115b6c008d5864bdd673c726ab6155c274aeb58ed14d012bc5f578009
writer cli      : D:\tools\writer.BAT
operation       : validation.validate-file
report id       : val_report_09f6ac0f556390675896fc79
status          : PASS  decision=ACCEPT gate=PASS
readiness/grade : 100/FREEZE_READY
findings        : total 0 (critical 0, high 0, medium 0, low 0, info 0)
review required : False
record          : ...\2506.21734_blog_20261018_190634.writer-handoff.json
```

This is a real Writer operation against a real Stephanie artifact, with Writer's
own report identity (`val_report_09f6ac0f556390675896fc79`) and the artifact hash
recorded for provenance. Note the CLI resolved through PATH (`D:\tools\writer.BAT`),
not the venv executable — both are supported by `WriterRuntimeDiscovery`.

## 5. Tests

```
tests/test_writer_handoff.py ......................................... 13 passed
full suite (handoff + sections-cache + blog + bridge + writer-runtime)  52 passed
```

Covered: exact CLI arguments; PASS verdict with no review required; FAIL verdict
recording `review_required` with reasons; record persistence (default and
`record_dir`); correlation-id generation/reuse; and every failure mode —
Writer missing, non-zero exit, timeout, invalid JSON, missing artifact — each
producing an explicit record rather than an exception. Plus: a REJECT decision is
never silently marked accepted.

Stage wiring smoke-check (no model calls): `config/information.yaml` composes to
`[…, paper_blog_generator, paper_blog_judge, writer_handoff]` and the `cls`
resolves to `WriterHandoffAgent`.

## 5b. Pipeline wiring contract (and the first failed run)

The first in-pipeline attempt failed **before any model call**:

```
omegaconf.errors.ConfigKeyError: Key 'writer_handoff' is not in struct
full_key: agents.writer_handoff
```

`Supervisor._parse_pipeline_stages` resolves every stage through
`cfg.agents[name]` (`supervisor.py:188`), which only exists when the stage has a
`config/agents/<name>.yaml` file listed under `defaults:` in
`config/information.yaml`. Declaring the stage in `pipeline.stages` is not enough,
and keys placed on the stage entry are **never read by the agent** — the agent
receives `cfg.agents.<name>`.

Fixed in `de86f8dd`: added `config/agents/writer_handoff.yaml`, listed it in the
defaults, and moved the handoff settings there. The same mistake was present on
the `paper_pipeline` stage entry (the section-cache flags); that is fixed too.
`tests/test_information_pipeline_config.py` now composes the real Hydra config and
asserts every enabled stage has a `cfg.agents` entry — the check that would have
caught this before the model stack loaded.

## 6. Not yet verified

* **The stage has not run inside a full pipeline execution.** Doing so needs a
  paid paper→blog run (~18 model calls) which has not been authorised. Everything
  the stage does is exercised by the live script run and the unit tests; what
  remains unproven is only the plumbing from the generator's context key into the
  stage inside a real run.
* `document_artifact_id` / `document_id` / `project_id` / `book_id` came back
  `null` — Writer did not persist the draft as a project artifact (the operation
  was run without `--persist-report`). If Stephanie wants Writer to own the
  artifact, the next increment should pass `--persist-report` and record those ids.
* Non-fiction/technical validation profiles were not probed; `fiction` is the
  Writer default and accepted the draft.

## 7. Next steps

1. Run the pipeline once with the stage enabled to close the plumbing gap
   (needs authorisation for the model spend).
2. Optionally add `--persist-report` so Writer returns artifact/document ids and
   the handoff can be linked to a Writer-side project rather than a bare file.
3. Surface the handoff record in SIS (a read-only row on the pipeline run page:
   status, report id, readiness, review-required).
4. Then the Next / Improve / Workbench / Sidecar integration matrix (SWF01
   phase 7) — still not attempted.
