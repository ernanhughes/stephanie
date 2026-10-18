# SWF01 — First Stephanie → Writer Editorial Handoff

**Status: IMPLEMENTED and VERIFIED END-TO-END IN THE PIPELINE.**
Runs 10039 and 10040 executed the full paper→blog pipeline with the
`writer_handoff` stage enabled. Writer's report identity, verdict and review
requirement are recorded against real artifacts. Three bugs were found **only by
actually running it**, and are fixed (see §5b and §5c).

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

## 4b. In-pipeline result (run 10040)

Full pipeline, stage enabled: 18 paid model calls, 0 Ollama fallbacks, exit 0.

```
WriterDraftHandoff WARN: report=val_report_1b18282324e9cc5a07581b00 decision=REVIEW review_required=True
```

The record written next to the fresh draft
(`runs/paper_blogs/10040/2506.21734_blog_20261018_221535.writer-handoff.json`):

```
correlation_id   : swh-1d80bfae2a54
artifact_path    : C:\Projects\stephanie\runs\paper_blogs\10040\2506.21734_blog_20261018_221535.md
artifact sha256  : 613aab1169e054114f3b4d836aabbb4ea37b40e147f3dcc3121e37f488e1e37c
writer_cli       : D:\tools\writer.BAT
writer_report_id : val_report_1b18282324e9cc5a07581b00
status/decision  : WARN / REVIEW   gate=REVIEW
readiness/grade  : 48/BLOCKED
coverage         : 7/7 validators successful (0 failed)
finding_counts   : total 2 (critical 1, medium 1)
review_required  : True
  - Writer status is WARN
  - Writer decision is REVIEW
  - Writer gate is REVIEW
  - 1 critical finding(s)
paper/run        : 2506.21734 / 10040
record_path      : ...\2506.21734_blog_20261018_221535.writer-handoff.json
```

Writer refused to let the draft advance (`readiness 48/100 BLOCKED`) and named
two real defects in Stephanie's own prose:

| Severity | Finding | Validator |
| --- | --- | --- |
| CRITICAL | `MISSING_SPACE_AFTER_PUNCTUATION` (line 253) | text_corruption v3 |
| MEDIUM | `UNBALANCED_QUOTES` (29 straight, 0 curly) | dialogue_quotes v2 |

The critical evidence is a genuine text-corruption artefact in the generated
prose: `guage modeling, where data is dense, i.i.d., and easy to batch. Adoption
in reinfo` — a word split/joined across a boundary, the same class of damage as
the PDF line-break artefacts noted for the paper pipeline. **Writer found a real
defect in Stephanie's output automatically** — which is the value of the handoff.

Stephanie recorded all of it, marked the draft as requiring human review, and did
**not** advance, rewrite or accept anything. The pipeline still exited 0.

## 5. Tests

```
tests/test_writer_handoff.py ......................................... 14 passed
tests/test_information_pipeline_config.py .............................  3 passed
full suite ............................................................ 56 passed
```

Covered: exact CLI arguments; PASS verdict with no review required; FAIL verdict
recording `review_required` with reasons; record persistence (default and
`record_dir`) including a self-describing `record_path`; correlation-id
generation/reuse; relative → absolute path resolution; and every failure mode —
Writer missing, non-zero exit, timeout, invalid JSON, missing artifact — each
producing an explicit record rather than an exception. Plus: a REJECT decision is
never silently marked accepted.

`tests/test_information_pipeline_config.py` composes the real Hydra config and
asserts every enabled stage has a `cfg.agents` entry (see §5b).

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

## 5c. Second in-pipeline bug: the relative draft path

Run 10039 (the first run past the config fix) reached the stage and produced a
record, but the handoff failed:

```
WriterDraftHandoff error: invalid JSON from writer
  (File not found: runs\paper_blogs\10039\2506.21734_blog_20261018_220918.md)
```

The generator publishes a path **relative to the process CWD**, and Writer is a
separate process with its own working directory, so it could not resolve the
draft. The stage itself worked: it found `context["paper_blog_artifact_path"]`,
hashed the artifact, recorded the paper/run identity and provenance, set
`review_required`, and did not fail the pipeline. Only the path was wrong.

Fixed in `ba641cdc`: `_resolve_artifact()` resolves to an absolute path before
hashing, invoking Writer and recording. Also fixed there: the persisted record now
contains its own `record_path` (it was serialised before being set, so it was
always `null`).

Re-running the exact failing relative path through the fixed code:

```
report id       : val_report_856ce535b9427eb114341c02
status          : PASS  decision=ACCEPT gate=PASS
readiness/grade : 100/FREEZE_READY
review required : False
```

## 6. Known gaps

* `document_artifact_id` / `document_id` / `project_id` / `book_id` come back
  `null` — Writer does not persist the draft as a project artifact (the operation
  runs without `--persist-report`). To let Writer own the artifact, pass
  `--persist-report` and record those ids.
* Only the `fiction` validation profile has been exercised.
* The record is written next to the artifact and returned in
  `context["writer_handoff"]`, but it is **not** yet persisted in Stephanie's
  database (CaseBook/PlanTrace) or surfaced in SIS.
* The CRITICAL `MISSING_SPACE_AFTER_PUNCTUATION` finding is a real defect in the
  generated prose and is **not yet fixed at source**.

## 7. Next steps

1. Surface the handoff record in SIS (read-only: status, report id, readiness,
   review-required) and/or persist it against the pipeline run.
2. Add `--persist-report` so Writer returns artifact/document ids.
3. Fix the source of the CRITICAL text-corruption finding (a word split/joined at
   a boundary) — the same class as the PDF line-break damage.
4. Then the Next / Improve / Workbench / Sidecar integration matrix (SWF01
   phase 7) — still not attempted.
