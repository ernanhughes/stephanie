# Paper → Blog Pipeline

Configuration: [`config/information.yaml`](../config/information.yaml)
(workflow name `paper_to_blog`).

Stages: `paper_pipeline` → `paper_spine_builder` → `paper_graph_linker` →
`paper_pipeline_report` → `paper_blog_generator` → `paper_blog_judge`.

Artifacts are written to `runs/paper_blogs/<run_id>/` as
`<arxiv_id>_blog_<timestamp>.md` and `<arxiv_id>_blog_with_refs_<timestamp>.md`.

## Defects fixed

The generator silently produced an off-topic, badly-heading blog. Root causes and
fixes (all in `stephanie/components/information/agents/paper_blog_generator.py`
unless noted):

| # | Defect | Fix |
|---|---|---|
| 1 | **Silent cross-paper fallback.** `root_sections` matched on a raw `paper_arxiv_id` string; when it matched nothing the code fell back to *every* paper's sections, so unrelated papers appeared as body headings (e.g. ARC-AGI, motile bacteria in a CoT/CoCoVa blog). | `normalize_arxiv_id()` (handles `arXiv:`, `vN`, URLs) plus `partition_sections()`. Root and related sections are now separated explicitly and the fallback is removed. |
| 2 | **Cross-paper synthesis was unlabelled.** | Related-paper sections are kept (configurable) but emitted under an explicit `## Related work from other papers` heading, each with a `*Source: <title> (<id>).*` line. |
| 3 | **Headings were generated summary sentences**, sometimes truncated mid-sentence (e.g. `## Vision-Language Models (VLMs) remain constrained to reasoning within`). | `sanitize_heading()` + `_section_heading()` prefer a real heading from `meta["heading"]`, cut at the first sentence boundary, trim to a word boundary, and strip markdown. `_force_heading()` then sets the heading deterministically so a sentence fragment can never leak through. |
| 4 | **Duplicate sections** (two near-identical chain-of-thought blocks). | `dedupe_sections()` removes exact heading/text repeats and near-repeats (Jaccard ≥ 0.8 or keyword containment). |
| 5 | **Mangled references.** Citations were split on physical lines, so wrapped entries merged and lost their titles; and the reference block was appended to *every* section then collapsed. | Reference parsing now groups wrapped continuation lines into single entries; the block is built once, at the end of the document. |
| 6 | **Judge was advisory only.** | The judge records an explicit `context["blog_quality"] = {score, gate_min_score, gate_mode, passed, reason}` so downstream stages/reporting can act on it. |

Smaller fixes: `_extract_pair()` looked up a corrupted key `" I b_id"` instead of
`"b_id"`; the output directory used `run_id` verbatim, which nested folders as
`runs/paper_blogs/<id>/runs/paper_blogs/...` — it now uses the final path segment.

## Configuration

Relevant knobs in `config/agents/paper_blog_generator.yaml`:

```yaml
paper_blog_generator:
  blog:
    max_sections: 16
  include_related: true        # keep related-paper sections (labelled)
  max_related_sections: 4      # how many labelled related blocks to add
```

Set `include_related: false` for a strictly single-paper blog.

## Known limitation (not yet wired)

`PaperBlogGeneratorConfig` defines `max_ticks`, `quality_threshold` and
`tick_min_delta`, but nothing consumes them: the intended
**generate → judge → regenerate** quality loop is not implemented, and the judge
runs as a later stage so it cannot trigger a rewrite. Disqualified blogs are now
*marked* (`blog_quality.passed = false`) rather than silently shipped, but a bad
draft is still written to disk. Wiring the loop is a follow-up.

## Tests

```powershell
venv\Scripts\python.exe -m pytest tests\test_paper_blog_generator.py -q
```

Covers id normalisation, heading sanitisation, de-duplication, root/related
partitioning, heading override + source labelling, and reference grouping.
