#!/usr/bin/env python3
"""
slim_context_json.py — shrink a single large context JSON dict (not JSONL).

It will:
- keep important identifiers (run_id, pipeline_run_id, etc.)
- compact trace items: keep agent + metadata + goal stub + REPORTS summary
- drop huge keys (LOGS, raw outputs, prompts, embeddings, full text, etc.)
- truncate long strings
- cap list/dict sizes
- optionally gzip output

Usage:
  python slim_context_json.py input.json output.json.gz
  python slim_context_json.py input.json output.json --pretty
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional


# ---- Defaults tuned for your context structure ----

KEEP_TOP_KEYS = {
    "run_id", "pipeline_run_id", "pipeline", "goal", "metadata",
    "trace", "REPORTS", "STAGE_DETAILS", "METRICS",
    "paper_id", "arxiv_id", "seed_id", "variant", "config_id", "blog_config_id",
    "artifact_path", "blog_path", "paper_blog_path", "report_path", "pipeline_report_path",
    "status", "error", "errors",
}

# Drop keys anywhere that often explode size
DROP_KEY_REGEX = re.compile(
    r"(?i)(^|_)(LOGS|messages|context|prompt|prompt_text|sys_preamble|system_prompt|"
    r"paper_text|paper_md|paper_markdown|full_text|pdf_text|raw_text|document_text|"
    r"blog_md|blog_markdown|generated_blog|output_text|completion|response|raw|outputs|"
    r"embeddings?|vectors?|tensor|images?|base64|html|xml)(_|$)"
)

TRUNCATE_KEY_REGEX = re.compile(r"(?i)(^|_)(message|rationale|summary|notes|goal_text)(_|$)")


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def summarize_str(s: str, max_string: int, head: int = 220, tail: int = 80) -> Any:
    if len(s) <= max_string:
        return s
    return {
        "__truncated__": True,
        "len": len(s),
        "sha256": sha256_hex(s),
        "head": s[:head],
        "tail": s[-tail:] if tail > 0 else "",
    }


def summarize_big_value(v: Any, max_string: int) -> Any:
    if isinstance(v, str):
        return summarize_str(v, max_string=max_string)
    if isinstance(v, list):
        return {"__dropped__": True, "type": "list", "len": len(v)}
    if isinstance(v, dict):
        return {"__dropped__": True, "type": "dict", "len": len(v)}
    return v


def prune(
    obj: Any,
    *,
    max_string: int,
    max_list: int,
    max_dict: int,
) -> Any:
    """Generic recursive prune/truncate based on key regexes + size caps."""
    if isinstance(obj, dict):
        items = list(obj.items())
        if len(items) > max_dict:
            items = sorted(items, key=lambda kv: str(kv[0]))[:max_dict]
        out: Dict[str, Any] = {}
        for k, v in items:
            ks = str(k)

            if DROP_KEY_REGEX.search(ks):
                out[ks] = summarize_big_value(v, max_string=max_string)
                continue

            if TRUNCATE_KEY_REGEX.search(ks) and isinstance(v, str):
                out[ks] = summarize_str(v, max_string=max_string)
                continue

            out[ks] = prune(v, max_string=max_string, max_list=max_list, max_dict=max_dict)
        return out

    if isinstance(obj, list):
        if len(obj) > max_list:
            kept = obj[:max_list]
            return {
                "__truncated_list__": True,
                "len": len(obj),
                "kept": [prune(x, max_string=max_string, max_list=max_list, max_dict=max_dict) for x in kept],
            }
        return [prune(x, max_string=max_string, max_list=max_list, max_dict=max_dict) for x in obj]

    if isinstance(obj, str):
        return summarize_str(obj, max_string=max_string)

    return obj


def goal_stub(goal: Any, max_string: int) -> Any:
    if not isinstance(goal, dict):
        return goal
    out = {
        "id": goal.get("id"),
        "goal_type": goal.get("goal_type"),
        "goal_category": goal.get("goal_category"),
        "source": goal.get("source"),
        "strategy": goal.get("strategy"),
        "scorer_signals": goal.get("scorer_signals"),
        "success_criteria": goal.get("success_criteria"),
        "expected_formats": goal.get("expected_formats"),
    }
    gt = goal.get("goal_text")
    if isinstance(gt, str):
        out["goal_text"] = summarize_str(gt, max_string=max_string)
    return out


def compact_reports(reports: Any, max_string: int, max_reports: int) -> Any:
    if not isinstance(reports, list):
        return reports
    slim: List[Dict[str, Any]] = []
    for r in reports[:max_reports]:
        if not isinstance(r, dict):
            continue
        slim.append({
            "stage": r.get("stage"),
            "agent": r.get("agent"),
            "status": r.get("status"),
            "start_time": r.get("start_time"),
            "end_time": r.get("end_time"),
            "error": summarize_str(r.get("error", ""), max_string=max_string) if isinstance(r.get("error"), str) else r.get("error"),
            "summary": summarize_str(r.get("summary", ""), max_string=max_string) if isinstance(r.get("summary"), str) else r.get("summary"),
            # keep metrics/outputs but let prune shrink them
            "metrics": r.get("metrics", {}),
            "outputs": r.get("outputs", {}),
        })
    return slim


def compact_trace_item(item: Any, *, max_string: int, max_reports: int, keep_stage_details: bool) -> Any:
    if not isinstance(item, dict):
        return item

    agent = item.get("agent")
    inputs = item.get("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}

    meta = inputs.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}

    run_id = inputs.get("run_id") or meta.get("run_id")
    pipeline_run_id = inputs.get("pipeline_run_id")
    pipeline = inputs.get("pipeline")
    goal = inputs.get("goal")

    reports = inputs.get("REPORTS")
    stage_details = inputs.get("STAGE_DETAILS") if keep_stage_details else None
    metrics = inputs.get("METRICS")

    compact: Dict[str, Any] = {
        "agent": agent,
        "run_id": run_id,
        "pipeline_run_id": pipeline_run_id,
        "pipeline": pipeline,
        "metadata": meta,
        "goal": goal_stub(goal, max_string=max_string),
        "REPORTS": compact_reports(reports, max_string=max_string, max_reports=max_reports),
    }

    if keep_stage_details:
        compact["STAGE_DETAILS"] = stage_details

    if metrics is not None:
        compact["METRICS"] = metrics

    return compact


def compact_context(
    ctx: Dict[str, Any],
    *,
    max_string: int,
    max_trace_items: int,
    max_reports: int,
    keep_stage_details: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # keep a few top keys if present
    for k in ("run_id", "pipeline_run_id", "pipeline", "variant", "config_id", "blog_config_id"):
        if k in ctx:
            out[k] = ctx.get(k)

    if "metadata" in ctx:
        out["metadata"] = ctx.get("metadata")

    if "goal" in ctx:
        out["goal"] = goal_stub(ctx.get("goal"), max_string=max_string)

    trace = ctx.get("trace")
    if isinstance(trace, list):
        out["trace"] = [compact_trace_item(t, max_string=max_string, max_reports=max_reports, keep_stage_details=keep_stage_details)
                        for t in trace[:max_trace_items]]
        if len(trace) > max_trace_items:
            out["trace_truncated_from"] = len(trace)

    # also preserve top-level REPORTS if they exist (sometimes they do)
    if "REPORTS" in ctx:
        out["REPORTS"] = compact_reports(ctx.get("REPORTS"), max_string=max_string, max_reports=max_reports)

    return out


def open_out(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, "w", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input JSON file (single dict)")
    ap.add_argument("output", help="Output JSON or JSON.GZ")
    ap.add_argument("--max-string", type=int, default=1200)
    ap.add_argument("--max-list", type=int, default=80)
    ap.add_argument("--max-dict", type=int, default=120)
    ap.add_argument("--max-trace", type=int, default=80, help="Max trace items to keep")
    ap.add_argument("--max-reports", type=int, default=60, help="Max REPORTS entries to keep per trace item")
    ap.add_argument("--keep-stage-details", action="store_true", help="Keep STAGE_DETAILS (can be large)")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print output JSON (larger)")
    args = ap.parse_args()

    in_bytes = os.path.getsize(args.input)

    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        ctx = json.load(f)

    if not isinstance(ctx, dict):
        raise SystemExit("Input JSON is not a dict/object at the top level.")

    # 1) structural compaction (huge win)
    compact = compact_context(
        ctx,
        max_string=args.max_string,
        max_trace_items=args.max_trace,
        max_reports=args.max_reports,
        keep_stage_details=args.keep_stage_details,
    )

    # 2) generic prune to shrink nested metrics/outputs/etc.
    slim = prune(compact, max_string=args.max_string, max_list=args.max_list, max_dict=args.max_dict)

    with open_out(args.output) as out:
        if args.pretty:
            json.dump(slim, out, ensure_ascii=False, indent=2)
        else:
            json.dump(slim, out, ensure_ascii=False, separators=(",", ":"))

    out_bytes = os.path.getsize(args.output)
    print(f"Input:  {args.input}  ({in_bytes/1024/1024:.2f} MB)")
    print(f"Output: {args.output} ({out_bytes/1024/1024:.2f} MB)")
    print(f"Shrink: {(out_bytes/in_bytes)*100:.1f}% of original size")


if __name__ == "__main__":
    main()
