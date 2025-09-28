# stephanie/utils/emit_utils.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import math

# Keys that ReportingService.emit(...) uses as explicit kwargs
_RESERVED = {
    "event",
    "stage",
    "status",
    "summary",
    "finalize",
    "context",
    "level",
}

def _sanitize(v: Any) -> Any:
    # unwrap numpy scalars, torch tensors, etc.
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v

def _deep_sanitize(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _deep_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = type(x)
        return t(_deep_sanitize(v) for v in x)
    return _sanitize(x)

def _maybe_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return str(v)
    except Exception:
        return None

def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def _get_attr(obj: Any, *names: str) -> Any:
    for n in names:
        if obj is None:
            break
        if hasattr(obj, n):
            try:
                return getattr(obj, n)
            except Exception:
                pass
    return None

def _augment_ids(body: Dict[str, Any], *, agent: Any, context: Optional[Dict[str, Any]]) -> None:
    """
    Fill run_id / paper_id / section_name if missing in the body.
    Priority: existing in body > context > agent attrs.
    Also mirror into a compact `meta` block (non-destructive).
    """
    ctx = context or {}
    # run_id
    run_id = _first(
        body.get("run_id"),
        ctx.get("run_id"),
        _get_attr(agent, "run_id", "pipeline_run_id"),
    )
    run_id = _maybe_str(run_id)
    if run_id is not None:
        body["run_id"] = run_id

    # paper_id
    paper_id = _first(
        body.get("paper_id"),
        ctx.get("paper_id"),
        _get_attr(agent, "paper_id"),
        _get_attr(_get_attr(agent, "progress", None), "current_paper_id"),
    )
    paper_id = _maybe_str(paper_id)
    if paper_id is not None:
        body["paper_id"] = paper_id

    # section_name
    section_name = _first(
        body.get("section_name"),
        ctx.get("section_name"),
        _get_attr(agent, "section_name"),
    )
    section_name = _maybe_str(section_name)
    if section_name is not None:
        body["section_name"] = section_name

    # meta (non-destructive add)
    meta = dict(body.get("meta") or {})
    if run_id is not None and "run_id" not in meta:
        meta["run_id"] = run_id
    if paper_id is not None and "paper_id" not in meta:
        meta["paper_id"] = paper_id
    if section_name is not None and "section_name" not in meta:
        meta["section_name"] = section_name
    if meta:
        body["meta"] = meta

def prepare_emit(
    event_name: Optional[str],
    payload: Dict[str, Any],
    *,
    target: str = "reporting",
    agent: Any = None,                 # agent object or None
    context: Dict[str, Any] = None,    # pipeline/section context
) -> Dict[str, Any]:
    """
    Build a single collision-safe payload.

    target:
      - "reporting": safe for ReportingService.emit(**payload)
                     (renames reserved kwargs to arena_*, moves 'summary' to 'arena_summary')
      - "bus"      : safe for bus publish, preserves original 'summary' and keys
    """
    # 0) base & event name
    body = dict(payload or {})
    ev = event_name or body.get("event") or "unknown"
    body["event"] = ev

    # 1) ensure core IDs are present & normalized
    _augment_ids(body, agent=agent, context=context)

    # 2) choose shape
    if target == "reporting":
        # Avoid kw collisions with ReportingService.emit(...)
        # Keep 'event' as-is; rename others if they collide.
        out: Dict[str, Any] = {}
        summary_val = body.pop("summary", None)  # we'll re-add as arena_summary

        for k, v in body.items():
            if k in _RESERVED and k != "event":
                out[f"arena_{k}"] = v
            else:
                out[k] = v

        # put summary into arena_summary (structured or string)
        if summary_val is not None:
            out["arena_summary"] = summary_val

        return _deep_sanitize(out)

    # BUS mode: preserve original keys (including 'summary')
    return _deep_sanitize(body)

def prepare_emit_split(
    event_name: Optional[str], payload: Dict[str, Any], *, agent: Any = None, context: Dict[str, Any] = None
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Legacy helper: returns (event_name, safe_for_reporting, safe_for_bus)
    """
    ev = event_name or payload.get("event") or "unknown"
    rep = prepare_emit(ev, payload, target="reporting", agent=agent, context=context)
    bus = prepare_emit(ev, payload, target="bus", agent=agent, context=context)
    return ev, rep, bus
