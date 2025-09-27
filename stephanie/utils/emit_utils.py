from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import math

# Keys that ReportingService.emit(...) uses as explicit kwargs
_RESERVED = {"event", "stage", "status", "summary", "finalize", "context", "level"}

def _sanitize(v: Any) -> Any:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v

def _sanitize_map(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _sanitize(v) for k, v in (d or {}).items()}

def prepare_emit(event_name: Optional[str], payload: Dict[str, Any], *, target: str = "reporting") -> Dict[str, Any]:
    """
    Build a single collision-safe payload.

    target:
      - "reporting": safe for ReportingService.emit(**payload)
                     (moves payload['summary'] -> 'arena_summary', keeps 'event')
      - "bus"      : safe for bus publish, preserves original 'summary' and sets 'event'
    """
    body = dict(payload or {})
    ev = event_name or body.get("event") or "unknown"
    body["event"] = ev

    # coerce run_id to str if present
    if "run_id" in body and body["run_id"] is not None:
        try:
            body["run_id"] = str(body["run_id"])
        except Exception:
            pass

    if target == "reporting":
        # avoid kw collisions with ReportingService.emit(summary=..., status=..., etc.)
        summary_val = body.pop("summary", None)
        out: Dict[str, Any] = {}
        for k, v in body.items():
            if k in _RESERVED and k not in {"event"}:
                out[f"arena_{k}"] = v  # rename reserved keys
            else:
                out[k] = v
        if summary_val is not None and not isinstance(summary_val, str):
            out["arena_summary"] = summary_val
        elif summary_val is not None:
            # string summaries are okay to keep as arena_summary too (optional)
            out["arena_summary"] = summary_val
        return _sanitize_map(out)

    # BUS mode: preserve original keys including 'summary'
    return _sanitize_map(body)

# --- Optional legacy compatibility: need both at once? ---
def prepare_emit_split(event_name: Optional[str], payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Legacy shape: (event_name, safe_for_reporting, safe_for_bus)
    Use ONLY if a caller truly needs both versions simultaneously.
    """
    ev = (event_name or payload.get("event") or "unknown")
    rep = prepare_emit(ev, payload, target="reporting")
    bus = prepare_emit(ev, payload, target="bus")
    return ev, rep, bus
