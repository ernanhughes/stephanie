# stephanie/components/gap/risk/provenance.py
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
import logging


_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def _sanitize(name: str) -> str:
    return _SAFE.sub("_", name)


def _ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_badge_svg(out_dir: Path, run_id: str, badge_data_uri: str, logger: logging.Logger) -> Optional[Path]:
    try:
        if not badge_data_uri.startswith("data:image/svg+xml;base64,"):
            return None
        b64 = badge_data_uri.split(",", 1)[1]
        svg_bytes = base64.b64decode(b64.encode("ascii"))
        p = out_dir / f"{_sanitize(run_id)}.badge.svg"
        with open(p, "wb") as f:
            f.write(svg_bytes)
        return p
    except Exception:
        logger.exception("Failed to save badge SVG")
        return None


class ProvenanceLogger:
    """
    Persists a compact JSON payload per run + appends a JSONL decision trace.

    Files written to {out_dir}/:
      - {run_id}.json                (full record + goal/reply/context)
      - {run_id}.badge.svg           (decoded SVG, optional)
      - decision_trace.jsonl         (one-line JSON per run; append-only)
    """

    def __init__(self, out_dir: str = "./runs/hallucinations", logger: Optional[logging.Logger] = None) -> None:
        self.out_dir = _ensure_dir(out_dir)
        self.trace_path = self.out_dir / "decision_trace.jsonl"
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    def log(self, *, record: Dict[str, Any], goal: str, reply: str, context: Optional[Dict[str, Any]] = None) -> None:
        run_id = str(record.get("run_id", "run"))
        payload = {
            **record,
            "goal": goal,
            "reply": reply,
            "context": context or {},
        }

        # Write full JSON
        full_path = self.out_dir / f"{_sanitize(run_id)}.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # Extract & persist SVG badge if present
        badge_uri = record.get("badge_svg")
        if isinstance(badge_uri, str):
            _save_badge_svg(self.out_dir, run_id, badge_uri, self.logger)

        # Append to decision trace
        try:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "run_id": run_id,
                    "decision": record.get("decision"),
                    "metrics": record.get("metrics"),
                    "thresholds": record.get("thresholds"),
                    "reasons": record.get("reasons"),
                    "model_alias": record.get("model_alias"),
                    "monitor_alias": record.get("monitor_alias"),
                }, ensure_ascii=False))
                f.write("\n")
        except Exception:
            self.logger.exception("Failed to append decision trace")
