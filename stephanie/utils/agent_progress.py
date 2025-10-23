# stephanie/utils/agent_progress.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional


class AgentProgress:
    """
    Lightweight progress reporter for multi-level runs.
    Emits to: logger.report (if present), SIS (if available), and optional callbacks.
    """

    def __init__(self, agent, *, enable_sis: bool = True):
        self.agent = agent
        self.enable_sis = enable_sis
        self.t0 = time.time()
        self.current_paper_id: Optional[str] = None
        self.total_sections: int = 0
        self.done_sections: int = 0

    # ---- generic emit ----
    def _emit(self, event: str, payload: Dict[str, Any]):
        # 1) BaseAgent.report (safe no-op if not implemented)
        try:
            self.agent.report({"event": event, **payload})
        except Exception:
            pass

        # 2) logger.log for searchable telemetry
        try:
            self.agent.logger.log(event, payload)
        except Exception:
            pass

        # 3) SIS card (optional, non-blocking)
        self.agent.memory.sis_cards.upsert_payload({**payload, "ts": payload.get("ts") or time.time()})

    def _ms(self) -> float:
        return round((time.time() - self.t0) * 1000.0, 1)

    # ---- paper level ----
    def start_paper(self, paper_id: str | int, title: str, total_sections: int):
        self.current_paper_id = str(paper_id)
        self.total_sections = max(1, int(total_sections or 1))
        self.done_sections = 0
        self._emit("LfL_Paper_Start", {
            "paper_id": self.current_paper_id,
            "title": title,
            "total_sections": self.total_sections,
            "progress_pct": 0.0,
            "elapsed_ms": self._ms(),
            "meta": {"paper_id": self.current_paper_id}
        })

    def end_paper(self, stats: Dict[str, Any]):
        pct = 100.0 * (self.done_sections / max(1, self.total_sections))
        self._emit("LfL_Paper_End", {
            "paper_id": self.current_paper_id,
            "progress_pct": pct,
            "elapsed_ms": self._ms(),
            "stats": stats,
            "meta": {"paper_id": self.current_paper_id}
        })

    # ---- section level ----
    def start_section(self, name: str, index: int):
        pct = 100.0 * (self.done_sections / max(1, self.total_sections))
        self._emit("LfL_Section_Start", {
            "paper_id": self.current_paper_id,
            "section_index": index,
            "section_name": name,
            "progress_pct": pct,
            "elapsed_ms": self._ms(),
            "stage": "start",
            "meta": {"paper_id": self.current_paper_id, "section_name": name}
        })

    def stage(self, name: str, section_name: str, **kv):
        pct = 100.0 * (self.done_sections / max(1, self.total_sections))
        self._emit("LfL_Section_Stage", {
            "paper_id": self.current_paper_id,
            "section_name": section_name,
            "stage": name,
            "progress_pct": pct,
            "elapsed_ms": self._ms(),
            **kv,
            "meta": {"paper_id": self.current_paper_id, "section_name": section_name}
        })

    def end_section(self, name: str, metrics: Dict[str, Any]):
        self.done_sections += 1
        pct = 100.0 * (self.done_sections / max(1, self.total_sections))
        self._emit("LfL_Section_End", {
            "paper_id": self.current_paper_id,
            "section_name": name,
            "progress_pct": pct,
            "elapsed_ms": self._ms(),
            "metrics": metrics,
            "meta": {"paper_id": self.current_paper_id, "section_name": name}
        })
