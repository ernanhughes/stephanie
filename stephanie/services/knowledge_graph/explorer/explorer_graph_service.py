# stephanie/services/knowledge_graph/explorer/explorer_graph_service.py
from __future__ import annotations

from typing import Any, Optional

from .explorer_graph_builder import ExplorerGraphBuilder
from .explorer_graph import ExplorerGraph


class ExplorerGraphService:
    def __init__(self, db_session: Any):
        self.db = db_session
        self.builder = ExplorerGraphBuilder()

    def build_and_attach(
        self,
        plan_trace: Any,
        *,
        include_evaluations: bool = True,
        include_scores: bool = True,
        meta_key: str = "explorer_graph_v1",
        commit: bool = True,
    ) -> ExplorerGraph:
        g = self.builder.build_from_plan_trace(
            plan_trace,
            include_evaluations=include_evaluations,
            include_scores=include_scores,
        )

        # Attach to PlanTrace.meta (JSON)
        meta = getattr(plan_trace, "meta", None) or {}
        meta[meta_key] = g.to_dict()
        plan_trace.meta = meta

        self.db.add(plan_trace)
        if commit:
            self.db.commit()

        return g

    def load_from_plan_trace(
        self,
        plan_trace: Any,
        *,
        meta_key: str = "explorer_graph_v1",
    ) -> Optional[ExplorerGraph]:
        meta = getattr(plan_trace, "meta", None) or {}
        payload = meta.get(meta_key)
        if not payload:
            return None
        return ExplorerGraph.from_dict(payload)
