from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.components.information.tasks.web_search_ingest_task import WebSearchIngestTask


def _sha256_hex(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class SearchResultsIngestAgent(BaseAgent):
    """
    Consumes context["search_results"] produced by SearchOrchestratorAgent and produces:
      - sources + source_candidates + source_quality
      - fetch_plan.json artifacts in runs/.../{run_id}/sources/...

    Expected context fields (from SearchOrchestratorAgent):
      context["search_results"] : list[dict] where each dict includes:
        query, source, result_type, title, summary, url, pid, goal_id, extra_data, ...
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.enabled = bool(cfg.get("enabled", True))

        # where artifacts go
        self.run_dir_tpl = cfg.get("run_dir", "runs/paper_blogs/${run_id}")
        self.out_subdir = cfg.get("out_subdir", "sources")
        self.per_query = bool(cfg.get("per_query", True))

        # quality task
        self.task = WebSearchIngestTask(
            source_store=self.memory.sources,                 # MUST exist (your SourceStore)
            candidate_store=self.memory.source_candidates,     # new
            quality_store=self.memory.source_quality,          # new
            logger=self.logger,
        )

        self.default_goal_type = cfg.get("default_goal_type", "research")

    async def run(self, context: dict) -> dict:
        if not self.enabled:
            return context

        run_id = context.get("run_id") or context.get("pipeline_run_id")
        if not run_id:
            self.logger.log("SearchResultsIngestSkipped", {"reason": "missing run_id"})
            return context

        goal = context.get(GOAL, {}) or {}
        goal_type = (goal.get("goal_type") or self.default_goal_type).strip()

        search_results: List[Dict[str, Any]] = context.get("search_results") or []
        if not search_results:
            self.logger.log("SearchResultsIngestSkipped", {"reason": "no search_results"})
            return context

        # resolve run_dir from template
        run_dir = self._resolve_run_dir(self.run_dir_tpl, run_id)
        out_dir = os.path.join(run_dir, self.out_subdir)
        os.makedirs(out_dir, exist_ok=True)

        # Group by query if desired (recommended: budgets apply per query)
        if self.per_query:
            grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in search_results:
                grouped[(r.get("query") or "").strip()].append(r)

            outputs = []
            for q, group in grouped.items():
                if not q:
                    continue
                task_results = self._ingest_one_query(
                    pipeline_run_id=int(run_id),
                    goal_type=goal_type,
                    query_text=q,
                    rows=group,
                    out_dir=out_dir,
                )
                outputs.append(task_results)

            context["source_ingest_outputs"] = outputs
            return context

        # Single combined ingest
        query_text = (context.get("search_query") or "combined_search").strip()
        task_results = self._ingest_one_query(
            pipeline_run_id=int(run_id),
            goal_type=goal_type,
            query_text=query_text,
            rows=search_results,
            out_dir=out_dir,
        )
        context["source_ingest_outputs"] = [task_results]
        return context

    # -------------------------
    # internals
    # -------------------------

    def _resolve_run_dir(self, tpl: str, run_id: int) -> str:
        # minimal safe interpolation (works even without OmegaConf interpolation)
        return (tpl or "runs/paper_blogs/${run_id}").replace("${run_id}", str(run_id))

    def _ingest_one_query(
        self,
        *,
        pipeline_run_id: int,
        goal_type: str,
        query_text: str,
        rows: List[Dict[str, Any]],
        out_dir: str,
    ) -> Dict[str, Any]:
        # Convert SearchOrchestrator "search_results" shape -> WebSearchIngestTask shape
        converted: List[Dict[str, Any]] = []
        for idx, r in enumerate(rows):
            url = (r.get("url") or "").strip()
            if not url:
                continue

            converted.append(
                {
                    "url": url,
                    "title": r.get("title") or "",
                    "snippet": r.get("summary") or "",
                    "rank": r.get("rank", idx),
                    "provider": r.get("source"),  # arxiv/web/wikipedia/huggingface/similar_papers
                    "result_type": r.get("result_type") or "unknown",
                    "pid": r.get("pid") or "",
                    "extra_data": r.get("extra_data") or {},
                }
            )

        # Per-query artifact subdir (stable name)
        qhash = _sha256_hex(query_text)[:16]
        qdir = os.path.join(out_dir, f"q_{qhash}")
        os.makedirs(qdir, exist_ok=True)

        out = self.task.run(
            pipeline_run_id=pipeline_run_id,
            goal_type=goal_type,
            query_text=query_text,
            results=converted,
            run_dir=qdir,
        )

        # Useful back to pipeline
        return {
            "query_text": query_text,
            "query_hash": qhash,
            **out,
        }
