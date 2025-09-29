# stephanie/agents/maintenance/casebook_seeder.py
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL, PIPELINE_RUN_ID
# Models (names from your codebase)
from stephanie.models.casebook import CaseBookORM, CaseORM
from stephanie.scoring.scorable import Scorable, ScorableType
from stephanie.utils.slug import simple_slugify

# Optional: if you want a separate goal-state table
try:
    from stephanie.models.case_goal_state import CaseGoalStateORM
except Exception:
    CaseGoalStateORM = None


class SeedCaseBookAgent(BaseAgent):
    """
    Usage: place this agent after KnowledgeDBLoaderAgent.
    Config:
      - documents_key: context key containing a list[dict] of docs (defaults to loader output key)
      - max_docs: cap number of docs to seed (default 20)
      - casebook_name: optional explicit casebook name; if omitted, derived from goal text
      - mode: "one_case_per_doc" (default) or "single_case_with_many" (kept for future)
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.documents_key = (
            cfg.get("documents_key") or self.input_key or self.name
        )  # be flexible
        self.max_docs = int(cfg.get("max_docs", 50))
        # self.casebook_name = cfg.get("casebook_name", "default_casebook")
        self.casebook_name = "default_casebook"
        self.mode = cfg.get("mode", "one_case_per_doc")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = context.get(GOAL, {}) or {}
        goal_text = goal.get("goal_text", "") or ""
        pipeline_run_id = context.get(PIPELINE_RUN_ID)

        # pull docs from context (be permissive about keys)
        docs = context.get(self.documents_key)

        # derive a casebook name if not provided
        if not self.casebook_name:
            date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.casebook_name = (
                f"cb_{simple_slugify(goal_text or 'goal')}_{date}"
            )

        # create/ensure
        pipeline_run_id = context.get(PIPELINE_RUN_ID)
        cb = self.memory.casebooks.ensure_casebook(
            name=self.casebook_name,
            pipeline_run_id=pipeline_run_id,
            description=goal_text[:240],
        )

        goal_id_str = str(goal.get("id"))
        # create a parent "goal" case (if not already present)
        pipeline_run_id = context.get(PIPELINE_RUN_ID)
        goal_case = self.memory.casebooks.ensure_goal_state_for_case(
            casebook_id=cb.id,
            goal_id=goal_id_str,
            goal_text=goal_text,
            pipeline_run_id=pipeline_run_id,
        )

        existing = self.memory.casebooks.list_cases(
            casebook_id=cb.id, goal_id=goal_id_str
        )

        created = 0
        for d in docs[: self.max_docs]:
            doc_id = d.get("id")
            if not doc_id or doc_id in existing:
                continue

            # Create a case per document
            title = d.get("title") or d.get("name") or f"Document {doc_id}"
            meta = {
                "score": d.get("score"),
                "source": d.get("source"),
                "snippet": (d.get("summary") or d.get("text", "")[:280]),
                "retrieval": {
                    "agent": self.name,
                    "documents_key": self.documents_key,
                },
            }
            scorable = Scorable(
                id=str(doc_id),
                text=d.get("text", title),
                target_type=ScorableType.DOCUMENT,
                meta=meta,
            )

            self.memory.casebooks.add_case(
                casebook_id=cb.id,
                goal_id=goal_id_str,
                goal_text=goal_text,
                agent_name=self.name,
                scorables=[scorable.to_dict()],
            )
            created += 1

            # Optional: record linkage in pipeline_references for audit
            try:
                self.memory.pipeline_references.insert(
                    {
                        "pipeline_run_id": pipeline_run_id,
                        "scorable_type": ScorableType.DOCUMENT,
                        "scorable_id": doc_id,
                        "relation_type": "seeded_case",
                        "source": self.name,
                    }
                )
            except Exception:
                pass
        self.logger.log(
            "CasebookSeeded",
            {
                "casebook_id": cb.id,
                "goal_case_id": goal_case.id,
                "inserted_cases": created,
                "skipped_existing": len(docs[: self.max_docs]) - created,
                "casebook_name": self.casebook_name,
            },
        )

        # Return handy references for downstream stages
        context.setdefault(
            "casebook", {"id": cb.id, "name": self.casebook_name}
        )
        context.setdefault("cases_seeded", 0)
        context["cases_seeded"] += created
        return context
