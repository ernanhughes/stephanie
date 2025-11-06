# stephanie/agents/scorable_loader.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import time

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import ScorableType
from stephanie.services.scoring_service import ScoringService
from stephanie.utils.progress_mixin import ProgressMixin

class ScorableLoaderAgent(BaseAgent, ProgressMixin):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.top_k = cfg.get("top_k", 10)
        self.include_full_text = cfg.get("include_full_text", True)
        self.target_type = cfg.get("target_type", ScorableType.CONVERSATION_TURN)
        self.include_ner = cfg.get("include_ner", False)
        self.save_pipeline_refs = cfg.get("save_pipeline_refs", False)
        self.doc_ids_scoring = cfg.get("doc_ids_scoring", False)
        self.doc_ids = cfg.get("doc_ids", [])
        self.scoring: ScoringService = self.container.get("scoring")
        self.limit = int(cfg.get("limit", 100))
        self.batch_size = int(cfg.get("batch_size", 100))

    async def run(self, context: dict) -> dict:
        scorables = []
        task = f"ScorableLoad:{context.get('pipeline_run_id','na')}"
        self._init_progress(self.container, self.logger)

        if self.target_type == ScorableType.CONVERSATION_TURN:
            # Unknown total, so use 'indeterminate' first, then switch as we go
            self.pstart(task=task, total=self.limit, meta={
                "target_type": str(self.target_type),
                "limit": self.limit,
                "batch_size": self.batch_size
            })

            produced = 0
            for batch in self.memory.chats.iter_turns_with_texts(
                total_limit=self.limit,
                batch_size=self.batch_size,
                include_texts=self.include_full_text,
                include_goal=True,
                require_assistant_text=True,
                require_nonempty_ner=self.include_ner,
                min_assistant_len=1,
                order_desc=True,
            ):
                for row in batch:
                    item = TextItem.from_chat_turn(row)
                    scorables.append(asdict(item))

                    # (Optional) lightweight per-item scoring or defer to later stage
                    # res = self.scoring.score("knowledge", item, context=context, dimensions=["knowledge"])
                    # self.logger.log("TurnScored", {"turn_id": item.scorable_id, "score": res.get("knowledge")})

                produced += len(batch)
                self.ptick(task=task, done=produced, total=self.limit)
                if produced >= self.limit:
                    break

            self.pdone(task=task)

        context[self.output_key] = scorables
        context["retrieved_ids"] = [d["scorable_id"] for d in scorables]

        if self.save_pipeline_refs:
            for d in scorables:
                self.memory.pipeline_references.insert({
                    "pipeline_run_id": context.get("pipeline_run_id"),
                    "scorable_type": d["scorable_type"],
                    "scorable_id": d["scorable_id"],
                    "relation_type": "retrieved",
                    "source": self.name,
                })

        self.logger.log("KnowledgeDBLoaded", {
            "count": len(scorables),
            "target_type": str(self.target_type),
            "limit": self.limit,
            "batch_size": self.batch_size
        })
        return context


class GoalKind(str, Enum):
    CONTEXT = "context"
    TURN_USER = "turn_user"
    SECTION = "section"
    CUSTOM = "custom"


@dataclass
class GoalRef:
    text: str = ""
    kind: GoalKind = GoalKind.CONTEXT


@dataclass
class TextItem:
    scorable_type: ScorableType = ScorableType.CONVERSATION_TURN
    scorable_id: Optional[int] = None
    conversation_id: Optional[int] = None
    external_id: Optional[str] = None

    order_index: int = 0
    created_at: Optional[datetime] = None

    text: str = ""
    user_text: str = ""
    goal_ref: GoalRef = field(default_factory=GoalRef)

    ner: Union[List[Dict[str, Any]], List[Any]] = field(default_factory=list)
    domains: Union[List[Dict[str, Any]], List[Any]] = field(default_factory=list)

    star: Optional[int] = None
    ai_score: Optional[float] = None
    ai_rationale: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chat_turn(cls, row: Dict[str, Any]) -> "TextItem":
        goal_text_user = (row.get("user_text") or "").strip()
        goal_text_title = (row.get("goal_text") or "").strip()
        return cls(
            scorable_type=ScorableType.CONVERSATION_TURN,
            scorable_id=int(row.get("id")) if row.get("id") is not None else None,
            conversation_id=row.get("conversation_id"),
            order_index=int(row.get("order_index") or 0),
            created_at=None,
            text=(row.get("assistant_text") or "").strip(),
            user_text=(row.get("user_text") or "").strip(),
            goal_ref=GoalRef(
                text=goal_text_user if goal_text_user else goal_text_title,
                kind=GoalKind.TURN_USER if goal_text_user else GoalKind.CONTEXT,
            ),
            ner=row.get("ner") or [],
            domains=row.get("domains") or [],
            star=(int(row["star"]) if row.get("star") is not None else None),
            ai_score=(float(row["ai_score"]) if row.get("ai_score") is not None else None),
            ai_rationale=row.get("ai_rationale") or "",
            meta={"conversation_title": goal_text_title},
        )
