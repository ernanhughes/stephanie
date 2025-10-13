# stephanie/agents/scorable_loader.py
from __future__ import annotations

# stephanie/scoring/types.py
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import ScorableType
from stephanie.services.scoring_service import ScoringService


class ScorableLoaderAgent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.top_k = cfg.get("top_k", 10)
        self.include_full_text = cfg.get("include_full_text", True)
        self.target_type = cfg.get(
            "target_type", ScorableType.CONVERSATION_TURN
        )  # or "section"
        self.include_ner = cfg.get("include_ner", False)
        self.save_pipeline_refs = cfg.get("save_pipeline_refs", False)

        self.doc_ids_scoring = cfg.get("doc_ids_scoring", False)
        self.doc_ids = cfg.get("doc_ids", [])
        self.scoring: ScoringService = self.container.get("scoring")


    async def run(self, context: dict) -> dict:
        scorables = []
        if self.target_type == ScorableType.CONVERSATION_TURN:
            truns = self.memory.chats.list_turns_with_texts(min_star=1)
            for turn in truns:
                item = TextItem.from_chat_turn(turn)
                scorables.append(asdict(item))

                result = self.scoring.score("knowledge", item, context=context, dimensions=["knowledge"])
                self.logger.log("TurnScored", {"turn_id": item.scorable_id, "score": result.get("knowledge")})

        # 2. Save retrieved doc dicts into context
        context[self.output_key] = scorables

        context["retrieved_ids"] = [d["scorable_id"] for d in scorables]

        if self.save_pipeline_refs:
            for d in scorables:
                self.memory.pipeline_references.insert(
                    {
                        "pipeline_run_id": context.get("pipeline_run_id"),
                        "scorable_type": d["scorable_type"],
                        "scorable_id": d["scorable_id"],
                        "relation_type": "retrieved",
                        "source": self.name,
                    }
                )

        self.logger.log(
            "KnowledgeDBLoaded",
            {"count": len(scorables), "search_method": self.target_type},
        )

        return context




class GoalKind(str, Enum):
    """What the 'goal' string represents for this item."""
    CONTEXT = "context"       # global conversation/document goal/title
    TURN_USER = "turn_user"   # the user message the assistant reply addresses
    SECTION = "section"       # a section heading/title
    CUSTOM = "custom"         # caller-specified goal string


@dataclass
class GoalRef:
    text: str = ""
    kind: GoalKind = GoalKind.CONTEXT


@dataclass
class TextItem:
    """
    De-facto scorable for any text in the system.
    For chat turns, `main_text` is the assistant response and `goal_ref`
    typically points to the *user message* (TURN_USER). For docs, `goal_ref`
    may point to a section title (SECTION) or a global context (CONTEXT).
    """
    # identity / provenance
    scorable_type: ScorableType = ScorableType.CONVERSATION_TURN
    scorable_id: Optional[int] = None                # primary id (e.g., chat turn id)
    conversation_id: Optional[int] = None
    external_id: Optional[str] = None            # optional foreign id

    # ordering / context
    order_index: int = 0
    created_at: Optional[datetime] = None

    # texts
    text: str = ""                          # the text to be scored (assistant_text for chat)
    user_text: str = ""                          # original user message (handy for audits)
    goal_ref: GoalRef = field(default_factory=GoalRef)

    # annotations
    ner: Union[List[Dict[str, Any]], List[Any]] = field(default_factory=list)
    domains: Union[List[Dict[str, Any]], List[Any]] = field(default_factory=list)

    # supervision signals
    star: Optional[int] = None                   # human star (-5..+5), if any
    ai_score: Optional[float] = None             # knowledge score (0..100), if any
    ai_rationale: str = ""                       # rationale/explanation, if any
    # misc meta (free-form)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------- factories ----------

    @classmethod
    def from_chat_turn(cls, row: Dict[str, Any]) -> TextItem:
        """
        Map directly from memory.chats.list_turns_with_texts() row.
        Fields expected (per your SQL): id, conversation_id, order_index, star,
        user_text, assistant_text, ner, domains, goal_text, ai_score, ai_rationale.
        """
        # The *scoring* goal for a chat assistant turn should come from the user message.
        # We still retain conversation goal/title in meta for audits.
        goal_text_user = (row.get("user_text") or "").strip()
        goal_text_title = (row.get("goal_text") or "").strip()

        return cls(
            scorable_type=ScorableType.CONVERSATION_TURN,
            scorable_id=int(row.get("id")) if row.get("id") is not None else None,
            conversation_id=row.get("conversation_id"),
            order_index=int(row.get("order_index") or 0),
            created_at=None,  # fill if you have it
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
            meta={
                # keep the conversation title around for visibility
                "conversation_title": goal_text_title,
                # anything else you want to carry forward
            },
        )

