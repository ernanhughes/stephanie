from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from stephanie.scoring.scorable import ScorableType


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
