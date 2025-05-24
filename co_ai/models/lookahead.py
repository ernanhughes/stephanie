from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class Lookahead:
    goal: str
    agent_name: str
    model_name: str
    input_pipeline: Optional[List[str]] = None
    suggested_pipeline: Optional[List[str]] = None
    rationale: Optional[str] = None
    reflection: Optional[str] = None
    backup_plans: Optional[List[str]] = None
    metadata: Optional[Dict] = None
    run_id: Optional[str] = None
    created_at: Optional[datetime] = None

    def store(self, memory, logger=None):
        """
        Stores this LookaheadResult in the `lookaheads` table using the memory layer.
        Resolves goal_id from goal text.
        """
        try:
            goal = memory.goal.get_by_text(self.goal)
            if goal is None:
                raise ValueError("Missing goal_id")

            memory.lookahead.insert(goal.id, self)

            if logger:
                logger.log("LookaheadSaved", {
                    "goal_id": goal.id,
                    "suggested_pipeline": self.suggested_pipeline,
                    "rationale_snippet": (self.rationale or "")[:100]
                })

        except Exception as e:
            if logger:
                logger.log("LookaheadStorageError", {"error": str(e)})
            raise
