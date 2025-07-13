# stephanie/agents/ats/solution_node.py

import time
from typing import Any, Dict, Optional


class SolutionNode:
    def __init__(
        self,
        plan: str,
        code: Optional[str] = None,
        metric: Optional[float] = None,
        output: Optional[str] = None,
        summary: Optional[str] = None,
        parent_id: Optional[str] = None,
        is_buggy: bool = False,
        timestamp: float = None,
    ):
        self.id = hash(self)
        self.plan = plan
        self.code = code
        self.metric = metric
        self.output = output
        self.summary = summary
        self.parent_id = parent_id
        self.is_buggy = is_buggy
        self.timestamp = timestamp or time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "plan": self.plan,
            "code": self.code,
            "metric": self.metric,
            "output": self.output,
            "summary": self.summary,
            "parent_id": self.parent_id,
            "is_buggy": self.is_buggy,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SolutionNode":
        return cls(**data)
