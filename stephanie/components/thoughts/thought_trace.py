from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .thought_types import Thought


@dataclass
class ThoughtTrace:
    goal_text: str
    thoughts: List[Thought] = field(default_factory=list)
    run_id: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    def add(self, t: Thought) -> None:
        self.thoughts.append(t)

    def to_records(self) -> List[Dict[str, Any]]:
        base = {"goal_text": self.goal_text, "run_id": self.run_id}
        return [{**base, **t.to_dict()} for t in self.thoughts]
