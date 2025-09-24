# stephanie/agents/compiler/scorer.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.compiler.reasoning_trace import ReasoningNode
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable import ScorableType


class ReasoningNodeScorer(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory=memory, logger=logger)

    def score(self, node: ReasoningNode, context: dict) -> dict:
        scorable = Scorable(text=node.response, type=ScorableType.HYPOTHESIS)
        return self._score(scorable=scorable, context=context)

    async def run(self, context):
        pass
