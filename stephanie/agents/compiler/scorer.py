# stephanie/agents/compiler/scorer.py
from __future__ import annotations

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.compiler.reasoning_trace import ReasoningNode
from stephanie.agents.mixins.scoring_mixin import ScoringMixin


class ReasoningNodeScorer(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory=memory, logger=logger)

    def score(self, node: ReasoningNode, context: dict) -> dict:
        hypothesis = {"text": node.response}
        # You can change 'metrics' to 'compiler' or a list like ['correctness', 'clarity']
        return self.score_item(hypothesis, context, metrics="compiler")

    async def run(self, context):
        pass
