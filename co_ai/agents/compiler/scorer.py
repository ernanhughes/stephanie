# co_ai/compiler/scorer.py
from co_ai.agents.mixins.scoring_mixin import ScoringMixin
from co_ai.agents.compiler.reasoning_trace import ReasoningNode
from co_ai.agents.base_agent import BaseAgent

class ReasoningNodeScorer(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory=memory, logger=logger)

    def score(self, node: ReasoningNode, context: dict) -> dict:
        hypothesis = {"text": node.response}
        # You can change 'metrics' to 'compiler' or a list like ['correctness', 'clarity']
        return self.score_hypothesis(hypothesis, context, metrics="compiler")

    async def run(self, context):
        pass
