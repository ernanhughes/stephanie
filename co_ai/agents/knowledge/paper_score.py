# co_ai/agents/knowledge/paper_scoring_adapter.py

from co_ai.agents.base_agent import BaseAgent
from co_ai.agents.mixins.paper_scoring_mixin import PaperScoringMixin

class PaperScoreAgent(BaseAgent, PaperScoringMixin):

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)


    async def run(self, context: dict) -> dict:
        papers = context.get(self.input_key, [])
        results = []
        for paper in papers:
            self.logger.log("ScoringPaper", {"title": paper.get("title")})
            score_result = self.score_paper(paper, context=context)
            results.append({
                "title": paper.get("title"),
                "scores": score_result,
            })
        context[self.output_key] = results
        return context
