# co_ai/agents/knowledge/paper_scoring_adapter.py

from co_ai.agents.base_agent import BaseAgent
from co_ai.agents.mixins.paper_scoring_mixin import PaperScoringMixin

class PaperScoringAdapterAgent(BaseAgent, PaperScoringMixin):
    def run(self, paper_docs: list, context: dict = {}):
        results = []

        for paper_doc in paper_docs:
            self.logger.log("ScoringPaper", {"title": paper_doc.get("title")})
            score_result = self.score_paper(paper_doc, context=context)
            results.append({
                "title": paper_doc.get("title"),
                "scores": score_result,
            })

        return results
