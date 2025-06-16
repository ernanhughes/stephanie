# co_ai/scoring/paper_score_evaluator.py
from co_ai.analysis.score_evaluator import ScoreEvaluator

class PaperScoreEvaluator(ScoreEvaluator):
    def evaluate(self, paper: dict, context: dict = {}, llm_fn=None):
        if self.output_format == "cor":
            return self._evaluate_cor(paper, context, llm_fn)
        else:
            return self._evaluate_simple(paper, context, llm_fn)

    def _evaluate_cor(self, paper: dict, context: dict = {}, llm_fn=None):
        context = {**context, "paper": paper}
        return super()._evaluate_cor(paper, context, llm_fn)

    def _evaluate_simple(self, paper: dict, context: dict = {}, llm_fn=None):
        context = {**context, "paper": paper}
        return super()._evaluate_simple(paper, context, llm_fn)

    def save_score_to_memory(self, results, paper, context):
        # optionally skip or override — for now just log
        self.logger.log("PaperScoresComputed", {
            "title": paper.get("title"),
            "scores": results
        })
