# stephanie/agents/mixins/paper_scoring_mixin.py

from stephanie.analysis.paper_score_evaluator import PaperScoreEvaluator


class PaperScoringMixin:
    def score_paper(self, paper_doc: dict, context: dict = None) -> dict:
        context = context or {}
        context["paper_score"] = paper_doc

        if not hasattr(self, "call_llm"):
            raise AttributeError("Agent must implement `call_llm(prompt, context)`")

        evaluator = PaperScoreEvaluator.from_file(
            filepath=self.cfg.get("score_config", "config/scoring/paper_review.yaml"),
            prompt_loader=self.prompt_loader,
            cfg=self.cfg,
            logger=self.logger,
            memory=self.memory,
            llm_fn=self.call_llm,
        )

        scores = evaluator.evaluate(
            document=paper_doc, context=context, llm_fn=self.call_llm
        )
        return scores

