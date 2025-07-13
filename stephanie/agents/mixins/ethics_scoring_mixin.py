# stephanie/agents/mixins/ethics_scoring_mixin.py

from stephanie.analysis.paper_score_evaluator import PaperScoreEvaluator


class EthicsScoringMixin:
    def score_ethics(self, doc: dict, context: dict = None) -> dict:
        context = context or {}
        context["ethics_score"] = doc

        if not hasattr(self, "call_llm"):
            raise AttributeError("Agent must implement `call_llm(prompt, context)`")

        evaluator = PaperScoreEvaluator.from_file(
            filepath=self.cfg.get("score_config", "config/scoring/ethics.yaml"),
            prompt_loader=self.prompt_loader,
            cfg=self.cfg,
            logger=self.logger,
            memory=self.memory,
        )

        return evaluator.evaluate(
            document=doc,
            context=context,
            llm_fn=self.call_llm,
            text_to_evaluate="summary",
        )
