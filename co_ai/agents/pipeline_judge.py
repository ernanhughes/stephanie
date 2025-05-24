import re

from co_ai.agents.base import BaseAgent
from co_ai.models import Score
from co_ai.constants import PIPELINE, RUN_ID
from dataclasses import asdict

class PipelineJudgeAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        goal = context["goal"]
        pipeline = context[PIPELINE]
        hypotheses = context.get("scored_hypotheses", []) or context.get(
            "hypotheses", []
        )
        # Get top-scoring or first hypothesis if available
        top_hypo = None
        if hypotheses:
            # top_hypo = max(hypotheses, key=lambda h: h.get("composite_score", 0))
            top_hypo = hypotheses[0]

        reflection = context.get("lookahead", {}).get("reflection", "")

        prompt_context = {
            "goal": goal["goal_text"],
            "pipeline": pipeline,
            "hypothesis": top_hypo,
            "lookahead": reflection
        }

        prompt = self.prompt_loader.load_prompt(self.cfg, prompt_context)
        judgement = self.call_llm(prompt, prompt_context).strip()

        # Parse score and rationale
        # Match "**Score: 0.8**", "Score = 0.8", etc.
        score_match = re.search(
            r"\*\*?score[:=]?\s*([0-9]*\.?[0-9]+)\*\*?", judgement, re.IGNORECASE
        )
        score = float(score_match.group(1)) if score_match else None
        rationale = (
            judgement
            if score is None
            else judgement[judgement.index(str(score)) + len(str(score)) :].strip()
        )

        # Store
        score_obj = Score(
            goal=prompt_context["goal"],
            hypothesis=prompt_context["hypothesis"],
            agent_name="PipelineJudgeAgent",
            model_name=self.cfg.get("model", {}).get("name"),
            evaluator_name="PipelineJudgeAgent",
            score_type="pipeline_judgment",
            score=score,
            rationale=rationale,
            run_id=context.get(RUN_ID),
            metadata={"raw_response": judgement},
        )
        score_obj.store(self.memory, self.logger)

        context[self.output_key] = {
            "score" : asdict(score_obj),
            "judgement": judgement
        }

        return context
