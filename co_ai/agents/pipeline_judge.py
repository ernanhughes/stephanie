import re

from co_ai.agents.base import BaseAgent
from co_ai.models import ScoreORM
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
        else:
            self.logger.log("JudgementSkipped", {"Error": "Non hypotheses found"})
            return context

        reflection = context.get("lookahead", {}).get("reflection", "")

        prompt_context = {
            "goal": goal,
            "pipeline": pipeline,
            "hypothesis": top_hypo,
            "lookahead": reflection
        }

        prompt = self.prompt_loader.load_prompt(self.cfg, prompt_context)
        judgement = self.call_llm(prompt, prompt_context).strip()

        # Parse score and rationale
        # Match "**Score: 0.8**", "Score = 0.8", etc.
        score_match = re.search(
            r"\*\*?score[:=]?\*\*?\s*([0-9]+(?:\.[0-9]+)?)", judgement, re.IGNORECASE
        )
        if not score_match:
            self.logger.log("⚠️ ScoreParseFailed", {
                "agent": self.name,
                "judgement": judgement,
                "goal_id": goal.get("id"),
                "run_id": context.get(RUN_ID),
                "emoji": "🚨❓🧠"
            })
            score = None
            rationale = judgement
        else:
            score = float(score_match.group(1))
            rationale_start = score_match.end()
            rationale = judgement[rationale_start:].strip()

        # Store
        score_obj = ScoreORM(
            goal_id=self.get_goal_id(goal),
            hypothesis_id=self.get_hypothesis_id(top_hypo),
            agent_name=self.name,
            model_name=self.model_name,
            evaluator_name="PipelineJudgeAgent",
            score_type="pipeline_judgment",
            score=score,
            rationale=rationale,
            run_id=context.get(RUN_ID),
            metadata={"raw_response": judgement}
        )
        self.memory.scores.insert(score_obj)

        context[self.output_key] = {
            "score" : score_obj.to_dict(),
            "judgement": judgement
        }

        return context
