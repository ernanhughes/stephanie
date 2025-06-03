import re
from dataclasses import asdict
from datetime import datetime

from co_ai.agents.base import BaseAgent
from co_ai.constants import PIPELINE, RUN_ID
from co_ai.models import RuleApplicationORM, ScoreORM


class PipelineJudgeAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        self.logger.log("PipelineJudgeAgentStart", {"run_id": context.get(RUN_ID)})

        goal = context["goal"]
        pipeline = context[PIPELINE]
        hypotheses = context.get("scored_hypotheses") or context.get("hypotheses") or []

        self.logger.log("HypothesesReceived", {
            "count": len(hypotheses),
            "source": "scored_hypotheses" if context.get("scored_hypotheses") else "hypotheses"
        })

        if not hypotheses:
            self.logger.log("JudgementSkipped", {
                "error": "No hypotheses found",
                "goal_id": goal.get("id"),
                "run_id": context.get(RUN_ID)
            })
            return context

        top_hypo = hypotheses[0]
        reflection = context.get("lookahead", {}).get("reflection", "")

        prompt_context = {
            "goal": goal,
            "pipeline": pipeline,
            "hypothesis": top_hypo,
            "lookahead": reflection,
        }

        prompt = self.prompt_loader.load_prompt(self.cfg, prompt_context)
        self.logger.log("PromptLoaded", {"prompt": prompt[:200]})

        judgement = self.call_llm(prompt, prompt_context).strip()
        self.logger.log("JudgementReceived", {"judgement": judgement[:250]})

        # Score extraction
        score_match = re.search(
            r"\*\*?score[:=]?\*\*?\s*([0-9]+(?:\.[0-9]+)?)", judgement, re.IGNORECASE
        )

        if not score_match:
            score = None
            rationale = judgement
            self.logger.log("ScoreParseFailed", {
                "agent": self.name,
                "judgement": judgement,
                "goal_id": goal.get("id"),
                "run_id": context.get(RUN_ID),
                "emoji": "🚨❓🧠"
            })
        else:
            score = float(score_match.group(1))
            rationale = judgement[score_match.end():].strip()
            self.logger.log("ScoreParsed", {"score": score, "rationale": rationale[:100]})

        # Link rule application if available
        rule_application_id = context.get("symbolic_rule_application_id")

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
            rule_application_id=rule_application_id,
            metadata={"raw_response": judgement}
        )

        self.memory.scores.insert(score_obj)
        self.logger.log("ScoreSaved", {
            "score_id": score_obj.id,
            "run_id": context.get(RUN_ID),
            "rule_application_id": rule_application_id,
        })

        context[self.output_key] = {
            "score": score_obj.to_dict(),
            "judgement": judgement
        }

        self.logger.log("PipelineJudgeAgentEnd", {"output_key": self.output_key})
        return context
