import re
from dataclasses import asdict
from datetime import datetime

from co_ai.agents.base import BaseAgent
from co_ai.constants import PIPELINE, RUN_ID
from co_ai.models import ScoreORM, RuleApplicationORM


class PipelineJudgeAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        self.logger.log("PipelineJudgeAgentStart", {"run_id": context.get(RUN_ID)})

        goal = context["goal"]
        pipeline = context[PIPELINE]
        hypotheses = context.get("scored_hypotheses", []) or context.get("hypotheses", [])

        self.logger.log("HypothesesReceived", {
            "count": len(hypotheses),
            "source": "scored_hypotheses" if context.get("scored_hypotheses") else "hypotheses"
        })

        top_hypo = hypotheses[0] if hypotheses else None
        if not top_hypo:
            self.logger.log("JudgementSkipped", {
                "error": "No hypotheses found",
                "goal_id": goal.get("id"),
                "run_id": context.get(RUN_ID)
            })
            return context

        reflection = context.get("lookahead", {}).get("reflection", "")
        prompt_context = {
            "pipeline": pipeline,
            "hypothesis": top_hypo.get("text"),
            "lookahead": reflection
        }
        merged = {**context, **prompt_context}

        prompt = self.prompt_loader.load_prompt(self.cfg, merged)
        self.logger.log("PromptLoaded", {"prompt": prompt[:200]})

        judgement = self.call_llm(prompt, merged).strip()
        self.logger.log("JudgementReceived", {"judgement": judgement[:250]})

        # Extract score and rationale
        score_match = re.search(
            r"\*\*?score[:=]?\*\*?\s*([0-9]+(?:\.[0-9]+)?)", judgement, re.IGNORECASE
        )
        if not score_match:
            self.logger.log("PipelineScoreParseFailed", {
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
            self.logger.log("ScoreParsed", {"score": score, "rationale": rationale[:100]})

        # Fetch latest RuleApplicationORM for this run (if it exists)
        pipeline_run_id = context.get("pipeline_run_id")
        rule_application = (
            self.memory.session.query(RuleApplicationORM)
            .filter_by(pipeline_run_id=pipeline_run_id)
            .order_by(RuleApplicationORM.applied_at.desc())
            .first()
        )

        # Save Score
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
            rule_application_id=rule_application.id if rule_application else None,  # ✅ Link
            metadata={"raw_response": judgement}
        )

        self.memory.scores.insert(score_obj)

        self.logger.log("ScoreSaved", {
            "score_id": score_obj.id,
            "run_id": context.get(RUN_ID),
            "linked_rule_application_id": rule_application.id if rule_application else None
        })

        context[self.output_key] = {
            "score": score_obj.to_dict(),
            "judgement": judgement
        }

        self.logger.log("PipelineJudgeAgentEnd", {"output_key": self.output_key})
        return context
