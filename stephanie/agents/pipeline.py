# stephanie/agents/pipeline.py
import json
import re
from dataclasses import asdict
from datetime import datetime

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import PIPELINE, PIPELINE_RUN_ID, RUN_ID
from stephanie.models import EvaluationORM, RuleApplicationORM


class PipelineJudgeAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        self.logger.log("PipelineJudgeAgentStart", {"run_id": context.get(RUN_ID)})

        goal = context["goal"]
        pipeline = context[PIPELINE]
        hypotheses = context.get("scored_hypotheses") or context.get("hypotheses") or []

        self.logger.log(
            "HypothesesReceived",
            {
                "count": len(hypotheses),
                "source": "scored_hypotheses"
                if context.get("scored_hypotheses")
                else "hypotheses",
            },
        )

        if not hypotheses:
            self.logger.log(
                "JudgementSkipped",
                {
                    "error": "No hypotheses found",
                    "goal_id": goal.get("id"),
                    "run_id": context.get(RUN_ID),
                },
            )
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
        self.logger.log("JudgementReceived", {"judgement": judgement[:300]})

        # Parse main score
        score_match = re.search(
            r"\*\*?score[:=]?\*\*?\s*([0-9]+(?:\.[0-9]+)?)", judgement, re.IGNORECASE
        )

        score = float(score_match.group(1)) if score_match else None
        rationale = judgement[score_match.end() :].strip() if score_match else judgement

        if score is None:
            self.logger.log(
                "ScoreParseFailed",
                {
                    "agent": self.name,
                    "judgement": judgement,
                    "goal_id": goal.get("id"),
                    "run_id": context.get(RUN_ID),
                    "emoji": "🚨❓🧠",
                },
            )
        else:
            self.logger.log(
                "ScoreParsed", {"score": score, "rationale": rationale[:100]}
            )

        # Parse extra dimensions (look for: relevance, clarity, originality, correctness, etc.)
        dimension_fields = [
            "relevance",
            "clarity",
            "originality",
            "correctness",
            "novelty",
            "feasibility",
        ]
        dimensions = {}
        for field in dimension_fields:
            match = re.search(
                rf"{field}\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", judgement, re.IGNORECASE
            )
            if match:
                dimensions[field.lower()] = float(match.group(1))

        self.logger.log("ScoreDimensionsParsed", {"dimensions": dimensions})

        # Link rule application if available
        rule_application_id = context.get("symbolic_rule_application_id")

        score_obj = EvaluationORM(
            goal_id=self.get_goal_id(goal),
            target_type="hypothesis",
            target_id=self.get_hypothesis_id(top_hypo),
            agent_name=self.name,
            model_name=self.model_name,
            evaluator_name="PipelineJudgeAgent",
            score_type="pipeline_judgment",
            score=score,
            rationale=rationale,
            pipeline_run_id=context.get(RUN_ID),
            symbolic_rule_id=rule_application_id,
            extra_data={"raw_response": judgement},
            dimensions=dimensions,  # new: parsed dimensions
        )

        self.memory.evaluations.insert(score_obj)
        self.logger.log(
            "ScoreSaved",
            {
                "score_id": score_obj.id,
                "pipeline_run_id": context.get(PIPELINE_RUN_ID),
                "rule_application_id": rule_application_id,
            },
        )

        context[self.output_key] = {
            "score": score_obj.to_dict(),
            "judgement": judgement,
        }

        self.logger.log("PipelineJudgeAgentEnd", {"output_key": self.output_key})
        return context
