import re
from collections import defaultdict

from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, PIPELINE, PIPELINE_RUN_ID
from co_ai.evaluator import ARMReasoningSelfEvaluator, MRQSelfEvaluator
from co_ai.models import ScoreORM, RuleApplicationORM
from co_ai.analysis.rule_effect_analyzer import RuleEffectAnalyzer


class PipelineJudgeAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        self.logger.log("PipelineJudgeAgentStart", {"pipeline_run_id": context.get(PIPELINE_RUN_ID)})

        goal = context["goal"]
        pipeline = context[PIPELINE]
        hypotheses = context.get("scored_hypotheses", []) or context.get(
            "hypotheses", []
        )

        self.logger.log(
            "HypothesesReceived",
            {
                "count": len(hypotheses),
                "source": "scored_hypotheses"
                if context.get("scored_hypotheses")
                else "hypotheses",
            },
        )

        top_hypo = hypotheses[0] if hypotheses else None
        if not top_hypo:
            self.logger.log(
                "JudgementSkipped",
                {
                    "error": "No hypotheses found",
                    "goal_id": goal.get("id"),
                    "pipeline_run_id": context.get(PIPELINE_RUN_ID),
                },
            )
            return context

        reflection = context.get("lookahead", {}).get("reflection", "")
        prompt_context = {
            "pipeline": pipeline,
            "hypothesis": top_hypo.get("text"),
            "lookahead": reflection,
        }
        merged = {**context, **prompt_context}

        prompt = self.prompt_loader.load_prompt(self.cfg, merged)
        self.logger.log("PromptLoaded", {"prompt": prompt[:200]})

        judgement = self.call_llm(prompt, merged).strip()
        self.logger.log("JudgementReceived", {"judgement": judgement[:250]})

        # Extract score and rationale
        score_match = re.search(
            r"(?:\*\*?score[:=]?\*\*?\s*)?([0-9]+(?:\.[0-9]+)?)",
            judgement,
            re.IGNORECASE,
        )
        if not score_match:
            self.logger.log(
                "PipelineScoreParseFailed",
                {
                    "agent": self.name,
                    "judgement": judgement,
                    "goal_id": goal.get("id"),
                    "run_id": context.get(PIPELINE_RUN_ID),
                    "emoji": "🚨❓🧠",
                },
            )
            score = None
            rationale = judgement
        else:
            score = float(score_match.group(1))
            rationale_start = score_match.end()
            rationale = judgement[rationale_start:].strip()
            self.logger.log(
                "ScoreParsed", {"score": score, "rationale": rationale[:100]}
            )

        # Fetch latest RuleApplicationORM for this run (if it exists)
        pipeline_run_id = context.get(PIPELINE_RUN_ID)

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
            pipeline_run_id=context.get(PIPELINE_RUN_ID),
            extra_data={"raw_response": judgement},
        )

        self.memory.scores.insert(score_obj)

        self.logger.log(
            "ScoreSaved",
            {
                "score_id": score_obj.id,
                "run_id": context.get(PIPELINE_RUN_ID),
            },
        )

        context[self.output_key] = {
            "score": score_obj.to_dict(),
            "judgement": judgement,
        }

        self.run_rule_effects_evaluation(context)

        self.logger.log("PipelineJudgeAgentEnd", {"output_key": self.output_key})
        return context

    @DeprecationWarning
    def link_score_to_rule_applications(self, score_obj, context):
        goal = context.get(GOAL, {})
        goal_id = goal.get("id")
        pipeline_run_id = context.get(PIPELINE_RUN_ID)

        if not goal_id or not pipeline_run_id:
            self.logger.log(
                "RuleScoreLinkerMissingContext", {"goal_id": goal_id, "pipeline_run_id": pipeline_run_id}
            )
            return

        applications = self.memory.rule_effects.get_by_run_and_goal(pipeline_run_id, goal_id)
        if not applications:
            self.logger.log(
                "NoRuleApplicationsFound", {"run_id": pipeline_run_id, "goal_id": goal_id}
            )
            return

        for app in applications:
            app.result_score = score_obj.score
            app.result_label = score_obj.score_type
            self.memory.rule_effects.update(app)

        self.logger.log(
            "RuleApplicationsScored",
            {
                "pipeline_run_id": pipeline_run_id,
                "goal_id": goal_id,
                "score": score_obj.score,
                "count": len(applications),
            },
        )


    def run_rule_effects_evaluation(self, context: dict) -> dict:
        """
        Runs the rule effects evaluation using the MRQSelfEvaluator.
        """
        analyzer = RuleEffectAnalyzer(session=self.memory.session, logger=self.logger)
        summary = analyzer.analyze()
        top_rules = sorted(summary.items(), key=lambda x: x[1]["avg_score"], reverse=True)
        for rule_id, data in top_rules[:5]:
            print(f"Rule {rule_id}: avg {data['avg_score']:.3f} over {data['count']} applications")
