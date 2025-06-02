import re
from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, PIPELINE, PIPELINE_RUN_ID
from co_ai.models import ScoreORM
from co_ai.analysis.rule_effect_analyzer import RuleEffectAnalyzer
from co_ai.analysis.rule_analytics import RuleAnalytics


class PipelineJudgeAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        self.logger.log("PipelineJudgeAgentStart", {"pipeline_run_id": context.get(PIPELINE_RUN_ID)})

        goal = context[GOAL]
        pipeline = context[PIPELINE]
        hypotheses = context.get("scored_hypotheses", []) or context.get("hypotheses", [])

        self.logger.log("HypothesesReceived", {
            "count": len(hypotheses),
            "source": "scored_hypotheses" if context.get("scored_hypotheses") else "hypotheses"
        })

        for hypo in hypotheses:
            reflection = context.get("lookahead", {}).get("reflection", "")
            prompt_context = {
                "pipeline": pipeline,
                "hypothesis": hypo.get("text"),
                "lookahead": reflection
            }
            merged = {**context, **prompt_context}

            prompt = self.prompt_loader.load_prompt(self.cfg, merged)
            self.logger.log("PromptLoaded", {"prompt": prompt[:200]})

            judgement = self.call_llm(prompt, merged).strip()
            self.logger.log("JudgementReceived", {"judgement": judgement[:250]})

            score_match = re.search(r"(?:\*\*?score[:=]?\*\*?\s*)?([0-9]+(?:\.[0-9]+)?)", judgement, re.IGNORECASE)
            if not score_match:
                self.logger.log("PipelineScoreParseFailed", {
                    "agent": self.name,
                    "judgement": judgement,
                    "goal_id": goal.get("id"),
                    "run_id": context.get(PIPELINE_RUN_ID),
                    "emoji": "\ud83d\udea8\u2753\ud83e\udde0"
                })
                score = None
                rationale = judgement
            else:
                score = float(score_match.group(1))
                rationale_start = score_match.end()
                rationale = judgement[rationale_start:].strip()
                self.logger.log("ScoreParsed", {"score": score, "rationale": rationale[:100]})

            score_obj = ScoreORM(
                goal_id=self.get_goal_id(goal),
                hypothesis_id=self.get_hypothesis_id(hypo),
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

            self.logger.log("ScoreSaved", {"score_id": score_obj.id, "run_id": context.get(PIPELINE_RUN_ID)})

        self.run_rule_effects_evaluation(context)
        self.report_rule_analytics()

        self.logger.log("PipelineJudgeAgentEnd", {"output_key": self.output_key})
        return context

    def report_rule_analytics(self):
        analytics = RuleAnalytics(db=self.memory, logger=self.logger)
        results = analytics.analyze_all_rules()
        for r in results:
            print(r)

    def run_rule_effects_evaluation(self, context: dict) -> dict:
        analyzer = RuleEffectAnalyzer(session=self.memory.session, logger=self.logger)
        summary = analyzer.analyze()
        top_rules = sorted(summary.items(), key=lambda x: x[1]["avg_score"], reverse=True)
        for rule_id, data in top_rules[:5]:
            print(f"Rule {rule_id}: avg {data['avg_score']:.3f} over {data['count']} applications")
