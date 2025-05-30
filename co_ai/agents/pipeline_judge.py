import re
from collections import defaultdict

from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, PIPELINE, PIPELINE_RUN_ID
from co_ai.models import ScoreORM, RuleApplicationORM


class PipelineJudgeAgent(BaseAgent):
    async def run(self, context: dict) -> dict:
        self.logger.log("PipelineJudgeAgentStart", {"pipeline_run_id": context.get(PIPELINE_RUN_ID)})

        goal = context["goal"]
        self.get_mrq_training(context)
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
            pipeline_run_id=context.get(PIPELINE_RUN_ID),
            rule_application_id=rule_application.id
            if rule_application
            else None,  # ✅ Link
            extra_data={"raw_response": judgement},
        )

        self.memory.scores.insert(score_obj)

        self.logger.log(
            "ScoreSaved",
            {
                "score_id": score_obj.id,
                "run_id": context.get(PIPELINE_RUN_ID),
                "linked_rule_application_id": rule_application.id
                if rule_application
                else None,
            },
        )

        self.link_score_to_rule_applications(score_obj, context)

        context[self.output_key] = {
            "score": score_obj.to_dict(),
            "judgement": judgement,
        }

        self.logger.log("PipelineJudgeAgentEnd", {"output_key": self.output_key})
        return context

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

    def get_mrq_training(self, context:dict):
        goal = context.get(GOAL)
        goal_id = goal.get("id") if goal else None

        all_scores = self.memory.scores.get_by_goal_id(goal_id=goal_id)
        grouped_by_prompt = defaultdict(list)

        for score in all_scores:
            grouped_by_prompt[score.prompt].append(score)

        pairs = []
        for prompt, scores in grouped_by_prompt.items():
            scores.sort(key=lambda s: s.score, reverse=True)
            for i in range(len(scores)):
                for j in range(i + 1, len(scores)):
                    s1, s2 = scores[i], scores[j]
                    if abs(s1.score - s2.score) < self.margin:
                        continue
                    preferred = "a" if s1.score > s2.score else "b"
                    pairs.append({
                        "prompt": prompt,
                        "output_a": s1.response,
                        "output_b": s2.response,
                        "preferred": preferred,
                        "pipeline_a": s1.metadata.get("pipeline"),
                        "pipeline_b": s2.metadata.get("pipeline"),
                        "score_a": s1.score,
                        "score_b": s2.score,
                        "goal_id": goal_id,
                    })
                    if len(pairs) >= self.max_pairs:
                        break
                if len(pairs) >= self.max_pairs:
                    break

        self.logger.log("MRQTrainingPairsExtracted", {
            "count": len(pairs),
            "goal_id": goal_id,
        })

        context["dpo_samples"] = pairs
        return context
