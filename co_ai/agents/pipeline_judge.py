import re

from tabulate import tabulate

from co_ai.agents.base import BaseAgent
from co_ai.analysis.rule_analytics import RuleAnalytics
from co_ai.analysis.rule_effect_analyzer import RuleEffectAnalyzer
from co_ai.constants import GOAL, PIPELINE, PIPELINE_RUN_ID, RUN_ID
from co_ai.models import EvaluationORM


def extract_dimensions(text: str) -> dict:
    dimensions = {}
    for line in text.splitlines():
        match = re.match(r"(\w+)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", line.strip(), re.IGNORECASE)
        if match:
            key = match.group(1).strip().lower()
            val = float(match.group(2))
            dimensions[key] = val
    return dimensions

class PipelineJudgeAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.print_results = cfg.get("print_results", True)

    async def run(self, context: dict) -> dict:
        self.logger.log("PipelineJudgeAgentStart", {PIPELINE_RUN_ID: context.get(PIPELINE_RUN_ID)})

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
                "lookahead": reflection,
                **context
            }

            prompt = self.prompt_loader.load_prompt(self.cfg, prompt_context)
            self.logger.log("PromptLoaded", {"prompt": prompt[:200]})

            judgement = self.call_llm(prompt, prompt_context).strip()
            self.logger.log("JudgementReceived", {"judgement": judgement[:250]})

            # Score extraction
            # Extract main score and rationale
            score_match = re.search(r"(?:\*\*?score[:=]?\*\*?\s*)?([0-9]+(?:\.[0-9]+)?)", judgement, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1))
                rationale = judgement[score_match.end():].strip()
                dimensions = extract_dimensions(rationale)
                self.logger.log("ScoreParsed", {
                    "score": score,
                    "dimensions": dimensions,
                    "rationale": rationale[:100]
                })
            else:
                score = None
                rationale = judgement
                dimensions = {}
                self.logger.log("PipelineScoreParseFailed", {
                    "agent": self.name,
                    "judgement": judgement,
                    "goal_id": goal.get("id"),
                    RUN_ID: context.get(RUN_ID),
                })

            # Save score
            score_obj = EvaluationORM(
                goal_id=self.get_goal_id(goal),
                hypothesis_id=self.get_hypothesis_id(hypo),
                agent_name=self.name,
                model_name=self.model_name,
                evaluator_name="PipelineJudgeAgent",
                scores=dimensions,
                pipeline_run_id=context.get(PIPELINE_RUN_ID),
                extra_data={"raw_response": judgement},
            )
            self.memory.evaluations.insert(score_obj)
            self.logger.log("ScoreSaved", {"score_id": score_obj.id, "run_id": context.get(RUN_ID)})

            # Update rule applications (if matched)
            applications = self.memory.rule_effects.get_by_run_and_goal(
                context.get(PIPELINE_RUN_ID), goal.get("id")
            )

            for app in applications:
                if app.hypothesis_id == self.get_hypothesis_id(hypo):
                    app.result_score = score
                    app.result_label = "pipeline_judgment"
                    app.rule_application_id = app.id  # ensure linkage
                    self.memory.rule_effects.update(app)
                    self.logger.log("RuleApplicationUpdated", {
                        "rule_id": app.rule_id,
                        "score": score,
                        "hypothesis_id": app.hypothesis_id
                    })

        self.report_rule_analytics()
        self.run_rule_effects_evaluation(context)

        self.logger.log("PipelineJudgeAgentEnd", {"output_key": self.output_key})
        return context

    def report_rule_analytics(self):
        analytics = RuleAnalytics(db=self.memory, logger=self.logger)
        results = analytics.analyze_all_rules()

        if isinstance(results, list) and self.print_results:
            print("\n=== Rule Analytics Summary ===")
            print(f"{'Rule ID':<10}{'Applications':<15}{'Avg Score':<12}")
            print("-" * 40)
            for result in results:
                rule_id = result.get("rule_id")
                count = result.get("count", 0)
                avg_score = result.get("avg_score", 0.0)
                print(f"{rule_id:<10}{count:<15}{avg_score:<12.2f}")
            print("-" * 40)

    from tabulate import tabulate

    def run_rule_effects_evaluation(self, context: dict):
        analyzer = RuleEffectAnalyzer(session=self.memory.session, logger=self.logger)
        summary = analyzer.analyze(context.get(PIPELINE_RUN_ID))

        # Sort by average score descending
        top_rules = sorted(
            summary.items(), key=lambda x: x[1]["avg_score"], reverse=True
        )

        # Prepare table data
        table_data = []
        for rule_id, data in top_rules[:5]:
            table_data.append(
                [
                    rule_id,
                    f"{data['avg_score']:.2f}",
                    data["count"],
                    f"{data['min']} / {data['max']}",
                    f"{data['std']:.2f}",
                    f"{data['success_rate']:.2%}",
                ]
            )

        # Define headers
        headers = [
            "Rule ID",
            "Avg Score",
            "Count",
            "Min / Max",
            "Std Dev",
            "Success Rate ≥50",
        ]

        # Print table
        print("\n📈 Top Performing Rules:")
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

        analyzer.pipeline_run_scores(context=context)
