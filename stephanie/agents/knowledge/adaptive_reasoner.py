from typing import Union

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL
from stephanie.dataloaders import ARMDataLoader
from stephanie.evaluator import ARMReasoningSelfEvaluator, LLMJudgeEvaluator


class AdaptiveReasonerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        self.modes = ["adaptive", "instruction_guided", "consensus_guided"]
        self.mode = self.cfg.get("mode", "adaptive")
        self.format_list = self.cfg.get(
            "format_list", ["direct", "short_cot", "code", "long_cot"]
        )
        self.judge = self._init_judge()

    async def run(self, context: dict):
        goal = context.get(GOAL) 


        self.judge.train_from_database(goal.get("goal_text"), self.cfg)

        prompt = goal.get("goal_text")

        response = ""
        if self.mode == "instruction_guided":
            format_name = self.cfg.get("format", "long_cot")
            response = self._generate_with_format(format_name, context)
        elif self.mode == "consensus_guided":
            response = self._run_consensus_mode(context)
        else:  # default to adaptive
            response = self._run_adaptive_mode(prompt, context)

        self.logger.log("AdaptiveReasoningResponse", response)

        context[self.output_key] = response
        return context

    def _generate_with_format(self, fmt, context):
        prompt = self.prompt_loader.from_file(fmt, self.cfg, context)
        response = self.call_llm(prompt, context)
        return {
            "prompt": prompt,
            "response": response,
            "format_used": ARMDataLoader.detect_format(response) or fmt,
        }

    def _run_consensus_mode(self, context:dict):
        outputs = {}
        for fmt in ["direct", "short_cot", "code"]:
            outputs[fmt] = self._generate_with_format(fmt, context)["response"]

        responses = list(outputs.values())
        unique_responses = set(responses)

        if len(unique_responses) == 1:
            return {
                "response": responses[0],
                "format": "consensus-simple",
                "source_formats": list(outputs.keys()),
            }
        else:
            long_cot_response = self._generate_with_format("long_cot", context)
            return {
                "response": long_cot_response["response"],
                "format": "long_cot",
                "source_formats": list(outputs.keys()),
                "fallback_reason": "no_consensus",
            }

    def _run_adaptive_mode(self, prompt:str, context:dict) -> dict[str, Union[str, float]]:
        prioritized_formats = ["direct", "short_cot", "code", "long_cot"]

        scores = {}
        for fmt in prioritized_formats:
            dict_response = self._generate_with_format(fmt, context)
            response = dict_response["response"]
            base_score = self.judge.score(prompt, response)

            token_len = len(response.split())
            rarity_bonus = 1.0 / (1 + self.judge.format_freq.get(fmt, 0))

            final_score = base_score - 0.01 * token_len + rarity_bonus
            scores[fmt] = final_score
            self.judge._update_format_stats(fmt, final_score)

        best_format = max(scores, key=scores.get)
        chosen_response = self._generate_with_format(best_format, context)
        # Log decision
        self.logger.log(
            "AdaptiveModeDecision",
            {"goal": prompt, "scores": scores, "chosen": best_format},
        )

        return {
            "response": chosen_response,
            "format_used": best_format,
            "scores": scores,
        }

    def get_format_for_goal(self, goal: dict):
        if "preferred_format" in goal:
            return goal["preferred_format"]
        goal_type = goal.get("goal_type", "default")
        if goal_type == "math":
            return "code"
        elif goal_type == "commonsense":
            return "short_cot"
        else:
            return "long_cot"

    def _get_prioritized_formats(self, context:dict):
        if "preferred_format" in context:
            return [context["preferred_format"]]

        priority_map = self.cfg.get("format_priority_by_difficulty", {})
        difficulty = context.get("difficulty", "default").lower()
        return priority_map.get(difficulty, priority_map.get("default", ["long_cot"]))

    def _init_judge(self):
        judge_strategy = self.cfg.get("judge", "mrq")
        if judge_strategy == "llm":
            llm = self.cfg.get("judge_model", self.cfg.get("model"))
            prompt_file = self.cfg.get(
                "judge_prompt_file", "judge_pairwise_comparison.txt"
            )
            self.logger.log(
                "EvaluatorInit", {"strategy": "LLM", "prompt_file": prompt_file}
            )
            return LLMJudgeEvaluator(
                self.cfg, llm, prompt_file, self.call_llm, self.logger
            )
        else:
            self.logger.log("EvaluatorInit", {"strategy": "ARM"})
            return ARMReasoningSelfEvaluator(self.cfg, self.memory, self.logger)

