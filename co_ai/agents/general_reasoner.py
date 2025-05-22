from torch.backends.opt_einsum import strategy

from co_ai.agents.base import BaseAgent
from co_ai.evaluator import LLMJudgeEvaluator
from co_ai.evaluator import MRQSelfEvaluator
from co_ai.constants import GOAL, GOAL_TYPE  
from co_ai.models import Hypothesis
from co_ai.prompts import PromptLoader
from itertools import combinations
from co_ai.models import Score

class GeneralReasonerAgent(BaseAgent):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.logger.log("AgentInit", {"agent": "GeneralReasonerAgent"})
        self.judge = self._init_judge()
        print(self.cfg)
        self.prompt_loader = PromptLoader(self.cfg, self.logger)

    async def run(self, context: dict):
        goal = context.get(GOAL)
        self.logger.log("AgentRunStarted", {"goal": goal})

        # Generate multiple reasoning outputs
        if self.cfg.get("thinking_mode") == "generate_and_judge":
            hypotheses = self.generate_hypotheses(goal, context)
        else:
            hypotheses = self.get_hypotheses(context)


        # Evaluate each hypothesis
        win_counts = {h.id: 0 for h in hypotheses}
        evaluations = []

        prompt_loader = PromptLoader(None, self.logger)
        judging_prompt_template = self.cfg.get(
            "evaluator_prompt_file", "judge_pairwise_comparison.txt"
        )

        for hyp_a, hyp_b in combinations(hypotheses, 2):
            # Render prompt for this pair
            context = {
                "goal": goal,
                "hypothesis_a": hyp_a.text,
                "hypothesis_b": hyp_b.text,
            }
            prompt_text = prompt_loader.from_file(
                judging_prompt_template, self.cfg, context
            )

            # Call the judge
            preferred, score = self.judge.judge(
                prompt_text, goal, hyp_a.text, hyp_b.text
            )
            s = Score.build(goal, hyp_a.text, self.cfg)
            s.set_score(score["score_a"])
            self.memory.hypotheses.insert_score(s)
            s = Score.build(goal, hyp_a.text, self.cfg)
            s.set_score(score["score_b"])
            self.memory.hypotheses.insert_score(s)

            evaluations.append(score)

            winner_id = hyp_a.id if score["winner"] == "A" else hyp_b.id
            win_counts[winner_id] += 1

            self.logger.log("GeneralReasoningJudgement",
                {
                    "event": "JudgedPair",
                    "goal": goal,
                    "hypothesis_a": hyp_a.text[:100],
                    "hypothesis_b": hyp_b.text[:100],
                    "winner": score["winner"],
                    "score_a": score["score_a"],
                    "score_b": score["score_b"],
                    "reason": score["reason"],
                    "evaluator": self.cfg.get("judge", "llm"),
                }
            )

        # Select best hypothesis by win count
        best_id = max(win_counts, key=win_counts.get)
        best_hypothesis = next(h for h in hypotheses if h.id == best_id)
        best_hypothesis.evaluation = {
            "wins": win_counts[best_id],
            "judged_pairs": len(hypotheses) - 1,
        }
        context[self.output_key] = best_hypothesis
        return context

    def generate_hypotheses(self, question, context):
        # Simple loop; replace with model call w/ temperature or variations
        strategies = self.cfg.get("generation_strategy_list", ["cot"])
        merged = {**context, **{"question": question}}
        hypotheses = []
        for strategy in strategies:
            prompt = self.prompt_loader.from_file(f"strategy_{strategy}.txt", self.cfg, merged)
            response = self.call_llm(prompt, merged)
            hypothesis = Hypothesis(text=response, goal=context.get(GOAL), goal_type=context.get(GOAL_TYPE), 
                                    strategy_used=strategy, features={"strategy": strategy},
                                    source=self.name)
            self.memory.hypotheses.store(hypothesis)
            hypotheses.append(hypothesis)
        return hypotheses


    def _init_judge(self):
        if self.cfg.get("judge", "mrq") == "llm":
            llm = self.cfg.get("judge_model", self.cfg.get("model"))
            prompt_file = self.cfg.get("judge_prompt_file", "judge_pairwise_comparison.txt")
            self.logger.log("EvaluatorInit", {"strategy": "LLM", "prompt_file": prompt_file})
            return LLMJudgeEvaluator(self.cfg, llm, prompt_file, self.call_llm, self.logger)
        else:
            self.logger.log("EvaluatorInit", {"strategy": "MRQ"})
            return MRQSelfEvaluator(self.memory, self.logger)
