from pyexpat import features

from co_ai.agents import BaseAgent
from co_ai.evaluator.mrq_self_evaluator import MRQSelfEvaluator

from co_ai.constants import GOAL


class ChainOfThoughtGeneratorAgent(BaseAgent):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.evaluator = MRQSelfEvaluator(memory, logger)
        self.num_candidates = cfg.get("num_candidates", 2)
        self.rubrics = self._load_enabled_rubrics()    

    async def run(self, context: dict):
        goal = context.get(GOAL)

        self.evaluator.train_from_database(goal=goal, cfg=self.cfg)

        prompt = self.prompt_loader.load_prompt(self.cfg, context)

        # Step 1: Generate candidates
        candidates = [
            self.call_llm(prompt, context) for _ in range(self.num_candidates)
        ]

        # Step 2: Evaluate pairwise with MRQ (extendable to tournament later)
        best = candidates[0]
        scores = {}
        for candidate in candidates[1:]:
            best, scores = self.evaluator.evaluate(
                prompt=prompt,
                goal=goal,
                output_a=best,
                output_b=candidate,
            )

        # Step 3: Classify the winning hypothesis
        pattern = self.classify(prompt, best, context)

        self.logger.log(
            "CoTClassifierResult",
            {
                "prompt": prompt,
                "best_output": best,
                "candidates": candidates,
                "pattern": pattern,
            },
        )

        # Step 5: Store in memory (optional but recommended)
        value_a = scores.get("value_a", 0)
        value_b = scores.get("value_b", 0)
        score = max(value_a, value_b)

        features = {
                "prompt": prompt,
                "best_output": best,
                "candidates": candidates,
                "pattern": pattern,
            }

        self.memory.hypotheses.store(goal, best, score, None, features, prompt)
        context[self.output_key] = [best]

        return context

    def classify(self, prompt, cot_response, context:dict):
        """Analyze the pattern of a single CoT."""
        results = {}
        pattern_file = self.cfg.get("pattern_prompt_file", "cot_pattern.txt")
        for rubric in self.rubrics:
            rubric["goal"] = prompt
            rubric["hypotheses"] = cot_response
            merged = {**context, **rubric}
            prompt_text = self.prompt_loader.from_file(
                pattern_file,
                self.cfg,
                merged
            )
            result = self.call_llm(prompt_text, merged)
            results[rubric["dimension"]] = result
        return results

    def _load_enabled_rubrics(self):
        """Load and return only the enabled rubrics as a structured list."""
        enabled_rubrics = []
        rubrics_cfg = self.cfg.get("rubrics", [])
        for entry in rubrics_cfg:
            if entry.get("enabled", False):
                enabled_rubrics.append({
                    "dimension": entry["dimension"],
                    "rubric": entry["rubric"],
                    "options": entry["options"]
                })
        return enabled_rubrics