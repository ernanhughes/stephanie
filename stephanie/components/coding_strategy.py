# --- components/coding_strategy.py ---

class SelfAdaptiveCoder:
    def __init__(self, config):
        self.threshold = config.get("complexity_threshold", 3)

    def score_complexity(self, plan):
        # Placeholder: use LLM or rubric to rate complexity (1–5)
        return 4 if "multi-stage" in plan.lower() else 2

    def generate_code(self, plan):
        complexity = self.score_complexity(plan)
        if complexity <= self.threshold:
            return self._generate_one_pass(plan)
        else:
            return self._generate_stepwise(plan)

    def _generate_one_pass(self, plan):
        return f"# One-pass code for plan\n# {plan}"

    def _generate_stepwise(self, plan):
        steps = [f"Step {i+1}: logic" for i in range(3)]  # Placeholder decomposition
        integrated_code = "\n".join([f"# {s}" for s in steps])
        return f"# Stepwise code for complex plan\n{integrated_code}"
