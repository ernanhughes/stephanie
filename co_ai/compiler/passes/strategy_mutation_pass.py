# co_ai/compiler/passes/strategy_mutation_pass.py

class StrategyMutationPass:
    def __init__(self, cfg:dict, compiler=None, evaluator=None, logger=None):
        self.cfg = cfg    
        self.compiler = compiler
        self.evaluator = evaluator
        self.logger = logger

    def apply(self, base_prompt: str, metadata: dict) -> list[dict]:
        """Generate and score prompt mutations using the evaluator."""
        mutations = []

        # Example symbolic mutations (could be expanded)
        candidates = [
            base_prompt.replace("Let's think step by step.", "Let's work through this carefully."),
            base_prompt + "\nProvide a rationale before giving your final answer.",
            base_prompt.replace("explain", "analyze"),
        ]

        for variant in candidates:
            try:
                score = self.evaluator.evaluate(variant, metadata=metadata) if self.evaluator else 0
                mutations.append({"prompt": variant, "score": score})
            except Exception as e:
                if self.logger:
                    self.logger.log("StrategyMutationEvalError", {
                        "error": str(e),
                        "prompt_snippet": variant[:100],
                    })

        # Optionally sort by score descending
        return sorted(mutations, key=lambda x: x["score"], reverse=True)
