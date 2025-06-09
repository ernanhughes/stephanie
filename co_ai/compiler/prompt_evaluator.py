from abc import ABC, abstractmethod


class EvaluationResult:
    def __init__(self, score: float, reason: str):
        self.score = score
        self.reason = reason

class BasePromptEvaluator(ABC):
    @abstractmethod
    def evaluate(self, program, context: dict = None) -> EvaluationResult:
        pass


class MRQPromptEvaluator(BasePromptEvaluator):
    def __init__(self, llm, prompt_loader, logger=None):
        self.llm = llm
        self.prompt_loader = prompt_loader
        self.logger = logger

    def evaluate(self, program, context: dict = None) -> EvaluationResult:
        context = context or {}
        try:
            evaluation_context = {
                **context,
                "goal": program.goal,
                "prompt": program.prompt_text,
                "hypothesis": program.hypothesis,
            }
            prompt = self.prompt_loader.load_prompt("prompt_evaluation", evaluation_context)
            response = self.llm(prompt)

            # Very basic scoring extraction
            import re
            match = re.search(r"score:(\d+(\.\d+)?)", response)
            score = float(match.group(1)) if match else 0.0

            return EvaluationResult(score=score, reason=response)

        except Exception as e:
            if self.logger:
                self.logger.log("PromptEvaluationFailed", {"error": str(e)})
            return EvaluationResult(score=0.0, reason=str(e))
