# stephanie/agents/icl/base_icl_strategy.py
class BaseICLStrategy:
    def __init__(self, memory, logger, cfg):
        self.memory = memory
        self.logger = logger
        self.cfg = cfg

    def generate_prompt(self, goal, context) -> str:
        """Return a prompt with in-context examples inserted"""
        raise NotImplementedError

    def observe_response(self, prompt, response, result_quality=None):
        """Store or score the example based on outcome"""
        raise NotImplementedError

    def extract_examples(self, goal, task_type=None) -> list[dict]:
        """Get ICL examples relevant to the task"""
        raise NotImplementedError

    def describe(self) -> str:
        """Optional: describe this strategy for traceability/logging"""
        return self.__class__.__name__
