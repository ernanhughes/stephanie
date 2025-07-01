import os

from stephanie.constants import GOAL


class StrategyResolver:
    def __init__(self, config):
        self.base_prompt_dir = getattr(config, "prompt_dir", "prompts/general_reasoner")
        self.default_strategy = getattr(config, "strategy", "cot")
        self.agent_name = getattr(config, "name", "general_reasoner")

    def resolve(self, context):
        """
        Determines the reasoning strategy and prompt path based on the context and config.
        Context should include goal metadata such as type, strategy, etc.

        Parameters:
            context (dict): Contains the goal object and optionally agent metadata.

        Returns:
            (str, str): Tuple of (strategy, prompt_file_path)
        """
        goal = context.get(GOAL).get("goal_text")
        is_reasoning_agent = context.get("agent_type", "reasoning") == "reasoning"

        if is_reasoning_agent:
            strategy = goal.get("reasoning_strategy", self.default_strategy)
        else:
            strategy = self.default_strategy

        filename = f"{strategy}_{self.agent_name}.j2"
        full_path = os.path.join(self.base_prompt_dir, filename)

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Prompt template not found for strategy '{strategy}': {full_path}")

        return strategy, full_path
