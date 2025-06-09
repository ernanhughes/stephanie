# co_ai/compiler/prompt_mutator.py

class PromptMutator:
    """
    A utility class to generate mutated versions of a prompt using symbolic or structural transformations.
    """

    def __init__(self, strategies: list[str] = None):
        self.strategies = strategies or [
            "Think step by step.",
            "Take a skeptical perspective.",
            "Consider multiple points of view.",
            "Use a detailed explanation.",
        ]

    def mutate_with_strategies(self, base_prompt: str) -> list[str]:
        """
        Prepend various reasoning strategies to the base prompt.
        """
        return [f"{strategy} {base_prompt}" for strategy in self.strategies]

    def mutate(self, base_prompt: str, metadata: dict = None) -> list[str]:
        """
        Apply one or more symbolic mutations to a prompt.
        """
        return self.mutate_with_strategies(base_prompt)

    def mutate_with_templates(self, base_prompt: str, template_list: list[str]) -> list[str]:
        """
        Apply a custom list of templates where `{prompt}` is replaced with the base prompt.
        """
        return [template.format(prompt=base_prompt) for template in template_list]
