from stephanie.compiler.prompt_evolver import PromptEvolver


class PromptEvolverMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_evolver = None  # Will be initialized on first use

    def init_evolver(self, llm, logger=None):
        self.prompt_evolver = PromptEvolver(llm, logger=logger)

    def evolve_prompts(self, examples: list[dict], context: dict = {}, sample_size: int = 10) -> list[str]:
        return self.prompt_evolver.evolve(examples, context=context, sample_size=sample_size)
