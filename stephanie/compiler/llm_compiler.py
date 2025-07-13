# stephanie/compiler/llm_compiler.py
from jinja2 import BaseLoader, Environment

from stephanie.models.prompt_program import PromptProgramORM


class LLMCompiler:
    def __init__(self, llm, evaluator=None, logger=None):
        """
        llm: callable that takes prompt_text and returns a response
        evaluator: optional scoring function (e.g., MR.Q or LLM judge)
        logger: optional logging tool
        """
        self.llm = llm
        self.evaluator = evaluator
        self.logger = logger
        self.jinja_env = Environment(loader=BaseLoader())

    def render_prompt(self, program: PromptProgramORM) -> str:
        try:
            template = self.jinja_env.from_string(program.template)
            rendered = template.render(**program.inputs)
            program.prompt_text = rendered
            return rendered
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "PromptRenderError", {"error": str(e), "template": program.template}
                )
            raise

    def execute(
        self, program: PromptProgramORM, context: dict = {}
    ) -> PromptProgramORM:
        try:
            # Step 1: Render prompt
            prompt = self.render_prompt(program)

            # Step 2: Call LLM
            response = self.llm(prompt)
            program.hypothesis = response
            program.execution_trace = response  # raw output; extend if needed

            # Step 3: Score hypothesis (optional)
            if self.evaluator:
                score_result = self.evaluator.evaluate(program, context=context)
                program.score = score_result.score
                program.rationale = score_result.reason

            if self.logger:
                self.logger.log(
                    "PromptProgramExecuted",
                    {
                        "program_id": program.id,
                        "score": program.score,
                        "rationale_snippet": program.rationale[:100]
                        if program.rationale
                        else None,
                    },
                )

            return program

        except Exception as e:
            if self.logger:
                self.logger.log("PromptExecutionError", {"error": str(e)})
            raise
