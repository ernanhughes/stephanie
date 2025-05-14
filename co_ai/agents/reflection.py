# co_ai/agents/reflection.py
from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, HYPOTHESES, REFLECTION


class ReflectionAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, "")
        hypotheses = context.get(HYPOTHESES, [])
        reflection = []
        for h in hypotheses:
            self.logger.log("ReflectionStart", {HYPOTHESES: h})
            prompt = self.prompt_loader.load_prompt(self.cfg, {**context, **{HYPOTHESES:h}})
            reflection_response = self.call_llm(prompt)
            self.memory.hypotheses.store_reflection(h, reflection_response)
            reflection.append({REFLECTION: reflection_response})


        context[REFLECTION] = reflection
        self.logger.log("GeneratedReflection", {
            GOAL: goal,
            REFLECTION: reflection
        })

        return context
