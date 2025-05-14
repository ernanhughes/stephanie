# co_ai/agents/reflection.py
from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, HYPOTHESES, REFLECTION


class ReflectionAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.cfg = cfg

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, "")
        hypotheses = context.get(HYPOTHESES, [])
        reflection = []
        for h in hypotheses:
            self.log(f"Generating reflection for: {h}")
            prompt = self.prompt_loader.load_prompt(self.cfg, {**context, **{HYPOTHESES:h}})
            reflection = self.call_llm(prompt).strip()
            self.memory.hypotheses.store_review(h, reflection)
            reflection.append({HYPOTHESES: h, "review": reflection})

        context[REFLECTION] = reflection
        self.logger.log("GeneratedReflection", {
            "goal": goal,
            "reflection": reflection
        })

        return context
