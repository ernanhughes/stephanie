from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, HYPOTHESES, REFLECTION


class ReflectionAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, "")
        hypotheses = self.get_hypotheses(goal, context)
        # Run reflection logic
        reflections = []
        for h in hypotheses:
            self.logger.log("ReflectingOnHypothesis", {HYPOTHESES: h})
            
            prompt = self.prompt_loader.load_prompt(self.cfg, {
                **context,
                **{HYPOTHESES: h}
            })

            response = self.call_llm(prompt, context).strip()
            self.memory.hypotheses.store_reflection(h, response)
            reflections.append({HYPOTHESES: h, REFLECTION: response})

            if self.source == "database":
                self.logger.log("BatchReflectionComplete", {
                    "goal": goal,
                    "reflected_count": len(reflections),
                    "preferences_used": context.get("preferences", [])
                })
            context[REFLECTION] = reflections
            self.logger.log("GeneratedReflection", {
                "goal": goal,
                "reflected_count": len(reflections)
            })
        return context

    def get_hypotheses(self, goal:str, context:dict) -> list[str]:
        if self.source == "context":
            hypotheses = context.get(HYPOTHESES, [])
            if not hypotheses:
                self.logger.log("NoHypothesesInContext", {"agent": self.name})
        elif self.source == "database":
            hypotheses = self.memory.hypotheses.get_unreflected(goal=goal, limit=self.batch_size)
            if not hypotheses:
                self.logger.log("NoUnReflectedInDatabase", {"agent": self.name})
            return []
        else:
            self.logger.log("InvalidSourceConfig", {"source": self.source})
        return []
