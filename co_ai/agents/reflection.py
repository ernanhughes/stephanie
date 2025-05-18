from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, HYPOTHESES, REFLECTION


class ReflectionAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.source = cfg.get("source", "context")
        self.batch_size = cfg.get("batch_size", 10)
        self.auto_run = cfg.get("auto_run", False)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL, "")
        
        if self.source == "context":
            hypotheses = context.get(HYPOTHESES, [])
            if not hypotheses:
                self.logger.log("NoHypothesesInContext", {"agent": self.name})
                return context
        elif self.source == "database":
            hypotheses = self._get_hypotheses_from_db(goal, self.batch_size)
            if not hypotheses:
                self.logger.log("NoHypothesesInDatabase", {"agent": self.name})
                return context
        else:
            self.logger.log("InvalidSourceConfig", {"source": self.source})
            return context

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

    def _get_hypotheses_from_db(self, goal: str, limit: int = 10) -> list:
        """Get hypotheses directly from database"""
        return self.memory.hypotheses.get_unreviewed(goal=goal, limit=limit)