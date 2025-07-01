# stephanie/agents/debate.py
from stephanie.agents.base_agent import BaseAgent


class OptimistDebater(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = context.get("hypotheses", [])
        reviews = []

        for h in hypotheses:
            prompt = (
                f"As an optimistic analyst, critique the following hypothesis:\n\n"
                f"{h}\n\n"
                f"Focus on strengths, positive implications, and reasons it might be valid."
            )
            review = self.call_llm(prompt, context)
            reviews.append({"hypotheses": h, "review": review, "persona": "Optimist"})

        return {"reviews": reviews}

class SkepticDebater(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = context.get("hypotheses", [])
        reviews = []

        for h in hypotheses:
            prompt = (
                f"As a skeptical analyst, critique the following hypothesis:\n\n"
                f"{h}\n\n"
                f"Focus on weaknesses, uncertainties, or reasons it might be flawed."
            )
            review = self.call_llm(prompt, context)
            reviews.append({"hypotheses": h, "review": review, "persona": "Skeptic"})

        return {"reviews": reviews}

class BalancedDebater(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = context.get("hypotheses", [])
        reviews = []

        for h in hypotheses:
            prompt = (
                f"As a balanced analyst, critique the following hypothesis:\n\n"
                f"{h}\n\n"
                f"Provide both positive and negative aspects."
            )
            review = self.call_llm(prompt, context)
            reviews.append({"hypotheses": h, "review": review, "persona": "Balanced"})

        return {"reviews": reviews}
    
class DebateAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.optimist = OptimistDebater(cfg, memory, logger)
        self.skeptic = SkepticDebater(cfg, memory, logger)
        self.balanced = BalancedDebater(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = context.get("hypotheses", [])
        optimist_reviews = await self.optimist.run({"hypotheses": hypotheses})
        skeptic_reviews = await self.skeptic.run({"hypotheses": hypotheses})
        balanced_reviews = await self.balanced.run({"hypotheses": hypotheses})

        return {
            "optimist_reviews": optimist_reviews,
            "skeptic_reviews": skeptic_reviews,
            "balanced_reviews": balanced_reviews
        }
    
