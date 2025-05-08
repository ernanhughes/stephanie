# co_ai/supervisor.py
from co_ai.agents.evolution import EvolutionAgent
from co_ai.agents.generation import GenerationAgent
from co_ai.agents.meta_review import MetaReviewAgent
from co_ai.agents.ranking import RankingAgent
from co_ai.agents.reflection import ReflectionAgent
from co_ai.logs import JSONLogger
from co_ai.memory.vector_store import VectorMemory


class Supervisor:
    def __init__(self):
        self.memory = VectorMemory()
        self.logger = JSONLogger()

    async def run_pipeline_config(self, goal, run_id="default_run", use_grafting=False):
        self.logger.log({"event": "supervisor_start", "goal": goal, "run_id": run_id})

        gen_agent = GenerationAgent(self.memory, self.logger)
        reflect_agent = ReflectionAgent(self.memory, self.logger)
        rank_agent = RankingAgent(self.memory, self.logger)
        evolve_agent = EvolutionAgent(self.memory, self.logger)
        review_agent = MetaReviewAgent(self.memory, self.logger)

        generated = await gen_agent.run({"goal": goal})
        if not isinstance(generated, dict):
            self.logger.log({"event": "generation_error", "message": "Expected dict but got", "type": str(type(generated))})
            return

        self.logger.log({
            "event": "generation_complete",
            "count": len(generated.get("hypotheses", []))
        })

        reflected = await reflect_agent.run({"hypotheses": generated.get("hypotheses", [])})
        self.logger.log({"event": "reflection_complete", "count": len(reflected.get("reviewed", []))})

        ranked = await rank_agent.run({"reviewed": reflected})
        self.logger.log({"event": "ranking_complete", "ranked": ranked.get("ranked", [])[:3]})

        evolved = await evolve_agent.run({"ranked": ranked.get("ranked", []), "use_grafting": use_grafting})
        self.logger.log({"event": "evolution_complete", "count": len(evolved.get("evolved", []))})

        summary = await review_agent.run({"evolved": evolved.get("evolved", [])})
        self.logger.log({"event": "meta_review_complete", "summary": summary.get("summary")})

        self.logger.log({"event": "supervisor_complete", "run_id": run_id})

        return summary.get("summary")
