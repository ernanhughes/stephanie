# co_ai/supervisor.py
from co_ai.agents.evolution import EvolutionAgent
from co_ai.agents.generation import GenerationAgent
from co_ai.agents.meta_review import MetaReviewAgent
from co_ai.agents.ranking import RankingAgent
from co_ai.agents.reflection import ReflectionAgent
from co_ai.logs import JSONLogger
from co_ai.memory.vector_store import VectorMemory


class Supervisor:
    def __init__(self, cfg, memory=VectorMemory(), logger=JSONLogger()):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

    async def run_pipeline_config(self, goal: str, run_id: str = "default", use_grafting: bool = False):
        gen_agent = GenerationAgent(self.cfg, self.memory, self.logger)
        reflect_agent = ReflectionAgent(self.cfg, self.memory, self.logger)
        rank_agent = RankingAgent(self.cfg, self.memory, self.logger)
        evolve_agent = EvolutionAgent(self.cfg, self.memory, self.logger)
        review_agent = MetaReviewAgent(self.cfg, self.memory, self.logger)

        print(f"[Pipeline] {run_id} Generating hypotheses for: {goal}")
        generated = await gen_agent.run({"goal": goal})

        print("[Pipeline] {run_id} Reflecting on hypotheses...")
        reflected = await reflect_agent.run({"hypotheses": generated["hypotheses"]})

        print("[Pipeline] {run_id}  Ranking hypotheses...")
        ranked = await rank_agent.run({"reviewed": reflected["reviewed"]})

        print("[Pipeline] {run_id} Evolving hypotheses...")
        evolved = await evolve_agent.run({"ranked": ranked["ranked"], "use_grafting": use_grafting})

        print("[Pipeline] Summarizing results...")
        summary = await review_agent.run({"evolved": evolved["evolved"]})

        return summary
