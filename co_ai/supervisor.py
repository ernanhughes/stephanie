# co_ai/supervisor.py
from co_ai.agents.evolution import EvolutionAgent
from co_ai.agents.generation import GenerationAgent
from co_ai.agents.meta_review import MetaReviewAgent
from co_ai.agents.ranking import RankingAgent
from co_ai.agents.reflection import ReflectionAgent
from co_ai.logs.json_logger import JSONLogger
from co_ai.memory.vector_store import VectorMemory


class Supervisor:
    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory or VectorMemory(cfg=cfg.db, logger=logger)
        self.logger = logger or JSONLogger(log_path=cfg.logger.log_path)
        self.logger.log("SupervisorInit", {"cfg": cfg})

        self.agents = {
            "generation": GenerationAgent(cfg.agents.generation, self.memory, self.logger),
            "reflection": ReflectionAgent(cfg.agents.reflection, self.memory, self.logger),
            "ranking": RankingAgent(cfg.agents.ranking, self.memory, self.logger),
            "evolution": EvolutionAgent(cfg.agents.evolution, self.memory, self.logger),
            "meta_review": MetaReviewAgent(cfg.agents.meta_review, self.memory, self.logger),
        }

    async def run_pipeline_config(self, goal: str, run_id: str = "default", use_grafting: bool = False):

        try:
            self.logger.log("PipelineStart", {"run_id": run_id, "goal": goal, "use_grafting": use_grafting})

            gen_agent = self.agents["generation"]
            reflect_agent = self.agents["reflection"]
            rank_agent = self.agents["ranking"]
            evolve_agent = self.agents["evolution"]
            review_agent = self.agents["meta_review"]

            generated = await gen_agent.run({"goal": goal})
            self.logger.log("GeneratedHypotheses", generated)

            reflected = await reflect_agent.run({"hypotheses": generated["hypotheses"]})
            self.logger.log("ReflectedHypotheses", reflected)

            ranked = await rank_agent.run({"reviewed": reflected["reviewed"]})
            self.logger.log("RankedHypotheses", ranked)

            evolved = await evolve_agent.run({"ranked": ranked["ranked"], "use_grafting": use_grafting})
            self.logger.log("EvolvedHypotheses", evolved)

            summary = await review_agent.run({"evolved": evolved["evolved"]})
            self.logger.log("Summary", summary)

            self.logger.log("PipelineSuccess", {"run_id": run_id})
            return summary

        except Exception as e:
            self.logger.log("PipelineError", {"run_id": run_id, "error": str(e)})
            raise
