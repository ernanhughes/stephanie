from typing import Optional, Dict, Any
from co_ai.agents.evolution import EvolutionAgent
from co_ai.agents.generation import GenerationAgent
from co_ai.agents.meta_review import MetaReviewAgent
from co_ai.agents.ranking import RankingAgent
from co_ai.agents.reflection import ReflectionAgent
from co_ai.logs import JSONLogger
from co_ai.memory.vector_store import VectorMemory


class Supervisor:
    """
    Orchestrates a multi-agent AI research pipeline based on the 'generate, debate, evolve' paradigm.
    
    This class implements a pipeline inspired by scientific reasoning: generating hypotheses,
    reflecting on their validity, ranking them through simulated debate, evolving promising ideas,
    and synthesizing insights into a coherent research overview.
    """

    def __init__(
        self,
        cfg,
        memory: Optional[VectorMemory] = None,
        logger: Optional[JSONLogger] = None,
        agents: Optional[Dict[str, Any]] = None
    ):
        self.cfg = cfg
        self.memory = memory or VectorMemory()
        self.logger = logger or JSONLogger()
        self.agents = agents or {}

    def _get_agent(self, name: str):
        """Lazy-load agent instance based on name."""
        cls_map = {
            "generation": GenerationAgent,
            "reflection": ReflectionAgent,
            "ranking": RankingAgent,
            "evolution": EvolutionAgent,
            "meta_review": MetaReviewAgent,
        }
        if name in self.agents:
            return self.agents[name]
        return cls_map[name](self.cfg, self.memory, self.logger)

    async def run_pipeline_config(
        self,
        goal: str,
        run_id: str = "default",
        use_grafting: bool = False,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Run the full hypothesis generation and evolution pipeline.
        
        Args:
            goal: Research goal as natural language input.
            run_id: Identifier for this pipeline execution.
            use_grafting: Whether to graft new ideas onto existing ones.
            iterations: Number of times to iterate through the pipeline.

        Returns:
            Final summary from the meta-review agent.
        """
        self.logger.log("PipelineStart", {"run_id": run_id, "goal": goal})

        try:
            for i in range(iterations):
                self.logger.log("IterationStart", {"iteration": i + 1})

                # Stage 1: Generate hypotheses
                gen_agent = self._get_agent("generation")
                generated = await gen_agent.run({"goal": goal})
                self.logger.log("GeneratedHypotheses", generated)

                # Stage 2: Reflect on hypotheses
                reflect_agent = self._get_agent("reflection")
                reflected = await reflect_agent.run({"hypotheses": generated["hypotheses"]})
                self.logger.log("ReflectedHypotheses", reflected)

                # Stage 3: Rank hypotheses
                rank_agent = self._get_agent("ranking")
                ranked = await rank_agent.run({"reviewed": reflected["reviewed"]})
                self.logger.log("RankedHypotheses", ranked)

                # Stage 4: Evolve top hypotheses
                evolve_agent = self._get_agent("evolution")
                evolved = await evolve_agent.run({"ranked": ranked["ranked"], "use_grafting": use_grafting})
                self.logger.log("EvolvedHypotheses", evolved)

                # Stage 5: Meta-review and synthesis
                review_agent = self._get_agent("meta_review")
                summary = await review_agent.run({"evolved": evolved["evolved"]})
                self.logger.log("MetaReviewSummary", summary)

                self.logger.log("IterationEnd", {"iteration": i + 1})

            self.logger.log("PipelineSuccess", {"run_id": run_id})
            return summary

        except Exception as e:
            self.logger.log("PipelineError", {"run_id": run_id, "error": str(e)})
            raise