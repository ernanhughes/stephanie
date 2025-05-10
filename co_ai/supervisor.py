# co_ai/supervisor.py

from typing import Any, Dict, List

from co_ai.agents.base import BaseAgent
from co_ai.agents.evolution import EvolutionAgent
from co_ai.agents.generation import GenerationAgent
from co_ai.agents.meta_review import MetaReviewAgent
from co_ai.agents.ranking import RankingAgent
from co_ai.agents.reflection import ReflectionAgent
from co_ai.logs.json_logger import JSONLogger
from co_ai.memory.vector_store import VectorMemory
from co_ai.utils.report_formatter import ReportFormatter

class PipelineStage:
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.enabled = config.get("enabled", True)
        self.iterations = config.get("iterations", 1)

class Supervisor:
    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory or VectorMemory(cfg=cfg.db, logger=logger)
        self.logger = logger or JSONLogger(log_path=cfg.logger.log_path)
        self.logger.log("SupervisorInit", {"cfg": cfg})

        self.agent_map = {
            "generation": GenerationAgent(cfg.agents.generation, self.memory, self.logger),
            "reflection": ReflectionAgent(cfg.agents.reflection, self.memory, self.logger),
            "ranking": RankingAgent(cfg.agents.ranking, self.memory, self.logger),
            "evolution": EvolutionAgent(cfg.agents.evolution, self.memory, self.logger),
            "meta_review": MetaReviewAgent(cfg.agents.meta_review, self.memory, self.logger),
        }
        # Parse pipeline stages from config
        self.pipeline_stages = self._parse_pipeline_stages(cfg.pipeline.stages)


    def _load_agent(self, name: str) -> BaseAgent:
        """Load agent based on name from config."""
        agent_class = {
            "generation": GenerationAgent(cfg=self.cfg.agents.generation, memory=self.memory, logger=self.logger),
            "reflection": ReflectionAgent(cfg=self.cfg.agents.reflection, memory=self.memory, logger=self.logger),
            "ranking": RankingAgent(cfg=self.cfg.agents.ranking, memory=self.memory, logger=self.logger),
            "evolution": EvolutionAgent(cfg=self.cfg.agents.evolution, memory=self.memory, logger=self.logger),
            "meta_review": MetaReviewAgent(cfg=self.cfg.agents.meta_review, memory=self.memory, logger=self.logger),
            # "debate": "co_ai.agents.debate.DebateAgent",  # Optional
        }[name]

        # Example using dynamic imports (you could also use hydra.utils.instantiate())
        module_name, class_name = agent_class.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls(self.cfg.agent[name], self.memory, self.logger)

    def _parse_pipeline_stages(self, stage_configs: List[Dict[str, Any]]) -> List[PipelineStage]:
            """Parse and validate pipeline stages from config."""
            stages = []
            for stage_config in stage_configs:
                name = stage_config.get("name")
                if name not in self.agent_map:
                    raise ValueError(f"Unknown agent '{name}' in pipeline config.")
                stages.append(PipelineStage(name, stage_config))
            return stages

    async def run_pipeline_config(self, goal: str, run_id: str = "default") -> Dict[str, Any]:
        """Run the pipeline based on config-defined stages."""
        self.logger.log("PipelineStart", {"run_id": run_id, "goal": goal})
        
        # Start with empty context
        context = {"goal": goal, "run_id": run_id}

        try:
            for stage in self.pipeline_stages:
                if not stage.enabled:
                    self.logger.log("PipelineStageSkipped", {
                        "stage": stage.name,
                        "reason": "disabled_in_config"
                    })
                    continue

                agent = self.agent_map[stage.name]
                if not agent:
                    self.logger.log("PipelineStageSkipped", {
                        "stage": stage.name,
                        "reason": "agent_not_found"
                    })
                    continue    
                
                self.logger.log("PipelineStageStart", {"stage": stage.name})

                for i in range(stage.iterations):
                    self.logger.log("PipelineIterationStart", {
                        "stage": stage.name,
                        "iteration": i + 1
                    })

                    # Pass context and expect it back modified
                    context = await agent.run(context)

                    self.logger.log("PipelineIterationEnd", {
                        "stage": stage.name,
                        "iteration": i + 1,
                        "context_snapshot": {k: v[:3] if isinstance(v, list) else v for k, v in context.items()}
                    })


                self.logger.log("PipelineStageEnd", {"stage": stage.name})
                self.logger.log("ContextAfterStage", {
                    "stage": stage.name,
                    "context_keys": list(context.keys())
            })

            self.logger.log("PipelineSuccess", {"run_id": run_id})
            return context

        except Exception as e:
            self.logger.log("PipelineError", {"error": str(e)})
            raise

    def generate_report(self, context: Dict[str, Any], run_id: str) -> str:
        """Generate a report based on the pipeline context."""
        formatter = ReportFormatter(self.cfg.report.path)
        report = formatter.format_report(context)
        self.memory.store_report(context.get("run_id", ""), context.get("goal", ""), report, self.cfg.report.path)
        self.logger.log("ReportGenerated", {
            "run_id": run_id,
            "report_snippet": report[:100]
        })
        return report
