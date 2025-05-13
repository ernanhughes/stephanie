# co_ai/supervisor.py

import hydra
from omegaconf import DictConfig, OmegaConf

from co_ai.logs.json_logger import JSONLogger
from co_ai.memory import MemoryTool
from co_ai.reports import ReportFormatter


class PipelineStage:
    def __init__(self, name: str, config: dict, stage_dict: dict):
        self.name = name
        self.cls = config.get("cls", "") 
        self.enabled = config.get("enabled", True)
        self.iterations = config.get("iterations", 1)
        self.stage_dict = stage_dict


class Supervisor:
    def __init__(self, cfg, memory=None, logger=None):
        self.cfg = cfg
        self.memory = memory or MemoryTool(cfg=cfg.db, logger=logger)
        self.logger = logger or JSONLogger(log_path=cfg.logger.log_path)
        self.logger.log("SupervisorInit", {"cfg": cfg})

        # Parse pipeline stages from config
        self.pipeline_stages = self._parse_pipeline_stages(cfg.pipeline.stages)

    def _parse_pipeline_stages(self, stage_configs: list[dict[str, any]]) -> list[PipelineStage]:
            """Parse and validate pipeline stages from config."""
            stages = []
            for stage_config in stage_configs:
                name = stage_config.name
                if not stage_config.enabled:
                    print(f"Skipping disabled stage: {name}")
                    continue
                stage_dict = self.cfg.agents[name]
                print(f"Stage dict: {stage_dict}")
                stages.append(PipelineStage(name, stage_config, stage_dict))
            return stages

    async def run_pipeline_config(self, input_data: dict) -> dict:
        """
        Run all stages defined in config.
        Each stage loads its class dynamically via hydra.utils.get_class()
        """
        self.logger.log("PipelineStart", input_data)
        
        # Start with empty context
        context = input_data.copy()
        context["prompts_dir"] = self.cfg.paths.prompts

        try:
            for stage in self.pipeline_stages:
                if not stage.enabled:
                    self.logger.log("PipelineStageSkipped", {
                        "stage": stage.name,
                        "reason": "disabled_in_config"
                    })
                    continue
                cls = hydra.utils.get_class(stage.cls)
                stage_dict = OmegaConf.to_container(stage.stage_dict, resolve=True)
                agent = cls(cfg=stage_dict, memory=self.memory, logger=self.logger)
                if not agent:
                    self.logger.log("PipelineStageError", {
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
                        # "context_snapshot": {k: v[:3] if isinstance(v, list) else v for k, v in context.items()}
                    })


                self.logger.log("PipelineStageEnd", {"stage": stage.name})
                self.logger.log("ContextAfterStage", {
                    "stage": stage.name,
                    "context_keys": list(context.keys())
            })

            self.logger.log("PipelineSuccess", context)
            return context

        except Exception as e:
            self.logger.log("PipelineError", {"error": str(e)})
            raise

    def generate_report(self, context: dict[str, any], run_id: str) -> str:
        """Generate a report based on the pipeline context."""
        formatter = ReportFormatter(self.cfg.report.path)
        report = formatter.format_report(context)
        self.memory.report.log(context.get("run_id", ""), context.get("goal", ""), report, self.cfg.report.path)
        self.logger.log("ReportGenerated", {
            "run_id": run_id,
            "report_snippet": report[:100]
        })
        return report
