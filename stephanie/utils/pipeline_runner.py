# stephanie/utils/pipeline_runner.py

from omegaconf import OmegaConf

from stephanie.supervisor import Supervisor


class PipelineRunner:
    """
    Runs Co-AI pipeline definitions dynamically from in-memory config structures.
    Can be reused across agents or CLI tools.
    """

    def __init__(self, full_cfg, memory=None, logger=None):
        self.full_cfg = full_cfg
        self.memory = memory
        self.logger = logger

    async def run(
        self, pipeline_def: list, context: dict, tag: str = "runtime"
    ) -> dict:
        """
        Run the provided pipeline definition within the given context.

        Args:
            pipeline_def (list): List of pipeline stage configurations.
            context (dict): The goal/context to apply the pipeline to.
            tag (str): A tag used to name/track this dynamic run.

        Returns:
            dict: Run result including score and selection.
        """
        pipeline_cfg = self.inject_pipeline_config(pipeline_def, tag=tag)
        merged_cfg = OmegaConf.merge(self.full_cfg, pipeline_cfg)

        supervisor = Supervisor(merged_cfg, memory=self.memory, logger=self.logger)
        result = await supervisor.run_pipeline_config(context)

        return {
            "status": "success",
            "result": result,
            "best_score": result.get("best_score", 0.0),
            "selected": result.get("selected"),
        }

    def inject_pipeline_config(
        self, pipeline_def: list, tag: str = "runtime"
    ) -> OmegaConf:
        """
        Injects a pipeline definition into a valid OmegaConf configuration.

        Args:
            pipeline_def (list): List of agent stage definitions.
            tag (str): Optional label for the pipeline instance.

        Returns:
            OmegaConf: A new config with the injected pipeline.
        """
        try:
            full_cfg_dict = OmegaConf.to_container(self.full_cfg, resolve=True)

            full_cfg_dict["pipeline"]["tag"] = tag
            full_cfg_dict["pipeline"]["stages"] = pipeline_def
            # full_cfg_dict["agents"] = {stage["name"]: stage for stage in pipeline_def}

            result = OmegaConf.create(full_cfg_dict)

            print(
                f"Injected pipeline config: {OmegaConf.to_yaml(result, resolve=True)}"
            )
            if self.logger:
                self.logger.log(
                    "PipelineInjectionSuccess",
                    {
                        "pipeline_def": pipeline_def,
                        "tag": tag,
                        "config": result,
                    },
                )
            return result

        except Exception as e:
            if self.logger:
                self.logger.log(
                    "PipelineInjectionError",
                    {
                        "error": str(e),
                        "pipeline_def": pipeline_def,
                        "tag": tag,
                    },
                )
            raise e
