# stephanie/agents/maintenance/memcube_trainer_agent.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.maintenance.srft_refinement_trainer import \
    SRFTRefinementTrainer
from stephanie.agents.mixins.memcube_trainer_mixin import MemCubeTrainerMixin
from stephanie.scoring.model.ebt_model import EBTModel


class MemCubeTrainerAgent(BaseAgent, MemCubeTrainerMixin):
    """
    Agent that trains dimension-specific models using MemCube data and SRFT-style refinement.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", ["usefulness", "ethics", "clarity"])
        self.model_cfg = cfg.get("training_config", {
            "epochs": 5,
            "learning_rate": 2e-5,
            "batch_size": 8,
            "margin": 1.0
        })

    async def run(self, context: dict) -> dict:
        """
        Train models for all target dimensions using SRFT and MemCube refinement data.
        """
        trained = []
        for dimension in self.dimensions:
            self.logger.log("MemCubeTrainingStart", {"dimension": dimension})
            self.train_with_memcube(
                dimension=dimension,
                model_cls=EBTModel,
                model_cfg=self.model_cfg,
                trainer_cls=SRFTRefinementTrainer,
                memcube_type="refinement"
            )
            trained.append(dimension)

        context["trained_dimensions"] = trained
        return context
