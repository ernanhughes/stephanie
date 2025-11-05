from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from scripts.train_vpm_thought_model import (  # reuse your iterable + loop
    GRPOTrainer, VPMThoughtDataset)
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.model.vpm_thought_policy import (VPMThoughtModelConfig,
                                                        VPMThoughtPolicy)


class VPMThoughtTrainerAgent(BaseAgent):
    """
    Thin agent wrapper that reuses your GRPO loop but logs via Stephanie + saves ckpts in expected places.
    """
    def __init__(self, cfg: DictConfig, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        mcfg = VPMThoughtModelConfig(**cfg.model)
        self.model = VPMThoughtPolicy(mcfg)
        self.device = torch.device(cfg.training.device)
        self.model.to(self.device)
        self.out_dir = Path(cfg.paths.models_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # dataset & dataloader
        ds = VPMThoughtDataset(self.cfg, curriculum_step=self.cfg.curriculum.steps[0])
        dl = DataLoader(ds, batch_size=int(self.cfg.training.batch_size), num_workers=int(self.cfg.training.num_workers))
        # trainer
        t = GRPOTrainer(self.model, self.cfg)  # uses same optimizer/logging as your script
        # a single stage for now (you can loop like in your script)
        loss = t.train_epoch(dl, epoch_idx=0)
        self.logger.log("VPMThoughtTrainer", {"loss": float(loss)})

        ckpt = self.out_dir / "vpm_thought_stephanie_final.pt"
        torch.save({
            "model_state": self.model.state_dict(),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }, ckpt)
        return {"status": "ok", "checkpoint": str(ckpt)}
