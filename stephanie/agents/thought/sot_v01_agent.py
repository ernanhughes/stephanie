# stephanie/agents/thought/sot_v01_agent.py
from __future__ import annotations
from stephanie.agents.base_agent import BaseAgent

from stephanie.agents.thought.sot_v01_dataset_builder_pg import SoTV01DatasetBuilderPg
from stephanie.agents.thought.sot_v01_trainer import SoTV01Trainer

class SoTV01Agent(BaseAgent):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.dataset_path = cfg.get("dataset_path", "data/sot_v01.jsonl")
        self.model_path   = cfg.get("model_path",   "models/sot_v01")
        self.max_convs    = cfg.get("max_conversations", 1000)
        self.window       = cfg.get("window", 2)

    async def run(self, context: dict) -> dict:
        # Build fast with pgvector + span exclusion
        builder = SoTV01DatasetBuilderPg(self.memory.session, self.memory, self.logger, window=self.window)
        builder.build_dataset(self.dataset_path, max_conversations=self.max_convs, top_spans=3)

        # Train LoRA multi-task
        trainer = SoTV01Trainer(model_name=self.cfg.get("model_name","Qwen/Qwen2.5-0.5B"))
        trainer.train(self.dataset_path, self.model_path, epochs=self.cfg.get("epochs",2), batch_size=self.cfg.get("batch_size",4))

        context["sot_v01_trained"] = True
        context["sot_v01_model"] = self.model_path
        return context
