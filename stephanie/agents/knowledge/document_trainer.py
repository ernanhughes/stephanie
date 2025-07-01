from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.document_pair_builder import DocumentPreferencePairBuilder
from stephanie.scoring.document_mrq_trainer import DocumentMRQTrainer
from stephanie.scoring.document_value_predictor import DocumentValuePredictor
from stephanie.evaluator.text_encoder import TextEncoder
import torch
import os
import torch
import json

class DocumentTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_save_path = cfg.get("model_save_path", "/models")
        self.model_prefix = cfg.get("model_prefix", "document_rm_")

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")
        builder = DocumentPreferencePairBuilder(db=self.memory.session, logger=self.logger)
        training_pairs = builder.get_training_pairs_by_dimension(goal=goal_text)

        all_contrast_pairs = []

        for dimension, pairs in training_pairs.items():
            for item in pairs:
                contrast_pair = {
                    "title": item["title"],         # Can be title or goal
                    "output_a": item["output_a"],
                    "output_b": item["output_b"],
                    "value_a": item["value_a"],
                    "value_b": item["value_b"],
                    "dimension": dimension
                }
                all_contrast_pairs.append(contrast_pair)
                self.logger.log("DocumentPairBuilderProgress", {
                    "dimension": dimension,
                    "pairs_count": len(pairs)
                })

        self.logger.log("DocumentPairBuilderComplete", {
            "dimensions": list(training_pairs.keys()),
            "total_pairs": sum(len(p) for p in training_pairs.values())
        })

        trainer = DocumentMRQTrainer(
            memory=self.memory,          # your MemoryTool
            logger=self.logger,          # your logger
            encoder=TextEncoder(),  # optional
            value_predictor=DocumentValuePredictor(),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        config = {
            "epochs": 10,
            "lr": 1e-4,
            "patience": 2,
            "min_delta": 0.001
        }

        assert isinstance(all_contrast_pairs, list), "Expected list for contrast pairs"
        assert len(all_contrast_pairs) > 0, "No contrast pairs found"


        trained_models, regression_tuners = trainer.train_multidimensional_model( all_contrast_pairs, cfg=config)
        self.logger.log("DocumentTrainingComplete", {
            "dimensions": list(trained_models.keys()), 
            "total_models": len(trained_models) 
        })

        os.makedirs(self.model_save_path, exist_ok=True)

        for dim, state_dict in trained_models.items():
            model_path = os.path.join(self.model_save_path, f"{self.model_prefix}{dim}.pt")
            torch.save(state_dict, model_path)
            self.logger.log("ModelSaved", {"dimension": dim, "path": model_path})

            # Save corresponding tuner
            tuner = regression_tuners.get(dim)
            if tuner:
                tuner_path = os.path.join(self.model_save_path, f"{self.model_prefix}{dim}_tuner.json")
                tuner.save(tuner_path) 
                self.logger.log("TunerSaved", {"dimension": dim, "path": tuner_path})

        context[self.output_key] = training_pairs
        self.logger.log("DocumentPairBuilderComplete", {
            "dimensions": list(training_pairs.keys()),
            "total_pairs": sum(len(p) for p in training_pairs.values())
        })
        return context