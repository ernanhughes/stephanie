# stephanie/agents/maintenance/mrq_trainer.py

import os

import torch

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.mrq.preference_pair_builder import PreferencePairBuilder
from stephanie.scoring.mrq.trainer_engine import MRQTrainerEngine
from stephanie.utils.file_utils import save_json
from stephanie.utils.model_utils import get_model_path


class MRQTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_path = cfg.get("model_path", "models")
        self.model_type = cfg.get("model_type", "mrq")
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")
        self.embedding_type = self.memory.embedding.type
        self.dimensions = cfg.get("dimensions", [])
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.epochs = cfg.get("epochs", 10)
        self.lr = cfg.get("lr", 1e-4)
        self.patience = cfg.get("patience", 2)  
        self.min_delta = cfg.get("min_delta", 0.001)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        self.logger.log(
            "MRQTrainerAgentInitialized",
            {
                "dimensions": self.dimensions,
                "model_type": self.model_type,
                "target_type": self.target_type,
                "model_version": self.model_version,
                "model_path": self.model_path,
                "epochs": self.epochs,
                "lr": self.lr,
                "patience": self.patience,
                "min_delta": self.min_delta,
            },
        )

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")

        builder = PreferencePairBuilder(db=self.memory.session, logger=self.logger)
        training_pairs_by_dim = {}

        for dim in self.dimensions:
            pairs = builder.get_training_pairs_by_dimension(goal=goal_text, dim=[dim])
            training_pairs_by_dim[dim] = pairs.get(dim, [])
        
        contrast_pairs = [
            {
                "title": item["title"],
                "output_a": item["output_a"],
                "output_b": item["output_b"],
                "value_a": item["value_a"],
                "value_b": item["value_b"],
                "dimension": dim,
            }
            for dim, pairs in training_pairs_by_dim.items()
            for item in pairs
        ]

        self.logger.log(
            "PreferencePairBuilder",
            {
                "dimensions": list(training_pairs_by_dim.keys()),
                "total_pairs": len(contrast_pairs),
            },
        )

        trainer = MRQTrainerEngine(
            memory=self.memory,
            logger=self.logger,
            device=self.device
        )


        assert contrast_pairs, "No contrast pairs found"

        trained_encoders, trained_models, regression_tuners = trainer.train_all(
            contrast_pairs, cfg=self.cfg
        )

        for dim in trained_models:
            model_path = get_model_path(
                self.model_path,
                self.model_type,
                self.target_type,
                dim,
                self.model_version,
                embedding_type=self.embedding_type
            )
            os.makedirs(model_path, exist_ok=True)

            predictor_path = os.path.join(model_path, f"{dim}.pt")
            encoder_path = os.path.join(model_path, f"{dim}_encoder.pt")
            tuner_path = os.path.join(model_path, f"{dim}_model.tuner.json")
            meta_path = os.path.join(model_path, f"{dim}.meta.json")

            # Save model weights
            torch.save(trained_models[dim], predictor_path)
            self.logger.log("ModelSaved", {"dimension": dim, "path": predictor_path})

            # Save encoder weights
            encoder_state = trained_encoders.get(dim)
            if encoder_state:
                torch.save(encoder_state, encoder_path)
                self.logger.log(
                    "EncoderSaved", {"dimension": dim, "path": encoder_path}
                )
            else:
                self.logger.log("EncoderMissing", {"dimension": dim})

            # Save regression tuner
            tuner = regression_tuners.get(dim)
            if tuner:
                tuner.save(tuner_path)

            # Save normalization metadata
            values = [
                (p["value_a"], p["value_b"])
                for p in contrast_pairs
                if p["dimension"] == dim
            ]
            flat_values = [v for pair in values for v in pair]
            save_json(
                {
                    "min_score": float(min(flat_values)),
                    "max_score": float(max(flat_values)),
                },
                meta_path,
            )

            self.logger.log(
                "DocumentModelSaved",
                {
                    "dimension": dim,
                    "model": predictor_path,
                    "encoder": encoder_path,
                    "tuner": tuner_path,
                    "meta": meta_path,
                },
            )

        context[self.output_key] = training_pairs_by_dim
        return context
