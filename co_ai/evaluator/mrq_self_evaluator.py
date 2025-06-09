import torch

from co_ai.evaluator.base import BaseEvaluator
from co_ai.evaluator.hypothesis_value_predictor import HypothesisValuePredictor
from co_ai.evaluator.mrq_trainer import MRQTrainer
from co_ai.evaluator.text_encoder import TextEncoder
from co_ai.models.sharpening_prediction import SharpeningPredictionORM


class MRQSelfEvaluator(BaseEvaluator):
    def __init__(self, memory, logger, device="cpu"):
        self.device = device
        self.memory = memory  # memory provides get_embedding
        self.logger = logger
        self.encoder = TextEncoder().to(self.device)
        self.value_predictor = HypothesisValuePredictor(512, 1024).to(self.device)
        self.trainer = MRQTrainer(
            memory=self.memory,
            logger=self.logger,
            value_predictor=self.value_predictor,
            encoder=self.encoder,
            device=self.device,
        )

    def judge(self, goal, prompt, output_a, output_b):
        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(prompt), device=self.device
        ).unsqueeze(0)
        output_a_emb = torch.tensor(
            self.memory.embedding.get_or_create(output_a), device=self.device
        ).unsqueeze(0)
        output_b_emb = torch.tensor(
            self.memory.embedding.get_or_create(output_b), device=self.device
        ).unsqueeze(0)

        zsa_a = self.encoder(prompt_emb, output_a_emb)
        zsa_b = self.encoder(prompt_emb, output_b_emb)

        value_a = self.value_predictor(zsa_a).item()
        value_b = self.value_predictor(zsa_b).item()

        preferred_output = output_a if value_a >= value_b else output_b
        scores = {"value_a": value_a, "value_b": value_b}

        if self.memory.mrq.log_evaluations():
            prediction = SharpeningPredictionORM(
                id=None,
                goal_id=-1,
                prompt_text=prompt,
                output_a=output_a,
                output_b=output_b,
                preferred="a" if value_a >= value_b else "b",
                predicted="a" if value_a >= value_b else "b",
                value_a=value_a,
                value_b=value_b,
            )

            self.memory.sharpening.insert_sharpening_prediction(
                prediction.to_dict(), goal
            )

        return preferred_output, scores

    def score_single(self, prompt: str, output: str, context: dict) -> float:
        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(prompt), device=self.device
        ).unsqueeze(0)
        output_emb = torch.tensor(
            self.memory.embedding.get_or_create(output), device=self.device
        ).unsqueeze(0)
        zsa = self.encoder(prompt_emb, output_emb)
        value = self.value_predictor(zsa).item()
        return value

    def train_from_database(self, goal: str, cfg: dict):
        samples = self.memory.mrq.get_training_pairs(
            goal=goal, limit=cfg.get("limit", 1000)
        )
        if not samples:
            self.logger.log(
                "MRQTrainingError",
                {
                    "error": "No training samples found for the given goal.",
                    "goal": goal,
                },
            )
            return

        dataloader = self.trainer.prepare_training_data(samples)
        self.trainer.train(dataloader, cfg)

    def train_from_context(self, context: dict, cfg: dict):
        samples = context.get("mrq_training_pairs", [])
        if not samples:
            self.logger.log(
                "MRQContextTrainingError",
                {"error": "No training samples found in context."},
            )
            return

        dataloader = self.trainer.prepare_training_data(samples)
        self.trainer.train(dataloader, cfg)
