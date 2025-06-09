# co_ai/evaluator/mrq_trainer.py
import copy
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, TensorDataset

from co_ai.evaluator.hypothesis_value_predictor import HypothesisValuePredictor
from co_ai.evaluator.text_encoder import TextEncoder


class MRQTrainer:
    def __init__(self, memory, logger, encoder=None, value_predictor=None, device="cpu"):
        self.memory = memory
        self.logger = logger
        self.device = device
        self.value_predictor = HypothesisValuePredictor(512, 1024).to(self.device)
        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = TextEncoder().to(device)
        if value_predictor is not None:            
            self.value_predictor = value_predictor.to(device)
        else:
            self.value_predictor = HypothesisValuePredictor(512, 1024).to(device)

    def prepare_training_data(self, samples):
        inputs, labels = [], []
        for item in samples:
            prompt_emb = self.memory.embedding.get_or_create(item["prompt"])
            output_a_emb = self.memory.embedding.get_or_create(item["output_a"])
            output_b_emb = self.memory.embedding.get_or_create(item["output_b"])
            preferred = item["preferred"]

            zsa_a = self.encoder(
                torch.tensor(prompt_emb).unsqueeze(0).to(self.device),
                torch.tensor(output_a_emb).unsqueeze(0).to(self.device),
            )
            zsa_b = self.encoder(
                torch.tensor(prompt_emb).unsqueeze(0).to(self.device),
                torch.tensor(output_b_emb).unsqueeze(0).to(self.device),
            )

            diff = zsa_a - zsa_b if preferred == "a" else zsa_b - zsa_a
            inputs.append(diff.squeeze(0).detach())
            labels.append(torch.tensor([1.0], device=self.device))

        dataset = TensorDataset(torch.stack(inputs), torch.stack(labels))
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def train(self, dataloader, cfg):
        epochs = cfg.get("epochs", 20)
        lr = cfg.get("lr", 1e-4)
        patience = cfg.get("patience", 3)
        min_delta = cfg.get("min_delta", 0.0001)

        opt = torch.optim.Adam(self.value_predictor.parameters(), lr=lr)
        self.value_predictor.train()

        best_loss = float("inf")
        epochs_no_improve = 0

        first = next(iter(dataloader))
        print("Sample batch:", first)

        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, y_batch in dataloader:
                preds = self.value_predictor(x_batch)
                loss = -torch.log(torch.sigmoid(preds)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.logger.log("MRQTrainerEpoch", {
                "epoch": epoch + 1,
                "avg_loss": round(avg_loss, 5)
            })

            if best_loss - avg_loss > min_delta:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    self.logger.log("MRQTrainerEarlyStopping", {
                        "stopped_epoch": epoch + 1,
                        "best_loss": round(best_loss, 5)
                    })
                    break

        self.logger.log(
            "MRQTrainerTrainingComplete",
            {"epochs_trained": epoch + 1, "final_loss": round(avg_loss, 5)},
        )

    def train_multidimensional_model(self, contrast_pairs, cfg=None):
        """
        Trains a separate model for each scoring dimension using the provided contrast pairs.
        Each pair is a dict with: output_a, output_b, prompt, preferred, dimension.
        """
        from collections import defaultdict

        # Group contrast pairs by dimension
        by_dimension = defaultdict(list)
        for pair in contrast_pairs:
            dim = pair["dimension"]
            by_dimension[dim].append(pair)

        trained_models = {}

        for dim, samples in by_dimension.items():
            if not samples:
                self.logger.log("DimensionSkippedNoSamples", {"dimension": dim})
                continue

            self.logger.log(
                "TrainingDimensionStart",
                {"dimension": dim, "num_samples": len(samples)},
            )

            dataloader = self.prepare_training_data(samples)
            self.train(dataloader, cfg or {})

            # Save model for this dimension if needed
            trained_models[dim] = self.value_predictor.state_dict()

            self.logger.log(
                "TrainingDimensionComplete", {"dimension": dim, "samples": len(samples)}
            )

        return trained_models
