# stephanie/evaluator/mrq_trainer.py
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stephanie.evaluator.hypothesis_value_predictor import \
    HypothesisValuePredictor
from stephanie.scoring.mrq.encoder import TextEncoder


class MRQTrainer:
    def __init__(
        self, memory, logger, encoder=None, value_predictor=None, device="cpu"
    ):
        self.memory = memory
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim
        self.logger = logger
        self.device = device

        # Use provided encoder or instantiate new TextEncoder
        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(device)

        # Use provided predictor or instantiate new HypothesisValuePredictor
        if value_predictor is not None:
            self.value_predictor = value_predictor.to(device)
        else:
            self.value_predictor = HypothesisValuePredictor(self.dim, self.hdim).to(device)

    def prepare_training_data(self, samples):
        inputs, labels = [], []
        total = len(samples)

        for idx, item in enumerate(samples):
            prompt_emb = self.memory.embedding.get_or_create(item["prompt"])
            output_a_emb = self.memory.embedding.get_or_create(item["output_a"])
            output_b_emb = self.memory.embedding.get_or_create(item["output_b"])

            preferred = "a" if item["value_a"] >= item["value_b"] else "b"

            # Convert to tensor and move to device
            prompt_tensor = torch.tensor(prompt_emb).unsqueeze(0).to(self.device)
            output_a_tensor = torch.tensor(output_a_emb).unsqueeze(0).to(self.device)
            output_b_tensor = torch.tensor(output_b_emb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                zsa_a = self.encoder(prompt_tensor, output_a_tensor)
                zsa_b = self.encoder(prompt_tensor, output_b_tensor)

            # Compute difference based on preference
            diff = zsa_a - zsa_b if preferred == "a" else zsa_b - zsa_a

            # Ensure correct shape before appending
            inputs.append(diff.squeeze(0).detach())
            labels.append(torch.tensor([1.0], device=self.device))

            # Log progress every 100 samples
            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                percent = round((idx + 1) / total * 100, 2)
                self.logger.log(
                    "TrainingDataProgress",
                    {"current": idx + 1, "total": total, "percent": percent},
                )

        # Create dataset and loader
        dataset = TensorDataset(torch.stack(inputs), torch.stack(labels))
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def train(self, dataloader, cfg):
        epochs = cfg.get("epochs", 20)
        lr = cfg.get("lr", 1e-4)
        patience = cfg.get("patience", 3)
        min_delta = cfg.get("min_delta", 0.0001)

        optimizer = torch.optim.Adam(self.value_predictor.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        self.value_predictor.train()

        best_loss = float("inf")
        epochs_no_improve = 0

        self.logger.log(
            "MRQTrainerStart",
            {
                "epochs": epochs,
                "learning_rate": lr,
                "patience": patience,
                "min_delta": min_delta,
            },
        )

        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, y_batch in dataloader:
                assert isinstance(x_batch, torch.Tensor), (
                    "x_batch must be a single tensor"
                )
                # Ensure x_batch has correct shape [batch_size, zsa_dim]
                assert len(x_batch.shape) == 2, (
                    f"Unexpected x_batch shape: {x_batch.shape}"
                )

                preds = self.value_predictor(x_batch)  # Only one input now
                loss = criterion(preds, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.logger.log(
                "MRQTrainerEpoch", {"epoch": epoch + 1, "avg_loss": round(avg_loss, 5)}
            )

            if best_loss - avg_loss > min_delta:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    self.logger.log(
                        "MRQTrainerEarlyStopping",
                        {"stopped_epoch": epoch + 1, "best_loss": round(best_loss, 5)},
                    )
                    break

        self.logger.log(
            "MRQTrainerTrainingComplete",
            {"epochs_trained": epoch + 1, "final_loss": round(avg_loss, 5)},
        )

    def train_multidimensional_model(self, contrast_pairs, cfg=None):
        """
        Trains separate models per scoring dimension using contrast pairs.
        Each pair contains: output_a, output_b, prompt, preferred, dimension.
        """
        by_dimension = defaultdict(list)
        for pair in contrast_pairs:
            dim = pair.get("dimension", "default")
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

            trained_models[dim] = self.value_predictor.state_dict()
            self.logger.log(
                "TrainingDimensionComplete", {"dimension": dim, "samples": len(samples)}
            )

        return trained_models
