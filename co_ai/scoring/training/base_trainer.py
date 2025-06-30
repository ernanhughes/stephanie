import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict


class BaseTrainer:
    def __init__(self, memory, logger, encoder=None, value_predictor=None, device="cpu"):
        self.memory = memory
        self.logger = logger
        self.device = device

        self.encoder = encoder.to(device) if encoder else self.init_encoder().to(device)
        self.value_predictor = value_predictor.to(device) if value_predictor else self.init_predictor().to(device)

    def init_encoder(self):
        raise NotImplementedError("Subclasses must implement init_encoder")

    def init_predictor(self):
        raise NotImplementedError("Subclasses must implement init_predictor")

    def prepare_training_data(self, samples: list[dict]) -> DataLoader:
        raise NotImplementedError("Subclasses must implement prepare_training_data")

    def train(self, dataloader: DataLoader, cfg: dict):
        epochs = cfg.get("epochs", 20)
        lr = cfg.get("lr", 1e-4)
        patience = cfg.get("patience", 3)
        min_delta = cfg.get("min_delta", 0.0001)

        optimizer = torch.optim.Adam(self.value_predictor.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        self.value_predictor.train()

        best_loss = float("inf")
        epochs_no_improve = 0

        self.logger.log("BaseTrainerTrainingStart", {
            "epochs": epochs,
            "lr": lr,
            "patience": patience,
            "min_delta": min_delta
        })

        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, y_batch in dataloader:
                assert isinstance(x_batch, torch.Tensor), "x_batch must be a tensor"
                assert len(x_batch.shape) == 2, f"Unexpected shape: {x_batch.shape}"

                preds = self.value_predictor(x_batch)
                loss = criterion(preds, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.logger.log("BaseTrainerEpoch", {
                "epoch": epoch + 1,
                "avg_loss": round(avg_loss, 5)
            })

            if best_loss - avg_loss > min_delta:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    self.logger.log("BaseTrainerEarlyStopping", {
                        "stopped_epoch": epoch + 1,
                        "best_loss": round(best_loss, 5)
                    })
                    break

        self.logger.log("BaseTrainerTrainingComplete", {
            "epochs_trained": epoch + 1,
            "final_loss": round(avg_loss, 5)
        })

    def train_multidimensional_model(self, contrast_pairs: list[dict], cfg: dict = None):
        by_dimension = defaultdict(list)
        for pair in contrast_pairs:
            dim = pair.get("dimension", "default")
            by_dimension[dim].append(pair)

        trained_models = {}

        for dim, samples in by_dimension.items():
            if not samples:
                self.logger.log("BaseTrainerSkipDimension", {"dimension": dim})
                continue

            self.logger.log("BaseTrainerTrainDimension", {
                "dimension": dim,
                "num_samples": len(samples)
            })

            dataloader = self.prepare_training_data(samples)
            self.train(dataloader, cfg or {})

            trained_models[dim] = self.value_predictor.state_dict()

        return trained_models
