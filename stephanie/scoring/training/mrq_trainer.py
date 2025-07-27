from datetime import datetime

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from stephanie.models.training_stats import TrainingStatsORM
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.mrq.model import MRQModel
from stephanie.scoring.mrq.value_predictor import ValuePredictor
from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class MRQTrainer(BaseTrainer):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.use_tuner = cfg.get("use_tuner", True)
        self.min_samples = cfg.get("min_samples", 5)
        self.batch_size = cfg.get("batch_size", 1)
        self.model = self._build_model()
        self.use_tuner = cfg.get("use_tuner", True)
        self.use_early_stopping = cfg.get("early_stopping", True)
        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.batch_size = cfg.get("batch_size", 2)
        self.epochs = cfg.get("epochs", 50)
        self.lr = cfg.get("lr", 1e-4)
        self.min_samples = cfg.get("min_samples", 5)

        
        self.logger.log("MRQTrainerInitialized", {
            "embedding_type": self.embedding_type,
            "use_tuner": self.use_tuner,
            "device": str(self.device)
        })

    def _create_dataloader(self, samples):
        inputs, labels = [], []

        for item in samples:
            prompt = item.get("title", "")
            output_a = item.get("output_a", "")
            output_b = item.get("output_b", "")
            value_a = item.get("value_a", 0)
            value_b = item.get("value_b", 0)

            if not prompt or not output_a or not output_b:
                continue

            try:
                prompt_emb = torch.tensor(self.memory.embedding.get_or_create(prompt)).unsqueeze(0).to(self.device)
                a_emb = torch.tensor(self.memory.embedding.get_or_create(output_a)).unsqueeze(0).to(self.device)
                b_emb = torch.tensor(self.memory.embedding.get_or_create(output_b)).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    zsa_a = self.model.encoder(prompt_emb, a_emb)
                    zsa_b = self.model.encoder(prompt_emb, b_emb)

                preferred = "a" if value_a >= value_b else "b"
                diff = zsa_a - zsa_b if preferred == "a" else zsa_b - zsa_a
                inputs.append(diff.squeeze(0).detach())
                labels.append(torch.tensor([1.0], device=self.device))

            except Exception as e:
                self.logger.log("PairPreparationError", {"error": str(e)})
                continue

        if len(inputs) < self.min_samples:
            self.logger.log("InsufficientSamples", {
                "sample_count": len(inputs),
                "threshold": self.min_samples
            })
            return None

        dataset = TensorDataset(torch.stack(inputs), torch.stack(labels))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _build_model(self):
        encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
        predictor = ValuePredictor(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
        return MRQModel(encoder, predictor, self.memory.embedding, device=self.device)

    def _train_epoch(self, model, dataloader):
        model.encoder.train()
        model.predictor.train()
        total_loss, count = 0.0, 0

        for inputs, scores in dataloader:
            inputs = inputs.to(self.device)
            scores = scores.to(self.device)

            predictions = model.predictor(inputs).squeeze()
            loss = F.mse_loss(predictions, scores)

            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 0.5)

            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

        return total_loss / count

    def train(self, samples, dimension):
        dataloader = self._create_dataloader(samples)
        if not dataloader:
            return {"error": "insufficient_data", "dimension": dimension}

        self.optimizer = optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.predictor.parameters()),
            lr=self.lr
        )

        best_loss = float("inf")
        early_stop_counter = 0
        losses = []

        for epoch in range(self.epochs):
            avg_loss = self._train_epoch(self.model, dataloader)
            losses.append(avg_loss)
            self.logger.log("MRQTrainingEpoch", {
                "epoch": epoch + 1,
                "loss": avg_loss
            })
            if avg_loss < best_loss - self.early_stopping_min_delta:
                best_loss = avg_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if self.use_early_stopping and early_stop_counter >= self.early_stopping_patience:
                    break

        locator = self.get_locator(dimension)            
        torch.save(self.model.encoder.state_dict(), locator.encoder_file())
        torch.save(self.model.predictor.state_dict(), locator.model_file())

        if self.use_tuner:
            tuner = RegressionTuner(dimension=dimension, logger=self.logger)
            for inputs, scores in dataloader:
                inputs = inputs.to(self.device)
                preds = self.model.predictor(inputs).squeeze().detach().cpu().numpy()
                actuals = scores.cpu().numpy()
                for p, a in zip(preds, actuals):
                    tuner.train_single(float(p), float(a))
            tuner.save(locator.tuner_file())

        scores_np = torch.tensor([s["value_a"] for s in samples])
        min_score = float(torch.min(scores_np))
        max_score = float(torch.max(scores_np))

        meta = {
            "dimension": dimension,
            "model_type": "mrq",
            "target_type": self.target_type,
            "embedding_type": self.embedding_type,
            "version": self.version,
            "dim": self.dim,
            "hdim": self.hdim,
            "min_score": min_score,
            "max_score": max_score,
            "avg_loss": best_loss,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._save_meta_file(meta, dimension)

        training_stat = TrainingStatsORM(
            model_type="mrq",
            target_type=self.target_type,
            dimension=dimension,
            version=self.version,
            embedding_type=self.embedding_type,
            
            avg_q_loss=best_loss
        )
        self.memory.session.add(training_stat)
        self.memory.session.commit()

        return meta
