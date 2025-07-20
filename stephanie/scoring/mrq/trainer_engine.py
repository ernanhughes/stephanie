# stephanie/scoring/mrq/trainer_engine.py

from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.value_predictor import ValuePredictor
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class MRQTrainerEngine:
    def __init__(self, memory, logger, device="cpu"):
        self.memory = memory
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim
        self.logger = logger
        self.device = device

    def build_encoder(self):
        return TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)

    def build_predictor(self):
        return ValuePredictor(zsa_dim=self.dim, hdim=self.hdim).to(self.device)

    def prepare_training_data(self, encoder, samples):
        inputs, labels = [], []

        for idx, item in enumerate(samples):
            context = item.get("title", "")
            context_emb = self.memory.embedding.get_or_create(context)
            doc_a_emb = self.memory.embedding.get_or_create(item["output_a"])
            doc_b_emb = self.memory.embedding.get_or_create(item["output_b"])

            context_tensor = torch.tensor(context_emb).unsqueeze(0).to(self.device)
            a_tensor = torch.tensor(doc_a_emb).unsqueeze(0).to(self.device)
            b_tensor = torch.tensor(doc_b_emb).unsqueeze(0).to(self.device)

            # print(f"[{idx}] context_tensor shape: {context_tensor.shape}")
            # print(f"[{idx}] a_tensor shape: {a_tensor.shape}")
            # print(f"[{idx}] b_tensor shape: {b_tensor.shape}")

            with torch.no_grad():
                zsa_a = encoder(context_tensor, a_tensor)
                zsa_b = encoder(context_tensor, b_tensor)

            # print(f"[{idx}] zsa_a shape: {zsa_a.shape}")
            # print(f"[{idx}] zsa_b shape: {zsa_b.shape}")

            diff = (
                zsa_a - zsa_b if item["value_a"] >= item["value_b"] else zsa_b - zsa_a
            )

            diff_squeezed = diff.squeeze(0).detach()
            # print(f"[{idx}] diff_squeezed shape: {diff_squeezed.shape}")

            inputs.append(diff_squeezed)
            labels.append(torch.tensor([1.0], device=self.device))

        stacked_inputs = torch.stack(inputs)
        stacked_labels = torch.stack(labels)
        print(f"Final input batch shape: {stacked_inputs.shape}")
        print(f"Final label batch shape: {stacked_labels.shape}")

        dataset = TensorDataset(stacked_inputs, stacked_labels)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def train_predictor(self, predictor, dataloader, cfg):
        epochs = cfg.get("epochs", 20)
        lr = cfg.get("lr", 1e-4)
        patience = cfg.get("patience", 3)
        min_delta = cfg.get("min_delta", 0.0001)

        optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        predictor.train()

        best_loss, epochs_no_improve = float("inf"), 0

        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, y_batch in dataloader:
                preds = predictor(x_batch)
                loss = criterion(preds, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.logger.log(
                "MRQTrainerEpoch", {"epoch": epoch + 1, "avg_loss": avg_loss}
            )

            if best_loss - avg_loss > min_delta:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

    def train_all(self, contrast_pairs, cfg):
        by_dimension = defaultdict(list)
        for p in contrast_pairs:
            dim = p.get("dimension", "default")
            by_dimension[dim].append(p)

        encoders, predictors, tuners = {}, {}, {}

        for dim, samples in by_dimension.items():
            encoder = self.build_encoder()
            predictor = self.build_predictor()
            tuner = RegressionTuner(dimension=dim, logger=self.logger)

            dataloader = self.prepare_training_data(encoder, samples)
            self.train_predictor(predictor, dataloader, cfg)

            for s in samples:
                for side in ["a", "b"]:
                    llm_score = s[f"value_{side}"]
                    doc_text = s[f"output_{side}"]
                    context_text = s.get("title", "")

                    context_emb = (
                        torch.tensor(self.memory.embedding.get_or_create(context_text))
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    doc_emb = (
                        torch.tensor(self.memory.embedding.get_or_create(doc_text))
                        .unsqueeze(0)
                        .to(self.device)
                    )

                    with torch.no_grad():
                        zsa = encoder(context_emb, doc_emb)
                        mrq_score = predictor(zsa).item()

                    tuner.train_single(mrq_score, llm_score)

            encoders[dim] = encoder.state_dict()
            predictors[dim] = predictor.state_dict()
            tuners[dim] = tuner

        return encoders, predictors, tuners

    def get_closest_llm_scores(
        self, hypothesis_text: str, dimension: str, top_k: int = 5
    ) -> list[float]:
        query_emb = self.memory.embedding.get_or_create(hypothesis_text)
        similar_items = self.memory.embedding.search_similar_prompts_with_scores(
            query_emb, top_k
        )

        scores = []
        for item in similar_items:
            matched_text = item.get("text")
            score_entry = self.memory.score.find_by_text_and_dimension(
                matched_text, dimension=dimension, source="llm"
            )
            if score_entry:
                scores.append(score_entry.score)
        return scores
