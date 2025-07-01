# stephanie/training/document_mrq_trainer.py

from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from stephanie.evaluator.text_encoder import TextEncoder
from stephanie.scoring.document_value_predictor import DocumentValuePredictor
from stephanie.scoring.transforms.regression_tuner import RegressionTuner



class DocumentMRQTrainer:
    def __init__(
        self,
        memory,
        logger,
        encoder=None,
        value_predictor=None,
        device="cpu"
    ):
        self.memory = memory
        self.logger = logger
        self.device = device

        self.encoder = encoder.to(device) if encoder else TextEncoder().to(device)
        self.value_predictor = value_predictor.to(device) if value_predictor else DocumentValuePredictor(512, 1024).to(device)
        self.regression_tuners = {}

    def prepare_training_data(self, samples: List[dict]):
        inputs, labels = [], []
        total = len(samples)

        for idx, item in enumerate(samples):
            context_text = item.get("title", "")  # Or item.get("goal_text", "")
            context_emb = self.memory.embedding.get_or_create(context_text)
            doc_a_emb = self.memory.embedding.get_or_create(item["output_a"])
            doc_b_emb = self.memory.embedding.get_or_create(item["output_b"])

            preferred = "a" if item["value_a"] >= item["value_b"] else "b"

            context_tensor = torch.tensor(context_emb).unsqueeze(0).to(self.device)
            a_tensor = torch.tensor(doc_a_emb).unsqueeze(0).to(self.device)
            b_tensor = torch.tensor(doc_b_emb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                zsa_a = self.encoder(context_tensor, a_tensor)
                zsa_b = self.encoder(context_tensor, b_tensor)

            diff = zsa_a - zsa_b if preferred == "a" else zsa_b - zsa_a

            inputs.append(diff.squeeze(0).detach())
            labels.append(torch.tensor([1.0], device=self.device))

            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                percent = round((idx + 1) / total * 100, 2)
                self.logger.log("DocumentTrainingProgress", {
                    "current": idx + 1,
                    "total": total,
                    "percent": percent
                })

        dataset = TensorDataset(torch.stack(inputs), torch.stack(labels))
        return DataLoader(dataset, batch_size=16, shuffle=True)

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

        self.logger.log("DocumentMRQTrainingStart", {
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
            self.logger.log("DocumentMRQTrainerEpoch", {
                "epoch": epoch + 1,
                "avg_loss": round(avg_loss, 5)
            })

            if best_loss - avg_loss > min_delta:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    self.logger.log("DocumentMRQEarlyStopping", {
                        "stopped_epoch": epoch + 1,
                        "best_loss": round(best_loss, 5)
                    })
                    break

        self.logger.log("DocumentMRQTrainingComplete", {
            "epochs_trained": epoch + 1,
            "final_loss": round(avg_loss, 5)
        })
    def train_multidimensional_model(self, contrast_pairs: List[dict], cfg=None):
        by_dimension = defaultdict(list)
        for pair in contrast_pairs:
            dim = pair.get("dimension", "default")
            by_dimension[dim].append(pair)

        trained_models = {}
        regression_tuners = {}

        for dim, samples in by_dimension.items():
            if not samples:
                self.logger.log("DocumentMRQSkipDimension", {"dimension": dim})
                continue

            self.logger.log("DocumentMRQTrainDimension", {
                "dimension": dim,
                "num_samples": len(samples)
            })

            tuner = RegressionTuner(dimension=dim, logger=self.logger)
            regression_tuners[dim] = tuner

            dataloader = self.prepare_training_data(samples)
            self.train(dataloader, cfg or {})
            trained_models[dim] = self.value_predictor.state_dict()

            # Train regression tuner
            for pair in samples:
                for side in ["a", "b"]:
                    llm_score = pair[f"value_{side}"]
                    doc_text = pair[f"output_{side}"]
                    context_text = pair.get("title", "")

                    # Compute MRQ score from model
                    context_emb = torch.tensor(self.memory.embedding.get_or_create(context_text)).unsqueeze(0).to(self.device)
                    doc_emb = torch.tensor(self.memory.embedding.get_or_create(doc_text)).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        zsa = self.encoder(context_emb, doc_emb)
                        mrq_score = self.value_predictor(zsa).item()

                    tuner.train_single(mrq_score, llm_score)

        return trained_models, regression_tuners

    def align_to_best_llm_neighbour(self, goal, hypothesis, dimension):
        """
        Fetch similar hypotheses that already have high LLM scores.
        Then align MR.Q prediction to the best of them.
        """
        llm_scores = self.get_closest_llm_scores(hypothesis["text"], dimension)
        if llm_scores:
            self.align_with_llm_score(dimension, goal, hypothesis, max(llm_scores))

    def get_closest_llm_scores(
        self, hypothesis_text: str, dimension: str, top_k: int = 5
    ) -> list[float]:
        """
        Finds the top_k LLM scores for hypotheses most similar to the given one.
        """
        query_emb = self.memory.embedding.get_or_create(hypothesis_text)
        similar_items = self.memory.embedding.search_similar_prompts_with_scores(query_emb, top_k)

        scores = []
        for item in similar_items:
            matched_text = item.get("text")
            score_entry = self.memory.score.find_by_text_and_dimension(
                matched_text, dimension=dimension, source="llm"
            )
            if score_entry:
                scores.append(score_entry.score)
        return scores

    def align_with_llm_score(self, dimension, goal, hypothesis, llm_score):
        mrq_score = self._estimate_score(goal, hypothesis, dimension)
        self.logger.log(
            "MRQAligningToLLM",
            {
                "goal": goal.get("goal_text"),
                "hypothesis": hypothesis.get("text"),
                "dimension": dimension,
                "mrq_raw": mrq_score,
                "llm_target": llm_score,
            },
        )
        self.regression_tuners[dimension].add_example(mrq_score, llm_score)
        self.logger.log(
            "MRQAlignmentAdded",
            {
                "dimension": dimension,
                "example_count": len(self.regression_tuners[dimension].examples),
            }, 
        )
