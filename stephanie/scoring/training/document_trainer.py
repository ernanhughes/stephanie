# stephanie/scoring/training/document_trainer.py
from typing import List

import torch
from torch.utils.data import DataLoader, TensorDataset

from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.value_predictor import ValuePredictor
from stephanie.scoring.training.base_trainer import BaseTrainer


class DocumentTrainer(BaseTrainer):
    def init_encoder(self):
        return TextEncoder().to(self.device)

    def init_predictor(self):
        return ValuePredictor().to(self.device)

    def prepare_training_data(self, samples: List[dict]) -> DataLoader:
        inputs, labels = [], []
        total = len(samples)

        for idx, item in enumerate(samples):
            context_text = item.get("title", "")
            context_emb = self.memory.embedding.get_or_create(context_text)
            doc_a_emb = self.memory.embedding.get_or_create(item["text_a"])
            doc_b_emb = self.memory.embedding.get_or_create(item["text_b"])

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
                self.logger.log(
                    "DocumentTrainingProgress",
                    {
                        "current": idx + 1,
                        "total": total,
                        "percent": round((idx + 1) / total * 100, 2),
                    },
                )

        dataset = TensorDataset(torch.stack(inputs), torch.stack(labels))
        return DataLoader(dataset, batch_size=16, shuffle=True)
