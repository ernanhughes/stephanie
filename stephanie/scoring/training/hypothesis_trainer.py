import torch
from torch.utils.data import DataLoader, TensorDataset

from stephanie.evaluator.hypothesis_value_predictor import \
    HypothesisValuePredictor
from stephanie.evaluator.text_encoder import TextEncoder
from stephanie.scoring.training.base_trainer import BaseTrainer


class HypothesisTrainer(BaseTrainer):
    def __init__(self, memory, logger, encoder=None, value_predictor=None, device="cpu"):
        encoder = encoder or TextEncoder()
        value_predictor = value_predictor or HypothesisValuePredictor(512, 1024)
        super().__init__(memory, logger, encoder, value_predictor, device)

    def prepare_training_data(self, samples):
        inputs, labels = [], []
        total = len(samples)

        for idx, item in enumerate(samples):
            prompt_emb = self.memory.embedding.get_or_create(item["prompt"])
            output_a_emb = self.memory.embedding.get_or_create(item["output_a"])
            output_b_emb = self.memory.embedding.get_or_create(item["output_b"])

            preferred = "a" if item["value_a"] >= item["value_b"] else "b"

            prompt_tensor = torch.tensor(prompt_emb).unsqueeze(0).to(self.device)
            a_tensor = torch.tensor(output_a_emb).unsqueeze(0).to(self.device)
            b_tensor = torch.tensor(output_b_emb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                zsa_a = self.encoder(prompt_tensor, a_tensor)
                zsa_b = self.encoder(prompt_tensor, b_tensor)

            diff = zsa_a - zsa_b if preferred == "a" else zsa_b - zsa_a

            inputs.append(diff.squeeze(0).detach())
            labels.append(torch.tensor([1.0], device=self.device))

            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                self.logger.log("HypothesisTrainerProgress", {
                    "current": idx + 1, "total": total, "percent": round((idx + 1) / total * 100, 2)
                })

        dataset = TensorDataset(torch.stack(inputs), torch.stack(labels))
        return DataLoader(dataset, batch_size=16, shuffle=True)
