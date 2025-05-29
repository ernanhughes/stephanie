import torch

from co_ai.evaluator.base import BaseEvaluator
from co_ai.evaluator import HypothesisValuePredictor, TextEncoder
from co_ai.dataloaders import ARMDataLoader

class ARMReasoningSelfEvaluator(BaseEvaluator):
    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.device = cfg.get("device", "cpu")

        self.format_freq = cfg.get(
            "format_freq", {"direct": 1, "short_cot": 1, "code": 1, "long_cot": 1}
        )
        self.format_rewards = cfg.get(
            "format_rewards",
            {"direct": [0.5], "short_cot": [0.5], "code": [0.5], "long_cot": [0.5]},
        )

        self.encoder = TextEncoder().to(self.device)
        self.value_predictor = HypothesisValuePredictor(512, 1024).to(self.device)

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
        scores = {
            "value_a": value_a,
            "value_b": value_b,
            "fmt_a": ARMDataLoader.detect_format(output_a),
            "fmt_b": ARMDataLoader.detect_format(output_b),
        }

        return preferred_output, scores

    def _update_format_stats(self, fmt: str, reward: float):
        """
        Track format usage and average reward per format.
        
        This enables format-aware reward shaping and prevents format collapse.
        """
        if fmt not in self.format_freq:
            self.format_freq[fmt] = 0
            self.format_rewards[fmt] = []

        self.format_freq[fmt] += 1
        self.format_rewards[fmt].append(reward)

    def train_from_database(self, goal_text: str, cfg: dict):
        limit = cfg.get("limit", 1000)
        epochs = cfg.get("epochs", 20)
        lr = cfg.get("lr", 1e-4)
        batch_size = cfg.get("batch_size", 16)

        samples = self.memory.mrq.get_training_pairs(goal=goal_text, limit=limit)
        if not samples:
            self.logger.log(
                "TrainingError", {"message": "No samples found", "goal": goal_text}
            )
            return

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

        dataset = torch.utils.data.TensorDataset(
            torch.stack(inputs), torch.stack(labels)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        opt = torch.optim.Adam(self.value_predictor.parameters(), lr=lr)
        self.value_predictor.train()

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
            self.logger.log(
                "TrainingEpoch",
                {"epoch": epoch + 1, "avg_loss": avg_loss, "goal": goal_text},
            )

        self.logger.log("TrainingComplete", {"goal": goal_text})

    def score(self, prompt: str, response: str) -> float:
        """
        Public scoring method used by agents like AdaptiveReasonerAgent.
        Returns a scalar score indicating how good a response is.
        """
        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(prompt), device=self.device
        ).unsqueeze(0)
        response_emb = torch.tensor(
            self.memory.embedding.get_or_create(response), device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            zsa = self.encoder(prompt_emb, response_emb)
            score = self.value_predictor(zsa).item()

        token_len = len(response.split())
        fmt = ARMDataLoader.detect_format(response)
        rarity_bonus = 1.0 / (1 + self.format_freq.get(fmt, 1))
        score -= 0.01 * token_len
        score += rarity_bonus

        self._update_format_stats(fmt, score)
        return score

    def _score_response(self, prompt_emb, response_emb):
        """Score a single response using prompt-response encoder + value predictor"""
        zsa = self.encoder(prompt_emb, response_emb)
        return self.value_predictor(zsa), zsa
