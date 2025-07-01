import json
from copy import deepcopy

import torch
import torch.nn.functional as F

from stephanie.dataloaders import ARMDataLoader
from stephanie.evaluator.base import BaseEvaluator
from stephanie.evaluator.hypothesis_value_predictor import HypothesisValuePredictor
from stephanie.evaluator.text_encoder import TextEncoder


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
            "format_rewards", {k: [0.5] for k in self.format_freq}
        )

        self.apply_penalty_bonus = cfg.get("apply_penalty_bonus", True)
        self.epsilon = cfg.get("epsilon", 0.1)
        self.kl_penalty_coeff = cfg.get("kl_penalty_coeff", 0.1)

        self.encoder = TextEncoder().to(self.device)
        self.value_predictor = HypothesisValuePredictor(512, 1024).to(self.device)
        self.ref_value_predictor = deepcopy(self.value_predictor)
        self.ref_value_predictor.eval()

    def judge(self, prompt, output_a, output_b, context: dict):
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

    def score_single(self, prompt: str, output: str, context) -> float:
        """Minimal ABC-compliant scoring method."""
        prompt_emb = torch.tensor(
            self.memory.embedding.get_or_create(prompt), device=self.device
        ).unsqueeze(0)
        output_emb = torch.tensor(
            self.memory.embedding.get_or_create(output), device=self.device
        ).unsqueeze(0)
        zsa = self.encoder(prompt_emb, output_emb)
        return self.value_predictor(zsa).item()

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
        """Framework-level scoring method with reward shaping."""
        base_score = self.score_single(prompt, response, context={})
        if not self.apply_penalty_bonus:
            return base_score

        token_len = len(response.split())
        fmt = ARMDataLoader.detect_format(response)
        rarity_bonus = 1.0 / (1 + self.format_freq.get(fmt, 1))
        shaped_score = base_score - 0.01 * token_len + rarity_bonus
        self._update_format_stats(fmt, shaped_score)
        return shaped_score

    def _score_response(self, prompt_emb, response_emb):
        """Score a single response using prompt-response encoder + value predictor"""
        zsa = self.encoder(prompt_emb, response_emb)
        return self.value_predictor(zsa), zsa

    def train_from_context(self, context: dict, cfg: dict):
        """
        Trains the value predictor using DPO samples stored in the context.
        Applies format-aware reward shaping and KL penalty.
        """
        dpo_samples = context.get("dpo_samples", [])
        if not dpo_samples:
            self.logger.log(
                "TrainingError", {"message": "No DPO samples found in context."}
            )
            return

        self.logger.log(
            "TrainingStarted", {"sample_count": len(dpo_samples), "config": cfg}
        )

        inputs, labels = [], []

        # Extract preference data
        for item in dpo_samples:
            prompt_emb = self.memory.embedding.get_or_create(item["prompt"])
            output_a_emb = self.memory.embedding.get_or_create(item["chosen"])
            output_b_emb = self.memory.embedding.get_or_create(item["rejected"])

            zsa_a = self.encoder(
                torch.tensor(prompt_emb).unsqueeze(0).to(self.device),
                torch.tensor(output_a_emb).unsqueeze(0).to(self.device),
            )
            zsa_b = self.encoder(
                torch.tensor(prompt_emb).unsqueeze(0).to(self.device),
                torch.tensor(output_b_emb).unsqueeze(0).to(self.device),
            )

            diff = zsa_a - zsa_b if item["preferred_format"] == "a" else zsa_b - zsa_a
            inputs.append(diff.squeeze(0).detach())
            labels.append(torch.tensor([1.0]))

        dataset = torch.utils.data.TensorDataset(
            torch.stack(inputs), torch.stack(labels)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.get("batch_size", 16), shuffle=True
        )

        opt = optim.Adam(self.value_predictor.parameters(), lr=cfg.get("lr", 1e-4))
        self.value_predictor.train()

        epochs = cfg.get("epochs", 20)
        best_loss = float("inf")
        patience_counter = 0
        patience = cfg.get("patience", 3)

        for epoch in range(epochs):
            total_loss = 0.0
            for x_batch, y_batch in dataloader:
                preds = self.value_predictor(x_batch)
                policy_log_probs = torch.log_softmax(preds, dim=-1)

                with torch.no_grad():
                    ref_preds = self.ref_value_predictor(x_batch)
                    ref_log_probs = torch.log_softmax(ref_preds, dim=-1)

                advantages = policy_log_probs - ref_log_probs
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-6
                )

                ratios = torch.exp(policy_log_probs - ref_log_probs)
                clipped_ratios = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                unclipped_loss = ratios * advantages
                clipped_loss = clipped_ratios * advantages

                policy_loss = -torch.min(unclipped_loss, clipped_loss).mean()
                kl = F.kl_div(ref_log_probs, policy_log_probs, reduction="batchmean")
                loss = policy_loss + self.kl_penalty_coeff * kl

                loss.backward()
                opt.step()
                opt.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.logger.log(
                "TrainingEpoch",
                {
                    "epoch": epoch + 1,
                    "avg_loss": round(avg_loss, 5),
                    "goal": "arm_dpo",
                    "format_usage": self.format_freq.copy(),
                    "format_rewards": {
                        k: round(sum(v) / len(v), 5) if v else 0
                        for k, v in self.format_rewards.items()
                    },
                },
            )

            if avg_loss < best_loss - 0.0001:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.log(
                    "EarlyStopping",
                    {"stopped_epoch": epoch + 1, "best_loss": round(best_loss, 5)},
                )
                break

        self.logger.log(
            "TrainingComplete",
            {"total_epochs": epoch + 1, "final_loss": round(avg_loss, 5)},
        )

    def export_samples_to_json(self, samples: list, output_path: str):
        """
        Exports raw preference pairs to a structured JSON file.

        Each entry includes:
            - Prompt
            - Output A / B
            - Format A / B
            - Preferred side
            - Token lengths
            - Rarity bonuses
            - Difficulty level
        """
        processed = []

        for item in samples:
            prompt = item.get("prompt", "")
            output_a = item.get("output_a", "")
            output_b = item.get("output_b", "")
            preferred = item.get("preferred", "a")

            # Detect format types
            fmt_a = ARMDataLoader.detect_format(output_a)
            fmt_b = ARMDataLoader.detect_format(output_b)

            # Count tokens
            token_len_a = len(output_a.split())
            token_len_b = len(output_b.split())

            # Add rarity bonus
            G = len(samples)
            F_a = (
                sum(
                    1
                    for s in samples
                    if ARMDataLoader.detect_format(s.get("output_a", "")) == fmt_a
                )
                + 1
            )
            F_b = (
                sum(
                    1
                    for s in samples
                    if ARMDataLoader.detect_format(s.get("output_b", "")) == fmt_b
                )
                + 1
            )

            rarity_bonus_a = G / F_a
            rarity_bonus_b = G / F_b

            # Infer difficulty from question length
            words = prompt.split()
            if len(words) < 20:
                difficulty = "easy"
            elif len(words) < 50:
                difficulty = "medium"
            else:
                difficulty = "hard"

            processed.append(
                {
                    "prompt": prompt,
                    "output_a": output_a,
                    "output_b": output_b,
                    "preferred": preferred,
                    "fmt_a": fmt_a,
                    "fmt_b": fmt_b,
                    "token_len_a": token_len_a,
                    "token_len_b": token_len_b,
                    "rarity_bonus_a": round(rarity_bonus_a, 3),
                    "rarity_bonus_b": round(rarity_bonus_b, 3),
                    "difficulty": difficulty,
                }
            )

        with open(output_path, "w") as fp:
            json.dump(processed, fp, indent=2)

        print(f"[INFO] Exported {len(processed)} samples to {output_path}")
