# stephanie/scoring/training/tiny_recursion_trainer.py
from __future__ import annotations
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from stephanie.models.tiny_recursion import TinyRecursionModel
from stephanie.models.training_stats import TrainingStatsORM
from stephanie.scoring.training.base_trainer import BaseTrainer


class TinyRecursionTrainer(BaseTrainer):
    """
    Trainer for TinyRecursionModel — aligns with MRQTrainer pattern
    but trains recursive reasoning dynamics instead of value predictors.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.epochs = cfg.get("epochs", 50)
        self.lr = cfg.get("lr", 1e-4)
        self.batch_size = cfg.get("batch_size", 4)
        self.dropout = cfg.get("dropout", 0.1)
        self.use_attention = cfg.get("use_attention", False)
        self.n_recursions = cfg.get("n_recursions", 6)
        self.vocab_size = cfg.get("vocab_size", 1024)

        self.model = TinyRecursionModel(
            d_model=self.dim,
            n_layers=2,
            n_recursions=self.n_recursions,
            vocab_size=self.vocab_size,
            use_attention=self.use_attention,
            dropout=self.dropout,
        ).to(self.device)

    # ------------------------------
    # Data prep: each sample = (x, y, z, target)
    # ------------------------------
    def _create_dataloader(self, samples):
        xs, ys, zs, labels = [], [], [], []
        dropped = 0
        for s in samples:
            try:
                x = torch.tensor(
                    self.memory.embedding.get_or_create(s["x"]),
                    dtype=torch.float,
                )
                y = torch.tensor(
                    self.memory.embedding.get_or_create(s["y"]),
                    dtype=torch.float,
                )
                z = torch.tensor(
                    self.memory.embedding.get_or_create(s["z"]),
                    dtype=torch.float,
                )

                t = s.get("target", None)
                if t is None:
                    dropped += 1
                    continue
                try:
                    t = int(t)
                    if not (0 <= t <= 100):
                        dropped += 1
                        continue
                except Exception:
                    dropped += 1
                    continue

                xs.append(x)
                ys.append(y)
                zs.append(z)
                labels.append(torch.tensor(t, dtype=torch.long))
            except Exception as e:
                dropped += 1
                if self.logger:
                    self.logger.log(
                        "TinyRecursionSampleError", {"error": str(e)}
                    )
                continue

        if len(xs) < 2:
            return None

        dataset = TensorDataset(
            torch.stack(xs).to(self.device),
            torch.stack(ys).to(self.device),
            torch.stack(zs).to(self.device),
            torch.stack(labels).to(self.device),
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    # ------------------------------
    # Train one epoch
    # ------------------------------
    def _train_epoch(self, model, dataloader):
        model.train()
        total_loss = 0.0
        count = 0

        for x, y, z, targets in dataloader:
            logits, halt_p, _ = model(x, y, z)
            ce_loss = F.cross_entropy(logits, targets)
            halt_loss = (
                torch.mean((halt_p - 1.0) ** 2) * 0.1
            )  # optional stabilization
            loss = ce_loss + halt_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            count += x.size(0)

        return total_loss / count

    # ------------------------------
    # Main train loop
    # ------------------------------
    def train(self, samples, dimension):
        dataloader = self._create_dataloader(samples)
        if not dataloader:
            return {"error": "insufficient_data"}

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        best_loss = float("inf")
        patience, wait = 3, 0

        for epoch in range(self.epochs):
            avg_loss = self._train_epoch(self.model, dataloader)
            self.logger.log(
                "TinyRecursionEpoch", {"epoch": epoch + 1, "loss": avg_loss}
            )
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        locator = self.get_locator(dimension)
        torch.save(self.model.state_dict(), locator.model_file())

        meta = {
            "dimension": dimension,
            "model_type": "tiny_recursion",
            "target_type": self.target_type,
            "embedding_type": self.embedding_type,
            "version": self.version,
            "epochs": self.epochs,
            "avg_loss": best_loss,
            "timestamp": datetime.now().isoformat(),
        }
        self._save_meta_file(meta, dimension)

        training_stat = TrainingStatsORM(
            model_type="tiny_recursion",
            target_type=self.target_type,
            dimension=dimension,
            version=self.version,
            embedding_type=self.embedding_type,
            avg_q_loss=best_loss,
        )
        self.memory.session.add(training_stat)
        self.memory.session.commit()
        return meta

    def _extract_target_0_100(
        self, sample, dimension: str | None
    ) -> int | None:
        """
        Priority:
        1) ai_knowledge_score   (gold)
        2) sample['target']     (explicit)
        3) scores[dimension].score (derived)
        Returns int in [0, 100] or None.
        """
        # 1) gold: ai_knowledge_score
        aks = sample.get("ai_knowledge_score", None)
        if aks is not None:
            try:
                v = float(aks)
                return int(max(0.0, min(100.0, v)))
            except Exception:
                pass

        # 2) explicit target
        raw = sample.get("target", None)
        if raw is not None:
            try:
                v = float(raw)
                return int(max(0.0, min(100.0, v)))
            except Exception:
                try:
                    v = float(str(raw).strip())
                    return int(max(0.0, min(100.0, v)))
                except Exception:
                    pass

        # 3) derived from scores[] for requested dimension
        if dimension:
            for sc in sample.get("scores") or []:
                if (sc.get("dimension") or "").lower() == dimension.lower():
                    v = sc.get("score")
                    if v is None:
                        continue
                    try:
                        vf = float(v)
                        return int(max(0.0, min(100.0, vf)))
                    except Exception:
                        continue

        return None
