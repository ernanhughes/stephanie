# stephanie/scoring/training/sicql_trainer.py
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from stephanie.models.belief_cartridge import BeliefCartridgeORM
from stephanie.scoring.model.sicql import (InContextQModel, PolicyHead, QHead,
                                           VHead)
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.scorable import ScorableFactory, ScorableType
from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class SICQLTrainer(BaseTrainer):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.embedding_type = self.memory.embedding.name
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim
        self.root_dir = cfg.get("model_path", "models")
        self.dimension = cfg.get("dimension", "alignment")
        self.model_type = "sicql"
        self.target_type = cfg.get("target_type", "document")
        self.version = cfg.get("model_version", "v1")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._init_config(cfg)

        self.best_loss = float("inf")
        self.early_stop_counter = 0
        self.models = {}
        self.tuners = {}
        self._load_tuners()

        if self.logger:
            self.logger.log("SICQLTrainerInitialized", {
                "dimension": self.cfg.get("dimension", "alignment"),
                "embedding_type": self.embedding_type,
                "use_gild": self.use_gild,
                "use_qmax": self.use_qmax,
                "device": str(self.device),
            },
        )

    def _init_config(self, cfg):
        """Initialize training parameters from config"""
        self.use_tuner = cfg.get("use_tuner", True)
        self.use_early_stopping = cfg.get("early_stopping", True)
        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.batch_size = cfg.get("batch_size", 32)
        self.epochs = cfg.get("epochs", 50)
        self.lr = cfg.get("lr", 1e-4)
        self.gamma = cfg.get("gamma", 0.95)  # Discount factor
        self.beta = cfg.get("beta", 1.0)  # Policy temperature
        self.entropy_weight = cfg.get("entropy_weight", 0.01)
        self.dimensions = cfg.get("dimensions", [])
        self.min_samples = cfg.get("min_samples", 10)
        self.expectile_tau = cfg.get("expectile_tau", 0.7)  # For V-head
        self.use_gild = cfg.get("use_gild", True)
        self.use_qmax = cfg.get("use_qmax", True)
        self.scorer_map = ["ebt", "svm", "mrq"]  # Policy head mapping

    def _create_sicql_dataloader(self, samples):
        """
        Accepts either:
        • singleton: {"title": str, "output": str, "score": float}
        • pairwise : {"title": str, "output_a": str, "output_b": str,
                        "value_a": float, "value_b": float}

        Builds a TensorDataset of (ctx_emb, doc_emb, score) on self.device.
        """
        import torch

        ctxs, docs, ys = [], [], []
        kept, skipped = 0, 0

        def _push(title, out, val):
            nonlocal kept
            g = torch.tensor(self.memory.embedding.get_or_create(title), dtype=torch.float32, device=self.device)
            d = torch.tensor(self.memory.embedding.get_or_create(out),   dtype=torch.float32, device=self.device)
            y = torch.tensor(float(val), dtype=torch.float32, device=self.device)
            ctxs.append(g)
            docs.append(d)
            ys.append(y)
            kept += 1

        for s in self.progress(samples, desc="Packing SICQL triples"):
            try:
                title = (s.get("title") or "").strip()
                if not title:
                    skipped += 1
                    continue

                if "output" in s and "score" in s:
                    out = (s.get("output") or "").strip()
                    val = s.get("score")
                    if out and val is not None:
                        _push(title, out, val)
                    else:
                        skipped += 1

                elif all(k in s for k in ("output_a", "output_b", "value_a", "value_b")):
                    a_out = (s.get("output_a") or "").strip()
                    b_out = (s.get("output_b") or "").strip()
                    a_val = s.get("value_a")
                    b_val = s.get("value_b")

                    # skip if both missing
                    if not a_out and not b_out:
                        skipped += 1
                        continue

                    # push A and/or B if present
                    if a_out and a_val is not None:
                        _push(title, a_out, a_val)
                    if b_out and b_val is not None:
                        _push(title, b_out, b_val)
                else:
                    skipped += 1

            except Exception as e:
                skipped += 1
                if self.logger:
                    self.logger.log("SICQLSampleError", {"error": str(e)})

        if kept < self.min_samples:
            if self.logger:
                self.logger.log("InsufficientSamples", {"kept": kept, "threshold": self.min_samples})
            return None

        X_ctx = torch.stack(ctxs)   # [N, D]
        X_doc = torch.stack(docs)   # [N, D]
        y     = torch.stack(ys)     # [N]

        dataset = TensorDataset(X_ctx, X_doc, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


    def _load_tuners(self):
        """Load regression tuners for each dimension"""
        for dim in self.dimensions:
            tuner_path = super().get_locator(dim).tuner_file()
            if os.path.exists(tuner_path):
                self.tuners[dim] = RegressionTuner(dimension=dim)
                self.tuners[dim].load(tuner_path)
            else:
                self.tuners[dim] = None
                self.logger.log(
                    "TunerMissing", {"dimension": dim, "path": tuner_path}
                )

    def _build_model(self, dimension):
        """Build or load SICQL model"""
        locator = super().get_locator(dimension)
        if locator.model_exists():
            # Load existing model
            encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
            q_head = QHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
            v_head = VHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
            pi_head = PolicyHead(
                zsa_dim=self.dim, hdim=self.hdim, num_actions=3
            ).to(self.device)

            # Load weights
            encoder.load_state_dict(
                torch.load(locator.encoder_file(), map_location=self.device)
            )
            q_head.load_state_dict(
                torch.load(locator.q_head_file(), map_location=self.device)
            )
            v_head.load_state_dict(
                torch.load(locator.v_head_file(), map_location=self.device)
            )
            pi_head.load_state_dict(
                torch.load(locator.pi_head_file(), map_location=self.device)
            )

            # Build model
            sicql_model = InContextQModel(
                encoder=encoder,
                q_head=q_head,
                v_head=v_head,
                pi_head=pi_head,
                embedding_store=self.memory.embedding,
                device=self.device,
            )
            return sicql_model

        # Build new model
        self.dim = self.memory.embedding.dim
        self.hdim = self.memory.embedding.hdim

        encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
        q_head = QHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
        v_head = VHead(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
        pi_head = PolicyHead(
            zsa_dim=self.dim, hdim=self.hdim, num_actions=3
        ).to(self.device)

        return InContextQModel(
            encoder=encoder,
            q_head=q_head,
            v_head=v_head,
            pi_head=pi_head,
            embedding_store=self.memory.embedding,
            device=self.device,
        )

    
    def _train_epoch(self, model, dataloader, epoch_idx=None):
        """Train for one epoch with all heads"""
        model.train()
        total_q_loss = 0.0
        total_v_loss = 0.0
        total_pi_loss = 0.0
        count = 0

        for ctx_emb, doc_emb, scores in tqdm(dataloader, desc=f"Training index {epoch_idx}"):
            ctx_emb = ctx_emb.to(self.device)
            doc_emb = doc_emb.to(self.device)
            scores = scores.to(self.device)

            outputs = model(ctx_emb, doc_emb)

            q_loss = F.mse_loss(outputs["q_value"], scores)

            v_loss = (
                self._expectile_loss(
                    scores - outputs["state_value"], tau=self.expectile_tau
                )
                if self.use_qmax
                else torch.tensor(0.0, device=self.device)
            )

            pi_loss = torch.tensor(0.0, device=self.device)
            if self.use_gild and "action_logits" in outputs:
                advantage = (
                    outputs["q_value"] - outputs["state_value"]
                ).detach()
                weights = torch.exp(self.beta * advantage)
                weights = weights / weights.sum()

                # Corrected reshape
                weights = weights.unsqueeze(-1)  # Ensure (batch_size, 1)

                log_probs = F.log_softmax(outputs["action_logits"], dim=-1)
                pi_loss = -(log_probs * weights).mean()

                # Optional entropy regularization
                entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
                pi_loss += self.entropy_weight * entropy

            loss = (
                q_loss * self.cfg.get("q_weight", 1.0)
                + v_loss * self.cfg.get("v_weight", 0.5)
                + pi_loss * self.cfg.get("pi_weight", 0.3)
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            self.optimizer.step()

            total_q_loss += q_loss.item() * ctx_emb.size(0)
            total_v_loss += v_loss.item() * ctx_emb.size(0)
            total_pi_loss += pi_loss.item() * ctx_emb.size(0)
            count += ctx_emb.size(0)

        avg_q = total_q_loss / count
        avg_v = total_v_loss / count
        avg_pi = total_pi_loss / count

        if self.use_qmax:
            self.scheduler["q"].step(avg_q)
        if self.use_gild:
            self.scheduler["pi"].step(avg_pi)

        return {"q": avg_q, "v": avg_v, "pi": avg_pi, "total": loss.item()}

    def _expectile_loss(self, diff, tau=0.7):
        """Compute expectile loss for V-head"""
        return torch.where(
            diff > 0, tau * diff.pow(2), (1 - tau) * diff.pow(2)
        ).mean()

    def _should_stop_early(self, current_avg):
        """Check for early stopping"""
        if not self.use_early_stopping:
            return False

        if current_avg < self.best_loss - self.early_stopping_min_delta:
            self.best_loss = current_avg
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        return self.early_stop_counter >= self.early_stopping_patience

    def _validate_tensor(self, tensor, name):
        """Validate tensor before use"""
        if tensor is None:
            self.logger.log(
                "InvalidTensor",
                {"tensor_name": name, "reason": "tensor_is_none"},
            )
            return False

        if torch.isnan(tensor).any():
            self.logger.log(
                "NaNInTensor", {"tensor_name": name, "tensor": tensor.tolist()}
            )
            return False

        return True

    def _calculate_policy_logits(self, model):
        """Calculate policy logits from policy head weights"""
        with torch.no_grad():
            policy_weights = model.pi_head.get_policy_weights()
            policy_probs = F.softmax(policy_weights, dim=-1)
            return policy_probs.tolist()

    def _calculate_policy_stability(self, policy_logits):
        """Calculate policy stability from logits"""
        if not policy_logits:
            return 0.0
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1)
        return policy_probs.max().item()

    def _calculate_policy_entropy(self, policy_logits):
        """Calculate policy entropy for versioning"""
        if not policy_logits:
            return 0.0
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1)
        return (
            -torch.sum(policy_probs * torch.log(policy_probs + 1e-8), dim=-1)
            .mean()
            .item()
        )

    def train(self, samples, dim):
        """
        Train SICQL model for a dimension
        Args:
            samples: List of training samples
            dim: Dimension to train
        Returns:
            Training statistics and model
        """
        self.logger.log("DimensionTrainingStarted", {"dimension": dim})

        # Prepare data
        dataloader = self._create_sicql_dataloader(samples)
        if not dataloader:
            return {"error": "insufficient_data", "dimension": dim}

        # Build model
        model = self._build_model(dim)
        model.train()

        # Optimizer for all heads
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = {
            "q": ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2
            ),
            "v": ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2
            ),
            "pi": ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2
            ),
        }

        # Training stats
        stats = {
            "q_losses": [],
            "v_losses": [],
            "pi_losses": [],
            "policy_entropies": [],
            "avg_q_loss": 0.0,
            "avg_v_loss": 0.0,
            "avg_pi_loss": 0.0,
            "policy_entropy": 0.0,
            "policy_stability": 0.0,
        }

        # Training loop
        losses = []
        best = float("inf")
        wait = 0

        epoch_iter = self.progress(range(1, self.epochs + 1), desc=f"SICQL[{dim}] epochs", leave=False)
        for epoch in epoch_iter:
            epoch_stats = self._train_epoch(model, dataloader, epoch_idx=epoch)
            q_avg = float(epoch_stats["q"])
            v_avg = float(epoch_stats["v"])
            pi_avg = float(epoch_stats["pi"])
            losses.append(q_avg)

            # record per-epoch stats
            stats["q_losses"].append(q_avg)
            stats["v_losses"].append(v_avg)
            stats["pi_losses"].append(pi_avg)

            # policy entropy snapshot (optional but helpful)
            try:
                with torch.no_grad():
                    w = model.pi_head.get_policy_weights()
                    p = F.softmax(w, dim=-1)
                    entropy = float(-(p * torch.log(p + 1e-8)).sum())
                stats["policy_entropies"].append(entropy)
            except Exception:
                pass

            # update epoch bar
            self.progress_postfix(
                epoch_iter, q=q_avg, v=v_avg, pi=pi_avg,
                best=(best if best < float("inf") else q_avg)
            )

            # (optional) step schedulers on epoch metrics
            try:
                self.scheduler["q"].step(q_avg)
                self.scheduler["v"].step(v_avg)
                self.scheduler["pi"].step(pi_avg)
            except Exception:
                pass

            # single early-stopper
            if q_avg < best - self.early_stopping_min_delta:
                best, wait = q_avg, 0
            else:
                wait += 1
                if self.use_early_stopping and wait >= self.early_stopping_patience:
                    self.logger.log("SICQLEarlyStopping", {"epoch": epoch, "best_loss": best})
                    break

            # per-epoch log
            self.logger.log("SICQLTrainingEpoch", {
                "epoch": epoch, "q_loss": q_avg, "v_loss": v_avg, "pi_loss": pi_avg
            })

        # Final stats (guard against empty lists)
        stats["avg_q_loss"] = float(np.mean(stats["q_losses"])) if stats["q_losses"] else float("nan")
        stats["avg_v_loss"] = float(np.mean(stats["v_losses"])) if stats["v_losses"] else float("nan")
        stats["avg_pi_loss"] = float(np.mean(stats["pi_losses"])) if stats["pi_losses"] else float("nan")
        stats["policy_entropy"] = float(np.mean(stats["policy_entropies"])) if stats["policy_entropies"] else float("nan")
        stats["policy_stability"] = (
            max(stats["policy_entropies"]) if stats["policy_entropies"] else float("nan")
        )

        # Save model
        meta = self._save_model(model, dim, stats)
        stats.update(meta)

        # Log to database (match your method's signature)
        self._log_training_stats(dim, meta)

        self.logger.log("DimensionTrainingComplete", {
            "dimension": dim,
            "final_q_loss": stats["avg_q_loss"],
            "final_v_loss": stats["avg_v_loss"],
            "final_pi_loss": stats["avg_pi_loss"],
        })

        # Final stats
        stats["avg_q_loss"] = np.mean(stats["q_losses"])
        stats["avg_v_loss"] = np.mean(stats["v_losses"])
        stats["avg_pi_loss"] = np.mean(stats["pi_losses"])
        stats["policy_entropy"] = np.mean(stats["policy_entropies"])
        stats["policy_stability"] = (
            max(stats["policy_entropies"])
            if stats["policy_entropies"]
            else 0.0
        )

        # Save model
        meta = self._save_model(model, dim, stats)
        stats.update(meta)

        # Log to database
        self._log_training_stats(dim, meta, samples=samples, dataloader=dataloader)

        self.logger.log(
            "DimensionTrainingComplete",
            {
                "dimension": dim,
                "final_q_loss": stats["avg_q_loss"],
                "final_v_loss": stats["avg_v_loss"],
                "final_pi_loss": stats["avg_pi_loss"],
            },
        )

        # Cache model
        self.models[dim] = model
        return stats

    def _log_training_stats(
        self,
        dim: str,
        meta: dict,
        *,
        samples: list | None = None,
        dataloader=None,
        pipeline_run_id: int | None = None,
        goal_id: int | None = None,
        model_version_id: int | None = None,
    ):
        """
        Log SICQL training stats using TrainingStats.add_from_result schema.

        Expects in `meta` (already computed in `train`):
        - avg_q_loss, avg_v_loss, avg_pi_loss
        - policy_entropy, policy_stability, policy_logits
        - (optionally) last_q_loss, last_v_loss, last_pi_loss
        - version (required)
        """

        # Derive counts
        sample_count = len(samples) if samples is not None else 0
        valid_samples = 0
        if dataloader is not None and getattr(dataloader, "dataset", None) is not None:
            try:
                valid_samples = len(dataloader.dataset)
            except Exception:
                valid_samples = 0
        invalid_samples = max(sample_count - valid_samples, 0) if sample_count else 0

        # Per-epoch “last” losses are optional—fall back to averages
        stats_payload = {
            "q_loss":      meta.get("last_q_loss", meta.get("avg_q_loss")),
            "v_loss":      meta.get("last_v_loss", meta.get("avg_v_loss")),
            "pi_loss":     meta.get("last_pi_loss", meta.get("avg_pi_loss")),
            "avg_q_loss":  meta.get("avg_q_loss"),
            "avg_v_loss":  meta.get("avg_v_loss"),
            "avg_pi_loss": meta.get("avg_pi_loss"),
            "policy_entropy":   meta.get("policy_entropy"),
            "policy_stability": meta.get("policy_stability"),
            "policy_logits":    meta.get("policy_logits"),
        }

        self.memory.training_stats.add_from_result(
            stats=stats_payload,
            model_type="sicql",
            target_type=self.target_type,
            dimension=dim,
            version=meta["version"],
            embedding_type=self.embedding_type,
            config={
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "use_tuner": self.use_tuner,
                "use_early_stopping": self.use_early_stopping,
                "patience": self.early_stopping_patience,
                "min_delta": self.early_stopping_min_delta,
                "dim": self.dim,
                "hdim": self.hdim,
                "expectile_tau": getattr(self, "expectile_tau", 0.7),
                "use_gild": getattr(self, "use_gild", True),
                "beta": getattr(self, "beta", 1.0),
                "entropy_weight": getattr(self, "entropy_weight", 0.01),
                "use_qmax": getattr(self, "use_qmax", True),
            },
            sample_count=sample_count,
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            goal_id=goal_id,
            model_version_id=model_version_id,
            start_time=meta.get("start_time"),      # if you stored it
            end_time=meta.get("end_time"),          # if you stored it
            pipeline_run_id=pipeline_run_id,
        )


    def _train_sicql(self, model, dataloader, output_dir):
        """Train SICQL model with all heads"""
        model.train()
        best_loss = float("inf")
        patience_counter = 0

        # Build optimizers
        optimizers = {
            "encoder": optim.Adam(model.encoder.parameters(), lr=self.lr),
            "q_head": optim.Adam(model.q_head.parameters(), lr=self.lr),
            "v_head": optim.Adam(model.v_head.parameters(), lr=self.lr),
            "pi_head": optim.Adam(model.pi_head.parameters(), lr=self.lr),
        }

        # Build schedulers
        self.scheduler = {
            "encoder": ReduceLROnPlateau(
                optimizers["encoder"], mode="min", factor=0.5, patience=2
            ),
            "q_head": ReduceLROnPlateau(
                optimizers["q_head"], mode="min", factor=0.5, patience=2
            ),
            "v_head": ReduceLROnPlateau(
                optimizers["v_head"], mode="min", factor=0.5, patience=2
            ),
            "pi_head": ReduceLROnPlateau(
                optimizers["pi_head"], mode="min", factor=0.5, patience=2
            ),
        }

        # Training loop
        for epoch in range(self.epochs):
            total_q_loss = 0.0
            total_v_loss = 0.0
            total_pi_loss = 0.0
            count = 0

            for ctx_emb, doc_emb, scores in tqdm(
                dataloader, desc=f"Epoch {epoch + 1}"
            ):
                # Device management
                ctx_emb = ctx_emb.to(self.device)
                doc_emb = doc_emb.to(self.device)
                scores = scores.to(self.device)

                # Forward pass
                outputs = model(ctx_emb, doc_emb)

                # Q-head loss
                q_loss = F.mse_loss(outputs["q_value"], scores)

                # V-head loss
                v_loss = self._expectile_loss(
                    scores - outputs["state_value"],
                    tau=self.cfg.get("expectile", 0.7),
                )

                # Policy head loss
                pi_loss = torch.tensor(0.0, device=self.device)
                if self.use_gild:
                    advantage = (
                        outputs["q_value"] - outputs["state_value"]
                    ).detach()
                    weights = torch.exp(self.beta * advantage)
                    weights = weights / weights.sum()

                    policy_probs = F.softmax(outputs["action_logits"], dim=-1)
                    entropy = -torch.sum(
                        policy_probs * torch.log(policy_probs + 1e-8), dim=-1
                    ).mean()

                    pi_loss = -(
                        F.log_softmax(outputs["action_logits"], dim=-1)
                        * weights
                    ).mean()
                    pi_loss += self.entropy_weight * entropy

                # Backward pass
                optimizers["q_head"].zero_grad()
                q_loss.backward()
                optimizers["q_head"].step()

                optimizers["v_head"].zero_grad()
                v_loss.backward()
                optimizers["v_head"].step()

                if self.use_gild:
                    optimizers["pi_head"].zero_grad()
                    pi_loss.backward()
                    optimizers["pi_head"].step()

                # Track losses
                total_q_loss += q_loss.item() * ctx_emb.size(0)
                total_v_loss += v_loss.item() * ctx_emb.size(0)
                total_pi_loss += pi_loss.item() * ctx_emb.size(0)
                count += ctx_emb.size(0)

            # End of epoch
            avg_q = total_q_loss / count
            avg_v = total_v_loss / count
            avg_pi = total_pi_loss / count

            # Early stopping
            if avg_q < best_loss - self.early_stopping_min_delta:
                best_loss = avg_q
                patience_counter = 0
                # Save best model
                torch.save(
                    model.encoder.state_dict(), f"{output_dir}/encoder.pt"
                )
                torch.save(
                    model.q_head.state_dict(), f"{output_dir}/q_head.pt"
                )
                torch.save(
                    model.v_head.state_dict(), f"{output_dir}/v_head.pt"
                )
                torch.save(
                    model.pi_head.state_dict(), f"{output_dir}/pi_head.pt"
                )
            else:
                patience_counter += 1

            # Log epoch
            self.logger.log(
                "SICQLTrainingEpoch",
                {
                    "epoch": epoch + 1,
                    "q_loss": avg_q,
                    "v_loss": avg_v,
                    "pi_loss": avg_pi,
                    "lr": optimizers["q_head"].param_groups[0]["lr"],
                },
            )

            # Check for early stopping
            if patience_counter >= self.early_stopping_patience:
                self.logger.log(
                    "SICQLEarlyStopping",
                    {"epoch": epoch + 1, "best_loss": best_loss},
                )
                break

        self.logger.log("SICQLTrainingComplete", {"best_loss": best_loss})
        return model

    def _save_model(self, model, dimension, stats):
        """Save SICQL model components"""
        locator = super().get_locator(dimension)
        # Save components separately
        torch.save(model.encoder.state_dict(), locator.encoder_file())
        torch.save(model.q_head.state_dict(), locator.q_head_file())
        torch.save(model.v_head.state_dict(), locator.v_head_file())
        torch.save(model.pi_head.state_dict(), locator.pi_head_file())

        # Calculate policy metrics
        policy_logits = model.pi_head.get_policy_weights().tolist()
        policy_probs_tensor = F.softmax(torch.tensor(policy_logits), dim=-1)
        policy_probs = policy_probs_tensor.tolist()
        policy_entropy = -torch.sum(
            policy_probs_tensor * torch.log(policy_probs_tensor + 1e-8)
        ).item()
        policy_stability = max(policy_probs)


        # Build metadata
        meta = {
            "dim": self.dim,
            "hdim": self.hdim,
            "dimension": dimension,
            "version": self.cfg.get("model_version", "v1"),
            "avg_q_loss": float(stats["avg_q_loss"]),
            "avg_v_loss": float(stats["avg_v_loss"]),
            "avg_pi_loss": float(stats["avg_pi_loss"]),
            "policy_entropy": float(policy_entropy),
            "policy_stability": float(policy_stability),
            "policy_logits": policy_logits,
            "policy_probs": policy_probs,
            "embedding_type": self.embedding_type,
            "max_value": 100,
            "min_value": 0,
            "device": str(self.device), 
            "timestamp": datetime.now().isoformat(),
        }

        super()._save_meta_file(meta, dimension)
        return meta

    def run(self, context: dict) -> dict:
        """Main entry point for training"""
        documents = context.get("documents", [])
        # Train each dimension
        results = {}
        for dim in self.dimensions:
            # Get training samples
            samples = self._get_samples(context, documents, dim)
            if not samples:
                continue

            # Train model
            stats = self.train(samples, dim)
            if "error" in stats:
                continue

            # Update belief cartridges
            self._update_belief_cartridge(context, dim, stats)
            results[dim] = stats

        # Update context with results
        context["training_stats"] = results
        return context

    def _get_samples(self, context, documents, dim):
        """Get training samples for dimension"""
        samples = []
        goal = context.get("goal", {})
        for doc in documents:
            scorable = ScorableFactory.from_dict(doc, ScorableType.DOCUMENT)
            score = self.memory.scores.get_score(goal.id, scorable.id)
            if score:
                samples.append(
                    {
                        "title": goal.get("goal_text", ""),
                        "output": scorable.text,
                        "score": score.score,
                    }
                )
        return samples

    def _update_belief_cartridge(self, context, dim, stats):
        """Update belief cartridges with policy stats"""
        policy_logits = stats.get("policy_logits", [0.3, 0.7, 0.0])
        policy_probs = F.softmax(torch.tensor(policy_logits), dim=-1).tolist()

        # Build belief cartridge
        cartridge = BeliefCartridgeORM(
            title=f"{dim} policy",
            content=f"Policy head weights: {policy_probs}",
            goal_id=context.get("goal_id"),
            domain=dim,
            policy_logits=policy_probs,
            policy_entropy=stats.get("policy_entropy", 1.05),
            policy_stability=stats.get("policy_stability", 0.82),
        )
        self.memory.session.add(cartridge)
        self.memory.session.commit()
