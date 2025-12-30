# stephanie/scoring/training/knowledge_trainer.py
from __future__ import annotations

import logging
import math
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from stephanie.scoring.calibration import ScoreCalibrator
from stephanie.model.knowledge import KnowledgeModel
from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner

logger = logging.getLogger(__name__)


def dpo_lite_loss(
    s_pos: torch.Tensor,
    s_neg: torch.Tensor,
    beta: float = 2.0,
    margin: float = 0.2,
) -> torch.Tensor:
    """DPO-lite loss: -log σ(β*(Δ - m)) implemented as softplus(-β*(Δ - m))"""
    return F.softplus(-(beta * (s_pos - s_neg - margin))).mean()


def _l2(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize tensor"""
    n = x.norm(dim=-1, keepdim=True)
    return x / (n + eps)


class KnowledgeTrainer(BaseTrainer):
    """
    Pairwise trainer for the 'knowledge' dimension with dual-head architecture:
    - Human head: trained on human star ratings (-5..+5)
    - AI head: trained on AI scores (0..100) with calibration

    Supports:
    - Contrastive DPO-lite training
    - AI score calibration to match human judgment
    - Alignment loss between heads
    - Proper metrics tracking
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # Configuration
        self.model_type = "knowledge"
        self.epochs = cfg.get("epochs", 30)
        self.lr = cfg.get("lr", 1e-4)
        self.batch_size = cfg.get("batch_size", 4)
        self.min_samples = cfg.get("min_samples", 16)
        self.use_tuner = cfg.get("use_tuner", True)
        self.use_early_stopping = cfg.get("early_stopping", True)
        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.beta = cfg.get("beta", 2.0)
        self.margin = cfg.get("margin", 0.2)
        self.ai_pair_weight = cfg.get("ai_pair_weight", 0.35)
        self.align_lambda = cfg.get("align_lambda", 0.05)
        self.hrm_weight = cfg.get("hrm_weight", 0.1)
        self.min_figure_score = cfg.get("min_figure_score", 0.8)

        # Auxiliary features (must match pair builder output)
        self.aux_features = cfg.get(
            "aux_features",
            [
                "human_stars",
                "pseudo_stars",
                "artifact_quality",
                "turn_pos_ratio",
                "has_retrieval",
                "retrieval_fidelity",
                "text_len_norm",
            ],
        )

        # Initialize model
        self.model = KnowledgeModel(
            dim=self.dim,
            hdim=self.hdim,
            embedding_store=self.memory.embedding,
            aux_feature_names=self.aux_features,
            device=str(self.device),
        )

        # Initialize calibrator
        self.calibrator = ScoreCalibrator()
        self._load_calibrator()

        # Logging
        self.logger.log(
            "KnowledgeTrainerInitialized",
            {
                "embedding_type": self.embedding_type,
                "device": str(self.device),
                "aux_features": self.aux_features,
                "ai_pair_weight": self.ai_pair_weight,
                "align_lambda": self.align_lambda,
            },
        )

    # --- dataloader helpers ---
    def _aux_vec(self, meta: dict) -> torch.Tensor:
        """Convert meta dict to tensor for auxiliary features"""
        vals = []
        for n in self.aux_features:
            try:
                vals.append(float(meta.get(n, 0.0)))
            except Exception:
                vals.append(0.0)
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def _pairs_to_dataset(self, pairs):
        """Convert pairs to TensorDataset"""
        G, A, B, AUXA, AUXB, S, W = [], [], [], [], [], [], []

        for p in pairs:
            goal = p.get("prompt") or p.get("goal_text") or ""
            oa, ob = p.get("output_a", ""), p.get("output_b", "")
            ma, mb = p.get("meta_a", {}) or {}, p.get("meta_b", {}) or {}
            label_source = p.get("label_source", "human")
            pair_weight = p.get("pair_weight", 1.0)

            if not (goal and oa and ob):
                continue

            try:
                # Get embeddings (L2 normalized)
                ge = torch.tensor(
                    self.memory.embedding.get_or_create(goal),
                    device=self.device,
                    dtype=torch.float32,
                )
                ea = torch.tensor(
                    self.memory.embedding.get_or_create(oa),
                    device=self.device,
                    dtype=torch.float32,
                )
                eb = torch.tensor(
                    self.memory.embedding.get_or_create(ob),
                    device=self.device,
                    dtype=torch.float32,
                )
                ge, ea, eb = _l2(ge), _l2(ea), _l2(eb)

                # Auxiliary features
                auxa = self._aux_vec(ma)
                auxb = self._aux_vec(mb)

                # Label source and weight
                source = 1.0 if label_source == "human" else 0.0
                weight = pair_weight

                G.append(ge)
                A.append(ea)
                B.append(eb)
                AUXA.append(auxa)
                AUXB.append(auxb)
                S.append(source)
                W.append(weight)

            except Exception as e:
                self.logger.log("KnowledgePairPrepError", {"error": str(e)})
                continue

        if len(G) < self.min_samples:
            self.logger.log(
                "InsufficientSamples",
                {"sample_count": len(G), "threshold": self.min_samples},
            )
            return None

        ds = TensorDataset(
            torch.stack(G),
            torch.stack(A),
            torch.stack(B),
            torch.stack(AUXA),
            torch.stack(AUXB),
            torch.tensor(S, device=self.device, dtype=torch.float32),
            torch.tensor(W, device=self.device, dtype=torch.float32),
        )
        return ds

    def _create_dataloaders(self, pairs):
        """Split pairs into train/validation dataloaders"""
        ds = self._pairs_to_dataset(pairs)
        if ds is None:
            return None, None

        n_total = len(ds)
        n_val = max(int(0.1 * n_total), 1)
        n_train = max(n_total - n_val, 1)

        train_ds, val_ds = random_split(ds, [n_train, n_val])

        train_dl = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        return train_dl, val_dl

    # --- metrics & evaluation ---
    @torch.no_grad()
    def _pair_acc(self, dl, head: str = "human") -> float:
        """Calculate pairwise accuracy for a specific head"""
        if dl is None:
            return float("nan")

        self.model.eval()
        correct, total = 0, 0

        for G, A, B, AUXA, AUXB, _, _ in dl:
            # Encode
            z_a = self.model.encoder(G, A)
            z_b = self.model.encoder(G, B)
            z_a = self.model.aux_proj(z_a, AUXA)
            z_b = self.model.aux_proj(z_b, AUXB)

            # Score with correct head
            if head == "human":
                s_a = self.model.score_h(z_a)
                s_b = self.model.score_h(z_b)
            else:  # "ai"
                s_a = self.model.score_a(z_a)
                s_b = self.model.score_a(z_b)

            # Count correct predictions
            correct += (s_a > s_b).sum().item()
            total += s_a.size(0)

        return (correct / total) if total else float("nan")

    @torch.no_grad()
    def _alignment_mse(self, dl) -> float:
        """Calculate MSE between human and AI heads on the same pairs"""
        if dl is None:
            return float("nan")

        self.model.eval()
        total_mse = 0.0
        count = 0

        for G, A, B, AUXA, AUXB, _, _ in dl:
            # Encode once
            z_a = self.model.encoder(G, A)
            z_b = self.model.encoder(G, B)
            z_a = self.model.aux_proj(z_a, AUXA)
            z_b = self.model.aux_proj(z_b, AUXB)

            # Get scores from both heads
            s_h_a = self.model.score_h(z_a)
            s_h_b = self.model.score_h(z_b)
            s_a_a = self.model.score_a(z_a)
            s_a_b = self.model.score_a(z_b)

            # Calculate MSE of relative scores
            diff_h = s_h_a - s_h_b
            diff_a = s_a_a - s_a_b
            mse = F.mse_loss(diff_h, diff_a)

            total_mse += mse.item()
            count += 1

        return total_mse / count if count > 0 else float("nan")

    @torch.no_grad()
    def _disagreement_rate(self, dl) -> float:
        """Rate at which human and AI heads disagree on pair ordering (elementwise across the batch)."""
        if dl is None:
            return float("nan")

        self.model.eval()
        disagreements = 0
        total = 0

        for G, A, B, AUXA, AUXB, _, _ in dl:
            # Encode once
            z_a = self.model.encoder(G, A)
            z_b = self.model.encoder(G, B)
            z_a = self.model.aux_proj(z_a, AUXA)
            z_b = self.model.aux_proj(z_b, AUXB)

            # Scores from both heads (shape: [batch] or [batch,1])
            s_h_a = self.model.score_h(z_a)
            s_h_b = self.model.score_h(z_b)
            s_a_a = self.model.score_a(z_a)
            s_a_b = self.model.score_a(z_b)

            # Elementwise order comparisons -> boolean vectors
            h_order = (s_h_a > s_h_b).view(-1)
            a_order = (s_a_a > s_a_b).view(-1)

            # Count disagreements elementwise
            disagreements += (h_order != a_order).sum().item()
            total += h_order.numel()

        return disagreements / total if total > 0 else float("nan")

    # --- training ---
    def _train_epoch(self, dl):
        self.model.train()
        opt = self.optimizer
        tot_loss, n_batches = 0.0, 0

        for G, A, B, AUXA, AUXB, source, weight in dl:
            z_a = self.model.encoder(G, A)
            z_b = self.model.encoder(G, B)
            z_a = self.model.aux_proj(z_a, AUXA)
            z_b = self.model.aux_proj(z_b, AUXB)

            s_h_a = self.model.score_h(z_a)
            s_h_b = self.model.score_h(z_b)
            s_a_a = self.model.score_a(z_a)
            s_a_b = self.model.score_a(z_b)

            # per-sample losses
            lh = dpo_lite_loss_per_sample(s_h_a, s_h_b, beta=self.beta, margin=self.margin)
            la = dpo_lite_loss_per_sample(s_a_a, s_a_b, beta=self.beta, margin=self.margin)

            # weights
            is_human = (source > 0.5).float()           # (batch,)
            human_w  = weight * is_human                # (batch,)
            ai_w     = weight * (1.0 - is_human)        # (batch,)

            # alignment + small reg (unweighted global)
            align = F.mse_loss(s_h_a - s_h_b, s_a_a - s_a_b)
            reg   = 1e-4 * (s_h_a.square().mean() + s_h_b.square().mean()
                            + s_a_a.square().mean() + s_a_b.square().mean())

            # combine with sample weights
            loss = ( (lh * human_w).sum() / (human_w.sum() + 1e-8)
                + self.ai_pair_weight * (la * ai_w).sum() / (ai_w.sum() + 1e-8)
                + self.align_lambda * align
                + reg )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.encoder.parameters())
                + list(self.model.aux_proj.parameters())
                + list(self.model.predictor_h.parameters())
                + list(self.model.predictor_a.parameters()),
                0.5,
            )
            opt.step()

            tot_loss += float(loss.item())
            n_batches += 1

        return tot_loss / max(1, n_batches)

    def _load_calibrator(self):
        """Load calibrator or initialize with CORRECT default ranges"""
        calibrator_path = self.cfg.get("calibrator_path")
        if calibrator_path:
            try:
                self.calibrator = ScoreCalibrator.load(calibrator_path)
                self.logger.log("CalibratorLoaded", {"path": calibrator_path})
                return
            except Exception as e:
                self.logger.log("CalibratorLoadFailed", {"error": str(e)})
        
        # Initialize with CORRECT ranges
        self.logger.log("CalibratorUsingDefaults", {})
        # Map: AI 0,40,50,75,100 → Human -5,0,1,3,5 (conservative)
        human_scores = [-5, 0, 1, 3, 5]
        ai_scores = [0, 40, 50, 75, 100]
        self.calibrator.fit(human_scores, ai_scores)

    def train(self, pairs, dimension: str = "knowledge"):
        """Train the knowledge model on contrastive pairs"""
        # Split into train/validation
        dl_train, dl_val = self._create_dataloaders(pairs)
        if dl_train is None:
            return {"error": "insufficient_data", "dimension": dimension}

        # Setup optimizer
        self.optimizer = optim.AdamW(
            list(self.model.encoder.parameters())
            + list(self.model.aux_proj.parameters())
            + list(self.model.predictor_h.parameters())
            + list(self.model.predictor_a.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )

        # Training loop
        best_val_metric = -math.inf
        patience = 0
        best_state = None
        val_pair_acc_h = float("nan")
        val_pair_acc_a = float("nan")
        alignment_mse = float("nan")
        disagreement_rate = float("nan")

        for epoch in range(self.epochs):
            # Train one epoch
            train_loss = self._train_epoch(dl_train)

            # Evaluate on validation
            val_pair_acc_h = (
                self._pair_acc(dl_val, "human") if dl_val else float("nan")
            )
            val_pair_acc_a = (
                self._pair_acc(dl_val, "ai") if dl_val else float("nan")
            )
            alignment_mse = (
                self._alignment_mse(dl_val) if dl_val else float("nan")
            )
            disagreement_rate = (
                self._disagreement_rate(dl_val) if dl_val else float("nan")
            )

            # Use human pair accuracy as primary metric
            val_metric = (
                val_pair_acc_h
                if not math.isnan(val_pair_acc_h)
                else -train_loss
            )

            # Log progress
            self.logger.log(
                "KnowledgeTrainingEpoch",
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_pair_acc_h": val_pair_acc_h,
                    "val_pair_acc_a": val_pair_acc_a,
                    "alignment_mse": alignment_mse,
                    "disagreement_rate": disagreement_rate,
                },
            )

            # Early stopping
            if val_metric > best_val_metric + self.early_stopping_min_delta:
                best_val_metric = val_metric
                patience = 0

                # Save best model state
                best_state = {
                    "encoder": {
                        k: v.cpu()
                        for k, v in self.model.encoder.state_dict().items()
                    },
                    "aux_proj": {
                        k: v.cpu()
                        for k, v in self.model.aux_proj.state_dict().items()
                    },
                    "predictor_h": {
                        k: v.cpu()
                        for k, v in self.model.predictor_h.state_dict().items()
                    },
                    "predictor_a": {
                        k: v.cpu()
                        for k, v in self.model.predictor_a.state_dict().items()
                    },
                }
            else:
                patience += 1
                if (
                    self.use_early_stopping
                    and patience >= self.early_stopping_patience
                ):
                    self.logger.log("EarlyStopping", {"patience": patience})
                    break

        # Restore best model
        if best_state is not None:
            self.model.encoder.load_state_dict(best_state["encoder"])
            self.model.aux_proj.load_state_dict(best_state["aux_proj"])
            self.model.predictor_h.load_state_dict(best_state["predictor_h"])
            self.model.predictor_a.load_state_dict(best_state["predictor_a"])

        # Save model
        locator = self.get_locator(dimension)
        self.model.save(
            encoder_path=locator.encoder_file(),
            head_h_path=locator.model_file(),     # human head
            head_a_path=locator.q_head_file(),    # AI head
            auxproj_path=locator.auxproj_file(),  # ← add this
            manifest_path=locator.meta_file(),# ← and this (optional but recommended)
            extra={
                "trained_pairs": len(pairs),
                "timestamp": datetime.now().isoformat(),
                "embedding_type": self.embedding_type,
                "version": self.version,
                "aux_features": self.aux_features,
            },
        )

        # Calibrate tuner on validation set
        if self.use_tuner and dl_val is not None:
            tuner = RegressionTuner(dimension=dimension, logger=self.logger)
            trained_count = 0

            with torch.no_grad():
                for G, A, B, AUXA, AUXB, _, _ in dl_val:
                    # Get scores from human head (gold standard)
                    z_a = self.model.encoder(G, A)
                    z_b = self.model.encoder(G, B)
                    z_a = self.model.aux_proj(z_a, AUXA)
                    z_b = self.model.aux_proj(z_b, AUXB)
                    s_a = self.model.score_h(z_a)
                    s_b = self.model.score_h(z_b)

                    # Train on relative differences
                    diffs = (s_a - s_b).cpu().numpy().tolist()
                    for d in diffs:
                        tuner.train_single(float(d), 1.0)  # target delta ~ 1.0
                        trained_count += 1

            if trained_count > 0:
                tuner.save(locator.tuner_file())
                self.logger.log(
                    "TunerSaved",
                    {
                        "file": locator.tuner_file(),
                        "trained_samples": trained_count,
                    },
                )

        # Prepare metadata
        meta = {
            "dimension": dimension,
            "model_type": "knowledge",
            "model_path": locator.model_file(),
            "q_head_path": locator.q_head_file(),
            "encoder_path": locator.encoder_file(),
            "tuner_path": locator.tuner_file() if self.use_tuner else None,
            "target_type": self.target_type,
            "embedding_type": self.embedding_type,
            "version": self.version,
            "dim": self.dim,
            "hdim": self.hdim,
            "aux_features": self.aux_features,
            "min_value": 0.0,
            "max_value": 1.0,
            "avg_loss": float(train_loss),
            "best_val_pair_acc_h": float(val_pair_acc_h)
            if not math.isnan(val_pair_acc_h)
            else None,
            "best_val_pair_acc_a": float(val_pair_acc_a)
            if not math.isnan(val_pair_acc_a)
            else None,
            "alignment_mse": float(alignment_mse)
            if not math.isnan(alignment_mse)
            else None,
            "disagreement_rate": float(disagreement_rate)
            if not math.isnan(disagreement_rate)
            else None,
            "trained_pairs": len(pairs),
            "beta": self.beta,
            "margin": self.margin,
            "ai_pair_weight": self.ai_pair_weight,
            "align_lambda": self.align_lambda,
            "timestamp": datetime.now().isoformat(),
        }

        # Save metadata
        self._save_meta_file(meta, dimension)

        # Log training stats
        self.memory.training_stats.add_from_result(
            stats={
                "best_val_pair_acc_h": meta["best_val_pair_acc_h"],
                "best_val_pair_acc_a": meta["best_val_pair_acc_a"],
                "alignment_mse": meta["alignment_mse"],
                "disagreement_rate": meta["disagreement_rate"],
            },
            model_type=self.model_type,
            target_type=self.target_type,
            dimension=dimension,
            version=self.version,
            embedding_type=self.memory.embedding.name,
            config={
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "beta": self.beta,
                "margin": self.margin,
                "ai_pair_weight": self.ai_pair_weight,
                "align_lambda": self.align_lambda,
            },
            sample_count=len(pairs),
            start_time=datetime.now(),
        )

        return meta

def dpo_lite_loss_per_sample(s_pos: torch.Tensor, s_neg: torch.Tensor, beta: float = 2.0, margin: float = 0.2):
    # returns vector loss (batch,)
    return F.softplus(-(beta * (s_pos - s_neg - margin)))
