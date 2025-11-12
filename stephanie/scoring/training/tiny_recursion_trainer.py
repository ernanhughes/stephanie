# stephanie/scoring/training/tiny_recursion_trainer.py
"""
TinyRecursionModel Trainer (Tiny+)

A specialized trainer for the TinyRecursionModel that implements multi-objective
training with heteroscedastic regression and auxiliary losses. This trainer handles
multiple data schemas and produces dimension-specific models with comprehensive
training telemetry.

Key Features:
- Heteroscedastic regression for score prediction with uncertainty estimation
- Multiple auxiliary objectives: bucket classification, disagreement, reconstruction
- Support for various input schemas (native, singleton, pairwise, HRM)
- Comprehensive training monitoring and validation
- Early stopping and model checkpointing

"""

from __future__ import annotations

import logging
import math
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from stephanie.scoring.model.tiny_recursion import TinyRecursionModel
from stephanie.scoring.training.base_trainer import BaseTrainer

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

log = logging.getLogger(__name__)

def _bucket3(y01: torch.Tensor) -> torch.Tensor:
    """
    Convert continuous scores to 3-class bucket labels.
    
    Args:
        y01: Tensor of scores in range [0, 1]
        
    Returns:
        Long tensor with bucket indices:
        - 0: scores < 1/3
        - 1: scores in [1/3, 2/3)
        - 2: scores >= 2/3
    """
    # <1/3 => 0, [1/3,2/3) => 1, >=2/3 => 2
    edges = torch.tensor([1/3, 2/3], device=y01.device, dtype=y01.dtype)
    return (y01 >= edges[0]).long() + (y01 >= edges[1]).long()


class TinyTrainer(BaseTrainer):
    """
    Trainer for TinyRecursionModel (Tiny+) with multi-objective optimization.
    
    This trainer implements a comprehensive training regimen that combines:
    - Main heteroscedastic regression objective
    - Multiple auxiliary objectives for regularization and feature learning
    - Support for various input data formats And see this was a complete waste of timed
    - Extensive monitoring and validation
    
    The model produces separate instances for each quality dimension.
    
    Attributes:
        model_type: Identifier for model architecture ("tiny")
        target_type: Type of scoring target ("document", "sentence", etc.)
        version: Model version identifier
        epochs: Number of training epochs
        lr: Learning rate for optimizer
        batch_size: Training batch size
        dropout: Dropout rate for model regularization
        use_attention: Whether to use attention mechanisms
        n_recursions: Number of recursion steps in model
        halt_lambda: Weight for halting regularization loss
        grad_clip: Gradient clipping value
        w_aux3: Weight for 3-class auxiliary classification
        w_disagree: Weight for disagreement prediction
        w_recon: Weight for reconstruction loss
        w_cons: Weight for consistency regularization
        w_sae_recon: Weight for sparse autoencoder reconstruction
        w_ood: Weight for out-of-distribution detection
    """

    def __init__(self, cfg, memory, container, logger):
        """Initialize TinyTrainer with configuration and dependencies."""
        super().__init__(cfg, memory, container, logger)

        # --- Identity / paths -------------------------------------------------
        self.model_type   = "tiny"
        self.target_type  = cfg.get("target_type", "document")
        self.version      = cfg.get("model_version", "v1")

        # --- Core knobs -------------------------------------------------------
        self.epochs        = int(cfg.get("epochs", 20))
        self.lr            = float(cfg.get("lr", 3e-5))           # conservative default
        self.batch_size    = int(cfg.get("batch_size", 16))
        self.dropout       = float(cfg.get("dropout", 0.1))
        self.use_attention = bool(cfg.get("use_attention", False))
        self.n_recursions  = int(cfg.get("n_recursions", 6))
        self.halt_lambda   = float(cfg.get("halt_lambda", 0.05))  # halting is a light regularizer
        self.grad_clip     = float(cfg.get("grad_clip", 0.5))

        # Aux loss weights
        self.w_aux3        = float(cfg.get("w_aux3", 0.3))
        self.w_disagree    = float(cfg.get("w_disagree", 0.3))
        self.w_recon       = float(cfg.get("w_recon", 0.2))
        self.w_cons        = float(cfg.get("w_consistency", 0.2))
        self.w_sae_recon   = float(cfg.get("w_sae_recon", 0.0))   # 0 = off by default
        self.w_ood         = float(cfg.get("w_ood", 0.0))         # 0 = off by default

        # --- Telemetry --------------------------------------------------------
        self.show_progress       = bool(cfg.get("show_progress", True))
        self.progress_every      = max(1, int(cfg.get("progress_every", 500)))
        self.log_every_steps     = max(1, int(cfg.get("log_every_steps", 50)))
        self.label_hist_bucket   = int(cfg.get("label_hist_bucket", 10))
        self.log_label_histogram = bool(cfg.get("log_label_histogram", True))

        # --- Validation / reproducibility ------------------------------------
        self.validation_ratio = float(cfg.get("validation_ratio", 0.1))
        self.seed             = int(cfg.get("seed", 42))
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        # --- Model ------------------------------------------------------------
        self.model = TinyRecursionModel(
            d_model=self.dim,
            n_layers=int(cfg.get("n_layers", 2)),
            n_recursions=self.n_recursions,
            vocab_size=int(cfg.get("vocab_size", 101)),   # kept for classifier compatibility
            use_attention=self.use_attention,
            dropout=self.dropout,
            attn_heads=int(cfg.get("attn_heads", 4)),
            step_scale=float(cfg.get("step_scale", 0.1)),
            consistency_mask_p=float(cfg.get("consistency_mask_p", 0.10)),
            len_norm_L=float(cfg.get("len_norm_L", 512.0)),
        ).to(self.device)

    # ------------------------------
    # Data prep
    # ------------------------------

    def _create_dataloader(self, samples: List[Dict[str, Any]]) -> Tuple[Optional[DataLoader], int, int]:
        """
        Create DataLoader from sample dictionaries with multiple schema support.
        
        Supports multiple input formats:
        - Native Tiny+ schema: x, y, z, target
        - Singleton format: goal_text/output with score
        - Pairwise format: output_a/output_b with comparative scores
        - HRM format: goal_text/scorable_text with target_score
        
        Args:
            samples: List of sample dictionaries with various possible schemas
            
        Returns:
            Tuple of (DataLoader, kept_count, dropped_count) or (None, kept, dropped) if insufficient data
        """
        xs, ys, zs = [], [], []
        y01, halt_targets, seq_lens = [], [], []
        kept = dropped = 0
        label_counts = Counter()

        use_tqdm = bool(self.show_progress and tqdm is not None)
        it = tqdm(samples, desc="Packing Tiny+ samples", unit="samp") if use_tqdm else samples

        def _push(goal: str, doc: str, target: float, *, z_text: Optional[str] = None, halt_t: float = 1.0, slen: int = 0):
            """Internal helper to process and validate a single sample."""
            nonlocal kept, dropped
            try:
                # Get embeddings for text inputs
                x = torch.tensor(self.memory.embedding.get_or_create(goal), dtype=torch.float32, device=self.device)
                y = torch.tensor(self.memory.embedding.get_or_create(doc),  dtype=torch.float32, device=self.device)
                z = torch.tensor(self.memory.embedding.get_or_create(z_text if z_text is not None else goal),
                                dtype=torch.float32, device=self.device)

                # ---- Normalize & sanitize inputs (prevents recursion amplification / NaNs)
                def _safe_vec(t):
                    """Safely normalize vector, handling NaN/inf values."""
                    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                    norm = t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    return t / norm

                x = _safe_vec(x); y = _safe_vec(y); z = _safe_vec(z)
                if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.isfinite(z).all():
                    dropped += 1
                    return

                # normalize target → [0,1]
                t = float(target)
                t = (max(0.0, min(100.0, t)) / 100.0) if t > 1.0 else max(0.0, min(1.0, t))

                xs.append(x); ys.append(y); zs.append(z)
                y01.append(t); halt_targets.append(float(halt_t)); seq_lens.append(int(slen))
                label_counts[int(round(t * 100))] += 1
                kept += 1
            except Exception as e:
                dropped += 1
                if self.logger: self.logger.log("TinyRecursionSampleError", {"error": str(e)})

        # Process all samples with schema detection
        for s in it:
            # Native Tiny+ schema
            if "x" in s and "y" in s and "z" in s and "target" in s:
                _push(s["x"], s["y"], s["target"], z_text=s.get("z"), halt_t=s.get("halt_target", 1.0), slen=s.get("seq_len", 0))
                continue

            # Singleton (SICQL/MRQ style)
            title = (s.get("goal_text") or s.get("title") or "").strip()
            if "output" in s and ("score" in s or "target_score" in s):
                out = (s.get("scorable_text") or s.get("output") or "").strip()
                val = s.get("target_score", s.get("score"))
                if title and out and (val is not None):
                    _push(title, out, val, z_text=title)
                else:
                    dropped += 1
                continue

            # Pairwise
            if all(k in s for k in ("output_a","output_b","value_a","value_b")):
                a_out = (s.get("output_a") or "").strip()
                b_out = (s.get("output_b") or "").strip()
                a_val = s.get("value_a"); b_val = s.get("value_b")
                if title:
                    if a_out and a_val is not None: _push(title, a_out, a_val, z_text=title)
                    if b_out and b_val is not None: _push(title, b_out, b_val, z_text=title)
                else:
                    dropped += 1
                continue

            # HRM/raw
            if ("goal_text" in s and "scorable_text" in s and ("target_score" in s or "score" in s)):
                out = (s.get("scorable_text") or "").strip()
                val = s.get("target_score", s.get("score"))
                _push(title, out, val, z_text=title)
                continue

            dropped += 1

            if use_tqdm: it.set_postfix(kept=kept, drop=dropped)

        if use_tqdm and hasattr(it, "close"): it.close()

        # Log label distribution for analysis
        if self.logger and self.log_label_histogram:
            exact = {int(k): int(v) for k, v in sorted(label_counts.items())}
            bucketed = self._bucketize_counts(label_counts, self.label_hist_bucket)
            self.logger.log("TinyPlusLabelHistogram", {
                "kept": int(kept), "dropped": int(dropped),
                "exact": exact, "bucket_size": int(self.label_hist_bucket),
                "bucketed": bucketed
            })

        if kept < self.min_samples:
            return None, kept, dropped

        # Create TensorDataset and DataLoader
        dataset = TensorDataset(
            torch.stack(xs), torch.stack(ys), torch.stack(zs),
            torch.tensor(y01, dtype=torch.float32, device=self.device).unsqueeze(-1),
            torch.tensor(halt_targets, dtype=torch.float32, device=self.device).unsqueeze(-1),
            torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader, kept, dropped

    # ------------------------------
    # Loss Functions
    # ------------------------------

    @staticmethod
    def _heteroscedastic_regression_loss(score: torch.Tensor, target01: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Compute heteroscedastic regression loss with uncertainty estimation.
        
        This loss adapts to the uncertainty in predictions by learning
        a variance term that scales the regression loss.
        
        Args:
            score: Predicted scores [B, 1]
            target01: Ground truth scores in [0, 1] [B, 1]
            log_var: Learned log variance [B, 1]
            
        Returns:
            Scalar loss value
        """
        log_var = log_var.clamp(-5.0, 5.0)  # defensive clamp to avoid precision explosion
        inv_var = torch.exp(-log_var)
        diff2   = (score - target01).pow(2)
        return (inv_var * diff2 + log_var).mean()

    @staticmethod
    def _cosine_recon_loss(y_recon: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine reconstruction loss.
        
        Measures how well the model can reconstruct the input embedding,
        encouraging meaningful internal representations.
        
        Args:
            y_recon: Reconstructed embedding
            y_true: Original embedding
            
        Returns:
            Cosine distance loss in range [0, 1]
        """
        # 1 - cosine in [0,2] → clamp to [0,1]
        cos = F.cosine_similarity(y_recon, y_true, dim=-1, eps=1e-8).unsqueeze(-1)
        return (1 - cos).clamp(0, 1).mean()

    # ------------------------------
    # Epoch training
    # ------------------------------

    def _train_epoch(self, model: TinyRecursionModel, dataloader: DataLoader, epoch_idx: int) -> float:
        """
        Train model for one epoch.
        
        Args:
            model: TinyRecursionModel instance
            dataloader: Training data loader
            epoch_idx: Current epoch index
            
        Returns:
            Average training loss for the epoch
        """
        model.train()
        total_loss = 0.0
        count = 0

        use_tqdm = bool(self.show_progress and tqdm is not None)
        it = tqdm(dataloader, desc=f"Epoch {epoch_idx}", unit="batch", leave=False) if use_tqdm else dataloader

        for step, batch in enumerate(it, start=1):
            x, y, z, target01, halt_target, seq_len = batch

            # Forward pass with auxiliary outputs
            logits, halt_logits, _, aux = model(x, y, z, seq_len=seq_len, return_aux=True)

            # Main loss: heteroscedastic regression on score/log_var
            L_main = self._heteroscedastic_regression_loss(aux["score"], target01, aux["log_var"])

            # Auxiliary losses for multi-objective training
            buckets = _bucket3(target01.squeeze(-1))
            L_aux3  = F.cross_entropy(aux["aux3_logits"], buckets)  # 3-class classification
            L_dis   = F.smooth_l1_loss(aux["disagree_hat"], (target01 - aux["score"].detach()).abs())  # Disagreement prediction
            L_recon = self._cosine_recon_loss(aux["y_recon"], y)  # Reconstruction quality
            L_cons  = F.mse_loss(aux["consistency_hat"], aux["consistency_target"])  # Consistency regularization

            # Optional losses (weight=0 means disabled)
            L_sae = torch.zeros((), device=self.device)
            if self.w_sae_recon > 0.0 and "concept_vec" in aux:
                L_sae = aux["concept_vec"].abs().mean()  # Sparse autoencoder reconstruction

            L_ood = torch.zeros((), device=self.device)
            if self.w_ood > 0.0 and "ood_hat" in aux:
                L_ood = F.binary_cross_entropy(aux["ood_hat"], torch.ones_like(aux["ood_hat"]))  # OOD detection

            L_halt = F.binary_cross_entropy_with_logits(halt_logits.unsqueeze(-1), halt_target)  # Halting regularization

            # Check components for finiteness & sanity
            all_terms = torch.stack([
                L_main.detach(),
                L_aux3.detach(),
                L_dis.detach(),
                L_recon.detach(),
                L_cons.detach(),
                L_sae.detach(),
                L_ood.detach(),
                L_halt.detach()
            ])
            if (not torch.isfinite(all_terms).all()) or (all_terms.abs().max() > 1e6):
                if self.logger:
                    self.logger.log("TinyPlusNaNBatch", {
                        "epoch": epoch_idx,
                        "step": step,
                        "any_nan": bool(not torch.isfinite(all_terms).all()),
                        "max_abs": float(all_terms.abs().max().item())
                    })
                self.optimizer.zero_grad(set_to_none=True)
                continue  # skip this batch

            # Combined loss with weighting
            loss = (
                L_main
                + self.w_aux3 * L_aux3
                + self.w_disagree * L_dis
                + self.w_recon * L_recon
                + self.w_cons * L_cons
                + self.w_sae_recon * L_sae
                + self.w_ood * L_ood
                + self.halt_lambda * L_halt
            )

            if (not torch.isfinite(loss)) or (abs(loss.item()) > 1e7):
                if self.logger:
                    self.logger.log("TinyPlusUnstableLoss", {
                        "epoch": epoch_idx,
                        "step": step,
                        "loss": float(loss.item()) if torch.isfinite(loss) else float('nan')
                    })
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            self.optimizer.step()

            bsz = x.size(0)
            total_loss += loss.item() * bsz
            count += bsz

            # Progress reporting
            if use_tqdm:
                it.set_postfix(loss=f"{loss.item():.4f}")
            elif self.logger and (step % self.log_every_steps == 0):
                self.logger.log("TinyPlusBatch", {
                    "epoch": epoch_idx,
                    "step": step,
                    "loss": float(loss.item()),
                    "L_main": float(L_main.item()),
                    "L_aux3": float(L_aux3.item()),
                    "L_dis": float(L_dis.item()),
                    "L_recon": float(L_recon.item()),
                    "L_cons": float(L_cons.item()),
                    "L_sae": float(L_sae.item()),
                    "L_ood": float(L_ood.item()),
                    "L_halt": float(L_halt.item()),
                })

        if use_tqdm and hasattr(it, "close"):
            it.close()

        return total_loss / max(1, count)

    # ------------------------------
    # Validation
    # ------------------------------

    @torch.no_grad()
    def _validate(self, model: TinyRecursionModel, dataloader: Optional[DataLoader]) -> Dict[str, float]:
        """
        Run validation and compute comprehensive metrics.
        
        Args:
            model: Model to validate
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        if not dataloader:
            return {}

        model.eval()
        scores, targets = [], []
        # 10 metric lists for comprehensive validation
        entropies, uncerts, disagree, recon_sim, cons_hat, temp01, jac, spars, ood, len_eff = (
            [] for _ in range(10)
        )

        for x, y, z, target01, _, seq_len in dataloader:
            _, _, _, aux = model(x, y, z, seq_len=seq_len, return_aux=True)
            s = aux["score"].detach().cpu().view(-1)
            t = target01.detach().cpu().view(-1)

            scores.append(s)
            targets.append(t)
            # Collect various auxiliary metrics for analysis
            entropies.append(aux["entropy_aux"].detach().cpu().view(-1))
            uncerts.append(aux["uncertainty"].detach().cpu().view(-1))
            disagree.append(aux["disagree_hat"].detach().cpu().view(-1))
            recon_sim.append(aux["recon_sim"].detach().cpu().view(-1))
            cons_hat.append(aux["consistency_hat"].detach().cpu().view(-1))
            temp01.append(aux["temp01"].detach().cpu().view(-1))
            jac.append(aux["jacobian_fd"].detach().cpu().view(-1))
            spars.append(aux["concept_sparsity"].detach().cpu().view(-1))
            ood.append(aux["ood_hat"].detach().cpu().view(-1))
            len_eff.append(aux["len_effect"].detach().cpu().view(-1))

        s = torch.cat(scores); t = torch.cat(targets)
        mae = F.l1_loss(s, t).item()
        rmse = torch.sqrt(F.mse_loss(s, t)).item()

        def mean_cat(arrs):
            """Helper to compute mean of concatenated tensor list."""
            return float(torch.cat(arrs).mean().item()) if arrs else 0.0

        return {
            "mae": mae,
            "rmse": rmse,
            "entropy_aux_mean":     mean_cat(entropies),
            "uncertainty_mean":     mean_cat(uncerts),
            "disagree_hat_mean":    mean_cat(disagree),
            "recon_sim_mean":       mean_cat(recon_sim),
            "consistency_hat_mean": mean_cat(cons_hat),
            "temp01_mean":          mean_cat(temp01),
            "jacobian_fd_mean":     mean_cat(jac),
            "concept_sparsity_mean":mean_cat(spars),
            "ood_hat_mean":         mean_cat(ood),
            "len_effect_mean":      mean_cat(len_eff),
        }

    # ------------------------------
    # Train/val split
    # ------------------------------

    def _create_train_val_split(self, samples: List[Dict[str, Any]]):
        """
        Split samples into training and validation sets.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Tuple of (train_samples, val_samples)
        """
        if not samples:
            return [], []
        if self.validation_ratio <= 0 or len(samples) < 10:
            return samples, []
        g = torch.Generator().manual_seed(self.seed)
        idx = torch.randperm(len(samples), generator=g).tolist()
        split = int(len(samples) * (1 - self.validation_ratio))
        return [samples[i] for i in idx[:split]], [samples[i] for i in idx[split:]]

    # ------------------------------
    # Main train loop (per dimension)
    # ------------------------------

    def train(self, samples, dimension):
        """
        Main training loop for a specific quality dimension.
        
        Args:
            samples: Training samples for the dimension
            dimension: Quality dimension name
            
        Returns:
            Training results dictionary
        """
        # Split data
        train_samples, val_samples = self._create_train_val_split(samples)

        # Create data loaders
        dataloader, kept, dropped = self._create_dataloader(train_samples)
        val_loader, val_kept, val_dropped = (None, 0, 0)
        if val_samples:
            val_loader, val_kept, val_dropped = self._create_dataloader(val_samples)

        if not dataloader:
            return {"error": "insufficient_data", "kept": kept, "dropped": dropped}

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2)

        best_metric = float("inf")
        patience, wait = int(self.cfg.get("patience", 3)), 0
        train_losses: List[float] = []
        saved_best = False

        locator = self.get_locator(dimension)  # create once; base_path will be ensured

        # Training loop with early stopping
        for epoch in range(1, self.epochs + 1):
            avg_loss = self._train_epoch(self.model, dataloader, epoch_idx=epoch)
            avg_loss = float(avg_loss)
            # Ensure epoch loss is finite for serialization/meta
            if not math.isfinite(avg_loss):
                avg_loss = float(train_losses[-1]) if train_losses else 0.0
                if self.logger:
                    self.logger.log("TinyPlusNaNEpoch", {"epoch": epoch})
            train_losses.append(avg_loss)

            # Validation
            val_metrics = self._validate(self.model, val_loader) if val_loader else {}
            if self.logger:
                payload = {"epoch": epoch, "train_loss": float(avg_loss)}
                payload.update({f"val_{k}": v for k, v in val_metrics.items()})
                self.logger.log("TinyPlusEpoch", payload)

            # Early stopping metric: prefer val MAE, fallback to train loss
            stop_metric = val_metrics.get("mae", avg_loss) if val_metrics else avg_loss
            if not math.isfinite(stop_metric):
                if self.logger:
                    self.logger.log("TinyPlusNonFiniteMetric", {"epoch": epoch})
                stop_metric = float("inf")

            improved = (not math.isfinite(best_metric)) or (stop_metric < best_metric - 1e-6)
            if improved:
                best_metric = stop_metric
                wait = 0
                best_path = locator.model_file(suffix="_tiny.pt")
                try:
                    torch.save(self.model.state_dict(), best_path)
                    saved_best = True
                    if self.logger:
                        self.logger.log("TinyPlusSaveCheckpoint", {"epoch": epoch, "path": best_path, "metric": float(best_metric)})
                except Exception as e:
                    if self.logger:
                        self.logger.log("TinyPlusSaveError", {"epoch": epoch, "path": best_path, "error": str(e)})
            else:
                wait += 1
                if wait >= patience:
                    if self.logger:
                        self.logger.log("TinyPlusEarlyStopping", {"epoch": epoch, "best_metric": float(best_metric)})
                    break

        # ---- ALWAYS save a 'last' checkpoint ----
        last_path = locator.model_file(suffix="_tiny_last.pt")
        try:
            torch.save(self.model.state_dict(), last_path)
            if self.logger:
                self.logger.log("TinyPlusSaveLast", {"path": last_path})
        except Exception as e:
            if self.logger:
                self.logger.log("TinyPlusSaveLastError", {"path": last_path, "error": str(e)})

        # If no 'best' was saved during training, backfill it now:
        best_path = locator.model_file(suffix="_tiny.pt")
        if not saved_best or not os.path.exists(best_path):
            try:
                torch.save(self.model.state_dict(), best_path)
                if self.logger:
                    self.logger.log("TinyPlusBackfillBest", {"path": best_path})
            except Exception as e:
                if self.logger:
                    self.logger.log("TinyPlusBackfillBestError", {"path": best_path, "error": str(e)})

        # --- Save training metadata -------------------------------------------
        safe_config = {
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "halt_lambda": self.halt_lambda,
            "n_layers": self.cfg.get("n_layers", 2),
            "n_recursions": self.n_recursions,
            "use_attention": self.use_attention,
            "dropout": self.dropout,
            "seed": self.seed,
            "vocab_size": int(self.cfg.get("vocab_size", 101)),
            "w_aux3": self.w_aux3,
            "w_disagree": self.w_disagree,
            "w_recon": self.w_recon,
            "w_consistency": self.w_cons,
            "w_sae_recon": self.w_sae_recon,
            "w_ood": self.w_ood,
            "grad_clip": self.grad_clip,
        }

        # Ensure train_loss_curve is finite-only floats
        finite_curve = []
        last_finite = 0.0
        for v in train_losses:
            if math.isfinite(v):
                last_finite = float(v)
            finite_curve.append(float(last_finite))

        meta = {
            "dimension": dimension,
            "model_type": "tiny_recursion",
            "expects_triplet": True,
            "embedding_type": self.embedding_type,
            "input_dim": self.dim,
            "concat_input_dim": self.dim * 2,
            "version": self.version,
            "epochs": self.epochs,
            "avg_loss": float(min(finite_curve or [best_metric])),
            "timestamp": datetime.now().isoformat(),
            "cfg": dict(self.cfg),
            "kept": int(kept),
            "best_metric": float(best_metric),
            "train_loss_curve": [float(x) for x in finite_curve],
            "dropped": int(dropped),
            "val_kept": int(val_kept),
            "val_dropped": int(val_dropped),
        }
        self._save_meta_file(meta, dimension)

        # TrainingStatsStore integration
        self.memory.training_stats.add_from_result(
            stats={
                "avg_q_loss": float(min(finite_curve or [best_metric])),
                "avg_loss":   float(min(finite_curve or [best_metric])),
            },
            model_type=self.model_type,
            target_type=self.target_type,
            dimension=dimension,
            version=self.version,
            embedding_type=self.embedding_type,
            config=safe_config,
            sample_count=len(samples),
            valid_samples=int(kept),
            invalid_samples=int(dropped),
            start_time=datetime.now(),
        )

        return {
            "best_metric": float(best_metric),
            "train_loss_curve": [float(x) for x in finite_curve],
            "kept": int(kept),
            "dropped": int(dropped),
            "val_kept": int(val_kept),
            "val_dropped": int(val_dropped),
        }

    # ------------------------------
    # Helper Methods
    # ------------------------------

    def _bucketize_label(self, y: int) -> int:
        """
        Bucketize label for histogram analysis.
        
        If bucket_size > 0, map 0..100 → 0..num_bins-1 using fixed-width bins.
        Ensure vocab_size >= num_bins when you enable bucketing.
        
        Args:
            y: Original label value
            
        Returns:
            Bucket index
        """
        b = int(self.bucket_size)
        if b <= 0:
            return max(0, min(100, y))              # <<< clamp to 0..100
        num_bins = (101 + b - 1) // b               # e.g., b=10 -> 11 bins (0..10)
        yb = min(max(y, 0) // b, num_bins - 1)
        return yb

    def _bucketize_counts(self, counts: Counter, bucket: int) -> dict:
        """
        Convert exact label counts to bucketized counts for visualization.
        
        Args:
            counts: Counter of exact label values
            bucket: Bucket size
            
        Returns:
            Dictionary mapping bucket ranges to counts
        """
        if bucket <= 1:
            return {str(k): int(v) for k, v in sorted(counts.items())}

        buckets = {}
        for label, c in counts.items():
            try:
                l = int(label)
            except Exception:
                continue
            start = (l // bucket) * bucket
            end = min(100, start + bucket - 1)
            key = f"{start}-{end}"
            buckets[key] = buckets.get(key, 0) + int(c)

        # Ensure all possible buckets are represented
        start = 0
        while start <= 100:
            end = min(100, start + bucket - 1)
            key = f"{start}-{end}"
            buckets.setdefault(key, 0)
            start += bucket

        return dict(sorted(buckets.items(), key=lambda kv: int(kv[0].split('-')[0])))