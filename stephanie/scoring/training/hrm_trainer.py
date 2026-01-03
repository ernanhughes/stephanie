# stephanie/training/hrm_trainer.py
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW  # As recommended by HRM paper
from torch.utils.data import DataLoader, TensorDataset

from stephanie.model.hrm import HRMModel  # Import the model
from stephanie.scoring.training.base_trainer import \
    BaseTrainer  # Assuming this exists or adapt

log = logging.getLogger(__name__)

class HRMTrainer(BaseTrainer): 
    """
    Trainer Agent for the Hierarchical Reasoning Model (HRM).
    Integrates with Stephanie's training framework.
    """
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        
        # --- HRM Specific Config ---
        self.model_type = "hrm"
        self.embedding_type = self.memory.embedding.name
        embedding_dim = memory.embedding.dim
        self.input_dim = embedding_dim * 2
        self.h_dim = cfg.get("hrm.h_dim", 256)
        self.l_dim = cfg.get("hrm.l_dim", 128)
        self.output_dim = cfg.get("hrm.output_dim", 1) # 1 for score prediction
        self.n_cycles = cfg.get("hrm.n_cycles", 4)
        self.t_steps = cfg.get("hrm.t_steps", 4)
        self.lr = cfg.get("hrm.lr", 1e-4)
        self.epochs = cfg.get("hrm.epochs", 10)
        self.batch_size = cfg.get("hrm.batch_size", 32)
        self.apply_sigmoid = cfg.get("hrm.apply_sigmoid", True) 
        self.scaler_p_lo = cfg.get("hrm.target_scale_p_lo", 10.0)
        self.scaler_p_hi = cfg.get("hrm.target_scale_p_hi", 90.0) 
        self._scaler_stats = {}  

        # Device setup (inherited or set)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the HRM model
        hrm_cfg = {
            "input_dim": self.input_dim,
            "h_dim": self.h_dim,
            "l_dim": self.l_dim,
            "output_dim": self.output_dim,
            "n_cycles": self.n_cycles,
            "t_steps": self.t_steps,
        }
        self.hrm_model = HRMModel(hrm_cfg, logger=self.logger).to(self.device)
        
        # Optimizer (AdamW as recommended)
        self.optimizer = AdamW(self.hrm_model.parameters(), lr=self.lr)
        
        # Loss function (MSE for regression, e.g., predicting a score)
        # Can be made configurable (e.g., CrossEntropy for classification)
        self.criterion = nn.MSELoss() 
        self.show_progress = self.cfg.get("show_progress", True)   # enables tqdm
        self.log_interval  = self.cfg.get("log_interval", 25)      # batch logging freq


        log.info("HRMTrainerInitialized model_type %s input_dim %d h_dim %d l_dim %d output_dim %d n_cycles %d t_steps %d lr %f device %s",
            self.model_type,
            self.input_dim,
            self.h_dim,
            self.l_dim,
            self.output_dim,
            self.n_cycles,
            self.t_steps,
            self.lr,
            str(self.device)
        )

    def train(self, samples, dimension) -> dict:
        log.info("HRMTrainingStarted epochs %d", self.epochs)

        dataloader = self._create_dataloader(samples, dimension)
        if dataloader is None:
            log.error("HRMTrainingError message Dataloader creation failed or insufficient samples.")

        losses = []
        best = float("inf")
        wait = 0

        # epoch progress bar
        epoch_iter = self.progress(range(self.epochs), desc=f"HRM[{dimension}] epochs", leave=False)
        for epoch in epoch_iter:
            epoch_loss, num_batches = 0.0, 0

            # batch progress bar
            batch_iter = self.progress(enumerate(dataloader), desc=f"epoch {epoch+1}", leave=False)
            for bidx, (x_batch, y_batch) in batch_iter:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                y_pred, _ = self.hrm_model(x_batch)
                if self.apply_sigmoid:
                    y_pred = torch.sigmoid(y_pred)

                loss = self.criterion(y_pred, y_batch)
                loss.backward()

                # (optional) clip to keep HRM stable
                torch.nn.utils.clip_grad_norm_(self.hrm_model.parameters(), max_norm=1.0)

                self.optimizer.step()

                epoch_loss += float(loss.item())
                num_batches += 1

                # live progress feedback
                if self.show_progress:
                    self.progress_postfix(
                        batch_iter,
                        loss=float(loss.item()),
                        gnorm=self._grad_global_norm(),
                        lr=float(self.optimizer.param_groups[0]["lr"]),
                    )

                # periodic logs
                if (bidx + 1) % max(1, self.log_interval) == 0:
                    log.info("HRMTrainingBatch epoch %s batch %s loss %f lr %f",
                        epoch,
                        bidx + 1,
                        float(loss.item()),
                        float(self.optimizer.param_groups[0]["lr"])
                    )

            avg_epoch_loss = epoch_loss / max(1, num_batches)
            losses.append(avg_epoch_loss)

            # update epoch bar postfix
            if self.show_progress:
                self.progress_postfix(
                    epoch_iter,
                    avg_loss=avg_epoch_loss,
                    best=best if best < float("inf") else avg_epoch_loss
                )

            log.info("HRMTrainingEpoch epoch %s avg_loss %f best_so_far %f",
                epoch,
                avg_epoch_loss,
                min(best, avg_epoch_loss)
            )

            # early stopping
            if avg_epoch_loss < best - self.early_stopping_min_delta:
                best, wait = avg_epoch_loss, 0
            else:
                wait += 1
                if self.use_early_stopping and wait >= self.early_stopping_patience:
                    log.info("HRMEarlyStopping epoch %s reason %s best_loss %f",
                        epoch,
                        "patience_reached",
                        best
                    )
                    break

        # --- save model ---
        self._save_model(dimension)

        # --- log scaler + training stats summary ---
        scaler_meta = {
            "method": "robust_minmax",
            "p_lo": self.scaler_p_lo, "p_hi": self.scaler_p_hi,
            "lo": self._scaler_stats[dimension]["lo"],
            "hi": self._scaler_stats[dimension]["hi"],
        }
        final_avg = float(losses[-1]) if losses else float("nan")

        # lightweight snapshot (on-train predictions vs targets)
        try:
            with torch.no_grad():
                preds, acts = [], []
                for xb, yb in dataloader:
                    yp, _ = self.hrm_model(xb.to(self.device))
                    if self.apply_sigmoid:
                        yp = torch.sigmoid(yp)
                    preds.append(yp.detach().cpu().view(-1))
                    acts.append(yb.detach().cpu().view(-1))
                if preds:
                    P = torch.cat(preds).numpy()
                    Y = torch.cat(acts).numpy()
                    diff = P - Y
                    mae  = float(np.mean(np.abs(diff)))
                    rmse = float(np.sqrt(np.mean(diff * diff)))
                    log.info("HRMTrainingSnapshot dimension %s train_mae %f train_rmse %f pred_mean %f target_mean %f size %d",
                        dimension,
                        mae,
                        rmse,
                        float(np.mean(P)),
                        float(np.mean(Y)),
                        int(P.size)
                    )
        except Exception as e:
            log.error("HRMSnapshotError error %s", str(e))

        self._log_training_stats(
            dimension,
            avg_loss=final_avg,
            sample_count=len(samples),
            valid_samples=len(dataloader.dataset),
            scaler_meta=scaler_meta,
        )

        self.logger.log("HRMTrainingCompleted", {"final_avg_loss": final_avg, "best_loss": best})
        return {"status": "trained", "final_loss": final_avg, "best_loss": best}

    def _create_dataloader(self, samples, dimension):
        """
        Accepts either:
        • singleton: {"title": str, "output": str, "score": float}
        • pairwise : {"title": str, "output_a": str, "output_b": str,
                        "value_a": float, "value_b": float}
        • HRM/raw  : {"goal_text": str, "scorable_text": str, "target_score" or "score": float}

        Builds a TensorDataset of (x_concat, y_norm) where:
        x_concat = concat([ctx_emb, doc_emb])  # shape [2*D]
        y_norm   = normalized target in [0,1], shape [1]
        Uses robust percentiles (p_lo..p_hi) for stability.
        """
        # ---- collect raw (goal, doc, value) triples ----
        triples = []      # (goal_text, doc_text, value)
        kept, skipped = 0, 0

        def _maybe_push(title, out, val):
            nonlocal kept, skipped
            if title and out and (val is not None):
                try:
                    triples.append((title, out, float(val)))
                    kept += 1
                except Exception:
                    skipped += 1
            else:
                skipped += 1

        for s in self.progress(samples, desc="Packing HRM triples"):
            # normalize keys across MRQ/SICQL/HRM shapes
            title = (s.get("goal_text") or s.get("title") or "").strip()
            # singleton path
            if "output" in s and ("score" in s or "target_score" in s):
                out = (s.get("scorable_text") or s.get("output") or "").strip()
                val = s.get("target_score", s.get("score", None))
                _maybe_push(title, out, val)
                continue
            # pairwise path
            if all(k in s for k in ("output_a", "output_b", "value_a", "value_b")):
                a_out = (s.get("output_a") or "").strip()
                b_out = (s.get("output_b") or "").strip()
                a_val = s.get("value_a", None)
                b_val = s.get("value_b", None)
                if a_out or b_out:
                    if a_out and (a_val is not None):
                        _maybe_push(title, a_out, a_val)
                    if b_out and (b_val is not None):
                        _maybe_push(title, b_out, b_val)
                else:
                    skipped += 1
                continue
            # explicit HRM/raw path
            if ("goal_text" in s and "scorable_text" in s and
                ("target_score" in s or "score" in s)):
                out = (s.get("scorable_text") or "").strip()
                val = s.get("target_score", s.get("score"))
                _maybe_push(title, out, val)
                continue
            # otherwise unrecognized
            skipped += 1

        if kept < self.min_samples:
            self.logger.log("HRMDataError", {
                "message": f"Insufficient samples after parsing: kept={kept}, skipped={skipped}",
                "kept": kept, "skipped": skipped, "threshold": self.min_samples
            })
            return None

        # ---- robust scaling over raw targets ----
        raw_vals = [v for _, _, v in triples]
        lo = float(np.percentile(raw_vals, self.scaler_p_lo)) if raw_vals else 0.0
        hi = float(np.percentile(raw_vals, self.scaler_p_hi)) if raw_vals else 1.0
        if hi - lo < 1e-9:
            hi = lo + 1.0

        self._scaler_stats[dimension] = {"lo": lo, "hi": hi}
        self.logger.log("HRMTargetScaling", {
            "dimension": dimension, "p_lo": self.scaler_p_lo, "p_hi": self.scaler_p_hi,
            "lo": lo, "hi": hi, "n": len(raw_vals)
        })

        def _norm(v: float) -> float:
            x = (v - lo) / (hi - lo)
            return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)

        # ---- encode & pack tensors ----
        xs, ys = [], []
        for goal_text, doc_text, value in triples:
            try:
                ctx = torch.tensor(self.memory.embedding.get_or_create(goal_text),
                                dtype=torch.float32, device=self.device)
                doc = torch.tensor(self.memory.embedding.get_or_create(doc_text),
                                dtype=torch.float32, device=self.device)
                x = torch.cat([ctx, doc], dim=-1)                 # [2*D]
                y = torch.tensor([_norm(value)], dtype=torch.float32, device=self.device)  # [1]
                xs.append(x)
                ys.append(y)
            except Exception as e:
                skipped += 1
                self.logger.log("HRMDataError", {"error": str(e), "sample_goal_preview": goal_text[:80]})

        if len(xs) < self.min_samples:
            self.logger.log("HRMDataError", {
                "message": f"Insufficient valid encoded samples: {len(xs)} < {self.min_samples}",
                "after_encoding_kept": len(xs), "skipped_total": skipped
            })
            return None

        X = torch.stack(xs)      # [N, 2*D]
        Y = torch.stack(ys)      # [N, 1]
        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        log.info("HRMDataLoaderCreated dimension %s num_samples %d num_batches %d kept %d skipped %d",
            dimension,
            len(dataset),
            len(loader),
            kept, skipped
        )
        return loader

    def _log_training_stats(self, dimension, *, avg_loss, sample_count, valid_samples, scaler_meta):
        self.memory.training_stats.add_from_result(
            stats={
                "avg_loss": avg_loss,
                "policy_entropy": None,
                "policy_stability": None,
                "policy_logits": None,
            },
            model_type="hrm",
            target_type=self.target_type,
            dimension=dimension,
            version=self.version,
            embedding_type=self.embedding_type,
            config={
                "lr": self.lr, "epochs": self.epochs, "batch_size": self.batch_size,
                "dim": self.dim, "hdim": self.hdim,
                "h_dim": self.h_dim, "l_dim": self.l_dim,
                "n_cycles": self.n_cycles, "t_steps": self.t_steps,
                "apply_sigmoid": self.apply_sigmoid,
                "target_scale": scaler_meta,
            },
            sample_count=sample_count,
            valid_samples=valid_samples,
        )

    def _grad_global_norm(self) -> float:
        num, den = 0.0, 0.0
        for p in self.hrm_model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                num += float(torch.sum(g * g))
            if p.is_floating_point(): # Only consider floating point parameters for norm
                den += float(torch.sum(p * p))
        return float(np.sqrt(num)) / (float(np.sqrt(den)) + 1e-8) if den > 0 else 0.0


    def _save_model(self, dimension: str):
        locator = self.get_locator(dimension)
        torch.save(self.hrm_model.state_dict(), locator.model_file(suffix="_hrm.pt"))

        meta = {
            "model_type": self.model_type,
            "input_dim": self.input_dim,
            "h_dim": self.h_dim,
            "l_dim": self.l_dim,
            "output_dim": self.output_dim,
            "n_cycles": self.n_cycles,
            "t_steps": self.t_steps,
            "lr": self.lr,
            "epochs": self.epochs,
            "apply_sigmoid": self.apply_sigmoid,               
            "target_scale": {                                 
                "method": "robust_minmax",
                "p_lo": self.scaler_p_lo,
                "p_hi": self.scaler_p_hi,
                "lo": self._scaler_stats.get(dimension, {}).get("lo"),
                "hi": self._scaler_stats.get(dimension, {}).get("hi"),
            },
        }
        self._save_meta_file(meta, dimension)
        log.info("HRMModelSaved path %s dimension %s", locator.base_path, dimension)
