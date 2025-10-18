# stephanie/scoring/training/mrq_trainer.py
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from stephanie.models.training_stats import TrainingStatsORM
from stephanie.scoring.model.mrq_model import MRQModel
from stephanie.scoring.model.text_encoder import TextEncoder
from stephanie.scoring.model.value_predictor import ValuePredictor
from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner

# tqdm (progress bars)
try:
    from tqdm.auto import tqdm
except Exception:  # if tqdm isn't available, fall back to None
    tqdm = None


class MRQTrainer(BaseTrainer):
    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.use_tuner = cfg.get("use_tuner", True)
        self.min_samples = cfg.get("min_samples", 5)
        self.batch_size = cfg.get("batch_size", 1)
        self.model = self._build_model()
        self.use_tuner = cfg.get("use_tuner", True)
        self.use_early_stopping = cfg.get("early_stopping", True)
        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.batch_size = cfg.get("batch_size", 2)
        self.epochs = cfg.get("epochs", 50)
        self.lr = cfg.get("lr", 1e-4)
        self.min_samples = cfg.get("min_samples", 5)
        self.show_progress  = bool(cfg.get("show_progress", True))
        self.progress_leave = bool(cfg.get("progress_leave", False))  # keep bars on screen after finishing?

        
        self.logger.log("MRQTrainerInitialized", {
            "embedding_type": self.embedding_type,
            "use_tuner": self.use_tuner,
            "device": str(self.device)
        })

    def _create_dataloader(self, samples):
        """
        Build (X, y, w) pairs for MRQ:
        - X: (zA - zB) embedding diffs
        - y: 1.0 if value_a > value_b else 0.0  (preference label)
        - w: sample weight proportional to |value_a - value_b|
        Shows a tqdm progress bar if enabled.
        """
        inputs, labels, weights = [], [], []

        iterable = samples
        pbar = None
        if self.show_progress and "tqdm" in globals() and tqdm is not None:
            pbar = tqdm(samples, desc="Packing MRQ pairs", unit="pair", leave=self.progress_leave)
            iterable = pbar

        kept = 0
        skipped = 0

        for item in iterable:
            prompt = item.get("title", "")
            A_text = item.get("output_a", "")
            B_text = item.get("output_b", "")
            vA = item.get("value_a", None)
            vB = item.get("value_b", None)

            if not prompt or not A_text or not B_text or vA is None or vB is None:
                skipped += 1
                if pbar is not None and (kept + skipped) % 200 == 0:
                    pbar.set_postfix(kept=kept, skipped=skipped)
                continue

            try:
                # Embeddings
                G = torch.tensor(self.memory.embedding.get_or_create(prompt), dtype=torch.float32, device=self.device).unsqueeze(0)
                A = torch.tensor(self.memory.embedding.get_or_create(A_text), dtype=torch.float32, device=self.device).unsqueeze(0)
                B = torch.tensor(self.memory.embedding.get_or_create(B_text), dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    zA = self.model.encoder(G, A)  # [1, D]
                    zB = self.model.encoder(G, B)  # [1, D]


                # zA, zB already computed; vA, vB are floats
                delta = vA - vB

                if torch.rand(()) < 0.5:
                    x = (zA - zB).squeeze(0).detach()
                    y = 1.0 if delta > 0 else 0.0
                else:
                    x = (zB - zA).squeeze(0).detach()
                    y = 1.0 if delta < 0 else 0.0

                gap = abs(delta)
                w = min(1.0 + 0.05 * gap, 3.0)

                inputs.append(x)
                labels.append(torch.tensor(y, dtype=torch.float32, device=self.device))
                weights.append(torch.tensor(w, dtype=torch.float32, device=self.device))

                kept += 1

                if self.logger:
                    pos = sum(1 for t in labels if float(t.item()) > 0.5)
                    neg = len(labels) - pos
                    self.logger.log("MRQLabelBalance", {"pos": int(pos), "neg": int(neg)})


                if pbar is not None and kept % 100 == 0:
                    pbar.set_postfix(kept=kept, skipped=skipped)

            except Exception as e:
                skipped += 1
                if self.logger:
                    self.logger.log("PairPreparationError", {"error": str(e)})
                if pbar is not None and (kept + skipped) % 200 == 0:
                    pbar.set_postfix(kept=kept, skipped=skipped)
                continue


        if pbar is not None:
            pbar.close()

        if len(inputs) < self.min_samples:
            if self.logger:
                self.logger.log("InsufficientSamples", {"sample_count": len(inputs), "threshold": self.min_samples})
            return None

        X = torch.stack(inputs)           # [N, D]
        y = torch.stack(labels).view(-1)  # [N]
        w = torch.stack(weights).view(-1) # [N]

        dataset = TensorDataset(X, y, w)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def _build_model(self):
        encoder = TextEncoder(dim=self.dim, hdim=self.hdim).to(self.device)
        predictor = ValuePredictor(zsa_dim=self.dim, hdim=self.hdim).to(self.device)
        return MRQModel(encoder, predictor, self.memory.embedding, device=self.device)

    def _train_epoch(self, model, dataloader, epoch_idx: int = 1):
        model.encoder.train()
        model.predictor.train()
        total_loss, count = 0.0, 0

        iterator = dataloader
        pbar = None
        if self.show_progress and "tqdm" in globals() and tqdm is not None:
            pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch_idx}",
            unit="batch", leave=self.progress_leave)


        for batch in (pbar if pbar is not None else iterator):
            if len(batch) == 3:
                X, y, w = batch
            else:
                X, y = batch
                w = torch.ones_like(y)

            X = X.to(self.device)
            y = y.to(self.device)
            w = w.to(self.device)

            logits = model.predictor(X).view(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, weight=w, pos_weight=getattr(self, "pos_weight_tensor", None)
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 0.5)
            self.optimizer.step()

            bsz = X.size(0)
            total_loss += loss.item() * bsz
            count += bsz

            if pbar is not None:
                # live classification metrics
                with torch.no_grad():
                    preds = (logits > 0).float()
                    acc = (preds == y).float().mean().item()
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")

        if pbar is not None:
            pbar.close()

        return total_loss / max(1, count)

    def train(self, samples, dimension):
        dataloader = self._create_dataloader(samples)
        if not dataloader:
            return {"error": "insufficient_data", "dimension": dimension}

        # --- Class prior & pos_weight ---
        # collect labels from dataloader once
        ys = []
        for batch in dataloader:
            if len(batch) == 3: _, y, _ = batch
            elif len(batch) == 2: _, y = batch
            else: continue
            ys.append(y.detach().cpu().view(-1))
        y_all = torch.cat(ys) if ys else torch.tensor([])
        p_pos = float((y_all > 0.5).float().mean()) if y_all.numel() > 0 else 0.5
        # pos_weight = N_neg / N_pos, clamp to sane range
        eps = 1e-6
        pos_weight_scalar = float((1.0 - p_pos) / max(p_pos, eps))
        pos_weight_scalar = float(min(max(pos_weight_scalar, 0.2), 5.0))  # clamp
        self.pos_weight_tensor = torch.tensor(pos_weight_scalar, device=self.device)

        # --- Initialize predictor bias to prior logit ---
        try:
            prior_logit = math.log(p_pos / max(1.0 - p_pos, eps))
            # adapt to your ValuePredictor: out layer likely named self.model.predictor.out
            if hasattr(self.model.predictor, "out") and hasattr(self.model.predictor.out, "bias"):
                with torch.no_grad():
                    self.model.predictor.out.bias.fill_(prior_logit)
        except Exception:
            pass


        self.optimizer = optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.predictor.parameters()),
            lr=self.lr
        )

        best_loss = float("inf")
        early_stop_counter = 0
        losses = []

        for epoch in range(self.epochs):
            avg_loss = self._train_epoch(self.model, dataloader, epoch_idx=epoch + 1)
            losses.append(avg_loss)
            self.logger.log("MRQTrainingEpoch", {
                "epoch": epoch + 1,
                "loss": avg_loss
            })
            if avg_loss < best_loss - self.early_stopping_min_delta:
                best_loss = avg_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if self.use_early_stopping and early_stop_counter >= self.early_stopping_patience:
                    break

        # --- Save weights ---
        locator = self.get_locator(dimension)
        torch.save(self.model.encoder.state_dict(), locator.encoder_file())
        torch.save(self.model.predictor.state_dict(), locator.model_file())

        train_preds01, train_acts = self.collect_preds_targets(
            self.model, dataloader, self.device, head="q", apply_sigmoid=True
        )
        reg_stats = self.regression_metrics(train_preds01, train_acts)

        # --- Pairwise classification metrics on logits (unchanged) ---
        pair_stats = self.binary_cls_metrics(
            dataloader, forward_fn=lambda X: self.model.predictor(X)
        )

        # --- Optional tuner calibration on *train* (or a held-out set if you have one) ---
        if self.use_tuner:
            tuner = RegressionTuner(dimension=dimension, logger=self.logger)  # expects train_single(x,y)
            with torch.no_grad():
                for X, y, _w in dataloader:
                    X = X.to(self.device)
                    logits = self.model.predictor(X).view(-1).cpu().numpy()
                    probs = 1 / (1 + np.exp(-logits))         # sigmoid
                    scores01 = probs                          # already 0..1
                    for s01 in scores01:
                        tuner.train_single(float(s01), float(s01))  # identity fit keeps API consistent
            tuner.save(locator.tuner_file())

        # --- Basic label min/max from samples (use both A & B for coverage) ---
        try:
            vals = []
            for s in samples:
                if "value_a" in s: vals.append(float(s["value_a"]))
                if "value_b" in s: vals.append(float(s["value_b"]))
            if vals:
                min_value, max_value = float(np.min(vals)), float(np.max(vals))
            else:
                min_value = max_value = float("nan")
        except Exception:
            min_value = max_value = float("nan")

        meta = {
            "dimension": dimension,
            "model_type": "mrq",
            "target_type": self.target_type,
            "embedding_type": self.embedding_type,
            "version": self.version,
            "dim": self.dim,
            "hdim": self.hdim,
            "min_value": min_value,
            "max_value": max_value,
            "epochs": self.epochs,
            "avg_loss": float(np.mean(losses)) if losses else float("nan"),
            "best_loss": float(best_loss),
            "loss_curve": [float(x) for x in losses],
            "train_count": reg_stats["count"],
            "train_mae": reg_stats["mae"],
            "train_mse": reg_stats["mse"],
            "train_rmse": reg_stats["rmse"],
            "train_r2": reg_stats["r2"],
            "train_pearson_r": reg_stats["pearson_r"],
            "train_spearman_rho": reg_stats["spearman_rho"],
            "train_within1": reg_stats["within1"],
            "train_within2": reg_stats["within2"],
            "train_within5": reg_stats["within5"],
            "pred_mean": reg_stats["pred_mean"],
            "pred_std": reg_stats["pred_std"],
            "pair_acc": pair_stats["pair_acc"],
            "pair_auc": pair_stats["auc"],
            "pos_rate": pair_stats["pos_rate"],
            "logloss": pair_stats["logloss"],
            "timestamp": datetime.now().isoformat(),
        }
        self._save_meta_file(meta, dimension)

        # --- Persist TrainingStats (clean, dashboard-friendly) ---
        self.memory.training_stats.add_from_result(
            stats={
                "train_mae": meta["train_mae"],
                "train_rmse": meta["train_rmse"],
                "train_r2": meta["train_r2"],
                "train_pearson_r": meta["train_pearson_r"],
                "train_spearman_rho": meta["train_spearman_rho"],
                "within1": meta["train_within1"],
                "within2": meta["train_within2"],
                "within5": meta["train_within5"],
                "pair_auc": pair_stats["auc"],
                "best_loss": meta["best_loss"],
                "avg_loss": meta["avg_loss"],
                "pair_acc": pair_stats["pair_acc"],
                "logloss": meta["logloss"],
            },
            model_type="mrq",
            target_type=self.target_type,
            dimension=dimension,
            version=self.version,
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
            },
            sample_count=len(samples),
            start_time=datetime.now(),
        )

        report = self.report_training_snapshot(
            dataloader,
            meta=meta,
            losses=losses,
            title=f"MRQ[{dimension}]",
            head=12,
            within_deltas=(1.0, 2.0, 5.0),
        )

        print(report)

        return meta
