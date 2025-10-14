# stephanie/scoring/training/tiny_recursion_trainer.py
from __future__ import annotations
from datetime import datetime
from collections import Counter
from typing import Tuple, List, Dict, Any

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from stephanie.scoring.model.tiny_recursion import TinyRecursionModel
from stephanie.scoring.training.base_trainer import BaseTrainer

# make tqdm optional
try:
    from tqdm.auto import tqdm  # optional
except Exception:
    tqdm = None


class TinyRecursionTrainer(BaseTrainer):
    """
    Trainer for TinyRecursionModel — aligns with MRQTrainer pattern
    but trains recursive reasoning dynamics instead of value predictors.
    Produces ONE MODEL PER DIMENSION.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)

        # --- Identity / paths -------------------------------------------------
        self.model_type   = "tiny"                                     # <<< ensures locator picks models/tiny/...
        self.target_type  = cfg.get("target_type", "document")         # <<< consistency with your scorers
        self.version      = cfg.get("model_version", "v1")             # <<< folder versioning

        # --- Core knobs -------------------------------------------------------
        self.epochs        = cfg.get("epochs", 20)
        self.lr            = cfg.get("lr", 1e-4)
        self.batch_size    = cfg.get("batch_size", 4)
        self.dropout       = cfg.get("dropout", 0.1)
        self.use_attention = cfg.get("use_attention", False)
        self.n_recursions  = cfg.get("n_recursions", 6)
        self.vocab_size    = cfg.get("vocab_size", 101)                # <<< 0..100 by default
        self.halt_lambda   = cfg.get("halt_lambda", 0.1)

        # --- Telemetry --------------------------------------------------------
        self.show_progress       = cfg.get("show_progress", True)
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

        # --- Optional label bucketing ----------------------------------------
        self.bucket_size = int(cfg.get("bucket_size", 0))  # 0 disables

        # --- Model ------------------------------------------------------------
        # self.dim comes from BaseTrainer (embedding dim). We pass that as d_model.
        self.model = TinyRecursionModel(
            d_model=self.dim,
            n_layers=cfg.get("n_layers", 2),
            n_recursions=self.n_recursions,
            vocab_size=self.vocab_size,
            use_attention=self.use_attention,
            dropout=self.dropout,
        ).to(self.device)

    # ------------------------------
    # Data prep: each sample = (x, y, z, target_class[0..100 or binned], halt_target)
    # ------------------------------
    def _create_dataloader(self, samples: List[Dict[str, Any]]) -> Tuple[Any, int, int]:
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        zs: List[torch.Tensor] = []
        targets: List[int] = []
        halt_targets: List[float] = []

        kept = 0
        dropped = 0
        label_counts = Counter()

        use_tqdm = bool(self.show_progress and tqdm is not None)
        pbar = tqdm(samples, desc="Packing TinyRecursion samples", unit="samp") if use_tqdm else None
        iterator = enumerate(pbar if pbar is not None else samples, start=1)

        for idx, s in iterator:
            try:
                x = torch.tensor(self.memory.embedding.get_or_create(s["x"]), dtype=torch.float, device=self.device)
                y = torch.tensor(self.memory.embedding.get_or_create(s["y"]), dtype=torch.float, device=self.device)
                z = torch.tensor(self.memory.embedding.get_or_create(s["z"]), dtype=torch.float, device=self.device)

                raw_t = s.get("target")
                if raw_t is None:
                    dropped += 1
                    if (not use_tqdm) and self.show_progress and (idx % self.progress_every == 0) and self.logger:
                        self.logger.log("TinyRecursionPackProgress", {"kept": kept, "dropped": dropped})
                    continue

                try:
                    t_int = int(float(raw_t))
                except Exception:
                    dropped += 1
                    if (not use_tqdm) and self.show_progress and (idx % self.progress_every == 0) and self.logger:
                        self.logger.log("TinyRecursionPackProgress", {"kept": kept, "dropped": dropped})
                    continue

                # optional bucketing
                t_int = self._bucketize_label(t_int)

                # ensure class id fits vocab
                if not (0 <= t_int < self.vocab_size):
                    dropped += 1
                    if (not use_tqdm) and self.show_progress and (idx % self.progress_every == 0) and self.logger:
                        self.logger.log("TinyRecursionPackProgress", {"kept": kept, "dropped": dropped})
                    continue

                halt_t = float(s.get("halt_target", 1.0))

                xs.append(x)
                ys.append(y)
                zs.append(z)
                targets.append(t_int)
                halt_targets.append(halt_t)
                label_counts[t_int] += 1
                kept += 1

                if pbar is not None:
                    pbar.set_postfix(kept=kept, drop=dropped)

            except Exception as e:
                dropped += 1
                if self.logger:
                    self.logger.log("TinyRecursionSampleError", {"error": str(e)})

        if pbar is not None:
            pbar.close()

        # label histogram logging
        if self.logger and self.log_label_histogram:
            exact = {int(k): int(v) for k, v in sorted(label_counts.items())}
            bucketed = self._bucketize_counts(label_counts, self.label_hist_bucket)
            self.logger.log("TinyRecursionLabelHistogram", {
                "kept": int(kept),
                "dropped": int(dropped),
                "exact": exact,
                "bucket_size": int(self.label_hist_bucket),
                "bucketed": bucketed
            })

        if kept < 2:
            return None, kept, dropped

        dataset = TensorDataset(
            torch.stack(xs),
            torch.stack(ys),
            torch.stack(zs),
            torch.tensor(targets, dtype=torch.long, device=self.device),
            torch.tensor(halt_targets, dtype=torch.float, device=self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader, kept, dropped

    # ------------------------------
    # Epoch training
    # ------------------------------
    def _train_epoch(self, model, dataloader, epoch_idx: int) -> float:
        model.train()
        total_loss = 0.0
        count = 0

        use_tqdm = bool(self.show_progress and tqdm is not None)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx}", unit="batch", leave=False) if use_tqdm else None
        iterator = enumerate(pbar if pbar is not None else dataloader, start=1)

        for step, batch in iterator:
            x, y, z, targets, halt_targets = batch
            logits, halt_logits, _ = model(x, y, z)                # <<< treat halt head as logits
            ce_loss = F.cross_entropy(logits, targets)
            halt_loss = F.binary_cross_entropy_with_logits(halt_logits, halt_targets)
            loss = ce_loss + self.halt_lambda * halt_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()

            bsz = x.size(0)
            total_loss += loss.item() * bsz
            count += bsz

            if pbar is not None:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            elif self.logger and (step % self.log_every_steps == 0):
                self.logger.log("TinyRecursionBatch", {"epoch": epoch_idx, "step": step, "batch_loss": float(loss.item())})

        if pbar is not None:
            pbar.close()

        return total_loss / max(1, count)

    # ------------------------------
    # Validation
    # ------------------------------
    def _validate(self, model, dataloader) -> Dict[str, float]:
        if not dataloader:
            return {}

        model.eval()
        preds, targs, halt_preds, halt_targs = [], [], [], []
        with torch.no_grad():
            for x, y, z, targets, halt_targets in dataloader:
                logits, halt_logits, _ = model(x, y, z)           # <<<
                preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
                targs.extend(targets.detach().cpu().tolist())
                halt_preds.extend(torch.sigmoid(halt_logits).detach().cpu().tolist())  # <<<
                halt_targs.extend(halt_targets.detach().cpu().tolist())

        if not targs:
            return {}

        pt = torch.tensor(preds, dtype=torch.float32)
        tt = torch.tensor(targs, dtype=torch.float32)
        hp = torch.tensor(halt_preds, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
        ht = torch.tensor(halt_targs, dtype=torch.float32)

        mae = F.l1_loss(pt, tt).item()
        within5 = (pt.sub(tt).abs() <= 5).float().mean().item()
        top1 = (pt == tt).float().mean().item()
        halt_bce = F.binary_cross_entropy(hp, ht).item()

        return {"mae": mae, "within_delta_5": within5, "top1_accuracy": top1, "halt_bce": halt_bce}

    # ------------------------------
    # Train/val split
    # ------------------------------
    def _create_train_val_split(self, samples: List[Dict[str, Any]]):
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
        # split
        train_samples, val_samples = self._create_train_val_split(samples)

        # loaders
        dataloader, kept, dropped = self._create_dataloader(train_samples)
        val_loader, val_kept, val_dropped = (None, 0, 0)
        if val_samples:
            val_loader, val_kept, val_dropped = self._create_dataloader(val_samples)

        if not dataloader:
            return {"error": "insufficient_data", "kept": kept, "dropped": dropped}

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        best_metric = float("inf")
        patience, wait = 3, 0
        train_losses = []

        for epoch in range(1, self.epochs + 1):
            avg_loss = self._train_epoch(self.model, dataloader, epoch_idx=epoch)
            train_losses.append(avg_loss)

            val_metrics = self._validate(self.model, val_loader) if val_loader else {}
            if self.logger:
                payload = {"epoch": epoch, "train_loss": float(avg_loss)}
                payload.update({f"val_{k}": v for k, v in val_metrics.items()})
                self.logger.log("TinyRecursionEpoch", payload)

            # early-stop on validation MAE (or train loss if no val)
            stop_metric = val_metrics.get("mae", avg_loss) if val_metrics else avg_loss

            if stop_metric < best_metric - 1e-4:
                best_metric = stop_metric
                wait = 0
                locator = self.get_locator(dimension)
                torch.save(self.model.state_dict(), locator.model_file(suffix="_tiny.pt"))  # <<< save per-dim, tiny suffix
            else:
                wait += 1
                if wait >= patience:
                    if self.logger:
                        self.logger.log("TinyRecursionEarlyStopping", {"epoch": epoch})
                    break

        # --- meta -------------------------------------------------------------
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
            "bucket_size": self.bucket_size,
            "vocab_size": self.vocab_size,                           # <<<
        }

        meta = {
            "dimension": dimension,
            "model_type": "tiny_recursion",
            "expects_triplet": True,                                  # <<< tells scorer interface is (x,y,z)
            "embedding_type": self.embedding_type,
            "input_dim": self.dim,                                    # each of x,y,z length
            "concat_input_dim": self.dim * 2,                         # useful if a scorer builds x=[ctx,doc]
            "version": self.version,
            "epochs": self.epochs,
            "avg_loss": float(min(train_losses or [best_metric])),
            "timestamp": datetime.now().isoformat(),
            "cfg": dict(self.cfg),
            "kept": int(kept),
            "dropped": int(dropped),
            "val_kept": int(val_kept),
            "val_dropped": int(val_dropped),
        }
        self._save_meta_file(meta, dimension)                         # <<< will write next to the model file

        # TrainingStatsStore integration
        self.memory.training_stats.add_from_result(
            stats={"avg_q_loss": float(min(train_losses or [best_metric]))},
            model_type=self.model_type,                               # <<<
            target_type=self.target_type,                             # <<<
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
            "train_loss_curve": [float(x) for x in train_losses],
            "kept": int(kept),
            "dropped": int(dropped),
            "val_kept": int(val_kept),
            "val_dropped": int(val_dropped),
        }

    # ------------------------------
    # Helpers
    # ------------------------------
    def _bucketize_label(self, y: int) -> int:
        """
        If bucket_size > 0, map 0..100 → 0..num_bins-1 using fixed-width bins.
        Ensure vocab_size >= num_bins when you enable bucketing.
        """
        b = int(self.bucket_size)
        if b <= 0:
            return max(0, min(100, y))              # <<< clamp to 0..100
        num_bins = (101 + b - 1) // b               # e.g., b=10 -> 11 bins (0..10)
        yb = min(max(y, 0) // b, num_bins - 1)
        return yb

    def _bucketize_counts(self, counts: Counter, bucket: int) -> dict:
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

        start = 0
        while start <= 100:
            end = min(100, start + bucket - 1)
            key = f"{start}-{end}"
            buckets.setdefault(key, 0)
            start += bucket

        return dict(sorted(buckets.items(), key=lambda kv: int(kv[0].split('-')[0])))
