# stephanie/scoring/training/base_trainer.py
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from stephanie.utils.json_sanitize import dumps_safe

# Optional tqdm (pretty progress if available)
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


class BaseTrainer:
    # ----------------------------- Model locator -----------------------------
    class Locator:
        def __init__(self, root_dir, model_type, target_type, dimension, version, embedding_type):
            self.root_dir = root_dir
            self.model_type = model_type
            self.target_type = target_type
            self.dimension = dimension
            self.version = version
            self.embedding_type = embedding_type

        @property
        def base_path(self) -> str:
            path = os.path.join(
                self.root_dir,
                self.embedding_type,
                self.model_type,
                self.target_type,
                self.dimension,
                self.version,
            )
            os.makedirs(path, exist_ok=True)
            return path

        def encoder_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_encoder.pt")

        def q_head_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_q.pt")

        def v_head_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_v.pt")

        def pi_head_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_pi.pt")

        def auxproj_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_auxproj.pt")

        def meta_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}.meta.json")

        def tuner_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}.tuner.json")

        def scaler_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_scaler.joblib")

        def joblib_file(self) -> str:
            return os.path.join(self.base_path, f"{self.dimension}_model.joblib")

        def model_file(self, suffix: str = ".pt") -> str:
            return os.path.join(self.base_path, f"{self.dimension}{suffix}")

        def model_exists(self) -> bool:
            # Treat an existing encoder (and at least one head OR generic model file) as "exists"
            return (
                os.path.exists(self.encoder_file()) and
                (
                    os.path.exists(self.q_head_file()) or
                    os.path.exists(self.v_head_file()) or
                    os.path.exists(self.pi_head_file()) or
                    os.path.exists(self.model_file())
                )
            )

    # ----------------------------- Init -----------------------------
    def __init__(self, cfg, memory, container, logger):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        self.embedding_type = self.memory.embedding.name
        self.dim = memory.embedding.dim
        self.hdim = memory.embedding.hdim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.root_dir = cfg.get("model_path", "models")
        self.version = cfg.get("model_version", "v1")
        self.target_type = cfg.get("target_type", "document")
        self.model_type = cfg.get("model_type", "base")

        self.dimensions = cfg.get("dimensions", [])
        self.min_samples = cfg.get("min_samples", 5)
        self.batch_size = int(cfg.get("batch_size", 32))

        # progress flags (shared)
        self.show_progress = bool(cfg.get("show_progress", True))
        self.progress_leave = bool(cfg.get("progress_leave", False))

        # for early stopping (shared)
        self.early_stopping_patience = int(cfg.get("patience", 3))
        self.use_early_stopping = bool(cfg.get("use_early_stopping", True))
        self.early_stopping_min_delta = float(cfg.get("min_delta", 1e-4))

    def get_locator(self, dimension: str) -> "BaseTrainer.Locator":
        return self.Locator(
            root_dir=self.root_dir,
            model_type=self.model_type,
            target_type=self.target_type,
            dimension=dimension,
            version=self.version,
            embedding_type=self.embedding_type,
        )

    # ----------------------------- Dataloader (generic: ctx/output/score) -----------------------------
    def _create_dataloader(self, samples: List[Dict[str, Any]]) -> DataLoader | None:
        """
        Expects samples shaped like:
          { "title": <str>, "output": <str>, "score": <float> }
        Returns a DataLoader yielding (ctx_emb, doc_emb, score) — all on CPU, float32.
        """
        ctxs, docs, ys = [], [], []
        for s in samples:
            ctx_text = s.get("title", "")
            doc_text = s.get("output", "")
            score = s.get("score", None)

            if not ctx_text or not doc_text or score is None:
                continue

            try:
                # keep tensors on CPU; move to device inside training loop
                ctx = torch.tensor(self.memory.embedding.get_or_create(ctx_text), dtype=torch.float32)  # [D]
                doc = torch.tensor(self.memory.embedding.get_or_create(doc_text), dtype=torch.float32)  # [D]
                y = torch.tensor(float(score), dtype=torch.float32)
                ctxs.append(ctx); docs.append(doc); ys.append(y)
            except Exception as e:
                if self.logger:
                    self.logger.log("CreateDataloaderSampleError", {"error": str(e)})

        if len(ctxs) < self.min_samples:
            if self.logger:
                self.logger.log("InsufficientSamples", {"sample_count": len(ctxs), "threshold": self.min_samples})
            return None

        dataset = TensorDataset(torch.stack(ctxs), torch.stack(docs), torch.stack(ys))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    # ----------------------------- Meta / Logging -----------------------------

    def _save_meta_file(self, meta: dict, dimension: str):
        locator = self.get_locator(dimension)
        with open(locator.meta_file(), "w", encoding="utf-8") as f:
            f.write(dumps_safe(meta, indent=2))

    def _calculate_policy_metrics(self, logits: List[float]) -> Tuple[float, float]:
        probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
        entropy = float(-(probs * torch.log(probs + 1e-8)).sum())
        stability = float(probs.max())
        return entropy, stability

    def log_event(self, name: str, payload: dict):
        if self.logger:
            self.logger.log(name, payload)

    # ----------------------------- Progress helpers -----------------------------
    def progress(self, iterable, desc: str = "", unit: str = "it", leave: bool = False):
        if self.show_progress and tqdm is not None:
            return tqdm(iterable, desc=desc, unit=unit, leave=leave)
        return iterable

    def progress_postfix(self, pbar, **kv):
        if pbar is not None and hasattr(pbar, "set_postfix"):
            pbar.set_postfix(**{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in kv.items()})

    # ----------------------------- Numerics -----------------------------
    def _as_1d_list(self, x):
        if isinstance(x, torch.Tensor): return x.detach().cpu().view(-1).tolist()
        if isinstance(x, np.ndarray):
            return x.reshape(-1).tolist()
        if isinstance(x, (int, float)):
            return [x]
        try:
            return list(x)
        except Exception:
            return [x]

    def _as_np_1d(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().view(-1).numpy()
        if isinstance(x, np.ndarray):
            return x.reshape(-1)
        if isinstance(x, (list, tuple)):
            return np.asarray(x, dtype=np.float32).reshape(-1)
        return np.asarray([x], dtype=np.float32)

    def _spearmanr(self, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        ra = a.argsort().argsort().astype(np.float64)
        rb = b.argsort().argsort().astype(np.float64)
        ra = (ra - ra.mean()) / (ra.std() + 1e-9)
        rb = (rb - rb.mean()) / (rb.std() + 1e-9)
        return float(np.clip(np.mean(ra * rb), -1.0, 1.0))

    # ----------------------------- Metrics (reg + cls) -----------------------------
    def regression_metrics(self, preds, acts):
        preds_v = self._as_np_1d(preds)
        acts_v  = self._as_np_1d(acts)
        n = min(len(preds_v), len(acts_v))
        if n == 0:
            return dict(count=0, mae=math.nan, mse=math.nan, rmse=math.nan, r2=math.nan,
                        pearson_r=math.nan, spearman_rho=math.nan,
                        within1=math.nan, within2=math.nan, within5=math.nan,
                        pred_mean=math.nan, pred_std=math.nan)

        p = preds_v[:n].astype(np.float64)
        y = acts_v[:n].astype(np.float64)
        d = p - y

        mae = float(np.mean(np.abs(d)))
        mse = float(np.mean(d*d))
        rmse = float(np.sqrt(mse))
        var = float(np.var(y))
        r2 = float(1.0 - (np.sum(d**2) / (np.sum((y - y.mean())**2) + 1e-9))) if var > 1e-12 else float("nan")
        pearson = float(np.corrcoef(p, y)[0,1]) if (np.std(p) > 1e-12 and np.std(y) > 1e-12) else float("nan")
        spearman = self._spearmanr(p, y)

        # --- auto-scale within deltas ---
        vmax = float(max(np.max(np.abs(p)), np.max(np.abs(y))))
        if vmax <= 1.5:
            d1, d2, d5 = 0.01, 0.02, 0.05  # 1/2/5 percentage points in 0..1
        else:
            d1, d2, d5 = 1.0, 2.0, 5.0     # original 0..100 behavior

        within1 = float(np.mean(np.abs(d) <= d1))
        within2 = float(np.mean(np.abs(d) <= d2))
        within5 = float(np.mean(np.abs(d) <= d5))

        return dict(count=int(n), mae=mae, mse=mse, rmse=rmse, r2=r2,
                    pearson_r=pearson, spearman_rho=spearman,
                    within1=within1, within2=within2, within5=within5,
                    pred_mean=float(np.mean(p)), pred_std=float(np.std(p)))

    @torch.no_grad()
    def collect_preds_targets(self, model, dataloader, device, head: str = "q", apply_sigmoid: bool = False):
        """
        Works with:
        - SICQL-style: (ctx, doc, y)  -> model(ctx, doc)[head]
        - MRQ-style:   (X, y, w?)     -> model.predictor(X)
        - MRQ pairs 2-tuple: (X, y)   -> model.predictor(X)
        """
        preds_all, acts_all = [], []
        model.eval()
        for batch in dataloader:
            # Unpack defensively
            if len(batch) == 3:
                a, b, c = batch
                # Heuristic: MRQ triple if model has predictor AND first elem looks like a feature matrix
                if hasattr(model, "predictor") and isinstance(a, torch.Tensor) and a.dim() >= 2:
                    X, y = a.to(device), b.to(device)
                    out = model.predictor(X)
                    p = out.view(-1).detach().cpu().numpy()
                    a_np = y.view(-1).detach().cpu().numpy()
                else:
                    # SICQL triple: (ctx, doc, y)
                    ctx, doc, y = a.to(device), b.to(device), c.to(device)
                    outs = model(ctx, doc)
                    # prefer 'q_value' head, else first value
                    if isinstance(outs, dict):
                        out = outs.get("q_value", next(iter(outs.values())))
                    else:
                        out = outs
                    p = out.view(-1).detach().cpu().numpy()
                    a_np = y.view(-1).detach().cpu().numpy()

            elif len(batch) == 2:
                # MRQ pair: (X, y)
                X, y = batch
                X, y = X.to(device), y.to(device)
                if hasattr(model, "predictor"):
                    out = model.predictor(X)
                else:
                    out = X.sum(dim=-1)
                p = out.view(-1).detach().cpu().numpy()
                a_np = y.view(-1).detach().cpu().numpy()

            else:
                # Fallback: try to handle like MRQ features
                X = batch[0].to(device)
                out = model.predictor(X) if hasattr(model, "predictor") else X.sum(dim=-1)
                p = out.view(-1).detach().cpu().numpy()
                a_np = batch[1].to(device).view(-1).detach().cpu().numpy()

            # Optionally map logits → probabilities
            if apply_sigmoid:
                p = 1.0 / (1.0 + np.exp(-np.clip(p, -50.0, 50.0)))
            preds_all.append(p)
            acts_all.append(a_np)

        if not preds_all:
            return np.array([]), np.array([])

        return np.concatenate(preds_all, axis=0), np.concatenate(acts_all, axis=0)

    @torch.no_grad()
    def binary_cls_metrics(self, dataloader, forward_fn):
        Ls, Ys = [], []
        for batch in dataloader:
            if len(batch) == 3:
                X, y, _ = batch
            else:
                X, y    = batch
            logits = forward_fn(X).view(-1).detach().cpu().numpy()
            Ls.append(logits)
            Ys.append(y.view(-1).detach().cpu().numpy().astype(np.float32))
        if not Ls:
            return {"pair_acc": float("nan"), "auc": float("nan"), "logloss": float("nan"), "pos_rate": float("nan")}
        L = np.concatenate(Ls)
        Y = np.concatenate(Ys)
        pair_acc = float(((L > 0).astype(np.float32) == Y).mean())
        pos_rate = float((L > 0).mean())
        # rank-based AUC (safe)
        try:
            order = np.argsort(L)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(L))
            pos = (Y == 1).astype(np.float64)
            neg = 1.0 - pos
            n_pos, n_neg = pos.sum(), neg.sum()
            auc = float(((ranks[pos == 1]).sum() - n_pos*(n_pos-1)/2) / max(n_pos*n_neg, 1.0)) if n_pos>0 and n_neg>0 else float("nan")
        except Exception:
            auc = float("nan")
        # logloss
        P = 1.0 / (1.0 + np.exp(-L))
        eps = 1e-7
        logloss = float(-np.mean(Y*np.log(P+eps) + (1-Y)*np.log(1-P+eps)))
        return {"pair_acc": pair_acc, "auc": auc, "logloss": logloss, "pos_rate": pos_rate}

    # ----------------------------- Early stopping / priors -----------------------------
    def early_stop_update(self, best, current, patience, min_delta):
        if current < best - min_delta:
            return current, 0, False
        patience += 1
        return best, patience, patience >= self.early_stopping_patience

    def init_pos_weight_and_bias(self, dataloader, out_layer, clamp=(0.2, 5.0)):
        ys = []
        for batch in dataloader:
            if len(batch) == 3:
                _, y, _ = batch
            elif len(batch) == 2:
                _, y = batch
            else:
                continue
            ys.append(y.detach().cpu().view(-1))
        y_all = torch.cat(ys) if ys else torch.tensor([])
        p_pos = float((y_all > 0.5).float().mean()) if y_all.numel() > 0 else 0.5
        eps = 1e-6
        pos_weight = float((1.0 - p_pos) / max(p_pos, eps))
        pos_weight = float(min(max(pos_weight, clamp[0]), clamp[1]))
        self.pos_weight_tensor = torch.tensor(pos_weight, device=self.device)
        try:
            prior_logit = math.log(p_pos / max(1.0 - p_pos, eps))
            if hasattr(out_layer, "bias") and out_layer.bias is not None:
                with torch.no_grad():
                    out_layer.bias.fill_(prior_logit)
        except Exception:
            pass
        return p_pos, pos_weight

    # ----------------------------- Snapshot -----------------------------
    def report_training_snapshot(
        self,
        dataloader,
        *,
        meta: dict | None = None,
        losses: List[float] | None = None,
        title: str = "Training",
        head: int = 12,
        within_deltas=(1.0, 2.0, 5.0),
    ):
        """
        Quick diagnostic (regression-flavored): runs collect_preds_targets(self.model, ...)
        """
        preds, acts = self.collect_preds_targets(self.model, dataloader, device=self.device, head="q")
        preds_v = self._as_np_1d(preds)
        acts_v  = self._as_np_1d(acts)
        n = int(min(preds_v.size, acts_v.size))

        if n == 0:
            msg = f"=== {title} Training Snapshot ===\n(no predictions/targets)"
            print(msg)
            self.logger and self.logger.log("TrainingSnapshotEmpty", {"title": title})
            return {"title": title, "count": 0, "message": msg}

        preds_v = preds_v[:n].astype(np.float64)
        acts_v  = acts_v[:n].astype(np.float64)
        diff    = preds_v - acts_v

        mae  = float(np.mean(np.abs(diff)))
        mse  = float(np.mean(diff * diff))
        rmse = float(np.sqrt(mse))
        var_y = float(np.var(acts_v))
        r2    = float(1.0 - (np.sum(diff**2) / (np.sum((acts_v - acts_v.mean())**2) + 1e-9))) if var_y > 1e-12 else float("nan")
        pearson = float(np.corrcoef(preds_v, acts_v)[0, 1]) if (np.std(preds_v) > 1e-12 and np.std(acts_v) > 1e-12) else float("nan")
        spearman = self._spearmanr(preds_v, acts_v)

        within = {f"within{int(d) if float(d).is_integer() else d}": float(np.mean(np.abs(diff) <= d))
                  for d in within_deltas}

        lc_head = [float(x) for x in (losses[:5] if losses else [])]
        lc_tail = [float(x) for x in (losses[-5:] if losses else [])]

        report = {
            "title": title,
            "count": n,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "pearson_r": float(pearson),
            "spearman_rho": float(spearman),
            **within,
            "pred_mean": float(np.mean(preds_v)),
            "pred_std": float(np.std(preds_v)),
            "target_mean": float(np.mean(acts_v)),
            "target_std": float(np.std(acts_v)),
            "loss_curve_head": lc_head,
            "loss_curve_tail": lc_tail,
            "loss_best": float(min(losses)) if losses else None,
            "loss_first": float(losses[0]) if losses else None,
            "loss_last": float(losses[-1]) if losses else None,
            "meta": meta or {},
            "head_pairs": list(zip(np.round(preds_v[:head], 3), np.round(acts_v[:head], 3))),
        }

        lines = []
        lines.append(f"=== {title} Training Snapshot ===")
        lines.append(f"count={report['count']}")
        lines.append(f"MAE={report['mae']:.4f}  RMSE={report['rmse']:.4f}  MSE={report['mse']:.4f}")
        lines.append(f"R2={report['r2']:.4f}  Pearson r={report['pearson_r']:.4f}  Spearman ρ={report['spearman_rho']:.4f}")
        wkeys = [k for k in report.keys() if k.startswith("within")]
        if wkeys:
            lines.append(" | ".join([f"{k}={report[k]:.3f}" for k in sorted(wkeys, key=lambda k: float(k.replace('within','')))]))
        lines.append(f"pred μ={report['pred_mean']:.3f} σ={report['pred_std']:.3f} | target μ={report['target_mean']:.3f} σ={report['target_std']:.3f}")
        if losses:
            lines.append(f"loss first/best/last: {report['loss_first']:.4f} / {report['loss_best']:.4f} / {report['loss_last']:.4f}")
            lines.append(f"loss head: {', '.join([f'{x:.4f}' for x in lc_head])}")
            lines.append(f"loss tail: {', '.join([f'{x:.4f}' for x in lc_tail])}")
        if meta:
            lines.append(f"meta keys: {', '.join(sorted(meta.keys()))}")
        lines.append(f"preds vs targets (head {min(head, n)}): {report['head_pairs']}")

        text = "\n".join(lines)
        print(text)
        if self.logger:
            try:
                self.logger.log("TrainingSnapshot", {
                    **{k: v for k, v in report.items() if k not in ("meta", "head_pairs")},
                    "title": title
                })
            except Exception:
                pass

        return report

    # ----------------------------- Abstract -----------------------------
    def train_step(self, max_steps: int = 50, dimension: str | None = None) -> dict:
        raise NotImplementedError("train_step must be implemented in subclasses")
