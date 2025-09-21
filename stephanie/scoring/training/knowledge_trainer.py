# stephanie/scoring/training/knowledge_trainer.py
from __future__ import annotations

from datetime import datetime
import math
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.scoring.model.knowledge import KnowledgeModel
from tqdm import tqdm


def dpo_lite_loss(
    s_pos: torch.Tensor,
    s_neg: torch.Tensor,
    beta: float = 2.0,
    margin: float = 0.2,
) -> torch.Tensor:
    # -log σ(β*(Δ - m)) implemented as softplus(-β*(Δ - m))
    return F.softplus(-(beta * (s_pos - s_neg - margin))).mean()


def _l2(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = x.norm(dim=-1, keepdim=True)
    return x / (n + eps)


class KnowledgeTrainer(BaseTrainer):
    """
    Pairwise trainer for the 'knowledge' dimension.
    Loss = DPO-lite (pairwise) + optional L2 regularizer on scores.
    Supports aux features (stars, artifact_quality, pos_ratio, fidelity, etc.).
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
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

        self.model = KnowledgeModel(
            dim=self.dim,
            hdim=self.hdim,
            embedding_store=self.memory.embedding,
            aux_feature_names=self.aux_features,
            device=str(self.device),
        )

        self.logger.log(
            "KnowledgeTrainerInitialized",
            {
                "embedding_type": self.embedding_type,
                "device": str(self.device),
                "aux_features": self.aux_features,
            },
        )

    # --- dataloader expects pairs with meta ---
    def _aux_vec(self, meta: dict) -> torch.Tensor:
        vals = []
        for n in self.aux_features:
            try:
                vals.append(float(meta.get(n, 0.0)))
            except Exception:
                vals.append(0.0)
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def _pairs_to_dataset(self, pairs):
        G, A, B, AUXA, AUXB = [], [], [], [], []
        for p in pairs:
            goal = p.get("title") or p.get("prompt") or ""
            oa, ob = p.get("output_a", ""), p.get("output_b", "")
            ma, mb = p.get("meta_a", {}) or {}, p.get("meta_b", {}) or {}
            if not (goal and oa and ob):
                continue
            try:
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
                # L2 normalize for consistency with builder & model usage
                ge, ea, eb = _l2(ge), _l2(ea), _l2(eb)
                auxa = self._aux_vec(ma)
                auxb = self._aux_vec(mb)
            except Exception as e:
                self.logger.log("KnowledgePairPrepError", {"error": str(e)})
                continue
            G.append(ge)
            A.append(ea)
            B.append(eb)
            AUXA.append(auxa)
            AUXB.append(auxb)

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
        )
        return ds

    def _create_dataloaders(self, pairs):
        ds = self._pairs_to_dataset(pairs)
        if ds is None:
            return None, None
        n_total = len(ds)
        n_val = max(int(0.1 * n_total), 1)
        n_train = max(n_total - n_val, 1)
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        dl_val = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        return dl, dl_val

    @torch.no_grad()
    def _pair_acc(self, dl) -> float:
        if dl is None:
            return float("nan")
        self.model.eval()
        correct, total = 0, 0
        for G, A, B, AUXA, AUXB in dl:
            z_a = self.model.encoder(G, A)
            z_b = self.model.encoder(G, B)
            z_a = self.model.aux_proj(z_a, AUXA)
            z_b = self.model.aux_proj(z_b, AUXB)
            s_a = self.model.predictor(z_a)
            s_b = self.model.predictor(z_b)
            correct += (s_a > s_b).sum().item()
            total += s_a.numel()
        return (correct / total) if total else float("nan")

    def _train_epoch(self, dl):
        self.model.train()
        opt = self.optimizer
        tot, n = 0.0, 
        for batch in tqdm(dl, desc="Training Epoch", leave=False):
            G, A, B, AUXA, AUXB = batch
            z_a = self.model.encoder(G, A)
            z_b = self.model.encoder(G, B)
            z_a = self.model.aux_proj(z_a, AUXA)
            z_b = self.model.aux_proj(z_b, AUXB)
            s_a = self.model.predictor(z_a)
            s_b = self.model.predictor(z_b)

            loss_pref = dpo_lite_loss(
                s_a, s_b, beta=self.beta, margin=self.margin
            )
            loss_reg = 1e-4 * (s_a.square().mean() + s_b.square().mean())
            loss = loss_pref + loss_reg

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.encoder.parameters())
                + list(self.model.aux_proj.parameters())
                + list(self.model.predictor.parameters()),
                0.5,
            )
            opt.step()
            bs = G.size(0)
            tot += loss.item() * bs
            n += bs
        return tot / max(1, n)

    def train(self, pairs, dimension: str = "knowledge"):
        dl, dl_val = self._create_dataloaders(pairs)
        if dl is None:
            return {"error": "insufficient_data", "dimension": dimension}

        # AdamW is a bit more forgiving; keep lr from cfg
        self.optimizer = optim.AdamW(
            list(self.model.encoder.parameters())
            + list(self.model.aux_proj.parameters())
            + list(self.model.predictor.parameters()),
            lr=self.lr,
            weight_decay=1e-4,
        )

        best_metric = -math.inf
        best_state = None
        patience = 0
        best_val_pair_acc = float("nan")

        for epoch in range(self.epochs):
            loss = self._train_epoch(dl)
            val_acc = self._pair_acc(dl_val)
            self.logger.log(
                "KnowledgeTrainingEpoch",
                {"epoch": epoch + 1, "loss": loss, "val_pair_acc": val_acc},
            )

            metric = (
                val_acc if not math.isnan(val_acc) else -loss
            )  # prefer val_acc when available
            if metric > best_metric + self.early_stopping_min_delta:
                best_metric = metric
                best_val_pair_acc = val_acc
                best_state = {
                    "enc": {
                        k: v.cpu()
                        for k, v in self.model.encoder.state_dict().items()
                    },
                    "aux": {
                        k: v.cpu()
                        for k, v in self.model.aux_proj.state_dict().items()
                    },
                    "pred": {
                        k: v.cpu()
                        for k, v in self.model.predictor.state_dict().items()
                    },
                }
                patience = 0
                # checkpoint best immediately
                locator = self.get_locator(dimension)
                torch.save(best_state["enc"], locator.encoder_file() + ".best")
                torch.save(best_state["pred"], locator.model_file() + ".best")
                torch.save(best_state["aux"], locator.q_head_file() + ".best")
            else:
                patience += 1
                if (
                    self.use_early_stopping
                    and patience >= self.early_stopping_patience
                ):
                    break

        # restore best (if any)
        if best_state is not None:
            self.model.encoder.load_state_dict(best_state["enc"])
            self.model.aux_proj.load_state_dict(best_state["aux"])
            self.model.predictor.load_state_dict(best_state["pred"])

        # persist final
        locator = self.get_locator(dimension)
        self.model.save(
            locator.encoder_file(), locator.model_file(), locator.q_head_file()
        )

        # Calibrate on VALIDATION, but only if we have batches and we actually trained the tuner
        if self.use_tuner and dl_val is not None:
            tuner = RegressionTuner(dimension=dimension, logger=self.logger)
            trained_count = 0
            with torch.no_grad():
                for G, A, B, AUXA, AUXB in dl_val:
                    z_a = self.model.encoder(G, A); z_b = self.model.encoder(G, B)
                    z_a = self.model.aux_proj(z_a, AUXA); z_b = self.model.aux_proj(z_b, AUXB)
                    s_a = self.model.predictor(z_a);     s_b = self.model.predictor(z_b)
                    diffs = (s_a - s_b).cpu().numpy().tolist()
                    for d in diffs:
                        tuner.train_single(float(d), 1.0)  # target delta ~ 1.0
                        trained_count += 1

            if trained_count > 0:
                tuner.save(locator.tuner_file())
                self.logger.log("KnowledgeTunerSaved", {
                    "file": locator.tuner_file(),
                    "trained_samples": trained_count
                })
            else:
                self.logger.log("KnowledgeTunerSkipped", {
                    "reason": "no_validation_batches_or_no_diffs",
                    "val_pairs": 0
                })

        meta = {
            "dimension": dimension,
            "model_type": "knowledge",
            "model_path": locator.model_file(),
            "q_head_path": locator.q_head_file(),
            "encoder_path": locator.encoder_file(),
            "target_type": self.target_type,
            "embedding_type": self.embedding_type,
            "version": self.version,
            "dim": self.dim,
            "hdim": self.hdim,
            "aux_features": self.aux_features,
            "min_value": 0.0,
            "max_value": 1.0,
            "avg_loss": float(
                best_metric if math.isnan(best_metric) else best_metric
            )
            if dl_val is None
            else float("nan"),
            "best_val_pair_acc": float(best_val_pair_acc)
            if not math.isnan(best_val_pair_acc)
            else None,
            "timestamp": datetime.now().isoformat(),
        }

        self._save_meta_file(meta, dimension)
        self.memory.training_stats.add_from_result(
            stats={"best_val_pair_acc": meta.get("best_val_pair_acc")},
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
            },
            sample_count=len(pairs),
            start_time=datetime.now(),
        )
        return meta
