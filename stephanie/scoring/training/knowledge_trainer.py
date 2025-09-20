from __future__ import annotations
from datetime import datetime
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from stephanie.models.training_stats import TrainingStatsORM
from stephanie.scoring.training.base_trainer import BaseTrainer
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.scoring.model.knowledge import KnowledgeModel

def dpo_lite_loss(s_pos: torch.Tensor, s_neg: torch.Tensor, beta: float = 2.0, margin: float = 0.2) -> torch.Tensor:
    # -log σ(β * (s_pos - s_neg - margin))
    diff = beta * (s_pos - s_neg - margin)
    return F.softplus(-diff).mean()

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
        self.min_samples = cfg.get("min_samples", 20)
        self.use_tuner = cfg.get("use_tuner", True)
        self.use_early_stopping = cfg.get("early_stopping", True)
        self.early_stopping_patience = cfg.get("patience", 3)
        self.early_stopping_min_delta = cfg.get("min_delta", 1e-4)
        self.beta = cfg.get("beta", 2.0)
        self.margin = cfg.get("margin", 0.2)
        self.aux_features = cfg.get("aux_features", [
            "human_stars", "pseudo_stars", "artifact_quality",
            "turn_pos_ratio", "has_retrieval", "retrieval_fidelity",
            "text_len_norm"
        ])

        self.model = KnowledgeModel(
            dim=self.dim, hdim=self.hdim, embedding_store=self.memory.embedding,
            aux_feature_names=self.aux_features, device=str(self.device)
        )

        self.logger.log("KnowledgeTrainerInitialized", {
            "embedding_type": self.embedding_type,
            "device": str(self.device),
            "aux_features": self.aux_features
        })

    # --- dataloader expects pairs with meta ---
    def _aux_vec(self, meta: dict) -> torch.Tensor:
        vals = []
        for n in self.aux_features:
            try:
                vals.append(float(meta.get(n, 0.0)))
            except Exception:
                vals.append(0.0)
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def _create_dataloader(self, pairs):
        """
        pairs: [{
          'title'|'prompt', 'output_a', 'output_b',
          'value_a', 'value_b', 'meta_a', 'meta_b'
        }, ...]
        """
        G, A, B, AUXA, AUXB = [], [], [], [], []

        for p in pairs:
            goal = p.get("title") or p.get("prompt") or ""
            oa, ob = p.get("output_a",""), p.get("output_b","")
            ma, mb = p.get("meta_a", {}) or {}, p.get("meta_b", {}) or {}
            if not (goal and oa and ob):
                continue
            try:
                ge = torch.tensor(self.memory.embedding.get_or_create(goal), device=self.device, dtype=torch.float32)
                ea = torch.tensor(self.memory.embedding.get_or_create(oa), device=self.device, dtype=torch.float32)
                eb = torch.tensor(self.memory.embedding.get_or_create(ob), device=self.device, dtype=torch.float32)
                auxa = self._aux_vec(ma)
                auxb = self._aux_vec(mb)
                G.append(ge); A.append(ea); B.append(eb); AUXA.append(auxa); AUXB.append(auxb)
            except Exception as e:
                self.logger.log("KnowledgePairPrepError", {"error": str(e)})
                continue

        if len(G) < self.min_samples:
            self.logger.log("InsufficientSamples", {"sample_count": len(G), "threshold": self.min_samples})
            return None

        ds = TensorDataset(
            torch.stack(G), torch.stack(A), torch.stack(B),
            torch.stack(AUXA), torch.stack(AUXB)
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True)

    def _train_epoch(self, dl):
        self.model.train()
        opt = self.optimizer
        tot, n = 0.0, 0
        for G, A, B, AUXA, AUXB in dl:
            # encode
            z_a = self.model.encoder(G, A)               # [B,H]
            z_b = self.model.encoder(G, B)               # [B,H]
            z_a = self.model.aux_proj(z_a, AUXA)         # [B,H]
            z_b = self.model.aux_proj(z_b, AUXB)         # [B,H]
            s_a = self.model.predictor(z_a)              # [B]
            s_b = self.model.predictor(z_b)              # [B]

            # prefer 'a' over 'b' by margin (we build pairs so that value_a >= value_b)
            loss_pref = dpo_lite_loss(s_a, s_b, beta=self.beta, margin=self.margin)
            loss_reg = 1e-4 * (s_a.square().mean() + s_b.square().mean())
            loss = loss_pref + loss_reg

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm__(self.model.encoder.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm__(self.model.predictor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm__(self.model.aux_proj.parameters(), 0.5)
            opt.step()

            bs = G.size(0)
            tot += loss.item() * bs; n += bs
        return tot / max(1, n)

    def train(self, pairs, dimension: str = "knowledge"):
        dl = self._create_dataloader(pairs)
        if not dl:
            return {"error": "insufficient_data", "dimension": dimension}

        self.optimizer = optim.Adam(
            list(self.model.encoder.parameters()) +
            list(self.model.aux_proj.parameters()) +
            list(self.model.predictor.parameters()),
            lr=self.lr
        )

        best, patience = float("inf"), 0
        for epoch in range(self.epochs):
            loss = self._train_epoch(dl)
            self.logger.log("KnowledgeTrainingEpoch", {"epoch": epoch + 1, "loss": loss})
            if loss < best - self.early_stopping_min_delta:
                best, patience = loss, 0
            else:
                patience += 1
                if self.use_early_stopping and patience >= self.early_stopping_patience:
                    break

        locator = self.get_locator(dimension)
        self.model.save(locator.encoder_file(), locator.model_file(), locator.q_head_file())  # reuse q_head_file for aux_proj

        if self.use_tuner:
            tuner = RegressionTuner(dimension=dimension, logger=self.logger)
            # Calibrate predicted diffs ~ 1.0 on training pairs
            for G, A, B, AUXA, AUXB in dl:
                with torch.no_grad():
                    z_a = self.model.encoder(G, A); z_b = self.model.encoder(G, B)
                    z_a = self.model.aux_proj(z_a, AUXA); z_b = self.model.aux_proj(z_b, AUXB)
                    s_a = self.model.predictor(z_a); s_b = self.model.predictor(z_b)
                    diffs = (s_a - s_b).cpu().numpy().tolist()
                for d in diffs:
                    tuner.train_single(float(d), 1.0)
            tuner.save(locator.tuner_file())

        meta = {
            "dimension": dimension, "model_type": "knowledge", "target_type": self.target_type,
            "embedding_type": self.embedding_type, "version": self.version,
            "dim": self.dim, "hdim": self.hdim, "aux_features": self.aux_features,
            "min_value": 0.0, "max_value": 1.0, "avg_loss": best, "timestamp": datetime.now().isoformat()
        }
        self._save_meta_file(meta, dimension)

        stat = TrainingStatsORM(
            model_type="knowledge", target_type=self.target_type, dimension=dimension,
            version=self.version, embedding_type=self.embedding_type, avg_q_loss=best
        )
        self.memory.session.add(stat); self.memory.session.commit()
        return meta
