# stephanie/scoring/ I model/knowledge.py
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, List

class CrossFeatureEncoder(nn.Module):
    """
    Joint encoder over (goal_emb, text_emb) with bilinear & elementwise interactions.
    Produces a compact H-dim representation for the predictor heads.
    """
    def __init__(self, dim: int, hdim: int):
        super().__init__()
        self.dim = dim
        self.hdim = hdim
        self.bilinear = nn.Bilinear(dim, dim, hdim, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(dim * 4, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim),
            nn.ReLU()
        )

    def forward(self, goal: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        # goal, text: [B, D]
        z_bi = self.bilinear(goal, text)  # [B, H]
        z_feats = torch.cat([goal, text, goal * text, torch.abs(goal - text)], dim=-1)  # [B, 4D]
        z_proj = self.proj(z_feats)  # [B, H]
        return z_bi + z_proj  # [B, H]


class AuxProjector(nn.Module):
    """
    Projects optional auxiliary numeric features into H and fuses via residual add.
    """
    def __init__(self, hdim: int, aux_dim: int):
        super().__init__()
        self.aux_dim = aux_dim
        if aux_dim > 0:
            self.mlp = nn.Sequential(
                nn.Linear(aux_dim, hdim),
                nn.ReLU(),
                nn.Linear(hdim, hdim)
            )
        else:
            self.mlp = None

    def forward(self, z: torch.Tensor, aux: Optional[torch.Tensor]) -> torch.Tensor:
        if self.mlp is None or aux is None:
            return z
        return z + self.mlp(aux)  # residual fusion


class KnowledgePredictor(nn.Module):
    """
    Main scalar head (continuous “knowledgefulness” score).
    """
    def __init__(self, hdim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hdim, hdim),
            nn.ReLU(),
            nn.Linear(hdim, hdim // 2),
            nn.ReLU(),
            nn.Linear(hdim // 2, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.mlp(z).squeeze(-1)  # [B]


class KnowledgeModel:
    """
    End-to-end model wrapper:
      - Uses your existing embedding_store (same interface as MRQModel).
      - Goal-conditioned scoring: score(goal_text, candidate_text, meta)
      - Aux feature injection for stability/controllability.
    """
    def __init__(self, dim: int, hdim: int, embedding_store, aux_feature_names: Optional[List[str]] = None, device: str = "cpu"):
        self.device = device
        self.embedding_store = embedding_store
        self.aux_feature_names = aux_feature_names or []
        self.encoder = CrossFeatureEncoder(dim, hdim).to(device)
        self.aux_proj = AuxProjector(hdim, aux_dim=len(self.aux_feature_names)).to(device)
        self.predictor = KnowledgePredictor(hdim).to(device)

    # ----- runtime API (MRQ-compatible) -----
    def _embed(self, text: str) -> torch.Tensor:
        v = self.embedding_store.get_or_create(text)
        return torch.tensor(v, device=self.device, dtype=torch.float32).unsqueeze(0)  # [1,D]

    def _aux_tensor(self, meta: Optional[dict]) -> Optional[torch.Tensor]:
        if not self.aux_feature_names:
            return None
        meta = meta or {}
        vals = []
        for name in self.aux_feature_names:
            try:
                vals.append(float(meta.get(name, 0.0)))
            except Exception:
                vals.append(0.0)
        return torch.tensor(vals, device=self.device, dtype=torch.float32).unsqueeze(0)  # [1,A]

    def predict(self, goal_text: str, candidate_text: str, meta: Optional[dict] = None) -> float:
        g = self._embed(goal_text)    # [1,D]
        x = self._embed(candidate_text)
        z = self.encoder(g, x)        # [1,H]
        aux = self._aux_tensor(meta)  # [1,A] or None
        z = self.aux_proj(z, aux)     # [1,H]
        score = self.predictor(z).item()
        return score

    def train(self):
        self.encoder.train(); self.aux_proj.train(); self.predictor.train()

    def eval(self):
        self.encoder.eval(); self.aux_proj.eval(); self.predictor.eval()

    # ----- checkpoints -----
    def save(self, encoder_path: str, predictor_path: str, auxproj_path: str):
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.predictor.state_dict(), predictor_path)
        torch.save(self.aux_proj.state_dict(), auxproj_path)

    def load(self, encoder_path: str, predictor_path: str, auxproj_path: str):
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.predictor.load_state_dict(torch.load(predictor_path, map_location=self.device))
        self.aux_proj.load_state_dict(torch.load(auxproj_path, map_location=self.device))
        self.eval()
