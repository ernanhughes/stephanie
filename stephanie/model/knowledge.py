# stephanie/model/knowledge.py
from __future__ import annotations

import json
import os
from typing import List, Optional

import torch
import torch.nn as nn


class CrossFeatureEncoder(nn.Module):
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
        z_bi = self.bilinear(goal, text)
        z_feats = torch.cat([goal, text, goal * text, torch.abs(goal - text)], dim=-1)
        z_proj = self.proj(z_feats)
        return z_bi + z_proj


class AuxProjector(nn.Module):
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
        return z + self.mlp(aux)


class KnowledgePredictor(nn.Module):
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
    Two-head knowledge scorer:
      - predictor_h: human (gold) head
      - predictor_a: AI (weak) head
    """
    def __init__(self, dim: int, hdim: int, embedding_store, aux_feature_names: Optional[List[str]] = None, device: str = "cpu"):
        self.device = device
        self.embedding_store = embedding_store
        self.aux_feature_names = aux_feature_names or []
        self.encoder = CrossFeatureEncoder(dim, hdim).to(device)
        self.aux_proj = AuxProjector(hdim, aux_dim=len(self.aux_feature_names)).to(device)
        # heads
        self.predictor_h = KnowledgePredictor(hdim).to(device)
        self.predictor_a = KnowledgePredictor(hdim).to(device)

    # ---- scoring API ----
    def score_h(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor_h(z).squeeze(-1)

    def score_a(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor_a(z).squeeze(-1)

    @torch.no_grad()
    def predict(
        self,
        goal_text: str,
        candidate_text: str,
        meta: Optional[dict] = None,
        *,
        return_components: bool = False,
    ) -> float | tuple[float, dict]:
        """
        Returns a blended probability in [0,1].
        If return_components=True, also returns a dict with attribution details.
        """
        meta = meta or {}

        # --- encode ---
        g = self._embed(goal_text)                 # [1,D]
        x = self._embed(candidate_text)            # [1,D]
        z = self.encoder(g, x)                     # [1,H]
        aux = self._aux_tensor(meta)               # [1,A] or None
        z = self.aux_proj(z, aux)                  # [1,H]

        # --- logits -> probs ---
        s_h = self.score_h(z)                      # [1]
        s_a = self.score_a(z)                      # [1]
        s_h_val = float(s_h.item())
        s_a_val = float(s_a.item())
        h_prob = float(torch.sigmoid(torch.tensor(s_h_val)).item())
        a_prob = float(torch.sigmoid(torch.tensor(s_a_val)).item())

        # --- blending rule (human-first) ---
        has_similar_human = bool(meta.get("has_similar_human", False))
        alpha = 1.0 if has_similar_human else 0.6
        p = alpha * h_prob + (1.0 - alpha) * a_prob

        if not return_components:
            return p

        # components & fractions
        human_component = alpha * h_prob
        ai_component    = (1.0 - alpha) * a_prob
        denom = human_component + ai_component
        if denom > 0.0:
            human_fraction = human_component / denom
            ai_fraction    = ai_component / denom
        else:
            human_fraction = ai_fraction = 0.5  # guard

        details = {
            # final
            "probability": float(p),

            # raw head signals
            "human_logit": round(s_h_val, 6),
            "ai_logit": round(s_a_val, 6),
            "human_prob": round(h_prob, 6),
            "ai_prob": round(a_prob, 6),

            # blending
            "alpha_human_weight": float(alpha),
            "has_similar_human": has_similar_human,

            # contributions
            "human_component": round(human_component, 6),
            "ai_component": round(ai_component, 6),
            "human_fraction": round(human_fraction, 6),
            "ai_fraction": round(ai_fraction, 6),
        }
        return p, details

    def _blend_scores(self, s_h: float, s_a: float, meta: Optional[dict] = None) -> float:
        meta = meta or {}
        has_similar_human = bool(meta.get("has_similar_human", False))
        # human-first sigmoid blending
        h = torch.sigmoid(torch.tensor(s_h)).item()
        a = torch.sigmoid(torch.tensor(s_a)).item()
        if has_similar_human:
            return h
        alpha = 0.6  # bias toward human
        return alpha * h + (1 - alpha) * a

    # ---- utils ----
    def _embed(self, text: str) -> torch.Tensor:
        v = self.embedding_store.get_or_create(text)
        t = torch.tensor(v, device=self.device, dtype=torch.float32).unsqueeze(0)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-12)
        return t

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
        return torch.tensor(vals, device=self.device, dtype=torch.float32).unsqueeze(0)

    def train(self):
        self.encoder.train()
        self.aux_proj.train()
        self.predictor_h.train()
        self.predictor_a.train()

    def eval(self):
        self.encoder.eval()
        self.aux_proj.eval()
        self.predictor_h.eval()
        self.predictor_a.eval()

    def save(
        self,
        *,
        encoder_path: str,
        head_h_path: str,     # human head
        head_a_path: str,     # AI head
        auxproj_path: str,
        manifest_path: str | None = None,
        extra: dict | None = None,       # e.g., version, dim, hdim, aux names
    ) -> dict:
        os.makedirs(os.path.dirname(os.path.abspath(encoder_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(head_h_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(head_a_path)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(auxproj_path)), exist_ok=True)

        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.predictor_h.state_dict(), head_h_path)
        torch.save(self.predictor_a.state_dict(), head_a_path)
        torch.save(self.aux_proj.state_dict(), auxproj_path)

        manifest = {
            "format": "knowledge.v1",
            "device": str(self.device),
            "dim": getattr(self.encoder, "dim", None),
            "hdim": getattr(self.encoder, "hdim", None),
            "aux_features": self.aux_feature_names,
            "files": {
                "encoder": os.path.basename(encoder_path),
                "head_h":  os.path.basename(head_h_path),
                "head_a":  os.path.basename(head_a_path),
                "auxproj": os.path.basename(auxproj_path),
            }, 
        }
        if extra:
            manifest["extra"] = extra

        if manifest_path:
            os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
            import json
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

        return manifest

    @classmethod
    def load(
        cls,
        *,
        dim: int,
        hdim: int,
        embedding_store,
        aux_feature_names: list[str] | None,
        device: str,
        encoder_path: str,
        head_h_path: str,
        head_a_path: str,
        auxproj_path: str,
    ) -> KnowledgeModel:
        model = cls(dim=dim, hdim=hdim, embedding_store=embedding_store,
                    aux_feature_names=aux_feature_names, device=device)
        model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        model.predictor_h.load_state_dict(torch.load(head_h_path, map_location=device))
        model.predictor_a.load_state_dict(torch.load(head_a_path, map_location=device))
        model.aux_proj.load_state_dict(torch.load(auxproj_path, map_location=device))
        model.eval()
        return model

    def save_bundle(
        self,
        dir_path: str,
        *,
        extra: dict | None = None,
    ) -> dict:
        os.makedirs(dir_path, exist_ok=True)
        encoder = os.path.join(dir_path, "encoder.pt")
        head_h  = os.path.join(dir_path, "head_h.pt")
        head_a  = os.path.join(dir_path, "head_a.pt")
        auxproj = os.path.join(dir_path, "auxproj.pt")
        manifest_path = os.path.join(dir_path, "manifest.json")

        return self.save(
            encoder_path=encoder,
            head_h_path=head_h,
            head_a_path=head_a,
            auxproj_path=auxproj,
            manifest_path=manifest_path,
            extra=extra,
        )

    @classmethod
    def load_bundle(
        cls,
        dir_path: str,
        *,
        embedding_store,
        device: str = "cpu",
    ) -> KnowledgeModel:
        manifest_path = os.path.join(dir_path, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        dim   = int(m.get("dim"))
        hdim  = int(m.get("hdim"))
        aux   = m.get("aux_features") or []
        files = m["files"]

        return cls.load(
            dim=dim,
            hdim=hdim,
            embedding_store=embedding_store,
            aux_feature_names=aux,
            device=device,
            encoder_path=os.path.join(dir_path, os.path.basename(files["encoder"])),
            head_h_path=os.path.join(dir_path, os.path.basename(files["head_h"])),
            head_a_path=os.path.join(dir_path, os.path.basename(files["head_a"])),
            auxproj_path=os.path.join(dir_path, os.path.basename(files["auxproj"])),
        )
