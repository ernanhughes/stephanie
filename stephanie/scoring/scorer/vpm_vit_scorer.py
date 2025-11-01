# stephanie/scoring/scorer/vpm_vit_scorer.py
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import torch
from PIL import Image

from stephanie.data.score_bundle import ScoreBundle
from stephanie.data.score_result import ScoreResult
from stephanie.scoring.model.vpm_vit import VPMViT
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorer.base_scorer import BaseScorer

class VPMViTScorer(BaseScorer):
    """
    Inference wrapper for VPM-ViT.
    Expects either scorable.get_image_array() or scorable.meta['vpm_path'].
    """
    def __init__(self, cfg: Dict[str, Any], memory, container, logger=None):
        super().__init__(cfg, memory, container, logger)
        self.weights = cfg.get("weights_path", "models/vpm_vit_final.pth")
        ckpt = torch.load(self.weights, map_location="cpu")

        params = ckpt.get("config", {"in_ch": 1})
        self.dims = ckpt.get("dims", ["reasoning","knowledge","clarity","faithfulness","coverage"])
        self.risk_labels = ckpt.get("risk_labels", ["OK","WATCH","RISK"])

        self.model = VPMViT(**params)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _load_img(self, scorable: Scorable, in_ch: int) -> np.ndarray:
        arr = getattr(scorable, "get_image_array", lambda: None)()
        if arr is None:
            p = (scorable.meta or {}).get("vpm_path")
            if not p:
                raise ValueError("VPMViTScorer requires image array or meta['vpm_path']")
            img = Image.open(p)
            if in_ch == 1:
                img = img.convert("L")
                arr = np.array(img, dtype=np.float32)[None, ...]
            else:
                img = img.convert("RGB")
                arr = np.transpose(np.array(img, dtype=np.float32), (2,0,1))
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.shape[0] == 1 and self.model.patch_embed.proj.in_channels == 3:
            arr = np.repeat(arr, 3, axis=0)
        return arr.astype(np.float32)

    def _score_core(self, context: dict, scorable: Scorable, dimensions: List[str]) -> ScoreBundle:
        with torch.no_grad():
            x = torch.from_numpy(self._load_img(scorable, self.model.patch_embed.proj.in_channels)).unsqueeze(0)
            x = x.to(self.device)
            out = self.model(x, mask=None)
            reg = out.get("reg")
            cls = out.get("cls")

        results: Dict[str, ScoreResult] = {}
        if reg is not None:
            vec = reg.squeeze(0).cpu().numpy().tolist()
            for i, d in enumerate(self.dims):
                if d in dimensions:
                    results[d] = ScoreResult(
                        dimension=d, score=float(np.clip(vec[i], 0.0, 1.0)),
                        rationale=f"VPM-ViT regression for {d}.",
                        source="vpm_vit"
                    )

        if cls is not None and ("risk" in dimensions or "risk_label" in dimensions):
            pred = int(cls.argmax(dim=-1).item())
            prob = torch.softmax(cls, dim=-1)[0, pred].item()
            results["risk"] = ScoreResult(
                dimension="risk", score=float(prob),
                rationale=f"Risk class={self.risk_labels[pred]} ({prob:.2f})",
                source="vpm_vit",
                attributes={"class_index": pred, "label": self.risk_labels[pred]},
            )

        # Optional overall
        if "vpm_overall" in dimensions:
            vals = [results[d].score for d in self.dims if d in results]
            overall = float(np.mean(vals)) if vals else 0.5
            results["vpm_overall"] = ScoreResult(
                dimension="vpm_overall", score=overall,
                rationale="Mean of regression targets.", source="vpm_vit"
            )

        return ScoreBundle(results=results)
