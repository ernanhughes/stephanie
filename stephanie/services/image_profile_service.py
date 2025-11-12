from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    from PIL import Image, ImageFilter
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False
    Image = None  # type: ignore

# Optional CLIP backends (either works). If neither is present, we fallback.
_CLIP_BACKEND = None
_CLIP = None
_PREPROC = None

def _try_load_clip():
    global _CLIP_BACKEND, _CLIP, _PREPROC
    if _CLIP is not None:
        return
    try:
        # try openai clip
        import clip  # type: ignore
        _CLIP_BACKEND = "openai-clip"
        _CLIP, _PREPROC = clip.load("ViT-B/32", device="cpu", jit=False)
    except Exception:
        try:
            # try open_clip
            import open_clip  # type: ignore
            _CLIP_BACKEND = "open-clip"
            _CLIP, _, _PREPROC = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        except Exception:
            _CLIP_BACKEND = None

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) or 1e-8
    nb = np.linalg.norm(b) or 1e-8
    return float((a @ b) / (na * nb))

class ImageProfileService:
    """
    Learns a visual 'voice' from champion images:
      - CLIP centroid (if available; fallback to color histogram centroid)
      - Palette signature (top LAB bins)
      - Simple composition priors

    Provides:
      - score_image(image_path, prompt) -> VS_img (style) + relevance (CLIP sim if available)
      - update_from_champions(list_of_paths)
    """
    def __init__(self, memory, logger):
        self.memory = memory
        self.logger = logger
        self._profile = self.memory.meta.get("image_profile") or {}
        _try_load_clip()

    def _save(self):
        self.memory.meta["image_profile"] = self._profile

    # ---------- building ----------
    def update_from_champions(self, image_paths: List[str]) -> Dict[str, Any]:
        if not image_paths:
            return self._profile
        try:
            feats = [self._feat(p) for p in image_paths if self._feat(p) is not None]
            if not feats:
                return self._profile
            # centroid of embeddings
            embs = [f["emb"] for f in feats if f.get("emb") is not None]
            if embs:
                dim = len(embs[0])
                centroid = [sum(e[i] for e in embs) / len(embs) for i in range(dim)]
            else:
                centroid = []

            # palette mean (LAB bins)
            palettes = [f["palette"] for f in feats if f.get("palette") is not None]
            palette = list(np.mean(np.stack(palettes), axis=0)) if palettes else []

            # composition averages
            comp_vals = [f["composition"] for f in feats if f.get("composition") is not None]
            comp_mu = float(np.mean(comp_vals)) if comp_vals else 0.5

            self._profile = {
                "clip_centroid": centroid,       # [] if CLIP unavailable
                "palette_mu": palette,           # LAB histogram means
                "composition_mu": comp_mu,       # 0..1
                "version": int(self._profile.get("version", 0)) + 1
            }
            self._save()
            self.logger.log("ImageProfileUpdated", {"version": self._profile["version"]})
            return self._profile
        except Exception as e:
            self.logger.log("ImageProfileUpdateError", {"error": str(e), "traceback": traceback.format_exc()})
            return self._profile

    # ---------- scoring ----------
    def score_image(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, float]:
        """
        Returns:
          - style_clip: sim(image, style_centroid) in [0,1] (0.5 neutral if not available)
          - palette_match: [0,1] higher = closer to palette mean
          - composition_match: [0,1]
          - relevance: CLIP(text,img) if prompt given, else 0.5
        """
        try:
            f = self._feat(image_path)
            if f is None:
                return {"style_clip": 0.5, "palette_match": 0.5, "composition_match": 0.5, "relevance": 0.5}

            style_clip = 0.5
            centroid = self._profile.get("clip_centroid") or []
            if centroid and f.get("emb") is not None:
                sim = _cos(np.array(centroid), np.array(f["emb"]))
                style_clip = (sim + 1.0) / 2.0

            palette_match = self._palette_match(np.array(self._profile.get("palette_mu", [])), np.array(f.get("palette", [])))

            composition_mu = float(self._profile.get("composition_mu", 0.5))
            # closer is better (L2 distance → similarity)
            composition_match = max(0.0, min(1.0, 1.0 - abs(float(f["composition"]) - composition_mu)))

            relevance = 0.5
            if prompt and f.get("emb") is not None and _CLIP_BACKEND:
                text_emb = self._text_emb(prompt)
                if text_emb is not None:
                    relevance = ( _cos(np.array(text_emb), np.array(f["emb"])) + 1.0 ) / 2.0

            return {
                "style_clip": float(style_clip),
                "palette_match": float(palette_match),
                "composition_match": float(composition_match),
                "relevance": float(relevance)
            }
        except Exception as e:
            self.logger.log("ImageProfileScoreError", {"error": str(e), "traceback": traceback.format_exc()})
            return {"style_clip": 0.5, "palette_match": 0.5, "composition_match": 0.5, "relevance": 0.5}

    # ---------- feature extraction ----------
    def _feat(self, image_path: str) -> Optional[Dict[str, Any]]:
        if not _HAS_PIL:
            return None
        try:
            img = Image.open(image_path).convert("RGB")
            emb = self._img_emb(img)
            palette = self._palette(img)
            comp = self._composition(img)
            return {"emb": emb, "palette": palette, "composition": comp}
        except Exception:
            return None

    def _palette(self, img: "Image.Image", bins: int = 6) -> np.ndarray:
        # very light palette signature in LAB-like space via linear RGB→approx LAB-ish (use simple transform)
        arr = np.asarray(img.resize((256, 256))).astype(np.float32) / 255.0
        # naive "LAB-ish" (not true LAB, but stable): L ~ gray, A ~ R-G, B ~ B - (R+G)/2
        R, G, B = arr[...,0], arr[...,1], arr[...,2]
        L = 0.299*R + 0.587*G + 0.114*B
        A = R - G
        Bl = B - 0.5*(R+G)
        hL, _ = np.histogram(L, bins=bins, range=(0,1), density=True)
        hA, _ = np.histogram(A, bins=bins, range=(-1,1), density=True)
        hB, _ = np.histogram(Bl, bins=bins, range=(-1,1), density=True)
        return np.concatenate([hL, hA, hB]).astype(np.float32)

    def _composition(self, img: "Image.Image") -> float:
        # reuse rule-of-thirds proxy
        w, h = img.size
        thirds = [(w//3, h//3), (2*w//3, h//3), (w//3, 2*h//3), (2*w//3, 2*h//3)]
        edges = img.filter(ImageFilter.FIND_EDGES).convert("L")
        arr = np.asarray(edges).astype(np.float32) / 255.0
        H, W = arr.shape
        score = 0.0
        for (x, y) in thirds:
            x0, x1 = max(0, x-16), min(W, x+16)
            y0, y1 = max(0, y-16), min(H, y+16)
            patch = arr[y0:y1, x0:x1]
            score += float(patch.mean())
        score /= len(thirds)
        return max(0.0, min(1.0, score))

    def _img_emb(self, img: Image.Image) -> Optional[np.ndarray]:
        if _CLIP_BACKEND is None:
            return None
        try:
            if _CLIP_BACKEND == "openai-clip":
                with torch.no_grad():
                    t = _PREPROC(img).unsqueeze(0)
                    feats = _CLIP.encode_image(t)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    return feats.cpu().numpy().reshape(-1)
            else:
                # open-clip
                with torch.no_grad():
                    t = _PREPROC(img).unsqueeze(0)
                    feats = _CLIP.encode_image(t)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    return feats.cpu().numpy().reshape(-1)
        except Exception:
            return None

    def _text_emb(self, text: str) -> Optional[np.ndarray]:
        if _CLIP_BACKEND is None:
            return None
        try:
            if _CLIP_BACKEND == "openai-clip":
                import clip  # type: ignore
                with torch.no_grad():
                    tok = clip.tokenize([text], truncate=True)
                    feats = _CLIP.encode_text(tok)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    return feats.cpu().numpy().reshape(-1)
            else:
                from open_clip import tokenize  # type: ignore
                with torch.no_grad():
                    tok = tokenize([text])
                    feats = _CLIP.encode_text(tok)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    return feats.cpu().numpy().reshape(-1)
        except Exception:
            return None

    def _palette_match(self, mu: np.ndarray, x: np.ndarray) -> float:
        if mu.size == 0 or x.size == 0 or mu.shape != x.shape:
            return 0.5
        # inverse normalized L2 distance
        d = float(np.linalg.norm(mu - x))
        # heuristic scale: distances ~ [0, ~2.5] on these histograms
        sim = max(0.0, min(1.0, 1.0 - (d / 2.5)))
        return sim
