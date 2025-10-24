# stephanie/components/gap/processors/epistemic_guard.py
"""
EpistemicGuard — full, self-contained version
---------------------------------------------
Unifies: Risk prediction, HalVis VPM tensors, HRM↔Tiny disagreement overlays,
badge rendering, evidence persistence, and routing advice.


Dependencies (optional-but-detected):
  - numpy, pillow, matplotlib, umap-learn (for stable projections), jsonschema (optional)
Everything else has safe fallbacks.

Author: you
"""

from __future__ import annotations
import os
import json
import math
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# --------------------------- Safe imports / light deps --------------------------- #
try:
    import numpy as np
except Exception:
    raise RuntimeError("EpistemicGuard requires numpy")

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None  # badge fallbacks will raise if used

try:
    import umap  # stable 2D projection cache
except Exception:
    umap = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

SAFE_ID = re.compile(r"^[a-zA-Z0-9._:-]{3,128}$")

# --------------------------- Subjects (NATS / streams) -------------------------- #
class Subjects:
    VPM_OUT      = "vpm.hallucination.{trace_id}"
    ALERTS_OUT   = "alerts.hallucination"
    RISK_OUT     = "risk.score.{trace_id}"
    BADGE_OUT    = "badge.update.{trace_id}"
    INGRESS_QA   = "qa.scored"
    INGRESS_TRACE= "trace.completed"
    SCHEMA_VER   = "eg.v1"

# --------------------------- Contracts ----------------------------------------- #
@dataclass
class GuardInput:
    trace_id: str
    question: str
    context: str
    reference: str
    hypothesis: str
    # Optional model “views” for disagreement
    hrm_view: Optional[Dict[str, Any]] = None
    tiny_view: Optional[Dict[str, Any]] = None
    # Optional UI/context metrics
    trust: Optional[float] = None
    recency: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None

    def context_payload(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "reference": self.reference,
            **(self.meta or {}),
        }

@dataclass
class GuardOutput:
    trace_id: str
    risk: float
    thresholds: Tuple[float, float]
    route: str  # FAST | MEDIUM | HIGH
    metrics: Dict[str, Any]
    vpm_path: str
    field_path: str
    strip_path: str
    legend_path: str
    badge_path: str
    evidence_id: Optional[str] = None
    schema: str = Subjects.SCHEMA_VER

# --------------------------- Helpers: safe I/O --------------------------------- #
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _safe_trace_id(s: str) -> str:
    if not SAFE_ID.match(s):
        raise ValueError(f"unsafe trace_id: {s}")
    return s

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# --------------------------- Risk predictor (pluggable) ------------------------ #
class RiskPredictor:
    """Interface. Replace with your  I I/XGB bundle."""
    def __init__(self, thresholds=(0.2, 0.6)):
        self.low, self.high = thresholds

    async def predict_risk(self, question: str, context: str) -> Tuple[float, Tuple[float, float]]:
        # TODO: plug in your real featurizer + calibrated model
        # Fallback: heuristic risk from context length
        ctx_len = len(context.strip())
        risk = float(min(0.99, 0.1 + 0.9 * (1.0 - math.tanh(ctx_len / 512.0))))
        return risk, (self.low, self.high)

# --------------------------- HalVis (pluggable wrapper) ------------------------ #
class HalVisModule:
    """
    Adapter around your real HalVis. Must return:
      dict(
        vpm_tensor: np.ndarray (N,4) [R,G,B,A],
        metrics: {"mean_energy": float, "max_energy": float, "entropy": float, ...}
      )
    Fallback: constructs a synthetic VPM from text lengths.
    """
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        _ensure_dir(out_dir)

    def detect_hallucination(self, reference: str, hypothesis: str) -> Dict[str, Any]:
        # TODO: wire to your production HalVis
        n = max(64, min(512, len(hypothesis.split()) * 4))
        rng = np.random.default_rng(len(hypothesis))
        R = np.clip(rng.normal(0.4, 0.2, n), 0, 1)  # hallucination energy proxy
        G = np.clip(rng.normal(0.3, 0.15, n), 0, 1)  # entropy proxy
        B = np.zeros_like(R)                         # filled later (disagreement)
        A = 1.0 - np.clip(R * 0.7 + G * 0.3, 0, 1)  # alpha = 1 - confidence
        vpm = np.stack([R, G, B, A], axis=1)
        metrics = {
            "mean_energy": float(R.mean()),
            "max_energy": float(R.max()),
            "entropy": float(G.mean()),
            "tokens": int(n),
        }
        return {"vpm_tensor": vpm, "metrics": metrics}

# --------------------------- Disagreement (HRM↔Tiny) --------------------------- #
def compute_disagreement(hrm_view: Optional[Dict[str, Any]],
                         tiny_view: Optional[Dict[str, Any]],
                         n_tokens: int) -> Dict[str, Any]:
    """
    Return a per-token B-channel and summary disagreement metrics.
    Plug in your true alignment / logit margin deltas here.
    Fallback: simple scalar expanded over tokens if logit traces absent.
    """
    if not hrm_view or not tiny_view:
        return {"B": np.zeros(n_tokens), "max_B": 0.0, "disagree_rate": 0.0}

    # Example: use provided 'delta_conf' or 'agree_margin' if present
    max_B = float(abs(hrm_view.get("confidence", 0.7) - tiny_view.get("confidence", 0.5)))
    disagree_rate = float(tiny_view.get("disagree_rate_spans", 0.0))
    B = np.full(n_tokens, np.clip(max_B, 0, 1), dtype=np.float32)
    return {"B": B, "max_B": max_B, "disagree_rate": disagree_rate}

# --------------------------- Visuals (stable projection & renders) ------------- #
class EGVisual:
    def __init__(self, out_dir: str, seed: int = 42):
        self.out_dir = out_dir
        _ensure_dir(out_dir)
        self.seed = seed
        self._umap = None

    def _project(self, vpm: np.ndarray) -> np.ndarray:
        if umap is None:
            # fallback: PCA-ish 2D using first two channels
            x = vpm[:, :2]
            x = (x - x.mean(0)) / (x.std(0) + 1e-8)
            return x
        if self._umap is None:
            self._umap = umap.UMAP(n_neighbors=15, min_dist=0.05, metric="euclidean", random_state=self.seed)
            return self._umap.fit_transform(vpm)
        return self._umap.transform(vpm)

    def _norm(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v)
        if v.size == 0: return v
        return (v - v.min()) / (v.ptp() + 1e-8)

    def render(self, trace_id: str, vpm: np.ndarray) -> Tuple[str, str, str]:
        """Return (field.png, strip.png, legend.png) paths."""
        field = os.path.join(self.out_dir, f"{trace_id}_field.png")
        strip = os.path.join(self.out_dir, f"{trace_id}_strip.png")
        legend = os.path.join(self.out_dir, f"{trace_id}_legend.png")

        R = self._norm(vpm[:,0])
        proj = self._project(vpm)

        if plt is None:
            # cannot render; create blank placeholders
            for p in (field, strip, legend):
                if Image:
                    Image.new("RGBA",(8,8),(0,0,0,0)).save(p)
            return field, strip, legend

        # Field
        plt.figure(figsize=(5.6,4.6))
        sc = plt.scatter(proj[:,0], proj[:,1], c=R, s=10, cmap="inferno", alpha=0.95)
        plt.axis("off"); plt.tight_layout(); plt.savefig(field, dpi=220); plt.close()

        # Strip (linear)
        plt.figure(figsize=(8,1.2))
        plt.imshow(R[np.newaxis,:], aspect="auto", cmap="inferno")
        plt.yticks([]); plt.xticks([]); plt.tight_layout(); plt.savefig(strip, dpi=220); plt.close()

        # Legend
        plt.figure(figsize=(4,2.8))
        text = "Channels:\nR=Δ-energy (hallucination)\nG=entropy/uncertainty\nB=HRM↔Tiny disagreement\nA=1−confidence"
        plt.axis("off"); plt.text(0.02,0.98,text,va="top",fontsize=8)
        plt.tight_layout(); plt.savefig(legend, dpi=200); plt.close()

        return field, strip, legend

# --------------------------- Badge generator ----------------------------------- #
class BadgeGenerator:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        _ensure_dir(out_dir)

    def render_badge(self, trace_id: str, risk: float, mean_energy: float,
                     max_B: float, trust: float = 0.8, recency: float = 0.0) -> str:
        if Image is None:
            path = os.path.join(self.out_dir, f"{trace_id}_badge.txt")
            with open(path, "w") as f: f.write(f"risk={risk:.2f}|E={mean_energy:.2f}|B={max_B:.2f}")
            return path

        # Hue from risk (green→red ~ 0.4..0.0)
        hue = max(0.0, min(0.4, (1 - risk) * 0.4))
        # Convert HSV to RGB
        import colorsys
        r,g,b = colorsys.hsv_to_rgb(hue, min(1.0, mean_energy*1.5), trust*(1-0.3*recency))
        base = Image.new("RGBA", (96,96), (int(r*255), int(g*255), int(b*255), int(255*(1-max_B))))
        draw = ImageDraw.Draw(base)
        halo = int(255 * min(1.0, mean_energy*0.7 + risk*0.3))
        draw.ellipse([8,8,88,88], outline=(255,halo,0,220), width=6)
        # glyph
        glyph = "✓" if trust >= 0.8 and risk < 0.4 else ("!" if risk >= 0.6 else "·")
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((40,34), glyph, fill=(255,255,255,255), font=font)
        path = os.path.join(self.out_dir, f"{trace_id}_badge.png")
        base.save(path)
        return path

# --------------------------- Evidence store (stubs) ---------------------------- #
class EvidenceStore:
    """Persist minimal evidence; replace with MemCubeClient in production."""
    def __init__(self, base_dir: str):
        self.base = base_dir
        _ensure_dir(self.base)

    def save_vpm(self, trace_id: str, vpm: np.ndarray) -> str:
        p = os.path.join(self.base, f"{trace_id}_vpm.npz")
        np.savez_compressed(p, vpm=vpm)
        return p

    def store_evidence(self, data: Dict[str, Any]) -> str:
        eid = data.get("evidence_id") or str(uuid.uuid4())
        p = os.path.join(self.base, f"{eid}.json")
        with open(p, "w") as f: json.dump(data, f, indent=2)
        return eid

# --------------------------- Core: EpistemicGuard ------------------------------ #
class EpistemicGuard:
    """
    One-call facade:
      out = await EpistemicGuard(...).assess(GuardInput(...))

    Returns risk, thresholds, route, VPM paths, image paths, and evidence id.
    """
    def __init__(self,
                 out_dir: str = "./static/eg",
                 thresholds: Tuple[float,float] = (0.2, 0.6),
                 seed: int = 42):
        self.out_dir = out_dir
        self.img_dir = os.path.join(out_dir, "img")
        self.vpm_dir = os.path.join(out_dir, "vpm")
        _ensure_dir(self.img_dir); _ensure_dir(self.vpm_dir)

        self.risk = RiskPredictor(thresholds=thresholds)
        self.halvis = HalVisModule(out_dir=self.img_dir)
        self.visual = EGVisual(out_dir=self.img_dir, seed=seed)
        self.badges = BadgeGenerator(out_dir=self.img_dir)
        self.store = EvidenceStore(base_dir=self.vpm_dir)

    # ---------- Routing policy ---------- #
    def _route(self, risk: float, low: float, high: float, max_B: float, beta: float = 0.55) -> str:
        if (risk < low) and (max_B < beta): return "FAST"
        if (risk < high) and (max_B < beta): return "MEDIUM"
        return "HIGH"

    # ---------- Main entry ---------- #
    async def assess(self, data: GuardInput) -> GuardOutput:
        trace_id = _safe_trace_id(data.trace_id)

        # 1) Risk (domain-calibrated model can be plugged here)
        risk, (low, high) = await self.risk.predict_risk(data.question, data.context)

        # 2) HalVis: hallucination VPM + metrics (R,G,A)
        hv = self.halvis.detect_hallucination(data.reference, data.hypothesis)
        vpm = hv["vpm_tensor"].copy()
        metrics = hv.get("metrics", {})
        n_tokens = vpm.shape[0]

        # 3) HRM↔Tiny disagreement (B-channel)
        disc = compute_disagreement(data.hrm_view, data.tiny_view, n_tokens)
        vpm[:,2] = disc["B"]  # set B

        # 4) Render visuals (stable projection)
        field_path, strip_path, legend_path = self.visual.render(trace_id, vpm)

        # 5) Persist VPM + evidence
        vpm_path = self.store.save_vpm(trace_id, vpm)
        evidence_id = self.store.store_evidence({
            "evidence_id": f"eg-{trace_id}",
            "schema": Subjects.SCHEMA_VER,
            "trace_id": trace_id,
            "created_at": _now_iso(),
            "risk": risk,
            "thresholds": {"low": low, "high": high},
            "metrics": {**metrics, "max_B": disc["max_B"], "disagree_rate": disc["disagree_rate"]},
            "vpm_path": vpm_path,
            "field_path": field_path,
            "strip_path": strip_path,
            "legend_path": legend_path,
            "context": data.context_payload()
        })

        # 6) Badge
        badge_path = self.badges.render_badge(
            trace_id=trace_id,
            risk=risk,
            mean_energy=float(metrics.get("mean_energy", 0.3)),
            max_B=float(disc["max_B"]),
            trust=float(data.trust or 0.8),
            recency=float(data.recency or 0.0),
        )

        # 7) Route
        route = self._route(risk, low, high, max_B=float(disc["max_B"]))

        return GuardOutput(
            trace_id=trace_id,
            risk=risk,
            thresholds=(low, high),
            route=route,
            metrics={**metrics, "max_B": disc["max_B"], "disagree_rate": disc["disagree_rate"]},
            vpm_path=vpm_path,
            field_path=field_path,
            strip_path=strip_path,
            legend_path=legend_path,
            badge_path=badge_path,
            evidence_id=evidence_id,
        )
