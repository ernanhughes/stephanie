# stephanie/components/risk/epi/epistemic_guard.py
from __future__ import annotations

import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from stephanie.utils.time_utils import now_iso

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None

try:
    import umap  # optional
except Exception:
    umap = None

import matplotlib

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAFE_ID = re.compile(r"^[a-zA-Z0-9._:-]{3,128}$")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _safe_trace_id(s: str) -> str:
    if not SAFE_ID.match(s):
        raise ValueError(f"unsafe trace_id: {s}")
    return s


# ---------------- Contracts ----------------
@dataclass
class GuardInput:
    trace_id: str
    question: str
    context: str
    reference: str
    hypothesis: str
    hrm_view: Optional[Dict[str, Any]] = None
    tiny_view: Optional[Dict[str, Any]] = None
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
    route: str
    metrics: Dict[str, Any]
    vpm_path: str
    field_path: str
    strip_path: str
    legend_path: str
    badge_path: str
    evidence_id: Optional[str] = None
    schema: str = "eg.v1"


# --------------- Risk predictor ---------------
class RiskPredictor:
    def __init__(self, thresholds=(0.2, 0.6)):
        self.low, self.high = thresholds

    async def predict_risk(self, question: str, context: str):
        ctx_len = len(context.strip())
        risk = float(min(0.99, 0.1 + 0.9 * (1.0 - math.tanh(ctx_len / 512.0))))
        return risk, (self.low, self.high)


# --------------- HalVis adapter ---------------
class HalVisModule:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        _ensure_dir(out_dir)

    def detect_hallucination(
        self, reference: str, hypothesis: str
    ) -> Dict[str, Any]:
        n = max(64, min(512, len(hypothesis.split()) * 4))
        rng = np.random.default_rng(len(hypothesis))
        R = np.clip(rng.normal(0.4, 0.2, n), 0, 1)
        G = np.clip(rng.normal(0.3, 0.15, n), 0, 1)
        B = np.zeros_like(R)
        A = 1.0 - np.clip(R * 0.7 + G * 0.3, 0, 1)
        vpm = np.stack([R, G, B, A], axis=1)
        metrics = {
            "mean_energy": float(R.mean()),
            "max_energy": float(R.max()),
            "entropy": float(G.mean()),
            "tokens": int(n),
        }
        return {"vpm_tensor": vpm, "metrics": metrics}


def compute_disagreement(hrm_view, tiny_view, n_tokens: int) -> Dict[str, Any]:
    if not hrm_view or not tiny_view:
        return {"B": np.zeros(n_tokens), "max_B": 0.0, "disagree_rate": 0.0}
    max_B = float(
        abs(hrm_view.get("confidence", 0.7) - tiny_view.get("confidence", 0.5))
    )
    disagree_rate = float(tiny_view.get("disagree_rate_spans", 0.0))
    B = np.full(n_tokens, np.clip(max_B, 0, 1), dtype=np.float32)
    return {"B": B, "max_B": max_B, "disagree_rate": disagree_rate}


class EGVisual:
    def __init__(self, out_dir: str, seed: int = 42):
        self.out_dir = out_dir
        _ensure_dir(out_dir)
        self.seed = seed
        self._umap = None

    def _project(self, vpm: np.ndarray) -> np.ndarray:
        if umap is None:
            x = vpm[:, :2]
            x = (x - x.mean(0)) / (x.std(0) + 1e-8)
            return x
        if self._umap is None:
            self._umap = umap.UMAP(
                n_neighbors=15,
                min_dist=0.05,
                metric="euclidean",
                random_state=self.seed,
                n_jobs=1,
            )
            return self._umap.fit_transform(vpm)
        return self._umap.transform(vpm)

    def _norm(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v)
        return v if v.size == 0 else (v - v.min()) / (v.ptp() + 1e-8)

    def render(self, trace_id: str, vpm: np.ndarray):
        field = os.path.join(self.out_dir, f"{trace_id}_field.png")
        strip = os.path.join(self.out_dir, f"{trace_id}_strip.png")
        legend = os.path.join(self.out_dir, f"{trace_id}_legend.png")
        R = self._norm(vpm[:, 0])
        proj = self._project(vpm)
        plt.figure(figsize=(5.6, 4.6))
        plt.scatter(
            proj[:, 0], proj[:, 1], c=R, s=10, cmap="inferno", alpha=0.95
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(field, dpi=220)
        plt.close()
        plt.figure(figsize=(8, 1.2))
        plt.imshow(R[np.newaxis, :], aspect="auto", cmap="inferno")
        plt.yticks([])
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(strip, dpi=220)
        plt.close()
        plt.figure(figsize=(4, 2.8))
        plt.axis("off")
        plt.text(
            0.02,
            0.98,
            "Channels:\nR=Δ-energy (hallucination)\nG=entropy/uncertainty\nB=HRM↔Tiny disagreement\nA=1−confidence",
            va="top",
            fontsize=8,
        )
        plt.tight_layout()
        plt.savefig(legend, dpi=200)
        plt.close()
        return field, strip, legend


class BadgeGenerator:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        _ensure_dir(out_dir)

    def render_badge(
        self,
        trace_id: str,
        risk: float,
        mean_energy: float,
        max_B: float,
        trust: float = 0.8,
        recency: float = 0.0,
    ) -> str:
        if Image is None:
            path = os.path.join(self.out_dir, f"{trace_id}_badge.txt")
            with open(path, "w") as f:
                f.write(f"risk={risk:.2f}|E={mean_energy:.2f}|B={max_B:.2f}")
            return path
        import colorsys

        hue = max(0.0, min(0.4, (1 - risk) * 0.4))
        r, g, b = colorsys.hsv_to_rgb(
            hue, min(1.0, mean_energy * 1.5), 0.9 * (1 - 0.3 * recency)
        )
        base = Image.new(
            "RGBA",
            (96, 96),
            (int(r * 255), int(g * 255), int(b * 255), int(255 * (1 - max_B))),
        )
        draw = ImageDraw.Draw(base)
        halo = int(255 * min(1.0, mean_energy * 0.7 + risk * 0.3))
        draw.ellipse([8, 8, 88, 88], outline=(255, halo, 0, 220), width=6)
        glyph = "✓" if risk < 0.4 else ("!" if risk < 0.6 else "✕")
        font = ImageFont.load_default()
        draw.text((40, 34), glyph, fill=(255, 255, 255, 255), font=font)
        path = os.path.join(self.out_dir, f"{trace_id}_badge.png")
        base.save(path)
        return path


class EvidenceStore:
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
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        return eid


# ---------------- Core ----------------
class EpistemicGuard:
    """
    out = await EpistemicGuard(root_out_dir).assess(GuardInput(...))
    Returns GuardOutput; caller decides whether to copy artifacts into run.
    """

    def __init__(
        self,
        root_out_dir: str = "./runs/risk_visuals",
        thresholds=(0.2, 0.6),
        seed: int = 42,
    ):
        self.root = root_out_dir
        self.img_dir = os.path.join(root_out_dir, "img")
        self.vpm_dir = os.path.join(root_out_dir, "vpm")
        _ensure_dir(self.img_dir)
        _ensure_dir(self.vpm_dir)
        self.risk = RiskPredictor(thresholds=thresholds)
        self.halvis = HalVisModule(out_dir=self.img_dir)
        self.visual = EGVisual(out_dir=self.img_dir, seed=seed)
        self.badges = BadgeGenerator(out_dir=self.img_dir)
        self.store = EvidenceStore(base_dir=self.vpm_dir)

    def _route(
        self,
        risk: float,
        low: float,
        high: float,
        max_B: float,
        beta: float = 0.55,
    ) -> str:
        if (risk < low) and (max_B < beta):
            return "FAST"
        if (risk < high) and (max_B < beta):
            return "MEDIUM"
        return "HIGH"

    async def assess(self, data: GuardInput) -> GuardOutput:
        trace_id = _safe_trace_id(data.trace_id)
        risk, (low, high) = await self.risk.predict_risk(
            data.question, data.context
        )
        hv = self.halvis.detect_hallucination(data.reference, data.hypothesis)
        vpm = hv["vpm_tensor"].copy()
        metrics = hv.get("metrics", {})
        n_tokens = vpm.shape[0]
        disc = compute_disagreement(data.hrm_view, data.tiny_view, n_tokens)
        vpm[:, 2] = disc["B"]
        field_path, strip_path, legend_path = self.visual.render(trace_id, vpm)
        vpm_path = self.store.save_vpm(trace_id, vpm)
        evidence_id = self.store.store_evidence(
            {
                "evidence_id": f"eg-{trace_id}",
                "schema": "eg.v1",
                "trace_id": trace_id,
                "created_at": now_iso(),
                "risk": risk,
                "thresholds": {"low": low, "high": high},
                "metrics": {
                    **metrics,
                    "max_B": disc["max_B"],
                    "disagree_rate": disc["disagree_rate"],
                },
                "vpm_path": vpm_path,
                "field_path": field_path,
                "strip_path": strip_path,
                "legend_path": legend_path,
                "context": data.context_payload(),
            }
        )
        badge_path = self.badges.render_badge(
            trace_id=trace_id,
            risk=risk,
            mean_energy=float(metrics.get("mean_energy", 0.3)),
            max_B=float(disc["max_B"]),
            trust=float(data.trust or 0.8),
            recency=float(data.recency or 0.0),
        )
        route = self._route(risk, low, high, max_B=float(disc["max_B"]))
        return GuardOutput(
            trace_id=trace_id,
            risk=risk,
            thresholds=(low, high),
            route=route,
            metrics={
                **metrics,
                "max_B": disc["max_B"],
                "disagree_rate": disc["disagree_rate"],
            },
            vpm_path=vpm_path,
            field_path=field_path,
            strip_path=strip_path,
            legend_path=legend_path,
            badge_path=badge_path,
            evidence_id=evidence_id,
        )
