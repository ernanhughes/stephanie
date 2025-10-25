# stephanie/scoring/scorer/vpm_zero_scorer.py
from __future__ import annotations
import math
import numpy as np
from typing import Any, Dict, List

from stephanie.scoring.scorer.base_scorer import BaseScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.vpm_scorable import VPMScorable
from stephanie.data.score_result import ScoreResult
from stephanie.data.score_bundle import ScoreBundle

class ZeroScorer(BaseScorer):
    """
    VPM scorer that:
      - Respects metric order + per-metric weights (provided via VPMScorable)
      - Delegates image/timeline rendering to ZeroModelService (container)
      - Computes cognitive dimensions from the weighted vector and memory
    Config (example):
      dimensions: ["clarity","coherence","alignment","confidence","novelty","contradiction","complexity","vpm_overall"]
      dimension_weights:
        clarity: 1.25
        coherence: 1.15
        alignment: 1.10
        confidence: 1.00
      order_decay: 0.92          # position i weight = order_decay**i (if scorable doesn't supply)
      alignment_metrics: ["scm.alignment01","alignment","policy_align"]
      contradiction_pairs:
        - ["scm.consistency01","scm.uncertainty01"]
    """

    def __init__(self, cfg: Dict[str, Any], memory, container, logger=None):
        super().__init__(cfg, memory, container, logger)
        self.dimensions = cfg.get("dimensions", [
            "clarity","coherence","alignment","confidence","novelty","contradiction","complexity","vpm_overall"
        ])
        self.dim_w = {k: float(v) for k, v in (cfg.get("dimension_weights") or {}).items()}
        self.order_decay = float(cfg.get("order_decay", 1.0))
        self.align_keys = list(cfg.get("alignment_metrics", []) or [])
        self.contra_pairs = list(cfg.get("contradiction_pairs", []) or [])
        self.novelty_window = int(cfg.get("novelty_window", 20))
        self.max_ref_rows = int(cfg.get("max_ref_rows", 100))

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _safe01(x: float) -> float:
        return float(min(1.0, max(0.0, x)))

    @staticmethod
    def _entropy01(vec: np.ndarray) -> float:
        v = vec.astype(np.float64)
        v = np.clip(v, 1e-9, None)
        p = v / np.sum(v)
        H = -np.sum(p * np.log(p))
        Hmax = math.log(len(v) + 1e-9)
        return float(H / (Hmax + 1e-9))

    @staticmethod
    def _top_mass01(vec: np.ndarray, frac: float = 0.25) -> float:
        k = max(1, int(round(len(vec) * float(frac))))
        v = np.sort(vec)[::-1]
        top = float(np.sum(v[:k]))
        tot = float(np.sum(v) + 1e-9)
        return min(1.0, max(0.0, top / tot))

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _alignment_score(self, names: List[str], vec: np.ndarray) -> float:
        if not self.align_keys:
            return float(np.mean(vec))
        idx = [i for i, n in enumerate(names) if any(key.lower() in n.lower() for key in self.align_keys)]
        if not idx: return float(np.mean(vec))
        return float(np.mean(vec[idx]))

    def _contradiction_score(self, names: List[str], vec: np.ndarray) -> float:
        # If pairs provided: high when paired dimensions diverge strongly
        if not self.contra_pairs:
            # fallback: contradiction ~ 1 - coherence
            return 0.0
        scores = []
        low = [n.lower() for n in names]
        for a_key, b_key in self.contra_pairs:
            try:
                ia = next(i for i, n in enumerate(low) if a_key.lower() in n)
                ib = next(i for i, n in enumerate(low) if b_key.lower() in n)
                scores.append(abs(vec[ia] - vec[ib]))
            except StopIteration:
                continue
        if not scores: return 0.0
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _novelty_score(self, sc: VPMScorable, weighted_vec: np.ndarray) -> float:
        # Compare against recent history from memory.vpm_store
        if not hasattr(self.memory, "vpms"):
            return 0.5
        names, mat = self.memory.vpms.matrix_for_run(sc.run_id, n=min(self.max_ref_rows, self.novelty_window))
        if not mat:
            return 0.5
        # align to shared width
        mat = np.asarray(mat, dtype=np.float32)
        d = min(mat.shape[1], weighted_vec.size)
        R = mat[:, :d]
        v = weighted_vec[:d]
        sims = [self._cosine(v, R[i]) for i in range(R.shape[0])]
        # novelty ↑ when similarity ↓
        return float(np.clip(1.0 - np.mean(sims), 0.0, 1.0))

    # --- main ------------------------------------------------------------
    def score(self, context: dict, scorable: Scorable, dimensions: List[str]) -> ScoreBundle:
        if not isinstance(scorable, VPMScorable):
            # try to adapt common dict form
            m = getattr(scorable, "metadata", {}) or {}
            metric_names = m.get("metric_names") or []
            values = m.get("values") or []
            run_id = m.get("run_id") or "unknown"
            step = int(m.get("step") or 0)
            order_w = None
            metric_w = m.get("metric_weights")
            scorable = VPMScorable(
                id=str(getattr(scorable, "id", "vpm")),
                run_id=run_id,
                step=step,
                metric_names=metric_names,
                values=values,
                order_weights=order_w,
                metric_weights=metric_w,
                metadata=m,
            )

        # If the scorable didn't provide order weights, synthesize from order_decay
        if (not scorable.order_weights.size) and self.order_decay not in (None, 1.0):
            ow = np.asarray([self.order_decay ** i for i in range(len(scorable.values))], dtype=np.float32)
            scorable.order_weights = ow

        # weighted & normalized vector
        wvec = scorable.get_metric_vector()
        names = scorable.get_names()

        # Dimension primitives
        entropy = self._entropy01(wvec)                 # 0..1 (high = complex/diffuse)
        clarity = self._top_mass01(wvec, 0.25)          # 0..1 (high = concentrated)
        confidence = float(np.mean(wvec))               # 0..1 (average intensity)
        coherence = float(1.0 - entropy)                # 0..1 (low entropy == coherent)
        alignment = self._alignment_score(names, wvec)  # 0..1
        novelty = self._novelty_score(scorable, wvec)   # 0..1
        contradiction = self._contradiction_score(names, wvec)  # 0..1
        complexity = float(entropy)                     # 0..1

        # Aggregate
        dims_all = {
            "clarity": clarity,
            "coherence": coherence,
            "alignment": alignment,
            "confidence": confidence,
            "novelty": novelty,
            "contradiction": contradiction,
            "complexity": complexity,
        }
        # overall as a weighted sum (only of requested dims if present)
        overall = 0.0
        wsum = 0.0
        for d, val in dims_all.items():
            if d in dimensions:
                w = float(self.dim_w.get(d, 1.0))
                overall += w * val
                wsum += w
        vpm_overall = overall / (wsum + 1e-9)

        # results
        results: Dict[str, ScoreResult] = {}
        for d in dimensions:
            if d == "vpm_overall":
                results[d] = ScoreResult(dimension=d, score=self._safe01(vpm_overall),
                                         rationale="Weighted aggregate over VPM dimensions (order-aware).",
                                         source="vpm_zero")
            elif d in dims_all:
                results[d] = ScoreResult(dimension=d, score=self._safe01(dims_all[d]),
                                         rationale=f"{d} computed from weighted VPM vector.",
                                         source="vpm_zero")
            else:
                results[d] = ScoreResult(dimension=d, score=0.5,
                                         rationale="Unknown dimension; neutral score.",
                                         source="vpm_zero")

        # Optional: kick ZeroModel to render/run side effects (no blocking needed)
        try:
            zm = self.container.get("zeromodel")
            if zm:  # best-effort — e.g., render summary once per run
                # don’t render per row here; keep fast. You can hook timeline_finalize elsewhere.
                _ = zm.health_check()
        except Exception:
            pass

        return ScoreBundle(results=results)

# factory for registry
def create_zero_scorer(cfg, memory, container, logger):
    return ZeroScorer(cfg, memory, container, logger)
