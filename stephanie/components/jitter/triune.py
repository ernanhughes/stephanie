# stephanie/components/jitter/triune.py
"""
TriuneCognition
===============
Service- and plugin-oriented triune cognitive system.

This version removes heavyweight embedded nets and delegates to Stephanie
services/stores:

- Reptilian Core: boundary threat via text- or embedding-based scorer (container 'scoring')
- Mammalian Layer: pattern confidence + emotional valence via memory VPM store (or scorer if configured)
- Primate Cortex: lightweight abstract reasoning quality from relevant VPMs (service/store)
- True veto cascade: reptilian > mammalian > primate
- Config-driven attention, thresholds, energy extraction

Expected container services (best-effort, optional):
- 'scoring'                 → generic scoring facade (text/embedding scorers)
- 'zeromodel-service-v2'    → VPM rendering/ops (if needed)

Expected memory hooks (best-effort, optional):
- memory.vpms or memory.vpm_manager with:
  - get_negative_vpms() / get_positive_vpms()     (each VPM has .embedding)
  - get_relevant_vpms(query_emb, top_k=5)
"""

from __future__ import annotations

import time
import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("stephanie.jas.triune")


# ----------------------------- Data model ----------------------------- #

@dataclass
class CognitiveState:
    reptilian: float                 # 0..1 (threat; 1 = high threat)
    mammalian: float                 # 0..1 (pattern confidence)
    primate: float                   # 0..1 (reasoning quality)
    integrated: float                # 0..1 (weighted integration)
    cognitive_energy: float          # 0..1 (extracted)
    attention_weights: Dict[str, float]
    layer_veto: str                  # 'none' | 'reptilian' | 'mammalian'
    latency_ms: float
    threat_level: float              # == reptilian
    emotional_valence: float         # -1..1
    reasoning_depth: int


# ----------------------------- Triune ----------------------------- #

class TriuneCognition(nn.Module):
    """
    Thin, configurable triune module that delegates work to services/stores.
    Keeps nn.Module so upstream can call self.triune(x) → CognitiveState.
    """

    def __init__(self, cfg: Dict[str, Any], container, memory, logger=None):
        super().__init__()
        self.cfg = dict(cfg or {})
        self.container = container
        self.memory = memory
        self.logger = logger or log
        self.emb_dim = self.memory.embedding.dim
        # veto thresholds
        vt = self.cfg.get("veto_thresholds", {})
        self.veto_thresholds = {
            "reptilian": float(vt.get("reptilian", self.cfg.get("reptilian_veto_threshold", 0.7))),
            "mammalian": float(vt.get("mammalian", self.cfg.get("mammalian_veto_threshold", 0.6))),
        }

        # attention weights (normalized dict)
        aw_cfg = self.cfg.get("attention", {}).get("weights", {})
        r_w = float(aw_cfg.get("reptilian", self.cfg.get("reptilian_weight", 0.3)))
        m_w = float(aw_cfg.get("mammalian", self.cfg.get("mammalian_weight", 0.3)))
        p_w = float(aw_cfg.get("primate",   self.cfg.get("primate_weight",   0.4)))
        self._attn = self._normalize_weights({"reptilian": r_w, "mammalian": m_w, "primate": p_w})

        # energy scale
        self.energy_gain = float(self.cfg.get("energy_gain_factor", 1.0))

        # primitive state history (for telemetry/reproduction)
        self.state_history: List[CognitiveState] = []
        self.max_history = int(self.cfg.get("max_state_history", 1000))

        # fast accessors
        self.vpm_store = self.memory.vpms
        self.scoring = None
        try:
            self.scoring = self.container.get("scoring")
        except Exception:
            pass

        # scorer aliases (optional; used if scoring service is present)
        r_cfg = self.cfg.get("reptilian", {})
        m_cfg = self.cfg.get("mammalian", {})
        p_cfg = self.cfg.get("primate", {})
        self.reptilian_alias = r_cfg.get("scorer_alias")  # e.g., 'ebt' or 'hf_reptilian'
        self.mammalian_alias = m_cfg.get("scorer_alias")  # e.g., 'svm'
        self.primate_alias   = p_cfg.get("scorer_alias")  # e.g., 'mrq'

        self.primate_top_k = int(p_cfg.get("top_k", self.cfg.get("primate_top_k", 5)))

        # small MLPs keyed to emb_dim (no hard-coded 1024)
        self.reptilian_head = nn.Sequential(
            nn.Linear(self.emb_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.mammalian_head = nn.Sequential(
            nn.Linear(self.emb_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.primate_head = nn.Sequential(
            nn.Linear(self.emb_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )
        log.info(
            "TriuneCognition ready | veto=(R:%.2f M:%.2f) attn=(R=%.2f M=%.2f P=%.2f)",
            self.veto_thresholds["reptilian"],
            self.veto_thresholds["mammalian"],
            self._attn["reptilian"], self._attn["mammalian"], self._attn["primate"]
        )

    # ------------------------- public API ------------------------- #

    def forward(self, input_emb: torch.Tensor) -> CognitiveState:
        """
        Processes an embedding through reptilian → mammalian → primate with veto cascade.
        """
        t0 = time.time()
        try:
            q = self._ensure_1d_cpu(input_emb)

            # 1) Reptilian — boundary threat (0..1; higher = worse)
            threat = float(self._process_reptilian(q))
            if threat >= self.veto_thresholds["reptilian"]:
                state = self._make_veto_state(
                    veto_layer="reptilian",
                    primary_value=threat,
                    latency_ms=(time.time() - t0) * 1000.0,
                )
                self._record_state(state)
                return state

            # 2) Mammalian — pattern confidence (0..1) + emotional valence (-1..1)
            pattern_conf, emo = self._process_mammalian(q)
            if emo <= -self.veto_thresholds["mammalian"]:
                # strong negative valence triggers veto
                state = self._make_veto_state(
                    veto_layer="mammalian",
                    primary_value=max(0.0, min(1.0, 1.0 - (abs(emo)))),
                    latency_ms=(time.time() - t0) * 1000.0,
                    emotional_valence=emo,
                )
                self._record_state(state)
                return state

            # 3) Primate — reasoning quality (0..1) + depth
            primate_q, depth = self._process_primate(q)

            # integrate
            integrated = self._integrate(threat, pattern_conf, primate_q)

            # energy
            cognitive_energy = max(0.0, min(1.0, integrated * self.energy_gain))

            state = CognitiveState(
                reptilian=threat,
                mammalian=pattern_conf,
                primate=primate_q,
                integrated=integrated,
                cognitive_energy=cognitive_energy,
                attention_weights=dict(self._attn),
                layer_veto="none",
                latency_ms=(time.time() - t0) * 1000.0,
                threat_level=threat,
                emotional_valence=float(emo),
                reasoning_depth=int(depth),
            )
            self._record_state(state)
            return state

        except Exception as e:
            log.error(f"TriuneCognition error: {e}", exc_info=True)
            # safe fallback
            state = CognitiveState(
                reptilian=0.5, mammalian=0.5, primate=0.5, integrated=0.5,
                cognitive_energy=0.0, attention_weights=dict(self._attn),
                layer_veto="error", latency_ms=(time.time() - t0) * 1000.0,
                threat_level=0.5, emotional_valence=0.0, reasoning_depth=0
            )
            self._record_state(state)
            return state

    # ---------------------- layer implementations ---------------------- #

    def _process_reptilian(self, emb1d: torch.Tensor) -> float:
        """
        Boundary threat assessment. Prefers a scorer via container ('scoring')
        if an alias is configured; falls back to embedding heuristics.
        Returns: 0..1 (1 = high threat).
        """
        # try a configured scorer (e.g., EBT)
        if self.scoring and self.reptilian_alias:
            try:
                # Minimal scorable: raw embedding or empty text
                scorable = {"embedding": emb1d.numpy().tolist(), "text": ""}
                bundle = self.scoring.score(
                    self.reptilian_alias,
                    context={"mode": "reptilian"},
                    scorable=scorable,
                    dimensions=["threat"],
                )
                # duck-typed extraction
                res = getattr(bundle, "results", None) or {}
                maybe = res.get("threat") or res.get("threat_score")
                if maybe is not None and hasattr(maybe, "score"):
                    v = float(maybe.score)
                    return max(0.0, min(1.0, v))
            except Exception as e:
                log.debug(f"Reptilian scorer path failed: {e}")

        # heuristic: distance from mean (stabilized) → sigmoid to 0..1
        try:
            v = emb1d.detach().cpu().numpy().astype(np.float32)
            z = float(np.linalg.norm(v) / (np.sqrt(v.size) + 1e-8))
            # squash: higher norm → *lower* threat (assume strong identity), invert
            inv = 1.0 - (1.0 / (1.0 + math.exp(-3.0 * (z - 1.0))))
            return max(0.0, min(1.0, inv))
        except Exception:
            return 0.5

    def _process_mammalian(self, emb1d: torch.Tensor) -> Tuple[float, float]:
        """
        Pattern recognition + emotional valence.
        - pattern_confidence: 0..1
        - emotional_valence:  -1..1  (positive – negative)
        Prefers VPM store similarity; falls back to scorer alias or neutral values.
        """
        # similarity-based valence via VPM store
        pos_sim, neg_sim = 0.0, 0.0
        if self.vpm_store:
            try:
                pos = getattr(self.vpm_store, "get_positive_vpms", lambda: [])()
                neg = getattr(self.vpm_store, "get_negative_vpms", lambda: [])()
                pos_sim = self._avg_cosine(emb1d, pos)
                neg_sim = self._avg_cosine(emb1d, neg)
            except Exception as e:
                log.debug(f"Mammalian VPM similarity failed: {e}")

        # valence: positive − negative (clip to [-1,1])
        valence = float(np.clip(pos_sim - neg_sim, -1.0, 1.0))
        # pattern confidence: how *decisive* the similarity difference is
        pattern_conf = float(np.clip(abs(pos_sim - neg_sim), 0.0, 1.0))

        # if a scorer alias is configured, allow it to refine pattern_conf
        if self.scoring and self.mammalian_alias:
            try:
                scorable = {"embedding": emb1d.numpy().tolist(), "text": ""}
                bundle = self.scoring.score(
                    self.mammalian_alias,
                    context={"mode": "mammalian"},
                    scorable=scorable,
                    dimensions=["pattern_confidence"],
                )
                res = getattr(bundle, "results", None) or {}
                maybe = res.get("pattern_confidence")
                if maybe is not None and hasattr(maybe, "score"):
                    pattern_conf = float(np.clip(maybe.score, 0.0, 1.0))
            except Exception as e:
                log.debug(f"Mammalian scorer path failed: {e}")

        return pattern_conf, valence

    def _process_primate(self, emb1d: torch.Tensor) -> Tuple[float, int]:
        """
        Lightweight abstract reasoning quality:
        - fetch top-K relevant VPMs, compute mean cosine with them (mapped to 0..1)
        - depth = K actually used
        """
        K = max(1, int(self.primate_top_k))
        used = 0
        mean_cos = 0.0

        if self.vpm_store and hasattr(self.vpm_store, "get_relevant_vpms"):
            try:
                vpms = self.vpm_store.get_relevant_vpms(emb1d, top_k=K) or []
                if vpms:
                    cos = [self._cosine(emb1d, getattr(v, "embedding", None)) for v in vpms]
                    cos = [c for c in cos if c is not None]
                    if cos:
                        mean_cos = float(np.mean(cos))
                        used = len(cos)
            except Exception as e:
                log.debug(f"Primate relevant_vpms failed: {e}")

        # map cosine [-1,1] → [0,1]
        quality = float((mean_cos + 1.0) * 0.5)
        return quality, used

    # ---------------------- helpers & utilities ---------------------- #

    def _integrate(self, reptilian: float, mammalian: float, primate: float) -> float:
        w = self._attn
        x = reptilian * w["reptilian"] + mammalian * w["mammalian"] + primate * w["primate"]
        return float(np.clip(x, 0.0, 1.0))

    def _make_veto_state(
        self,
        *,
        veto_layer: str,
        primary_value: float,
        latency_ms: float,
        emotional_valence: float = 0.0,
    ) -> CognitiveState:
        rept = primary_value if veto_layer == "reptilian" else 0.0
        mamm = primary_value if veto_layer == "mammalian" else 0.0
        return CognitiveState(
            reptilian=float(np.clip(rept, 0.0, 1.0)),
            mammalian=float(np.clip(mamm, 0.0, 1.0)),
            primate=0.0,
            integrated=float(np.clip(primary_value, 0.0, 1.0)),
            cognitive_energy=0.0,  # no gain during hard veto
            attention_weights=dict(self._attn),
            layer_veto=veto_layer,
            latency_ms=float(latency_ms),
            threat_level=float(np.clip(rept, 0.0, 1.0)) if veto_layer == "reptilian" else 0.5,
            emotional_valence=float(np.clip(emotional_valence, -1.0, 1.0)) if veto_layer == "mammalian" else 0.0,
            reasoning_depth=0,
        )

    def _record_state(self, state: CognitiveState):
        self.state_history.append(state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)

    def get_recent_states(self, n: int = 10) -> List[CognitiveState]:
        return self.state_history[-n:] if self.state_history else []

    def update_attention(self, reward_signal: float):
        """
        Simple reinforcement: nudge all weights uniformly by reward and renormalize.
        (Keep it bounded/stable; this is a placeholder policy.)
        """
        r = float(self._attn["reptilian"])
        m = float(self._attn["mammalian"])
        p = float(self._attn["primate"])
        delta = float(reward_signal) * 0.01
        self._attn = self._normalize_weights({"reptilian": r + delta, "mammalian": m + delta, "primate": p + delta})

    def get_health_metrics(self) -> Dict[str, float]:
        if not self.state_history:
            return {
                "stability": 0.5,
                "efficiency": 0.5,
                "balance": 0.5,
                "veto_frequency": {"reptilian": 0.0, "mammalian": 0.0},
            }

        recent = self.state_history[-50:]
        integrated = np.array([s.integrated for s in recent], dtype=np.float32)
        stability = float(1.0 / (1.0 + float(np.var(integrated))))

        energy = np.array([s.cognitive_energy for s in recent], dtype=np.float32)
        efficiency = float(np.clip(float(np.mean(energy)), 0.0, 1.0)) if energy.size else 0.5

        # balance across average attention weights (std low → well balanced)
        attn_keys = list(recent[0].attention_weights.keys())
        attn_mat = np.array([[s.attention_weights[k] for k in attn_keys] for s in recent], dtype=np.float32)
        avg_attn = attn_mat.mean(axis=0)
        balance = float(np.clip(1.0 - float(np.std(avg_attn)) * 3.0, 0.0, 1.0))

        veto_r = sum(1 for s in recent if s.layer_veto == "reptilian") / len(recent)
        veto_m = sum(1 for s in recent if s.layer_veto == "mammalian") / len(recent)

        return {
            "stability": stability,
            "efficiency": efficiency,
            "balance": balance,
            "veto_frequency": {"reptilian": float(veto_r), "mammalian": float(veto_m)},
        }

    # ---------------------- low-level utils ---------------------- #

    @staticmethod
    def _ensure_1d_cpu(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim > 1:
            x = x.reshape(-1)
        return x.detach().float().cpu()

    @staticmethod
    def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
        s = sum(max(1e-8, float(v)) for v in w.values())
        return {k: float(max(1e-8, float(v)) / s) for k, v in w.items()}

    @staticmethod
    def _cosine(a: torch.Tensor, b: Optional[torch.Tensor]) -> Optional[float]:
        if b is None:
            return None
        aa = a.detach().cpu().float()
        bb = b.detach().cpu().float()
        if aa.ndim > 1: aa = aa.reshape(-1)
        if bb.ndim > 1: bb = bb.reshape(-1)
        na = torch.norm(aa) + 1e-8
        nb = torch.norm(bb) + 1e-8
        return float(torch.dot(aa, bb) / (na * nb))

    def _avg_cosine(self, q: torch.Tensor, vpms: List[Any]) -> float:
        if not vpms:
            return 0.0
        vals: List[float] = []
        for v in vpms:
            emb = getattr(v, "embedding", None)
            c = self._cosine(q, emb)
            if c is not None and math.isfinite(c):
                vals.append(c)
        if not vals:
            return 0.0
        # map cosine [-1,1] → [0,1] for a similarity-like measure
        return float(np.mean([(c + 1.0) * 0.5 for c in vals]))
