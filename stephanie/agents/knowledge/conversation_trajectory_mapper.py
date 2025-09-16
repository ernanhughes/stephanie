# stephanie/agents/knowledge/conversation_trajectory_mapper.py
"""
ConversationTrajectoryMapper (Upgraded)
---------------------------------------
Maps conversation trajectories to paper sections with causal relevance scoring.
Embedding alignment + HRM/MRQ (if available) + sentence-level evidence linking.
Emits graph-friendly mappings and can publish edges to the knowledge bus.

Input context:
  - paper_section: { id?, section_name, section_text, paper_id, goal? }
  - chat_corpus: [ { role, text, timestamp, id? }, ... ]  # not strictly required
  - conversation_trajectories: [
      {
        start_idx, end_idx,
        messages: [ { text, role?, id?, ts?, score?, is_critical? }, ... ],
        score, goal_achieved
      }, ...
    ]

Output added to context:
  - trajectory_mappings: [ { ... see return schema below ... } ]
  - critical_trajectory_mappings: [subset]
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stephanie.agents.base_agent import BaseAgent

_logger = logging.getLogger(__name__)


# ----------------------------
# Utilities
# ----------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b\w+\b", re.UNICODE)

def _sentences(text: str, max_sents: int = 80) -> List[str]:
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 2]
    if max_sents and len(sents) > max_sents:
        return sents[:max_sents]
    return sents

def _words(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())

def _lexical_overlap(a: str, b: str) -> float:
    A, B = set(_words(a)), set(_words(b))
    if not A:
        return 0.0
    return len(A & B) / max(1, len(A))

def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    if u is None or v is None:
        return 0.0
    num = float(np.dot(u, v))
    den = (float(np.dot(u, u)) ** 0.5) * (float(np.dot(v, v)) ** 0.5) + 1e-8
    return num / den


@dataclass
class CTMConfig:
    min_causal_strength: float = 0.60         # floor for causal thresholding
    critical_sigma: float = 0.5               # z-threshold: mean + sigma * std
    evidence_top_k: int = 3                   # top-K span pairs to keep
    max_traj_sents: int = 80                  # cap per-trajectory sentence calc
    max_section_sents: int = 120              # cap section sentence calc
    use_embeddings: bool = True               # embedding similarity if available
    use_hrm: bool = True                      # try HRM scorer if available
    use_mrq: bool = True                      # try MRQ scorer if available
    publish_edges: bool = True                # publish graph edges to bus
    bus_subject: str = "knowledge.trajectory.mapping"


class ConversationTrajectoryMapper(BaseAgent):
    """
    Embedding + scorer hybrid mapper from conversation trajectories to paper sections.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, logger: logging.Logger):
        super().__init__(cfg, memory, container, logger)
        self.kfg = CTMConfig(
            min_causal_strength=cfg.get("min_causal_strength", 0.60),
            critical_sigma=cfg.get("critical_sigma", 0.5),
            evidence_top_k=cfg.get("evidence_top_k", 3),
            max_traj_sents=cfg.get("max_traj_sents", 80),
            max_section_sents=cfg.get("max_section_sents", 120),
            use_embeddings=cfg.get("use_embeddings", True),
            use_hrm=cfg.get("use_hrm", True),
            use_mrq=cfg.get("use_mrq", True),
            publish_edges=cfg.get("publish_edges", True),
            bus_subject=cfg.get("bus_subject", "knowledge.trajectory.mapping"),
        )

        # Optional scorers (best-effort discovery)
        self.hrm_scorer = cfg.get("hrm_scorer")
        self.mrq_scorer = cfg.get("mrq_scorer")
        if self.kfg.use_hrm and self.hrm_scorer is None:
            self.hrm_scorer = getattr(self.memory, "hrm_scorer", None)
        if self.kfg.use_mrq and self.mrq_scorer is None:
            self.mrq_scorer = getattr(self.memory, "mrq_scorer", None)

        # Bus is optional; HybridKnowledgeBus recommended (NATS/InProcess)
        self.bus = getattr(self.memory, "bus", None)

        self.logger.info("ConversationTrajectoryMapper initialized", {
            "config": self.kfg.__dict__,
            "hrm_available": bool(self.hrm_scorer),
            "mrq_available": bool(self.mrq_scorer),
            "bus_available": bool(self.bus),
        })

    # ----------------------------
    # Agent entry point
    # ----------------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        paper_section = context.get("paper_section") or {}
        trajectories = context.get("conversation_trajectories", [])
        section_text = (paper_section.get("section_text") or "").strip()

        if not section_text or not trajectories:
            self.logger.log("TrajectoryMappingSkipped", {
                "reason": "missing_inputs",
                "has_section_text": bool(section_text),
                "num_trajectories": len(trajectories)
            })
            return context

        # Pre-embed section if embeddings are enabled
        section_emb = None
        section_sents = _sentences(section_text, self.kfg.max_section_sents)
        section_sent_embs: List[np.ndarray] = []

        if self.kfg.use_embeddings:
            try:
                section_emb = self._embed(section_text)
                for s in section_sents:
                    section_sent_embs.append(self._embed(s))
            except Exception as e:
                self.logger.log("SectionEmbeddingFailed", {"error": str(e)})
                section_emb = None
                section_sent_embs = []

        # Score each trajectory
        mappings: List[Dict[str, Any]] = []
        for i, traj in enumerate(trajectories):
            try:
                mapping = self._map_trajectory_to_section(
                    idx=i,
                    trajectory=traj,
                    section_text=section_text,
                    section_emb=section_emb,
                    section_sents=section_sents,
                    section_sent_embs=section_sent_embs
                )
                mappings.append(mapping)
            except Exception as e:
                self.logger.log("TrajectoryMappingError", {
                    "trajectory_index": i, "error": str(e)
                })

        # Adaptive critical thresholding on causal_strength
        strengths = [m["causal_strength"] for m in mappings if "causal_strength" in m]
        if strengths:
            mu = float(np.mean(strengths))
            sigma = float(np.std(strengths))
            dynamic_thr = max(self.kfg.min_causal_strength, mu + self.kfg.critical_sigma * sigma)
        else:
            mu, sigma, dynamic_thr = 0.0, 0.0, self.kfg.min_causal_strength

        for m in mappings:
            m["is_critical"] = bool(m["causal_strength"] >= dynamic_thr)

        critical = [m for m in mappings if m["is_critical"]]

        # Optionally publish graph edges for each mapping
        if self.kfg.publish_edges and self.bus:
            await self._publish_edges(paper_section, mappings)

        # Update context
        self.logger.log("TrajectoryMappingComplete", {
            "section": paper_section.get("section_name"),
            "trajectories_in": len(trajectories),
            "mappings_out": len(mappings),
            "critical": len(critical),
            "mu": mu, "sigma": sigma, "dynamic_thr": dynamic_thr
        })
        context["trajectory_mappings"] = mappings
        context["critical_trajectory_mappings"] = critical
        return context

    # ----------------------------
    # Core mapping logic
    # ----------------------------
    def _map_trajectory_to_section(
        self,
        idx: int,
        trajectory: Dict[str, Any],
        section_text: str,
        section_emb: Optional[np.ndarray],
        section_sents: List[str],
        section_sent_embs: List[np.ndarray],
    ) -> Dict[str, Any]:
        msgs = trajectory.get("messages", []) or []
        traj_text = "\n".join(m.get("text", "") for m in msgs if m.get("text"))
        traj_sents = _sentences(traj_text, self.kfg.max_traj_sents)

        # --- Relevance (embedding + lexical)
        section_relevance = self._section_relevance(
            msgs=msgs,
            traj_sents=traj_sents,
            section_text=section_text,
            section_emb=section_emb
        )

        # --- Causal strength (hybrid)
        causal_strength = self._causal_strength(
            trajectory_text=traj_text,
            section_text=section_text,
            hrm_score=self._score_hrm(trajectory),
            mrq_score=self._score_mrq(trajectory, section_text),
        )

        # --- Evidence linking (top-K sentence pairs)
        evidence = self._evidence_links(
            traj_sents=traj_sents,
            section_sents=section_sents,
            section_sent_embs=section_sent_embs
        )

        return {
            "trajectory_id": f"traj_{idx}",
            "section_relevance": float(section_relevance),
            "causal_strength": float(causal_strength),
            "supporting_evidence": evidence,  # [{trajectory_span, section_span, strength}]
            "trajectory": trajectory,
        }

    # ----------------------------
    # Scoring subroutines
    # ----------------------------
    def _section_relevance(
        self,
        msgs: List[Dict[str, Any]],
        traj_sents: List[str],
        section_text: str,
        section_emb: Optional[np.ndarray]
    ) -> float:
        # Message-level embedding max/avg sim (if available)
        emb_sims: List[float] = []
        if self.kfg.use_embeddings and section_emb is not None:
            for m in msgs:
                t = (m.get("text") or "").strip()
                if not t:
                    continue
                try:
                    e = self._embed(t[:2000])
                    emb_sims.append(_cosine(e, section_emb))
                except Exception:
                    pass

        # Lexical overlap fallback
        lex = _lexical_overlap("\n".join(traj_sents), section_text)

        if emb_sims:
            return 0.75 * float(np.mean(emb_sims)) + 0.25 * lex
        return lex

    def _causal_strength(
        self,
        trajectory_text: str,
        section_text: str,
        hrm_score: float,
        mrq_score: float,
    ) -> float:
        # Causal cues (regex)
        patterns = [
            r"\bso\s+we\s+(decided|concluded|implemented)",
            r"\btherefore\b",
            r"\bthis\s+led\s+to\b",
            r"\bas\s+a\s+result\b",
            r"\bbecause\s+of\s+this\b",
            r"\bconsequently\b",
            r"\bthus\b",
            r"\bhence\b",
        ]
        cue_hits = sum(1 for p in patterns if re.search(p, trajectory_text, re.IGNORECASE))
        cue_score = min(1.0, cue_hits / max(1, len(patterns) / 2.0))

        # Content overlap
        overlap = _lexical_overlap(trajectory_text, section_text)

        # Hybrid weighting:
        #   cues + overlap as base, HRM and MRQ (if present) as boosters
        base = 0.55 * cue_score + 0.45 * overlap
        booster = 0.0
        if hrm_score > 0:
            booster += 0.20 * hrm_score   # reasoning progress signal
        if mrq_score > 0:
            booster += 0.15 * mrq_score   # question/goal satisfaction signal

        return max(0.0, min(1.0, base + booster))

    def _evidence_links(
        self,
        traj_sents: List[str],
        section_sents: List[str],
        section_sent_embs: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        if not traj_sents or not section_sents:
            return []

        pairs: List[Tuple[str, str, float]] = []

        # If embeddings available, match by cosine; else lexical overlap
        if self.kfg.use_embeddings and section_sent_embs:
            # Pre-embed trajectory sentences
            traj_embs: List[np.ndarray] = []
            for s in traj_sents:
                try:
                    traj_embs.append(self._embed(s))
                except Exception:
                    traj_embs.append(None)

            for ti, (ts, te) in enumerate(zip(traj_sents, traj_embs)):
                # best match over section sentences
                best = 0.0
                best_ss = ""
                if te is not None:
                    for ss, se in zip(section_sents, section_sent_embs):
                        if se is None:
                            continue
                        sim = _cosine(te, se)
                        if sim > best:
                            best, best_ss = sim, ss
                else:
                    # fallback to lexical for this sentence
                    for ss in section_sents:
                        sim = _lexical_overlap(ts, ss)
                        if sim > best:
                            best, best_ss = sim, ss

                if best > 0:
                    pairs.append((ts, best_ss, float(best)))
        else:
            # lexical only
            for ts in traj_sents:
                best = 0.0
                best_ss = ""
                for ss in section_sents:
                    sim = _lexical_overlap(ts, ss)
                    if sim > best:
                        best, best_ss = sim, ss
                if best > 0:
                    pairs.append((ts, best_ss, float(best)))

        # Sort and take top-K
        pairs.sort(key=lambda x: x[2], reverse=True)
        top = pairs[: self.kfg.evidence_top_k]
        return [
            {"trajectory_span": t, "section_span": s, "strength": round(float(sc), 4)}
            for t, s, sc in top
        ]

    # ----------------------------
    # Optional scorer hooks
    # ----------------------------
    def _score_hrm(self, trajectory: Dict[str, Any]) -> float:
        """Human-Reasoning Model (progress) score if available; else heuristic."""
        try:
            if self.hrm_scorer and hasattr(self.hrm_scorer, "score"):
                # Expect interface: score(messages=[...]) -> {"score": float}
                out = self.hrm_scorer.score(messages=trajectory.get("messages", []))
                return float(out.get("score", 0.0))
        except Exception as e:
            self.logger.log("HRMScoreError", {"error": str(e)})

        # Heuristic fallback: look for resolution signals
        txt = "\n".join((m.get("text") or "").lower() for m in trajectory.get("messages", []))
        pos = int(bool(re.search(r"\b(solved|fixed|works|resolved|implemented)\b", txt)))
        neg = int(bool(re.search(r"\b(error|not working|issue|failed)\b", txt)))
        return max(0.0, min(1.0, 0.6 * pos + 0.2 * (1 - neg)))

    def _score_mrq(self, trajectory: Dict[str, Any], section_text: str) -> float:
        """Goal/Q alignment score if available; else heuristic overlap."""
        try:
            if self.mrq_scorer and hasattr(self.mrq_scorer, "score"):
                out = self.mrq_scorer.score(
                    messages=trajectory.get("messages", []),
                    section_text=section_text
                )
                return float(out.get("score", 0.0))
        except Exception as e:
            self.logger.log("MRQScoreError", {"error": str(e)})
        traj_text = "\n".join(m.get("text", "") for m in trajectory.get("messages", []))
        return _lexical_overlap(traj_text, section_text)

    # ----------------------------
    # Embedding
    # ----------------------------
    def _embed(self, text: str) -> np.ndarray:
        """Use memory.embedding to get/create embeddings (NumPy vector)."""
        if not text or not self.kfg.use_embeddings:
            return None
        vec = self.memory.embedding.get_or_create(text)
        # Ensure numpy array
        if isinstance(vec, np.ndarray):
            return vec
        try:
            return np.asarray(vec, dtype=np.float32)
        except Exception:
            return None

    # ----------------------------
    # Bus publishing
    # ----------------------------
    async def _publish_edges(self, paper_section: Dict[str, Any], mappings: List[Dict[str, Any]]) -> None:
        if not self.bus or not hasattr(self.bus, "publish"):
            return
        section_id = paper_section.get("id") or paper_section.get("paper_id")
        tasks = []
        for m in mappings:
            payload = {
                "event_type": "conversation.trajectory.mapping",
                "payload": {
                    "section_id": section_id,
                    "section_name": paper_section.get("section_name"),
                    "trajectory_id": m.get("trajectory_id"),
                    "causal_strength": m.get("causal_strength"),
                    "section_relevance": m.get("section_relevance"),
                    "is_critical": m.get("is_critical", False),
                    "evidence": m.get("supporting_evidence", []),
                    "timestamp": None,
                    "source_agent": "ConversationTrajectoryMapper",
                }
            }
            try:
                # HybridKnowledgeBus.publish(subject, payload) is async
                tasks.append(self.bus.publish(self.kfg.bus_subject, payload))
            except Exception as e:
                self.logger.log("TrajectoryMappingPublishError", {"error": str(e)})
        # Best-effort: publish concurrently
        if tasks:
            try:
                import asyncio
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                self.logger.log("TrajectoryMappingPublishGatherError", {"error": str(e)})
