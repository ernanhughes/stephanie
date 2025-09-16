"""
ConversationFilterAgent (Enhanced)
----------------------------------
Scores chat messages for relevance to a paper section and keeps only the
high-signal ones for downstream KnowledgeFusion/Drafting.

Key improvements:
- Windowed processing for long conversations (memory-safe)
- Hybrid scoring: embeddings + VPM + evidence/technical cues
- Dynamic thresholds (IQR-based) per section quality
- Optional domain-specific adjustments
- Critical path identification (causal learning trajectory)
- Robust embedding fallbacks (won’t crash on failures)

Input context:
  paper_section: { section_name, section_text, paper_id, domain? }
  chat_corpus:   [ { id?, role, text, timestamp? }, ... ]
  goal_template?: str (default "academic_summary")

Output context additions:
  scored_messages:   [ { ...message, score, similarity, vpm_score, reason, ... } ]
  critical_messages: high-signal subset (score ≥ dynamic threshold)
  critical_path:     ordered “learning path” through critical messages
  filter_threshold:  the dynamic threshold used
"""

from __future__ import annotations

import logging
import re
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.paper_improver.goals import GoalScorer

# ---- cues / patterns ----
EVIDENCE_HINTS = (
    "see figure", "fig.", "figure", "table", "tbl.", "[#]", "as shown",
    "we report", "results in", "demonstrated in", "shown in", "according to"
)
PROGRESS_HINTS = (
    "solution", "solve", "fixed", "works", "implemented", "approach",
    "method", "technique", "here's what we tried", "next we", "then we",
    "now we can", "this led to", "therefore", "consequently", "thus", "hence"
)
FACTUAL_HINTS = (
    "result", "show", "prove", "achiev", "increase", "decrease",
    "outperform", "error", "accuracy", "loss", "significant", "statistically"
)
TECHNICAL_TERMS = (
    "transformer", "adapter", "loss", "optimization", "pipeline",
    "retrieval", "graph", "policy", "reward", "ablation", "evaluation"
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _sentences(t: str) -> List[str]:
    if not t:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(t) if len(s.strip()) > 2]

def _lex_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    aw = set(re.findall(r"\b\w+\b", a.lower()))
    bw = set(re.findall(r"\b\w+\b", b.lower()))
    return (len(aw & bw) / max(1, len(aw))) if aw else 0.0

def _cos(u: np.ndarray, v: np.ndarray) -> float:
    if u is None or v is None:
        return 0.0
    num = float(np.dot(u, v))
    den = (float(np.dot(u, u)) ** 0.5) * (float(np.dot(v, v)) ** 0.5) + 1e-8
    return num / den


@dataclass
class CFConfig:
    # thresholds
    min_keep: float = 0.55
    z_sigma: float = 0.5  # legacy (kept; dynamic threshold takes precedence)
    # weights
    w_embed: float = 0.55
    w_vpm: float   = 0.30
    w_extra: float = 0.15
    # limits
    max_msgs: int = 1200
    window_size: int = 60
    window_stride: int = 30
    # toggles
    use_embeddings: bool = True
    use_temporal_boost: bool = True
    use_domain_rules: bool = True
    use_critical_path: bool = True
    # evidence / penalties
    evidence_bonus: float = 0.10
    chatter_penalty: float = 0.08
    length_cutoff: int = 18
    # logging
    log_top_k: int = 5


class ConversationFilterAgent(BaseAgent):
    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger):
        super().__init__(cfg, memory, container, logger)
        self.kfg = CFConfig(**cfg.get("conversation_filter", {}))
        self.goal_scorer = GoalScorer(logger=logger)
        self.logger.info("ConversationFilterAgent initialized", {"config": asdict(self.kfg)})

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()

        section = context.get("paper_section") or {}
        chat    = (context.get("chat_corpus") or [])[: self.kfg.max_msgs]
        goal_template = context.get("goal_template") or "academic_summary"

        section_text = (section.get("section_text") or "").strip()
        section_name = section.get("section_name", "section")
        section_domain = (section.get("domain") or "general").lower()

        if not section_text or not chat:
            self.logger.log("ConversationFilterSkipped", {
                "has_section_text": bool(section_text), "chat_count": len(chat)
            })
            return context

        # drop empties/very short
        chat = [m for m in chat if m.get("text") and len(m["text"].strip()) > 5]

        # section embedding (robust)
        sec_emb = None
        if self.kfg.use_embeddings:
            try:
                sec_emb = self._section_embedding(section_text)
            except Exception as e:
                self.logger.log("SectionEmbeddingFailed", {"error": str(e), "section": section_name})

        try:
            scored = self._score_messages(chat, section_text, sec_emb, goal_template)

            if self.kfg.use_domain_rules:
                scored = self._apply_domain_rules(scored, section_domain)

            thr = self._dynamic_threshold([m["score"] for m in scored], self.kfg.min_keep)
            critical = [m for m in scored if m["score"] >= thr]
            critical.sort(key=lambda x: x["score"], reverse=True)

            critical_path = []
            if self.kfg.use_critical_path and critical:
                critical_path = self._critical_path(critical)

            context.update({
                "scored_messages": scored,
                "critical_messages": critical,
                "critical_path": critical_path,
                "filter_threshold": thr,
                "section_domain": section_domain,
            })

            self.logger.log("ConversationFilterComplete", {
                "section": section_name,
                "domain": section_domain,
                "chat_in": len(chat),
                "kept": len(critical),
                "threshold": round(thr, 3),
                "top_samples": [
                    {"score": round(m["score"],3), "sim": round(m.get("similarity",0.0),3), "text": m["text"][:120]}
                    for m in critical[: self.kfg.log_top_k]
                ],
                "elapsed_s": round(time.time() - start, 2),
            })
            return context

        except Exception as e:
            self.logger.log("ConversationFilterError", {
                "error": str(e), "traceback": traceback.format_exc(), "section": section_name
            })
            context["filter_error"] = str(e)
            return context

    # ---------------- internal: scoring ----------------

    def _score_messages(self, chat: List[Dict[str, Any]], section_text: str,
                        sec_emb: Optional[np.ndarray], goal_template: str) -> List[Dict[str, Any]]:
        """Score all messages with overlapping windows to limit memory and add locality bias."""
        if len(chat) <= self.kfg.window_size:
            return self._score_window(chat, section_text, sec_emb, goal_template)

        all_scored: List[Dict[str, Any]] = []
        for start in range(0, len(chat), self.kfg.window_stride):
            end = min(start + self.kfg.window_size, len(chat))
            window = chat[start:end]
            window_scored = self._score_window(window, section_text, sec_emb, goal_template)

            # slight position bias toward window center (more coherent clusters)
            n = len(window_scored) or 1
            mid = (n - 1) / 2.0
            for i, m in enumerate(window_scored):
                m["score"] *= (1.0 - (abs(i - mid) / max(1.0, mid)) * 0.1)
            all_scored.extend(window_scored)

        return self._dedup_by_id(all_scored)

    def _score_window(self, chat: List[Dict[str, Any]], section_text: str,
                      sec_emb: Optional[np.ndarray], goal_template: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for msg in chat:
            text = (msg.get("text") or "").strip()
            if not text:
                continue

            # embedding similarity (with robust fallback)
            sim = 0.0
            if self.kfg.use_embeddings and sec_emb is not None:
                sim = self._embedding_similarity(text, sec_emb, section_text)

            # VPM dimensions + score
            dims = self._vpm_dims(text, section_text)
            vpm_score = self.goal_scorer.score(kind="text", goal=goal_template, vpm_row=dims).get("score", 0.0)

            # extras: evidence/technical/chatter/factual cues
            extra = self._extra_score(text, dims)

            score = (
                self.kfg.w_embed * sim +
                self.kfg.w_vpm   * vpm_score +
                self.kfg.w_extra * extra
            )

            # light temporal smoothing: boost msg after a strong one
            if self.kfg.use_temporal_boost and out and out[-1]["score"] > 0.6:
                score = min(1.0, score * 1.15)

            out.append({
                **msg,
                "score": float(score),
                "similarity": float(sim),
                "vpm_score": float(vpm_score),
                "extra_score": float(extra),
                "vpm_dims": dims,
                "reason": self._reason_from_dims(dims, extra),
            })
        return out

    # ---------------- internal: features ----------------

    def _section_embedding(self, section_text: str) -> Optional[np.ndarray]:
        if not section_text or len(section_text) < 10:
            return None
        try:
            if len(section_text) <= 2000:
                return self._embed(section_text)
            sents = _sentences(section_text)
            if len(sents) >= 3:
                first = self._embed(sents[0][:2000])
                mid   = self._embed(sents[len(sents)//2][:2000])
                last  = self._embed(sents[-1][:2000])
                return (first + mid + last) / 3.0
            return self._embed(section_text[:2000])
        except Exception:
            return None

    def _embedding_similarity(self, text: str, sec_emb: np.ndarray, section_text: str) -> float:
        try:
            if len(text) > 50:
                return _cos(self._embed(text[:2000]), sec_emb)
            # very short → lexical overlap proxy
            if len(text.split()) < 5:
                return _lex_overlap(text, section_text)
            sents = _sentences(text)
            if sents:
                return _cos(self._embed(sents[0][:200]), sec_emb)
        except Exception:
            self.logger.log("EmbeddingFallback", {"len": len(text)})
            return _lex_overlap(text, section_text)
        return 0.0

    def _vpm_dims(self, text: str, section_text: str) -> Dict[str, float]:
        t = text.lower()
        # correctness: overlap boosted when factual terms appear
        correctness = 0.35 + 0.65 * _lex_overlap(text, section_text)
        if any(k in t for k in FACTUAL_HINTS):
            correctness = min(1.0, correctness * 1.2)
        # progress: pushes toward solutions; penalize raw “error” unless analyzed
        if any(k in t for k in PROGRESS_HINTS):
            progress = 0.75
        elif "error" in t or "not working" in t:
            progress = 0.25
        else:
            progress = 0.5
        # evidence: hard references; otherwise overlap proxy
        evidence = 0.9 if any(h in t for h in EVIDENCE_HINTS) else min(1.0, 1.2 * _lex_overlap(text, section_text))
        # novelty: “we found/new/alternative…”
        novelty = 0.8 if any(k in t for k in ("new approach", "alternative", "novel", "innovative", "we found", "surprising", "counterintuitive")) else 0.35
        return {
            "correctness": float(correctness),
            "progress": float(progress),
            "evidence": float(evidence),
            "novelty": float(novelty),
        }

    def _extra_score(self, text: str, dims: Dict[str, float]) -> float:
        t = text.lower()
        extra = 0.0
        if any(h in t for h in EVIDENCE_HINTS):
            extra += self.kfg.evidence_bonus
        if len(text) < self.kfg.length_cutoff or t in {"ok", "thanks", "got it", "yup", "cool"}:
            extra -= self.kfg.chatter_penalty
        if any(term in t for term in TECHNICAL_TERMS):
            extra += 0.05
        if any(kw in t for kw in FACTUAL_HINTS) and dims["correctness"] > 0.7:
            extra += 0.07
        return max(-0.2, min(0.2, extra))

    def _reason_from_dims(self, dims: Dict[str, float], extra: float) -> str:
        bits = []
        if dims["correctness"] > 0.7: bits.append("factually accurate")
        if dims["progress"]   > 0.7: bits.append("advances understanding")
        if dims["evidence"]   > 0.7: bits.append("well-supported")
        if dims["novelty"]    > 0.7: bits.append("provides new insight")
        if extra > 0.05: bits.append("strong evidence refs")
        if extra < -0.05: bits.append("likely chatter")
        return " and ".join(bits) if bits else "moderate relevance"

    # ---------------- internal: post-processing ----------------

    def _dynamic_threshold(self, scores: List[float], floor: float) -> float:
        if not scores:
            return floor
        s = sorted(scores)
        q1 = s[int(0.25 * (len(s)-1))]
        q3 = s[int(0.75 * (len(s)-1))]
        iqr = q3 - q1
        thr = q3 + 0.5 * iqr
        return max(floor, min(0.9, thr))

    def _apply_domain_rules(self, scored: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        changed = 0
        for m in scored:
            before = m["score"]
            tl = m["text"].lower()
            if "machine learning" in domain:
                if "correlates with" in tl and "caus" not in tl:
                    m["score"] = max(0.0, m["score"] - 0.15)
                if any(t in tl for t in ["p-value", "confidence interval", "statistically significant"]):
                    m["score"] = min(1.0, m["score"] + 0.10)
            elif "biology" in domain:
                if any(t in tl for t in ["gene expression", "protein folding", "cellular mechanism"]):
                    m["score"] = min(1.0, m["score"] + 0.15)
            elif "math" in domain or "theory" in domain:
                if any(t in tl for t in ["proof", "theorem", "lemma", "corollary"]):
                    m["score"] = min(1.0, m["score"] + 0.12)
                if "basically" in tl or "kind of" in tl:
                    m["score"] = max(0.0, m["score"] - 0.10)
            if m["score"] != before:
                changed += 1
                m["domain_adjustment"] = round(m["score"] - before, 4)
        if changed:
            self.logger.log("DomainRulesApplied", {"domain": domain, "adjusted": changed})
        return scored

    def _critical_path(self, msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Greedy chain: high score + forward in time + causal/lexical link."""
        if not msgs:
            return []
        # sort by score desc, then by timestamp asc
        ranked = sorted(msgs, key=lambda x: (-x["score"], x.get("timestamp", 0)))
        path = [ranked[0]]
        last_ts = ranked[0].get("timestamp", 0)
        for m in ranked[1:]:
            ts = m.get("timestamp", 0)
            if ts < last_ts:
                continue
            if self._linked(path[-1]["text"], m["text"]):
                path.append(m)
                last_ts = ts
        # final path sorted by timestamp
        return sorted(path, key=lambda x: x.get("timestamp", 0))

    def _linked(self, prev: str, curr: str) -> bool:
        prev_l, curr_l = prev.lower(), curr.lower()
        causal_patterns = [
            r"so\s+we", r"therefore", r"as\s+a\s+result", r"because\s+of\s+this",
            r"consequently", r"thus", r"hence", r"this\s+led\s+to", r"follows\s+that"
        ]
        if any(re.search(p, prev_l + " " + curr_l) for p in causal_patterns):
            return True
        prev_words = set(prev_l.split()[:10])
        curr_words = set(curr_l.split())
        return len(prev_words & curr_words) > 2

    def _dedup_by_id(self, scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        uniq: List[Dict[str, Any]] = []
        # keep best-scoring duplicate
        for m in sorted(scored, key=lambda x: -x["score"]):
            mid = m.get("id") or str(hash(m["text"][:64]))
            if mid in seen:
                continue
            seen.add(mid)
            uniq.append(m)
        return uniq

    def _embed(self, text: str) -> np.ndarray:
        vec = self.memory.embedding.get_or_create(text)
        return vec if isinstance(vec, np.ndarray) else np.asarray(vec, dtype=np.float32)
