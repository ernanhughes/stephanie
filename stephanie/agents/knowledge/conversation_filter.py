# stephanie/agents/knowledge/conversation_filter.py
"""
ConversationFilterAgent (Enhanced)
-------------------------------
Scores chat messages for relevance to a paper section and keeps only
the high-signal ones for downstream KnowledgeFusion/Drafting.

Key improvements over original:
- Windowed processing for long conversations (avoids memory issues)
- Domain-aware filtering rules
- Critical path identification (finds actual learning trajectory)
- Self-calibrating thresholds (adapts to section quality)
- Robust embedding fallback strategy
- Better evidence extraction
"""

from __future__ import annotations
import json
import re
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import logging
import traceback

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.paper_improver.goals import GoalScorer
from stephanie.knowledge.casebook_store import CaseBookStore
from stephanie.scoring.scorable_factory import TargetType
from dataclasses import dataclass, asdict

# Global constants - make these configurable via CFConfig later
EVIDENCE_HINTS = ("see figure", "fig.", "figure", "table", "tbl.", "[#]", "as shown", 
                 "we report", "results in", "demonstrated in", "shown in", "according to")
PROGRESS_HINTS = ("solution", "solve", "fixed", "works", "implemented", "approach", 
                 "method", "technique", "here's what we tried", "next we", "then we", 
                 "now we can", "this led to", "therefore", "consequently")
FACTUAL_HINTS = ("result", "show", "prove", "achiev", "increase", "decrease", 
                "outperform", "error", "accuracy", "loss", "significant", "statistically")
TECHNICAL_TERMS = ("transformer", "adapter", "loss", "optimization", "pipeline", 
                 "retrieval", "graph", "policy", "reward", "ablation", "evaluation")


@dataclass
class CFConfig:
    """Configuration for ConversationFilterAgent."""
    # Thresholds
    min_keep: float = 0.55  # Floor for keeping a message
    z_sigma: float = 0.5    # Adaptive keep = max(min_keep, mean + z_sigma*std)
    
    # Weights
    w_embed: float = 0.55   # Embedding similarity weight
    w_vpm: float = 0.30     # VPM scoring weight
    w_extra: float = 0.15   # Evidence/temporal/chatter adjustments weight
    
    # Caps/limits
    max_msgs: int = 1000    # Quick guardrail
    window_size: int = 50   # For processing long conversations
    window_stride: int = 25
    
    # Toggles
    use_embeddings: bool = True
    use_temporal_boost: bool = True
    use_domain_rules: bool = True
    use_critical_path: bool = True
    
    # Evidence/penalties
    evidence_bonus: float = 0.10
    chatter_penalty: float = 0.08  # Very short messages or "thanks/ok"
    length_cutoff: int = 18        # <18 chars = likely chatter
    
    # Logging
    log_top_k: int = 5


class ConversationFilterAgent(BaseAgent):
    """
    Scores chat messages for relevance to a paper section and keeps only
    the high-signal ones for downstream KnowledgeFusion/Drafting.
    
    Uses a hybrid approach:
    1. Embedding similarity to section content
    2. VPM scoring across multiple dimensions
    3. Evidence-based filtering (figure/table references)
    4. Critical path identification (actual learning trajectory)
    """
    
    def __init__(self, cfg: Dict[str, Any], memory: Any, logger: logging.Logger):
        super().__init__(cfg, memory, logger)
        
        # Configuration
        self.kfg = CFConfig(**cfg.get("conversation_filter", {}))
        
        # Components
        self.goal_scorer = GoalScorer(logger=logger)
        self.casebooks: CaseBookStore = cfg.get("casebooks") or CaseBookStore()
        
        # Stats
        self.stats = {
            "total_messages": 0,
            "filtered_messages": 0,
            "processing_time": 0.0,
            "domain_rules_applied": 0
        }
        
        self.logger.info("ConversationFilterAgent initialized", {
            "config": asdict(self.kfg),
            "message": "Ready to filter conversation messages"
        })

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filters and scores chat messages for relevance to paper section.
        
        Expected context:
          - paper_section: { section_name, section_text, paper_id, domain? }
          - chat_corpus: [ { role, text, timestamp }, ... ]
          - goal_template: str (e.g., "academic_summary")
          
        Returns context with:
          - scored_messages: [ { text, score, reason, ... } ]
          - critical_messages: filtered high-signal messages
          - critical_path: identified learning trajectory
        """
        start_time = time.time()
        
        paper_section = context.get("paper_section")
        chat_corpus = context.get("chat_corpus", [])
        goal_template = context.get("goal_template", "academic_summary")
        
        if not paper_section or not chat_corpus:
            self.logger.log("ConversationFilterSkipped", {
                "reason": "missing_inputs",
                "has_section": bool(paper_section),
                "has_chat": bool(chat_corpus)
            })
            return context
            
        # Clean and prepare data
        section_text = paper_section.get("section_text", "")
        section_name = paper_section.get("section_name", "Unknown")
        section_domain = paper_section.get("domain", "general")
        
        # Remove empty messages
        chat_corpus = [msg for msg in chat_corpus 
                      if msg.get("text") and len(msg["text"].strip()) > 5]
        
        self.stats["total_messages"] = len(chat_corpus)
        
        try:
            # 1. Pre-embed section (if using embeddings)
            sec_emb = None
            if self.kfg.use_embeddings:
                try:
                    sec_emb = self._get_section_embedding(section_text)
                except Exception as e:
                    self.logger.warning("SectionEmbeddingFailed", {
                        "error": str(e),
                        "section": section_name
                    })
            
            # 2. Score each message with windowed processing
            scored = self._score_messages(
                chat_corpus, 
                section_text, 
                sec_emb,
                goal_template
            )
            
            # 3. Apply domain-specific rules
            if self.kfg.use_domain_rules:
                scored = self._apply_domain_rules(scored, paper_section)
                self.stats["domain_rules_applied"] += 1
                
            # 4. Calculate dynamic threshold
            dynamic_threshold = self._calculate_dynamic_threshold(
                [m["score"] for m in scored]
            )
            
            # 5. Filter to critical messages
            critical = [m for m in scored if m["score"] >= dynamic_threshold]
            
            # 6. Identify critical learning path (if enabled)
            critical_path = []
            if self.kfg.use_critical_path and critical:
                critical_path = self._identify_critical_path(critical)
            
            # 7. Update context
            context.update({
                "scored_messages": scored,
                "critical_messages": critical,
                "critical_path": critical_path,
                "filter_threshold": dynamic_threshold,
                "section_domain": section_domain
            })
            
            # 8. Log results
            processing_time = time.time() - start_time
            self.stats["processing_time"] = processing_time
            self.stats["filtered_messages"] = len(critical)
            
            self.logger.log("ConversationFilterComplete", {
                "section": section_name,
                "domain": section_domain,
                "total_messages": len(chat_corpus),
                "critical_messages": len(critical),
                "critical_path_length": len(critical_path),
                "threshold": dynamic_threshold,
                "processing_time": f"{processing_time:.2f}s"
            })
            
            # Log top messages for debugging
            if self.kfg.log_top_k > 0:
                top_msgs = sorted(critical, key=lambda x: -x["score"])[:self.kfg.log_top_k]
                for i, msg in enumerate(top_msgs):
                    self.logger.log("TopCriticalMessage", {
                        "rank": i+1,
                        "score": msg["score"],
                        "text": msg["text"][:100] + "..." if len(msg["text"]) > 100 else msg["text"]
                    })
            
            return context
            
        except Exception as e:
            self.logger.log("ConversationFilterError", {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "section": section_name
            })
            context["filter_error"] = str(e)
            return context

    def _get_section_embedding(self, section_text: str) -> Optional[np.ndarray]:
        """Get embedding for section with robust fallbacks."""
        if not section_text or len(section_text.strip()) < 10:
            return None
            
        try:
            # Try full section first
            if len(section_text) <= 2000:
                return self.memory.embedding.get_or_create(section_text)
                
            # Try key sentences if section is long
            sentences = _sentences(section_text)
            if len(sentences) > 3:
                # Get embeddings for first, middle, and last sentences
                first = self.memory.embedding.get_or_create(sentences[0][:2000])
                middle = self.memory.embedding.get_or_create(sentences[len(sentences)//2][:2000])
                last = self.memory.embedding.get_or_create(sentences[-1][:2000])
                # Average them
                return (first + middle + last) / 3.0
                
            # Fallback to first sentence
            return self.memory.embedding.get_or_create(sentences[0][:2000] if sentences else "")
            
        except Exception as e:
            self.logger.warning("SectionEmbeddingError", {
                "error": str(e),
                "text_length": len(section_text)
            })
            return None

    def _score_messages(self, 
                       chat: List[Dict[str, Any]], 
                       section_text: str,
                       sec_emb: Optional[np.ndarray],
                       goal_template: str) -> List[Dict[str, Any]]:
        """Score all messages with windowed processing for long conversations."""
        if len(chat) <= self.kfg.window_size:
            return self._score_window(chat, section_text, sec_emb, goal_template)
            
        # Process in overlapping windows
        all_scored = []
        for start_idx in range(0, len(chat), self.kfg.window_stride):
            end_idx = min(start_idx + self.kfg.window_size, len(chat))
            window = chat[start_idx:end_idx]
            
            # Score this window
            window_scored = self._score_window(window, section_text, sec_emb, goal_template)
            
            # Adjust scores based on position (center gets slight boost)
            for i, msg in enumerate(window_scored):
                position_factor = 1.0 - abs(i - len(window)/2) / (len(window)/2) * 0.1
                msg["score"] *= position_factor
                msg["window_pos"] = i
                
            all_scored.extend(window_scored)
        
        # Deduplicate messages that appeared in multiple windows
        return self._deduplicate_scores(all_scored)

    def _score_window(self,
                     chat: List[Dict[str, Any]],
                     section_text: str,
                     sec_emb: Optional[np.ndarray],
                     goal_template: str) -> List[Dict[str, Any]]:
        """Score messages within a single window."""
        scored = []
        for msg in chat:
            text = (msg.get("text") or "").strip()
            if not text:
                continue
                
            # 1) Embedding similarity
            sim = 0.0
            if self.kfg.use_embeddings and sec_emb is not None:
                try:
                    sim = self._get_embedding_similarity(text, sec_emb)
                except Exception as e:
                    self.logger.warning("EmbeddingScoreError", {
                        "error": str(e),
                        "text_length": len(text)
                    })
            
            # 2) VPM scoring
            dims = self._vpm_dims(text, section_text)
            vpm_result = self.goal_scorer.score(
                kind="text", 
                goal=goal_template, 
                vpm_row=dims
            )
            vpm_score = vpm_result.get("score", 0.0)
            
            # 3) Extra adjustments
            extra = self._calculate_extra_score(text, dims)
            
            # 4) Combined score
            total_score = (
                self.kfg.w_embed * sim +
                self.kfg.w_vpm * vpm_score +
                self.kfg.w_extra * extra
            )
            
            # 5) Temporal boost (if enabled)
            if self.kfg.use_temporal_boost and len(scored) > 0:
                last_score = scored[-1]["score"]
                if last_score > 0.6:  # Previous message was high quality
                    total_score = min(1.0, total_score * 1.15)
            
            scored.append({
                **msg,
                "score": total_score,
                "similarity": sim,
                "vpm_score": vpm_score,
                "extra_score": extra,
                "vpm_dims": dims,
                "reason": self._generate_reason(vpm_result, extra)
            })
            
        return scored

    def _get_embedding_similarity(self, text: str, sec_emb: np.ndarray) -> float:
        """Get embedding similarity with robust fallback strategy."""
        try:
            # 1. Try full embedding
            if len(text) > 50:
                text_emb = self.memory.embedding.get_or_create(text[:2000])
                return _cos(text_emb, sec_emb)
            
            # 2. Try keyword-based if text too short
            if len(text.split()) < 5:
                return _lexical_overlap(text, self.section_text)
                
            # 3. Fallback to sentence embedding
            sentences = _sentences(text)
            if sentences:
                sentence_emb = self.memory.embedding.get_or_create(sentences[0][:200])
                return _cos(sentence_emb, sec_emb)
                
        except Exception as e:
            self.logger.warning("EmbeddingFallback", {
                "error": str(e),
                "text_length": len(text),
                "fallback": "lexical_overlap"
            })
            return _lexical_overlap(text, self.section_text)
            
        return 0.0

    def _vpm_dims(self, text: str, section_text: str) -> Dict[str, float]:
        """Calculate VPM dimensions for message scoring."""
        t = text.lower()
        s = section_text.lower()
        
        # Correctness: factual accuracy relative to section
        correctness = 0.35 + 0.65 * _lexical_overlap(text, section_text)
        if any(k in t for k in FACTUAL_HINTS):
            correctness = min(1.0, correctness * 1.2)
        
        # Progress: movement toward understanding/solution
        if any(k in t for k in PROGRESS_HINTS):
            progress = 0.75
        elif "error" in t or "not working" in t:
            progress = 0.25
        else:
            progress = 0.5
            
        # Evidence: references to paper content
        if any(h in t for h in EVIDENCE_HINTS):
            evidence = 0.9
        else:
            evidence = min(1.0, 1.2 * _lexical_overlap(text, section_text))
        
        # Novelty: new insights vs. repetition
        novelty_terms = ("new approach", "alternative", "novel", "innovative", 
                        "we found", "surprising", "counterintuitive")
        novelty = 0.8 if any(k in t for k in novelty_terms) else 0.35
        
        return {
            "correctness": float(correctness),
            "progress": float(progress),
            "evidence": float(evidence),
            "novelty": float(novelty),
        }

    def _calculate_extra_score(self, text: str, dims: Dict[str, float]) -> float:
        """Calculate extra score adjustments (evidence, chatter, etc.)."""
        extra = 0.0
        t = text.lower()
        
        # Evidence bonus
        if any(h in t for h in EVIDENCE_HINTS):
            extra += self.kfg.evidence_bonus
            
        # Chatter penalty
        if len(text) < self.kfg.length_cutoff or t in {"ok", "thanks", "got it", "yup", "cool"}:
            extra -= self.kfg.chatter_penalty
            
        # Technical depth bonus
        if any(term in t for term in TECHNICAL_TERMS):
            extra += 0.05
            
        # Factual claim bonus
        if any(kw in t for kw in FACTUAL_HINTS) and dims["correctness"] > 0.7:
            extra += 0.07
            
        return max(-0.2, min(0.2, extra))  # Cap adjustments

    def _generate_reason(self, vpm_result: Dict[str, Any], extra: float) -> str:
        """Generate human-readable reason for score."""
        reasons = []
        
        # Highlight strong dimensions
        for dim, score in vpm_result["normalized"].items():
            if score > 0.7:
                if dim == "correctness":
                    reasons.append("factually accurate")
                elif dim == "progress":
                    reasons.append("advances understanding")
                elif dim == "evidence":
                    reasons.append("well-supported")
                elif dim == "novelty":
                    reasons.append("provides new insight")
        
        # Add extra factor explanations
        if extra > 0.05:
            reasons.append("strong evidence references")
        elif extra < -0.05:
            reasons.append("likely chatter/noise")
            
        if not reasons:
            return "moderate relevance"
        return " and ".join(reasons)[:100]  # Keep concise

    def _calculate_dynamic_threshold(self, all_scores: List[float]) -> float:
        """Calculate threshold based on score distribution."""
        if not all_scores:
            return self.kfg.min_keep
            
        # Use interquartile range to find natural clusters
        scores = sorted(all_scores)
        q1 = scores[int(len(scores) * 0.25)]
        q3 = scores[int(len(scores) * 0.75)]
        iqr = q3 - q1
        
        # Threshold is Q3 + 0.5*IQR (finds the upper cluster)
        dynamic_threshold = q3 + 0.5 * iqr
        
        # Never go below min_keep or above 0.9
        return max(self.kfg.min_keep, min(0.9, dynamic_threshold))

    def _apply_domain_rules(self, scored: List[dict], section: dict) -> List[dict]:
        """Apply domain-specific filtering rules."""
        domain = section.get("domain", "general").lower()
        modified = 0
        
        for msg in scored:
            original_score = msg["score"]
            
            # ML-specific rules
            if "machine learning" in domain:
                # Penalize confusing correlation with causation
                if "correlates with" in msg["text"].lower() and "causes" not in msg["text"].lower():
                    msg["score"] = max(0.0, msg["score"] - 0.15)
                
                # Boost proper statistical language
                if any(term in msg["text"].lower() for term in ["p-value", "confidence interval", "statistically significant"]):
                    msg["score"] = min(1.0, msg["score"] + 0.1)
            
            # Biology-specific rules
            elif "biology" in domain:
                # Boost proper biological terminology
                if any(term in msg["text"].lower() for term in ["gene expression", "protein folding", "cellular mechanism"]):
                    msg["score"] = min(1.0, msg["score"] + 0.15)
            
            # Math/theory-specific rules
            elif "math" in domain or "theory" in domain:
                # Boost formal reasoning
                if any(term in msg["text"].lower() for term in ["proof", "theorem", "lemma", "corollary"]):
                    msg["score"] = min(1.0, msg["score"] + 0.12)
                # Penalize hand-wavy explanations
                if "basically" in msg["text"].lower() or "kind of" in msg["text"].lower():
                    msg["score"] = max(0.0, msg["score"] - 0.1)
            
            if msg["score"] != original_score:
                modified += 1
                msg["domain_adjustment"] = msg["score"] - original_score
        
        if modified > 0:
            self.logger.log("DomainRulesApplied", {
                "domain": domain,
                "rules_applied": modified,
                "message": f"Applied {modified} domain-specific adjustments"
            })
        
        return scored

    def _identify_critical_path(self, scored: List[dict]) -> List[dict]:
        """Identify the actual learning path through the conversation."""
        if not scored:
            return []
            
        # Sort by score and timestamp
        scored = sorted(scored, key=lambda x: (-x["score"], x.get("timestamp", 0)))
        
        # Start with highest scoring message
        path = [scored[0]]
        last_ts = scored[0].get("timestamp", 0)
        
        # Build path by finding messages that logically follow
        for msg in scored[1:]:
            # Must be after last message in path
            if msg.get("timestamp", 0) < last_ts:
                continue
                
            # Must have logical connection to previous
            if not self._has_logical_connection(msg, path[-1]):
                continue
                
            path.append(msg)
            last_ts = msg.get("timestamp", 0)
        
        # Sort final path by timestamp
        return sorted(path, key=lambda x: x.get("timestamp", 0))

    def _has_logical_connection(self, msg1: dict, msg2: dict) -> bool:
        """Check if two messages have a logical connection."""
        # Check for causal language
        causal_patterns = [
            r"so\s+we", r"therefore", r"as\s+a\s+result", r"because\s+of\s+this",
            r"consequently", r"thus", r"hence", r"this\s+led\s+to", r"follows\s+that"
        ]
        
        combined = f"{msg1['text'].lower()} {msg2['text'].lower()}"
        if any(re.search(pattern, combined) for pattern in causal_patterns):
            return True
            
        # Check for reference to previous message content
        prev_words = set(msg1["text"].lower().split()[:10])
        curr_words = set(msg2["text"].lower().split())
        return len(prev_words & curr_words) > 2

    def _deduplicate_scores(self, scored: List[dict]) -> List[dict]:
        """Remove duplicate messages that appeared in multiple windows."""
        seen_ids = set()
        unique = []
        
        # Sort by score (highest first) so we keep best version
        scored = sorted(scored, key=lambda x: -x["score"])
        
        for msg in scored:
            msg_id = msg.get("id") or str(hash(msg["text"][:50]))
            if msg_id not in seen_ids:
                seen_ids.add(msg_id)
                unique.append(msg)
                
        return unique

    def health_check(self) -> Dict[str, Any]:
        """Return health status and metrics for the filter agent."""
        return {
            "status": "healthy",
            "stats": self.stats,
            "config": asdict(self.kfg),
            "message": "Conversation filter agent operational"
        }


# ======================
# UTILITY FUNCTIONS
# ======================

def _sentences(text: str) -> List[str]:
    """Split text into sentences."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]

def _lexical_overlap(a: str, b: str) -> float:
    """Calculate lexical overlap between two texts."""
    if not a or not b:
        return 0.0
    a_words = set(re.findall(r"\b\w+\b", a.lower()))
    b_words = set(re.findall(r"\b\w+\b", b.lower()))
    return len(a_words & b_words) / max(1, len(a_words))

def _cos(u: np.ndarray, v: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    if u is None or v is None:
        return 0.0
    num = float(np.dot(u, v))
    den = (float(np.dot(u, u)) ** 0.5) * (float(np.dot(v, v)) ** 0.5) + 1e-8
    return num / den 