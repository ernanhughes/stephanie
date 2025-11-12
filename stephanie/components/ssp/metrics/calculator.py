# stephanie/components/ssp/metrics/calculator.py
from __future__ import annotations

import math
import re
from typing import Any, Dict

from stephanie.components.ssp.metrics.registry import (SSP_METRIC_ORDER,
                                                       SSP_METRIC_VERSION,
                                                       MetricVector)
from stephanie.components.ssp.metrics.scorable import SSPScorable


class SSPMetricsCalculator:
    """
    Deterministic, versioned SSP metrics in [0,1]. Always returns the same order.
    Enhanced with paper-proven metrics that track capability growth.
    """

    def __init__(self, cfg: Dict[str, Any] | None = None):
        c = cfg or {}
        self.max_question_words = int(c.get("max_question_words", 128))
        self.max_answer_words   = int(c.get("max_answer_words", 128))
        self.max_evidence       = int(c.get("max_evidence", 8))
        self.max_steps          = int(c.get("max_steps", 64))
        self.max_depth          = int(c.get("max_depth", 16))
        self.noise_docs_count   = int(c.get("noise_docs_count", 4))  # From SSP paper Table 3

    def score(self, s: SSPScorable) -> MetricVector:
        names = list(SSP_METRIC_ORDER)
        vmap: Dict[str, float] = {}

        # 1) reward
        vmap["ssp.reward"] = _clamp01(_fallback(s.reward, 0.0))

        # 2) verified
        vmap["ssp.verified"] = 1.0 if s.verified else 0.0

        # 3) curriculum_difficulty
        vmap["ssp.curriculum_difficulty"] = _clamp01(_fallback(s.difficulty, 0.0))

        # 4) question_len
        q_words = _word_count(s.question)
        vmap["ssp.question_len"] = _clamp01(q_words / max(1, self.max_question_words))

        # 5) answer_len
        a_words = _word_count(s.predicted_answer)
        vmap["ssp.answer_len"] = _clamp01(a_words / max(1, self.max_answer_words))

        # 6) evidence_count
        evid_cnt = len(s.evidence_docs or [])
        vmap["ssp.evidence_count"] = _clamp01(evid_cnt / max(1, self.max_evidence))

        # 7) solver_steps (normalized; note: direction is DOWN in registry)
        steps = int(s.solver_steps or 0)
        vmap["ssp.solver_steps"] = _clamp01(steps / max(1, self.max_steps))

        # 8) score  (optional)
        score = s.score
        vmap["ssp.score"] = _clamp01(_fallback(score, 0.0))

        # 9) best_score (optional)
        best_score = s.best_score
        vmap["ssp.best_score"] = _clamp01(_fallback(best_score, vmap["ssp.score"]))

        # 10) improvement (derived, monotone in [0,1])
        # If both are present use relative lift; else 0
        base = vmap["ssp.score"]
        best = vmap["ssp.best_score"]
        if best >= base and best > 0:
            # relative delta wrt best, clipped to [0,1]
            #  = (best - base) / (1 - base) gives "room-to-1" lift
            denom = max(1e-6, 1.0 - base)
            vmap["ssp.improvement"] = _clamp01((best - base) / denom)
        else:
            vmap["ssp.improvement"] = 0.0

        # 11) depth
        depth = int(s.depth or 0)
        vmap["ssp.depth"] = _clamp01(depth / max(1, self.max_depth))

        # 12) novelty (optional)
        novelty = _to01((s.meta or {}).get("novelty"))
        vmap["ssp.novelty"] = _clamp01(_fallback(novelty, 0.0))
        
        # === NEW METRICS FROM SSP PAPER ===
        # 13) search_turns - directly from paper Figure 4a
        # Tracks actual search tool calls (not just solver steps)
        search_turns = self._count_search_turns(s)
        vmap["ssp.search_turns"] = _clamp01(search_turns / max(1, self.max_steps))
        
        # 14) f1_score - from paper's LLM-as-a-judge evaluation
        # Critical for tracking lexical accuracy without verification
        f1 = self._calculate_f1(s.predicted_answer, s.seed_answer)
        vmap["ssp.f1_score"] = _clamp01(f1)
        
        # 15) format_compliance - from paper Section 4.4 rule-based filtering
        # Must have <question> tags and not contain the answer
        format_ok = self._check_format_compliance(s)
        vmap["ssp.format_compliance"] = 1.0 if format_ok else 0.0
        
        # 16) noise_tolerance - from paper Table 3 (4 noisy documents optimal)
        # Measures robustness to irrelevant information
        noise_tolerance = self._calculate_noise_tolerance(s)
        vmap["ssp.noise_tolerance"] = _clamp01(noise_tolerance)
        
        # 17) rag_verification - from paper's RAG verification process
        # Critical for ensuring questions are answerable with provided evidence
        rag_verified = self._check_rag_verification(s)
        vmap["ssp.rag_verification"] = 1.0 if rag_verified else 0.0

        # Assemble fixed-order values
        values = [float(vmap.get(k, 0.0)) for k in names]
        return MetricVector(
            version=SSP_METRIC_VERSION,
            names=names,
            values=values,
            vector=vmap,
        )
    
    def _count_search_turns(self, s: SSPScorable) -> int:
        """Count actual search tool invocations from evidence or metadata"""
        # If evidence_docs contains search results, count them
        if s.evidence_docs and len(s.evidence_docs) > 0:
            return len(s.evidence_docs)
        
        # Check if metadata has search_turns info
        if s.meta and "search_turns" in s.meta:
            return int(s.meta["search_turns"])
            
        # Fallback to solver_steps (less accurate)
        return s.solver_steps or 0
    
    def _calculate_f1(self, response: str, ground_truth: str) -> float:
        """Lexical F1 calculation as used in paper's LLM-as-a-judge evaluation"""
        if not ground_truth or not response:
            return 0.0
            
        # Clean and tokenize
        gt_tokens = set(re.findall(r'\w+', ground_truth.lower()))
        resp_tokens = set(re.findall(r'\w+', response.lower()))
        
        if not gt_tokens or not resp_tokens:
            return 0.0
            
        common = len(gt_tokens & resp_tokens)
        precision = common / len(resp_tokens) if resp_tokens else 0.0
        recall = common / len(gt_tokens) if gt_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
            
        return (2 * precision * recall) / (precision + recall)
    
    def _check_format_compliance(self, s: SSPScorable) -> bool:
        """
        Check if response follows required format per SSP paper Section 4.4:
        - Must not contain original answer in question
        - Must have proper structure (e.g.,  tags)
        - Must require search (not answerable from general knowledge)
        """
        # Rule 1: No original answer in question
        if s.seed_answer and s.seed_answer.lower() in s.question.lower():
            return False
            
        # Rule 2: Must have proper answer tags (if using SSP format)
        if "" not in s.predicted_answer or "" not in s.predicted_answer:
            return False
            
        # Rule 3: Evidence must be non-empty (requires search)
        if not s.evidence_docs or len(s.evidence_docs) == 0:
            return False
            
        # Rule 4: Question must be sufficiently long
        if _word_count(s.question) < 5:
            return False
            
        return True
    
    def _calculate_noise_tolerance(self, s: SSPScorable) -> float:
        """
        Calculate how well the system handles noisy evidence.
        Based on paper's finding that 4 noisy documents is optimal.
        """
        if not s.meta:
            return 0.0
            
        # Look for noise-related metrics in metadata
        noise_count = s.meta.get("noise_doc_count", 0)
        noise_success = s.meta.get("noise_success", 0.0)
        
        # If we have specific noise tolerance data
        if noise_count > 0:
            return noise_success
            
        # Fallback: if we know how many noise docs were added
        if noise_count >= self.noise_docs_count:
            # Higher score if it succeeded with the optimal noise count
            return 1.0 if s.verified else 0.5
            
        return 1.0 if s.verified else 0.0
    
    def _check_rag_verification(self, s: SSPScorable) -> bool:
        """
        Check if RAG verification passed per SSP paper methodology.
        This is the critical quality gate for generated problems.
        """
        # If we have explicit RAG verification result
        if s.meta and "rag_verified" in s.meta:
            return bool(s.meta["rag_verified"])
            
        # Default: if verified and has evidence, assume RAG passed
        return s.verified and bool(s.evidence_docs and len(s.evidence_docs) > 0)

def _word_count(text: str) -> int:
    if not text: return 0
    return max(0, len(str(text).strip().split()))

def _clamp01(x: float) -> float:
    return 0.0 if not math.isfinite(x) else 1.0 if x > 1.0 else (0.0 if x < 0.0 else x)

def _to01(x):
    if x is None:
        return None
    try:
        return _clamp01(float(x))
    except Exception:
        return None

def _fallback(x, default):
    return default if x is None else x