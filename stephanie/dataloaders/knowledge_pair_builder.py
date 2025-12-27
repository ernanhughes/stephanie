# stephanie/dataloaders/knowledge_pair_builder.py
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from stephanie.utils.coerce_utils import to_float
from stephanie.utils.hash_utils import hash_text

log = logging.getLogger(__name__)

def _simple_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation as delimiters"""
    parts = re.split(r'(?<=[.!?])\s+', (text or '').strip())
    return [p for p in parts if p]

def _entity_texts_from_ner(ner) -> List[str]:
    """
    Parse NER data into unique, lower-cased entity surface forms.
    
    Supports: list[dict], JSON string, or None.
    """
    if not ner:
        return []
    if isinstance(ner, str):
        try:
            ner = json.loads(ner)
        except Exception:
            return []
    out = []
    for e in ner if isinstance(ner, list) else []:
        try:
            t = (e.get("text") or "").strip().lower()
            if len(t) >= 2:
                out.append(t)
        except Exception:
            continue
    return list({t for t in out})

def _primary_domain_from_row(row: Dict[str, Any]) -> Optional[str]:
    """
    Extract first non-empty domain name (lower-cased).
    
    Supports: list[dict], JSON string, or None.
    """
    doms = row.get("domains") or []
    if isinstance(doms, str):
        try:
            doms = json.loads(doms)
        except Exception:
            doms = []
    for d in doms if isinstance(doms, list) else []:
        try:
            name = str(d.get("domain") or "").strip().lower()
            if name:
                return name
        except Exception:
            continue
    return None

def _len_norm(t: str, cap: int = 2000) -> float:
    """Normalize text length to [0,1] range with cap"""
    t = t or ""
    n = min(len(t), cap)
    return n / cap

class KnowledgePairBuilder:
    """
    Builds contrastive (pos, neg) pairs for DPO-lite training with dual-head support.
    
    Key features:
    - Entity-aware pooling (min_entity_overlap constraint)
    - Dual-source pairs (human-rated + AI-rated)
    - Source tagging & weighting
    - AI confidence propagation
    - Efficient embedding caching
    
    Usage:
        builder = KnowledgePairBuilder(memory)
        pairs = builder.build_pairs(
            min_star_pos=2,
            max_star_neg=-1,
            limit=50000,
            max_negs_per_pos=3
        )
    """
    
    def __init__(self, memory: Any, min_entity_overlap: int = 1, seed: int = 1337):
        """
        Initialize the pair builder.
        
        Args:
            memory: Memory interface with chat access
            min_entity_overlap: Minimum entities that must overlap between turns
            seed: Random seed for deterministic shuffling
        """
        self.memory = memory
        self.min_entity_overlap = int(min_entity_overlap)
        random.seed(seed)
        self._emb_cache: dict[int, np.ndarray] = {}  # turn_id -> emb
        self._calibrator = None  # Will be set if available
    
    def set_calibrator(self, calibrator):
        """Set the ScoreCalibrator for AI score normalization"""
        self._calibrator = calibrator
    
    def build_pairs(
        self,
        min_star_pos: int = 2,
        max_star_neg: int = -1,
        limit: int = 500,
        max_negs_per_pos: int = 3,
        shuffle: bool = True,
        ai_score_threshold: float = 70.0,
        ai_confidence_threshold: float = 0.6,
        force_refresh: bool = False,   # ← add this (unused for now)
    ) -> List[Dict[str, Any]]:
        """
        Build contrastive pairs for DPO training.
        
        Args:
            min_star_pos: Minimum human star for positive examples
            max_star_neg: Maximum human star for negative examples
            limit: Maximum pairs to build
            max_negs_per_pos: Maximum negatives per positive
            shuffle: Whether to shuffle inputs
            ai_score_threshold: Minimum AI score to consider as positive
            ai_confidence_threshold: Minimum AI confidence to trust
            
        Returns:
            List of contrastive pairs with metadata
        """
        log.debug(
            f"Building knowledge pairs: pos≥{min_star_pos} (human) or ≥{ai_score_threshold} (AI), "
            f"neg≤{max_star_neg} (human) or ≤{100-ai_score_threshold} (AI), limit={limit}"
        )



        # Pull projected rows with assistant_text + ner/domains
        pos_turns = self._get_positive_turns(min_star_pos, ai_score_threshold)
        neg_turns = self._get_negative_turns(max_star_neg, ai_score_threshold)
        
        if not pos_turns or not neg_turns:
            log.warning("No positive or negative turns found.")
            return []

        if shuffle:
            random.shuffle(pos_turns)
            random.shuffle(neg_turns)

        # Bucket negatives by (conversation_id, primary_domain)
        neg_buckets: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
        for neg in neg_turns:
            key = (neg["conversation_id"], _primary_domain_from_row(neg))
            neg_buckets.setdefault(key, []).append(neg)

        log.debug(
            f"Pos={len(pos_turns)} ({self._count_human_pos(pos_turns)}/human, {self._count_ai_pos(pos_turns)}/AI) | "
            f"Neg={len(neg_turns)} ({self._count_human_neg(neg_turns)}/human, {self._count_ai_neg(neg_turns)}/AI) | "
            f"Buckets={len(neg_buckets)}"
        )

        pairs: List[Dict[str, Any]] = []
        seen: set[str] = set()  # dedupe by (pos_id, neg_id)
        
        for pi, pos in enumerate(tqdm(pos_turns, desc="Building knowledge pairs")):
            if len(pairs) >= limit:
                break

            user_prompt = (pos.get("goal_text") or "").strip()
            if not user_prompt or not pos.get("assistant_text", "").strip():
                continue

            # Find matching negatives
            key = (pos["conversation_id"], _primary_domain_from_row(pos))
            candidates = neg_buckets.get(key, [])
            if not candidates:
                continue

            # Filter by entity overlap
            pos_ents = set(_entity_texts_from_ner(pos.get("ner")))
            if not pos_ents:
                continue
                
            filtered = self._filter_by_entity_overlap(pos_ents, candidates)
            if not filtered:
                continue

            # Shuffle and select negatives
            if shuffle:
                random.shuffle(filtered)
            chosen = filtered[:max_negs_per_pos]

            # Process positive turn
            pos_emb = self._embed_turn_cached(pos)
            if pos_emb is None:
                continue

            # Log entity overlap drop rate
            if len(candidates) > 0 and len(filtered) == 0:
                log.debug(f"Entity overlap filtered {len(candidates)} candidates for pos_id={pos['id']}")

            # Build pairs
            for neg in chosen:
                if len(pairs) >= limit:
                    break
                if not neg.get("assistant_text", "").strip():
                    continue
                    
                neg_emb = self._embed_turn_cached(neg)
                if neg_emb is None:
                    continue

                pair_hash = hash_text(f"{pos['id']}:{neg['id']}")
                if pair_hash in seen:
                    continue
                seen.add(pair_hash)

                # Build the pair
                pair = self._build_pair(pos, neg, pair_hash)
                if pair:
                    pairs.append(pair)

            # Logging
            if (pi + 1) % 1000 == 0:
                log.debug(f"Processed {pi+1} positives → pairs={len(pairs)}")

        log.debug(f"✅ Built {len(pairs)} contrastive pairs.")
        return pairs

    # ---------------- internal helpers ----------------
    
    def _get_positive_turns(self, min_star_pos: int, ai_score_threshold: float) -> List[Dict[str, Any]]:
        """Get turns that qualify as positive examples (human or AI rated)"""
        human_pos = self.memory.chats.list_turns_with_texts(
            min_star=min_star_pos,
            require_assistant_text=True,
            require_nonempty_ner=True,
            limit=1_000_000
        )
        
        ai_pos = self.memory.chats.list_turns_with_texts(
            min_ai_score=ai_score_threshold,
            require_assistant_text=True,
            require_nonempty_ner=True,
            limit=1_000_000
        )
        
        # Deduplicate (a turn might have both human and AI scores)
        seen_ids = set()
        all_pos = []
        for turn in human_pos + ai_pos:
            if turn["id"] not in seen_ids:
                seen_ids.add(turn["id"])
                all_pos.append(turn)
                
        return all_pos
    
    def _get_negative_turns(self, max_star_neg: int, ai_score_threshold: float) -> List[Dict[str, Any]]:
        """Get turns that qualify as negative examples (human or AI rated)"""
        human_neg = self.memory.chats.list_turns_with_texts(
            max_star=max_star_neg,
            require_assistant_text=True,
            require_nonempty_ner=True,
            limit=1_000_000
        )
        
        ai_neg = self.memory.chats.list_turns_with_texts(
            max_ai_score=100 - ai_score_threshold,
            require_assistant_text=True,
            require_nonempty_ner=True,
            limit=1_000_000
        )
        
        # Deduplicate
        seen_ids = set()
        all_neg = []
        for turn in human_neg + ai_neg:
            if turn["id"] not in seen_ids:
                seen_ids.add(turn["id"])
                all_neg.append(turn)
                
        return all_neg
    
    def _count_human_pos(self, turns: List[Dict]) -> int:
        """Count human-rated positive turns"""
        return sum(1 for t in turns if t.get("star") is not None and t["star"] >= 2)
    
    def _count_ai_pos(self, turns: List[Dict]) -> int:
        """Count AI-rated positive turns"""
        return sum(1 for t in turns if t.get("ai_score") is not None and t["ai_score"] >= 70)
    
    def _count_human_neg(self, turns: List[Dict]) -> int:
        """Count human-rated negative turns"""
        return sum(1 for t in turns if t.get("star") is not None and t["star"] <= -1)
    
    def _count_ai_neg(self, turns: List[Dict]) -> int:
        """Count AI-rated negative turns"""
        return sum(1 for t in turns if t.get("ai_score") is not None and t["ai_score"] <= 30)
    
    def _filter_by_entity_overlap(
        self, 
        pos_ents: set, 
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter candidates by entity overlap with positive turn"""
        filtered = []
        for neg in candidates:
            neg_ents = set(_entity_texts_from_ner(neg.get("ner")))
            if len(pos_ents & neg_ents) >= self.min_entity_overlap:
                filtered.append(neg)
        return filtered
    
    def _build_pair(
        self,
        pos: Dict[str, Any],
        neg: Dict[str, Any],
        pair_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Build a single contrastive pair with all metadata"""
        # Goal text (from positive turn)
        user_prompt = (pos.get("goal_text") or "").strip()
        
        # Determine label source and confidence
        label_source, pair_weight, ai_conf = self._determine_label_source(pos, neg)
        
        # Check if we have similar human examples (for blending)
        has_similar_human = self._check_similar_human(pos, neg)
        
        # Build the pair
        return {
            # Goal/query embedding anchor
            "prompt": user_prompt,
            "goal_text": user_prompt,
            
            # Assistant outputs
            "output_a": pos["assistant_text"],  # preferred
            "output_b": neg["assistant_text"],  # non-preferred
            
            # Label source and confidence
            "label_source": label_source,
            "pair_weight": pair_weight,
            "ai_conf": ai_conf,
            "has_similar_human": has_similar_human,
            
            # Numeric labels (for auditing)
            "value_a": self._get_value(pos, label_source),
            "value_b": self._get_value(neg, label_source),
            
            # Auxiliary features
            "meta_a": self._build_meta(pos, label_source, "a"),
            "meta_b": self._build_meta(neg, label_source, "b"),
            
            # Tracing
            "domain": _primary_domain_from_row(pos),
            "goal_id": None,
            "pos_id": pos["id"],
            "neg_id": neg["id"],
            "pair_hash": pair_hash,
            "created_at": datetime.now().isoformat(),
        }
    
    def _determine_label_source(
        self,
        pos: Dict[str, Any],
        neg: Dict[str, Any]
    ) -> Tuple[str, float, float]:
        """
        Determine label source and confidence for the pair.
        
        Returns:
            (label_source, pair_weight, ai_conf)
        """
        # If both turns have human stars, use human source
        if pos.get("star") is not None and neg.get("star") is not None:
            return "human", 1.0, 0.0
        
        # If both turns have AI scores above confidence threshold, use AI source
        if (pos.get("ai_score") is not None and 
            neg.get("ai_score") is not None and
            pos.get("ai_score_conf", 0.0) >= 0.6 and
            neg.get("ai_score_conf", 0.0) >= 0.6):
            return "ai", 0.35, (pos["ai_score_conf"] + neg["ai_score_conf"]) / 2
        
        # Mixed case: prefer human if available on either side
        if pos.get("star") is not None or neg.get("star") is not None:
            return "human", 1.0, 0.0
            
        # Fallback to AI with lower weight if confidence is marginal
        if pos.get("ai_score") is not None and neg.get("ai_score") is not None:
            avg_conf = (pos.get("ai_score_conf", 0.0) + neg.get("ai_score_conf", 0.0)) / 2
            return "ai", max(0.1, 0.35 * avg_conf), avg_conf
        
        # Should not happen given our filtering, but safety check
        return "unknown", 0.0, 0.0
    
    def _check_similar_human(self, pos: Dict[str, Any], neg: Dict[str, Any]) -> bool:
        """
        Check if there are similar human-labeled examples for this context.
        
        This helps with blending strategy at inference time.
        """
        # Check if either turn has human stars
        if pos.get("star") is not None or neg.get("star") is not None:
            return True
            
        # Check if there are other human-labeled turns in the same conversation/domain
        domain = _primary_domain_from_row(pos)
        similar_human = self.memory.chats.count_human_labeled_turns(
            conversation_id=pos["conversation_id"],
            domain=domain,
            min_star=2,
            max_star=-1
        )
        return similar_human > 0
    
    def _get_value(self, turn: Dict[str, Any], label_source: str) -> float:
        """Get normalized value for the turn based on label source"""
        if label_source == "human" and turn.get("star") is not None:
            # Human: -5..+5 → 0..1 (0.5 = neutral)
            return (turn["star"] + 5) / 10.0
        elif label_source == "ai" and turn.get("ai_score") is not None:
            # AI: 0-100 → 0-1 (calibrated if calibrator available)
            raw_score = turn["ai_score"]
            if self._calibrator:
                return self._calibrator.calibrate(raw_score)
            return raw_score / 100.0
        return 0.5  # neutral fallback
    
    def _build_meta(self, turn: Dict[str, Any], label_source: str, role: str) -> Dict[str, float]:
        """Build metadata dictionary for a turn"""
        # Base features
        meta = {
            "human_stars": to_float(turn.get("star", 0)) if label_source == "human" else 0.0,
            "ai_score": to_float(turn.get("ai_score", 50.0)),
            "ai_score_conf": to_float(turn.get("ai_score_conf", 0.7)),
            "artifact_quality": to_float(turn.get("artifact_quality", 0.0)),
            "turn_pos_ratio": to_float(turn.get("order_index", 0)) / max(1, to_float(turn.get("conv_length", 1))),
            "has_retrieval": 1.0 if turn.get("has_retrieval") else 0.0,
            "retrieval_fidelity": to_float(turn.get("retrieval_fidelity", 0.0)),
            "text_len_norm": _len_norm(turn.get("assistant_text", "")),
        }
        
        # Add calibrated AI score if available and relevant
        if label_source == "ai" and turn.get("ai_score") is not None and self._calibrator:
            meta["calibrated_ai_score"] = self._calibrator.calibrate(turn["ai_score"])
        
        # Add pseudo stars if available (from active learning)
        if turn.get("pseudo_stars") is not None:
            meta["pseudo_stars"] = float(turn["pseudo_stars"])
        else:
            meta["pseudo_stars"] = 0.0
            
        return meta

    def _embed_turn_cached(self, row: Dict[str, Any]) -> Optional[np.ndarray]:
        """Get embedding from cache or compute and cache it"""
        tid = int(row["id"])
        if tid in self._emb_cache:
            return self._emb_cache[tid]
        emb = self._embed_turn(row)
        if emb is not None:
            self._emb_cache[tid] = emb
        return emb

    def _embed_turn(self, row: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Embed the assistant response text with L2 normalization.
        
        Returns:
            L2-normalized embedding vector or None if invalid
        """
        text = (row.get("assistant_text") or "").strip()
        if not text:
            return None

        try:
            # Get embedding
            vec = self.memory.embedding.get_or_create(text)
            if vec is None:
                return None

            # Convert to numpy and normalize
            vec = np.asarray(vec, dtype=np.float32)
            n = float(np.linalg.norm(vec))
            
            # Validate
            if not np.isfinite(n) or n == 0.0:
                return None
                
            return vec / n
        except Exception as e:
            log.error(f"Embedding failed for turn {row.get('id')}: {str(e)}")
            return None 