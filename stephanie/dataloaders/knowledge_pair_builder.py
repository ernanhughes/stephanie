# stephanie/dataloaders/knowledge_pair_builder.py
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import logging
import random
import hashlib
import numpy as np
import json
import re

_logger = logging.getLogger(__name__)

def _simple_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', (text or '').strip())
    return [p for p in parts if p]

def _entity_texts_from_ner(ner) -> List[str]:
    """
    ner supports: list[dict], JSON string, or None.
    Returns unique, lower-cased entity surface forms.
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
    domains supports: list[dict], JSON string, or None.
    Returns the first non-empty domain name (lower-cased).
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

class KnowledgePairBuilder:
    """
    Builds contrastive (pos, neg) pairs for DPO-lite training.
    Uses assistant response text for embeddings + NER for entity-aware pooling.
    """

    def __init__(self, memory: Any, min_entity_overlap: int = 1, seed: int = 1337):
        self.memory = memory
        self.min_entity_overlap = int(min_entity_overlap)
        random.seed(seed)
        self._emb_cache: dict[int, np.ndarray] = {}  # turn_id -> emb

    def build_pairs(
        self,
        min_star_pos: int = 1,
        max_star_neg: int = -1,
        limit: int = 50_000,
        max_negs_per_pos: int = 3,
        shuffle: bool = True,
    ) -> List[Dict[str, Any]]:
        _logger.info(f"Building knowledge pairs: pos≥{min_star_pos}, neg≤{max_star_neg}, limit={limit}")


        # Pull projected rows with assistant_text + ner/domains (no lazy loads)
        pos_turns = self.memory.chats.list_turns_with_texts(
            min_star=min_star_pos, require_assistant_text=True, require_nonempty_ner=True, limit=1_000_000
        )
        neg_turns = self.memory.chats.list_turns_with_texts(
            max_star=max_star_neg, require_assistant_text=True, require_nonempty_ner=True, limit=1_000_000
        )

        if shuffle:
            random.shuffle(pos_turns)
            random.shuffle(neg_turns)

        if not pos_turns or not neg_turns:
            _logger.warning("No positive or negative turns found.")
            return []

        # Bucket negatives by (conversation_id, primary_domain)
        neg_buckets: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
        for neg in neg_turns:
            key = (neg["conversation_id"], _primary_domain_from_row(neg))
            neg_buckets.setdefault(key, []).append(neg)

        _logger.info(f"Pos={len(pos_turns)} Neg={len(neg_turns)} Buckets={len(neg_buckets)}")

        pairs: List[Dict[str, Any]] = []
        seen: set[str] = set()  # dedupe by (pos_id, neg_id)
        for pi, pos in enumerate(pos_turns):
            if len(pairs) >= limit:
                break

            key = (pos["conversation_id"], _primary_domain_from_row(pos))
            candidates = neg_buckets.get(key, [])
            if not candidates:
                continue

            pos_ents = set(_entity_texts_from_ner(pos.get("ner")))
            if not pos_ents:
                continue

            filtered: List[Dict[str, Any]] = []
            for neg in candidates:
                neg_ents = set(_entity_texts_from_ner(neg.get("ner")))
                if len(pos_ents & neg_ents) >= self.min_entity_overlap:
                    filtered.append(neg)

            if not filtered:
                continue

            if shuffle:
                random.shuffle(filtered)
            chosen = filtered[:max_negs_per_pos]

            pos_emb = self._embed_turn_cached(pos)
            if pos_emb is None:
                continue

            for neg in chosen:
                if len(pairs) >= limit:
                    break
                neg_emb = self._embed_turn_cached(neg)
                if neg_emb is None:
                    continue

                pair_hash = hashlib.sha1(f"{pos['id']}:{neg['id']}".encode("utf-8")).hexdigest()[:16]
                if pair_hash in seen:
                    continue
                seen.add(pair_hash)

                # inside build_pairs(), replace the `pair = { ... }` you have now with:

                # A is the preferred sample (pos), B is the non-preferred (neg)
                user_prompt = (pos.get("goal_text") or "").strip()  # good "goal" signal for the trainer

                def _len_norm(t: str, cap: int = 2000) -> float:
                    t = t or ""
                    n = min(len(t), cap)
                    return n / cap

                pair = {
                    # goal / query embedding anchor
                    "prompt": user_prompt,                     # trainer will embed this as G

                    # assistant outputs
                    "output_a": pos["assistant_text"],         # preferred
                    "output_b": neg["assistant_text"],         # non-preferred

                    # optional numeric labels if you have them (not used by loss, but handy for audits)
                    "value_a": float(max(0, pos.get("star", 1))),   # e.g., star as a soft label
                    "value_b": float(min(0, neg.get("star", -1))),

                    # aux features (names must match KnowledgeTrainer.aux_features)
                    "meta_a": {
                        "human_stars": float(pos.get("star", 1)),
                        "pseudo_stars": 0.0,
                        "artifact_quality": 0.0,
                        "turn_pos_ratio": 1.0,
                        "has_retrieval": 0.0,
                        "retrieval_fidelity": 0.0,
                        "text_len_norm": _len_norm(pos["assistant_text"]),
                    },
                    "meta_b": {
                        "human_stars": float(neg.get("star", -1)),
                        "pseudo_stars": 0.0,
                        "artifact_quality": 0.0,
                        "turn_pos_ratio": 0.0,
                        "has_retrieval": 0.0,
                        "retrieval_fidelity": 0.0,
                        "text_len_norm": _len_norm(neg["assistant_text"]),
                    },

                    # optional tracing
                    "domain": _primary_domain_from_row(pos),
                    "goal_id": None,
                    "pos_id": pos["id"],
                    "neg_id": neg["id"],
                    "pair_hash": hashlib.sha1(f"{pos['id']}:{neg['id']}".encode("utf-8")).hexdigest()[:16],
                }
                pairs.append(pair)

            if (pi + 1) % 1000 == 0:
                _logger.info(f"Processed {pi+1} positives → pairs={len(pairs)}")

        _logger.info(f"✅ Built {len(pairs)} contrastive pairs.")
        return pairs

    # ---------------- internal ----------------

    def _embed_turn_cached(self, row: Dict[str, Any]) -> Optional[np.ndarray]:
        tid = int(row["id"])
        if tid in self._emb_cache:
            return self._emb_cache[tid]
        emb = self._embed_turn(row)
        if emb is None:
            return None
        self._emb_cache[tid] = emb
        return emb


    def _embed_turn(self, row: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        v1: Embed the full assistant response text.
        Keeps simple guards for empty text / NaNs and L2-normalizes.
        """
        text = (row.get("assistant_text") or "").strip()
        if not text:
            return None

        # Must return a vector (list/np array)
        vec = self.memory.embedding.get_or_create(text)
        if vec is None:
            return None

        vec = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(vec))
        if not np.isfinite(n) or n == 0.0:
            return None

        return vec / n
