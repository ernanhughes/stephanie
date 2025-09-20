# stephanie/dataloaders/knowledge_pair_builder.py
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import logging
import random
import hashlib
import numpy as np
import re

_logger = logging.getLogger(__name__)

def _simple_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', (text or '').strip())
    return [p for p in parts if p]

def _entity_texts_from_ner(ner) -> List[str]:
    # ner is list[dict] like {"text": "...", "label": "...", "start": int, "end": int}
    if not ner:
        return []
    out = []
    for e in ner if isinstance(ner, list) else []:
        t = (e.get("text") or "").strip().lower()
        if len(t) >= 2:
            out.append(t)
    return list({t for t in out})

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

        # bucket negatives by (conversation_id, primary_domain) — adjust if you add goal/casebook
        def _primary_domain(row) -> Optional[str]:
            doms = row.get("domains") or []
            for d in doms:
                name = str(d.get("domain") or "").strip().lower()
                if name:
                    return name
            return None

        neg_buckets: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
        for neg in neg_turns:
            key = (neg["conversation_id"], _primary_domain(neg))
            neg_buckets.setdefault(key, []).append(neg)

        pairs: List[Dict[str, Any]] = []
        for pi, pos in enumerate(pos_turns):
            if len(pairs) >= limit:
                break

            key = (pos["conversation_id"], _primary_domain(pos))
            candidates = neg_buckets.get(key, [])
            if not candidates:
                continue

            pos_ents = set(_entity_texts_from_ner(pos.get("ner")))
            if not pos_ents:
                continue

            filtered = []
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

                pair = {
                    "pos_text": pos["assistant_text"],
                    "neg_text": neg["assistant_text"],
                    "pos_emb": pos_emb,
                    "neg_emb": neg_emb,
                    "domain": _primary_domain(pos),
                    "goal_id": None,  # fill when you add it to schema
                    "pos_id": pos["id"],
                    "neg_id": neg["id"],
                }
                pair["pair_hash"] = hashlib.sha1(f"{pair['pos_id']}:{pair['neg_id']}".encode("utf-8")).hexdigest()[:16]
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
        Embed the assistant response, focusing on sentences containing NER entities.
        """
        text = (row.get("assistant_text") or "").strip()
        if not text:
            return None

        ents = _entity_texts_from_ner(row.get("ner"))

        # choose embedder: memory.embedding.get_or_create
        def _embed(s: str):
            # FIX: return the vector
            return self.memory.embedding.get_or_create(s)

        if not ents:
            vec = _embed(text)
        else:
            # Extract sentences containing entities
            try:
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(text)
            except Exception:
                sentences = _simple_sentences(text)

            lower_ents = set(ents)
            relevant = [s for s in sentences if any(e in s.lower() for e in lower_ents)]
            if not relevant:
                relevant = [text]

            embs = [np.asarray(_embed(s), dtype=np.float32) for s in relevant]
            embs = [e for e in embs if e is not None and np.isfinite(e).all()]
            if not embs:
                return None
            vec = np.mean(embs, axis=0)

        vec = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(vec))
        if n == 0.0 or not np.isfinite(n):
            return None
        return vec / n
