# stephanie/utils/similarity_utils.py

"""
similarity_utils.py
-------------------
Utility functions for computing similarity scores between text items
based on their embedding vectors.

This module is primarily used by scorers (e.g., ProximityScorer) to
analyze semantic relationships between hypotheses, goals, and other
knowledge units in Stephanie.

Functions:
    - compute_similarity_matrix: Given a list of texts, computes
      pairwise cosine similarity scores using embeddings retrieved
      from memory.

Notes:
    - Embeddings are retrieved from `memory.embedding.get_or_create`.
    - Any text without an embedding is skipped, with a log entry emitted.
    - Returns a sorted list of similarity tuples, highest similarity first.
"""

from __future__ import annotations
import itertools
import numpy as np
import re
import math
from typing import Sequence, Union, Optional, Tuple, List


TextLike = Union[str, Sequence[str]]

# ---------- core tokenization ----------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def compute_similarity_matrix(
    texts: list[str], memory, logger
) -> list[tuple[str, str, float]]:
    """
    Compute pairwise cosine similarity between embeddings of input texts.

    Args:
        texts (list[str]): List of text strings to compare.
        memory: Stephanie memory object with an embedding interface.
                Expected to have `memory.embedding.get_or_create(text)` 
                which returns a vector or None.
        logger: Logger used to capture warnings or missing embeddings.

    Returns:
        list[tuple[str, str, float]]:
            A list of tuples of the form (text1, text2, similarity),
            sorted in descending order of similarity score.
    """
    vectors = []      # store embedding vectors
    valid_texts = []  # store only those texts that had valid embeddings

    # 1. Generate embeddings for all texts
    for h in texts:
        vec = memory.embedding.get_or_create(h)
        if vec is None:
            # Log missing embeddings but continue gracefully
            logger.log("MissingEmbedding", {"texts_snippet": h[:60]})
            continue
        vectors.append(vec)
        valid_texts.append(h)

    similarities = []

    # 2. Compute cosine similarity for every unique pair
    for i, j in itertools.combinations(range(len(valid_texts)), 2):
        h1 = valid_texts[i]
        h2 = valid_texts[j]
        sim = float(
            np.dot(vectors[i], vectors[j])
            / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
        )
        similarities.append((h1, h2, sim))

    # 3. Sort results by similarity score (highest first)
    return sorted(similarities, key=lambda x: x[2], reverse=True)



def _tokens(s: str) -> List[str]:
    if not s:
        return []
    return _WORD_RE.findall(s.lower())

# ---------- cosine (no sklearn dependency) ----------
def _cosine(u, v) -> float:
    try:
        if np is not None:
            u = np.asarray(u, dtype=float)
            v = np.asarray(v, dtype=float)
            dot = float((u * v).sum())
            nu = float(np.linalg.norm(u))
            nv = float(np.linalg.norm(v))
        else:
            dot = sum(ui * vi for ui, vi in zip(u, v))
            nu = math.sqrt(sum(ui * ui for ui in u))
            nv = math.sqrt(sum(vi * vi for vi in v))
        denom = nu * nv
        return (dot / denom) if denom > 0 else 0.0
    except Exception:
        return 0.0

# ---------- lexical signal ----------
def lexical_f1(a_text: str, b_text: str) -> float:
    a = set(_tokens(a_text))
    b = set(_tokens(b_text))
    if not a or not b:
        return 0.0
    inter = len(a & b)
    prec = inter / len(a)
    rec  = inter / len(b)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

# ---------- embedding signal (uses your memory.embedding) ----------
def embedding_sim(a_text: str, b_text: str, memory) -> Optional[float]:
    try:
        emb = getattr(memory, "embedding", None)
        if emb is None:
            return None
        ea = emb.get_or_create(a_text)
        eb = emb.get_or_create(b_text)
        if ea is None or eb is None:
            return None
        sim = _cosine(ea, eb)  # [-1, 1]
        return 0.5 * (sim + 1.0)  # → [0, 1]
    except Exception:
        return None

# ---------- unified API ----------
def overlap_score(
    snippets: TextLike,
    target: str,
    *,
    memory=None,
    strategy: str = "hybrid",   # "lexical" | "semantic" | "hybrid"
    alpha: float = 0.5,         # weight for lexical in hybrid
) -> float:
    """
    Score evidence snippets vs target.
    - snippets: str or list[str]
    - strategy:
        - "lexical": lexical F1 only
        - "semantic": embedding cosine only (requires memory.embedding)
        - "hybrid": alpha * lexical + (1-alpha) * semantic (fallback to lexical if no embeddings)
    Returns best score across snippets in [0,1].
    """
    # normalize input
    if isinstance(snippets, str):
        texts = [snippets]
    else:
        texts = [s for s in (snippets or []) if isinstance(s, str)]

    if not texts or not (target or "").strip():
        return 0.0

    best = 0.0
    for s in texts:
        lex = lexical_f1(s, target)
        if strategy == "lexical":
            score = lex
        elif strategy == "semantic":
            sem = embedding_sim(s, target, memory)
            score = (sem if sem is not None else 0.0)
        else:  # hybrid
            sem = embedding_sim(s, target, memory)
            score = alpha * lex + (1 - alpha) * (sem if sem is not None else lex)
        if score > best:
            best = score
    return float(best)

def rank_snippets(
    snippets: Sequence[str],
    target: str,
    *,
    memory=None,
    strategy: str = "hybrid",
    alpha: float = 0.5,
) -> List[Tuple[int, float]]:
    """
    Return [(idx, score)] sorted desc by score.
    """
    scores = []
    for i, s in enumerate(snippets or []):
        sc = overlap_score(s, target, memory=memory, strategy=strategy, alpha=alpha)
        scores.append((i, sc))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def best_snippet(
    snippets: Sequence[str],
    target: str,
    *,
    memory=None,
    strategy: str = "hybrid",
    alpha: float = 0.5,
) -> Tuple[Optional[str], float]:
    if not snippets:
        return None, 0.0
    ranked = rank_snippets(snippets, target, memory=memory, strategy=strategy, alpha=alpha)
    idx, sc = ranked[0] if ranked else (None, 0.0)
    return (snippets[idx] if idx is not None else None, sc)
