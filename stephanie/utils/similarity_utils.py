# stephanie/utils/similarity_utils.py

"""
similarity_utils.py
-------------------
Utility functions for computing similarity scores and distance metrics between text items
based on their embedding vectors and lexical features.

This module provides comprehensive similarity analysis capabilities including:
- Semantic similarity using embedding vectors
- Lexical similarity using token-based F1 scores
- Hybrid approaches combining multiple signals
- Distance metrics for vector comparison

Primary use cases:
- ProximityScorer for semantic relationships between hypotheses, goals, and knowledge units
- Evidence retrieval and ranking
- Semantic search and clustering

Key features:
- Multiple similarity strategies (lexical, semantic, hybrid)
- Robust error handling and logging
- Memory-efficient pairwise comparisons
- Configurable weighting for hybrid approaches

Notes:
    - Embeddings are retrieved from `memory.embedding.get_or_create`.
    - Any text without an embedding is handled gracefully with appropriate logging.
    - All similarity scores are normalized to [0,1] range where applicable.
"""

from __future__ import annotations

import itertools
import logging
import math
import re
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

TextLike = Union[str, Sequence[str]]

# ---------- core tokenization ----------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def compute_similarity_matrix(
    texts: list[str], memory, logger
) -> list[tuple[str, str, float]]:
    """
    Compute pairwise cosine similarity between embeddings of input texts.

    This function generates a comprehensive similarity matrix for all unique
    pairs of input texts, using their embedding vectors retrieved from memory.

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

    Example:
        >>> texts = ["hello world", "hi world", "completely different"]
        >>> similarities = compute_similarity_matrix(texts, memory, log)
        >>> for text1, text2, sim in similarities:
        ...     print(f"{text1} <-> {text2}: {sim:.3f}")
    """
    log.debug(f"Computing similarity matrix for {len(texts)} texts")
    
    vectors = []      # store embedding vectors
    valid_texts = []  # store only those texts that had valid embeddings

    # 1. Generate embeddings for all texts
    for i, text in enumerate(texts):
        log.debug(f"Getting embedding for text {i+1}/{len(texts)}: '{text[:50]}...'")
        vec = memory.embedding.get_or_create(text)
        if vec is None:
            # Log missing embeddings but continue gracefully
            logger.log("MissingEmbedding", {"texts_snippet": text[:60]})
            log.warning(f"Could not generate embedding for text: '{text[:50]}...'")
            continue
        vectors.append(vec)
        valid_texts.append(text)

    log.debug(f"Successfully generated embeddings for {len(valid_texts)}/{len(texts)} texts")

    similarities = []
    total_pairs = math.comb(len(valid_texts), 2) if len(valid_texts) > 1 else 0
    log.debug(f"Computing {total_pairs} pairwise similarities")

    # 2. Compute cosine similarity for every unique pair
    for pair_num, (i, j) in enumerate(itertools.combinations(range(len(valid_texts)), 2)):
        if pair_num % 100 == 0:  # Log progress every 100 pairs
            log.debug(f"Processed {pair_num}/{total_pairs} pairs")
            
        h1 = valid_texts[i]
        h2 = valid_texts[j]
        
        try:
            sim = float(
                np.dot(vectors[i], vectors[j])
                / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
            )
            similarities.append((h1, h2, sim))
        except Exception as e:
            log.error(f"Error computing similarity between '{h1[:30]}...' and '{h2[:30]}...': {e}")
            # Continue with other pairs even if one fails
            continue

    # 3. Sort results by similarity score (highest first)
    sorted_similarities = sorted(similarities, key=lambda x: x[2], reverse=True)
    
    log.debug(f"Computed {len(sorted_similarities)} similarity pairs. "
              f"Max similarity: {sorted_similarities[0][2] if sorted_similarities else 0:.3f}, "
              f"Min similarity: {sorted_similarities[-1][2] if sorted_similarities else 0:.3f}")
    
    return sorted_similarities


def _tokens(s: str) -> List[str]:
    """
    Extract tokens from a string using simple word-based tokenization.
    
    Args:
        s (str): Input string to tokenize
        
    Returns:
        List[str]: List of lowercase alphanumeric tokens
        
    Example:
        >>> _tokens("Hello World! Test 123")
        ['hello', 'world', 'test', '123']
    """
    if not s:
        log.debug("Empty string provided for tokenization")
        return []
    
    tokens = _WORD_RE.findall(s.lower())
    log.debug(f"Extracted {len(tokens)} tokens from string: '{s[:50]}...'")
    return tokens


def cosine(u, v) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Robust implementation that works with both numpy arrays and Python lists,
    with comprehensive error handling.
    
    Args:
        u: First vector (numpy array, list, or iterable)
        v: Second vector (numpy array, list, or iterable)
        
    Returns:
        float: Cosine similarity in range [-1, 1] (0 on error)
        
    Example:
        >>> cosine([1, 0, 0], [1, 0, 0])
        1.0
        >>> cosine([1, 0], [0, 1])
        0.0
    """
    log.debug(f"Computing cosine similarity between vectors of lengths {len(u)} and {len(v)}")
    
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
        
        if denom <= 0:
            log.warning("Zero or negative denominator in cosine calculation")
            return 0.0
            
        result = dot / denom
        log.debug(f"Cosine similarity: {result:.4f}")
        return result
        
    except Exception as e:
        log.error(f"Error computing cosine similarity: {e}")
        return 0.0


def lexical_f1(a_text: str, b_text: str) -> float:
    """
    Compute lexical F1 score between two text strings.
    
    This measures token overlap using F1 metric, which balances
    precision and recall of token matches.
    
    Args:
        a_text (str): First text string
        b_text (str): Second text string
        
    Returns:
        float: F1 score in range [0, 1]
        
    Example:
        >>> lexical_f1("hello world", "world hello")
        1.0
        >>> lexical_f1("hello world", "goodbye world")
        0.666...
    """
    log.debug(f"Computing lexical F1 between texts: '{a_text[:30]}...' and '{b_text[:30]}...'")
    
    a = set(_tokens(a_text))
    b = set(_tokens(b_text))
    
    if not a or not b:
        log.debug("One or both texts have no tokens after processing")
        return 0.0
        
    inter = len(a & b)
    prec = inter / len(a)
    rec = inter / len(b)
    
    result = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    log.debug(f"Lexical F1: {result:.4f} (precision: {prec:.4f}, recall: {rec:.4f})")
    return result


def embedding_sim(a_text: str, b_text: str, memory) -> Optional[float]:
    """
    Compute semantic similarity between two texts using embeddings.
    
    Transforms cosine similarity from [-1,1] to [0,1] range for
    easier interpretation and combination with other metrics.
    
    Args:
        a_text (str): First text string
        b_text (str): Second text string
        memory: Memory object with embedding interface
        
    Returns:
        Optional[float]: Normalized similarity in [0,1] or None on error
        
    Example:
        >>> embedding_sim("machine learning", "artificial intelligence", memory)
        0.85
    """
    log.debug(f"Computing embedding similarity between: '{a_text[:30]}...' and '{b_text[:30]}...'")
    
    try:
        emb = getattr(memory, "embedding", None)
        if emb is None:
            log.error("Memory object has no embedding interface")
            return None
            
        ea = emb.get_or_create(a_text)
        eb = emb.get_or_create(b_text)
        
        if ea is None or eb is None:
            log.warning(f"Could not generate embeddings for one or both texts")
            return None
            
        sim = cosine(ea, eb)  # [-1, 1]
        normalized_sim = 0.5 * (sim + 1.0)  # → [0, 1]
        
        log.debug(f"Embedding similarity: {normalized_sim:.4f} (raw cosine: {sim:.4f})")
        return normalized_sim
        
    except Exception as e:
        log.error(f"Error computing embedding similarity: {e}")
        return None


def overlap_score(
    snippets: TextLike,
    target: str,
    *,
    memory=None,
    strategy: str = "hybrid",   # "lexical" | "semantic" | "hybrid"
    alpha: float = 0.5,         # weight for lexical in hybrid
) -> float:
    """
    Score evidence snippets against a target using specified strategy.
    
    Computes the best matching score between any snippet and the target,
    using lexical, semantic, or hybrid similarity measures.
    
    Args:
        snippets: Single string or list of strings to score
        target: Target string to compare against
        memory: Memory object for embedding access (required for semantic/hybrid)
        strategy: Similarity strategy - "lexical", "semantic", or "hybrid"
        alpha: Weight for lexical component in hybrid strategy (0-1)
        
    Returns:
        float: Best similarity score in range [0,1]
        
    Raises:
        ValueError: If invalid strategy provided
        
    Example:
        >>> overlap_score(["machine learning", "AI"], "artificial intelligence", 
        ...               memory=memory, strategy="hybrid", alpha=0.3)
        0.72
    """
    log.debug(f"Computing overlap score with strategy '{strategy}', alpha={alpha}, "
              f"target: '{target[:50]}...'")
    
    # Validate strategy
    valid_strategies = {"lexical", "semantic", "hybrid"}
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")
    
    # Normalize input
    if isinstance(snippets, str):
        texts = [snippets]
    else:
        texts = [s for s in (snippets or []) if isinstance(s, str)]

    if not texts:
        log.debug("No valid snippets provided")
        return 0.0
        
    if not (target or "").strip():
        log.debug("Empty target provided")
        return 0.0

    log.debug(f"Scoring {len(texts)} snippets against target")
    best = 0.0
    
    for i, snippet in enumerate(texts):
        lex = lexical_f1(snippet, target)
        
        if strategy == "lexical":
            score = lex
            log.debug(f"Snippet {i+1}: lexical score = {score:.4f}")
            
        elif strategy == "semantic":
            sem = embedding_sim(snippet, target, memory)
            score = sem if sem is not None else 0.0
            log.debug(f"Snippet {i+1}: semantic score = {score:.4f}")
            
        else:  # hybrid
            sem = embedding_sim(snippet, target, memory)
            semantic_component = sem if sem is not None else lex
            score = alpha * lex + (1 - alpha) * semantic_component
            log.debug(f"Snippet {i+1}: hybrid score = {score:.4f} "
                      f"(lexical: {lex:.4f}, semantic: {semantic_component:.4f})")
        
        if score > best:
            best = score
            log.debug(f"New best score: {best:.4f}")

    log.debug(f"Final best overlap score: {best:.4f}")
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
    Rank snippets by their similarity to target.
    
    Args:
        snippets: Sequence of text snippets to rank
        target: Target text to compare against
        memory: Memory object for embedding access
        strategy: Similarity strategy
        alpha: Weight for lexical component in hybrid strategy
        
    Returns:
        List[Tuple[int, float]]: List of (index, score) pairs sorted descending by score
        
    Example:
        >>> rank_snippets(["cat", "dog", "elephant"], "animal", memory=memory)
        [(0, 0.8), (1, 0.7), (2, 0.3)]
    """
    log.debug(f"Ranking {len(snippets)} snippets against target: '{target[:50]}...'")
    
    scores = []
    for i, snippet in enumerate(snippets or []):
        score = overlap_score(snippet, target, memory=memory, strategy=strategy, alpha=alpha)
        scores.append((i, score))
    
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Log ranking summary
    if ranked:
        log.debug(f"Ranking complete. Top score: {ranked[0][1]:.4f}, "
                  f"Bottom score: {ranked[-1][1]:.4f}")
    else:
        log.debug("No snippets to rank")
        
    return ranked


def best_snippet(
    snippets: Sequence[str],
    target: str,
    *,
    memory=None,
    strategy: str = "hybrid",
    alpha: float = 0.5,
) -> Tuple[Optional[str], float]:
    """
    Find the best matching snippet for a target.
    
    Args:
        snippets: Sequence of text snippets
        target: Target text to match
        memory: Memory object for embedding access
        strategy: Similarity strategy
        alpha: Weight for lexical component in hybrid strategy
        
    Returns:
        Tuple[Optional[str], float]: (best_snippet, score) or (None, 0.0) if no snippets
        
    Example:
        >>> best_snippet(["cat", "dog", "elephant"], "animal", memory=memory)
        ("cat", 0.8)
    """
    log.debug(f"Finding best snippet from {len(snippets)} options for target: '{target[:50]}...'")
    
    if not snippets:
        log.debug("No snippets provided")
        return None, 0.0
        
    ranked = rank_snippets(snippets, target, memory=memory, strategy=strategy, alpha=alpha)
    idx, score = ranked[0] if ranked else (None, 0.0)
    result = snippets[idx] if idx is not None else None
    
    log.debug(f"Best snippet: '{result[:50]}...' with score: {score:.4f}")
    return result, score


def euclidean_distance(u, v) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Args:
        u: First vector
        v: Second vector
        
    Returns:
        float: Non-negative distance (0 means identical)
        
    Example:
        >>> euclidean_distance([0, 0], [3, 4])
        5.0
    """
    log.debug(f"Computing Euclidean distance between vectors of lengths {len(u)} and {len(v)}")
    
    try:
        if np is not None:
            u_arr = np.asarray(u, dtype=float)
            v_arr = np.asarray(v, dtype=float)
            distance = float(np.linalg.norm(u_arr - v_arr))
        else:
            diff = [float(ua) - float(va) for ua, va in zip(u, v)]
            distance = math.sqrt(sum(d * d for d in diff))
            
        log.debug(f"Euclidean distance: {distance:.4f}")
        return distance
        
    except Exception as e:
        log.error(f"Error computing Euclidean distance: {e}")
        return 0.0


def huber_loss(u, v, delta: float = 1.0) -> float:
    """
    Compute mean Huber loss between two vectors.
    
    Huber loss is less sensitive to outliers than MSE and provides
    a smooth transition between L1 and L2 loss.
    
    Args:
        u, v: Vectors to compare
        delta: Threshold parameter for Huber loss
        
    Returns:
        float: Non-negative Huber loss (0 means identical)
        
    Example:
        >>> huber_loss([1, 2], [1, 3], delta=1.0)
        0.5
    """
    log.debug(f"Computing Huber loss with delta={delta} for vectors of lengths {len(u)} and {len(v)}")
    
    try:
        if np is not None:
            u_arr = np.asarray(u, dtype=float)
            v_arr = np.asarray(v, dtype=float)
            diff = u_arr - v_arr
            abs_diff = np.abs(diff)
            quadratic = np.minimum(abs_diff, delta)
            linear = abs_diff - quadratic
            loss = 0.5 * quadratic**2 + delta * linear
            result = float(loss.mean())
        else:
            diffs = [abs(float(ua) - float(va)) for ua, va in zip(u, v)]
            if not diffs:
                return 0.0
            total = 0.0
            for d in diffs:
                q = min(d, delta)
                l = d - q
                total += 0.5 * q * q + delta * l
            result = total / len(diffs)
            
        log.debug(f"Huber loss: {result:.4f}")
        return result
        
    except Exception as e:
        log.error(f"Error computing Huber loss: {e}")
        return 0.0


def jaccard_similarity(a_text: str, b_text: str) -> float:
    """
    Compute Jaccard similarity between two texts based on token overlap.
    
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    
    Args:
        a_text (str): First text string
        b_text (str): Second text string
        
    Returns:
        float: Jaccard similarity in range [0, 1]
        
    Example:
        >>> jaccard_similarity("hello world", "world hello")
        1.0
        >>> jaccard_similarity("hello world", "goodbye world")
        0.5
    """
    log.debug(f"Computing Jaccard similarity between: '{a_text[:30]}...' and '{b_text[:30]}...'")
    
    a_tokens = set(_tokens(a_text))
    b_tokens = set(_tokens(b_text))
    
    if not a_tokens and not b_tokens:
        log.debug("Both texts have no tokens")
        return 1.0  # Both empty texts are considered identical
    elif not a_tokens or not b_tokens:
        log.debug("One text has no tokens")
        return 0.0
        
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    
    similarity = intersection / union
    log.debug(f"Jaccard similarity: {similarity:.4f} (intersection: {intersection}, union: {union})")
    return similarity


def manhattan_distance(u, v) -> float:
    """
    Compute Manhattan distance (L1 norm) between two vectors.
    
    Args:
        u: First vector
        v: Second vector
        
    Returns:
        float: Non-negative Manhattan distance
        
    Example:
        >>> manhattan_distance([0, 0], [3, 4])
        7.0
    """
    log.debug(f"Computing Manhattan distance between vectors of lengths {len(u)} and {len(v)}")
    
    try:
        if np is not None:
            u_arr = np.asarray(u, dtype=float)
            v_arr = np.asarray(v, dtype=float)
            distance = float(np.sum(np.abs(u_arr - v_arr)))
        else:
            distance = sum(abs(float(ua) - float(va)) for ua, va in zip(u, v))
            
        log.debug(f"Manhattan distance: {distance:.4f}")
        return distance
        
    except Exception as e:
        log.error(f"Error computing Manhattan distance: {e}")
        return 0.0


def normalize_vector(v) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        v: Input vector
        
    Returns:
        np.ndarray: Normalized vector with unit L2 norm
        
    Raises:
        ValueError: If vector has zero norm
        
    Example:
        >>> normalize_vector([3, 4])
        array([0.6, 0.8])
    """
    log.debug(f"Normalizing vector of length {len(v)}")
    
    try:
        v_arr = np.asarray(v, dtype=float)
        norm = np.linalg.norm(v_arr)
        
        if norm == 0:
            log.warning("Cannot normalize zero vector")
            raise ValueError("Cannot normalize zero vector")
            
        normalized = v_arr / norm
        log.debug(f"Vector normalized (original norm: {norm:.4f})")
        return normalized
        
    except Exception as e:
        log.error(f"Error normalizing vector: {e}")
        raise


def batch_cosine_similarity(vectors_a: List, vectors_b: List) -> np.ndarray:
    """
    Compute cosine similarity between two batches of vectors efficiently.
    
    Args:
        vectors_a: First batch of vectors
        vectors_b: Second batch of vectors
        
    Returns:
        np.ndarray: Matrix of cosine similarities where [i,j] = cosine(vectors_a[i], vectors_b[j])
        
    Example:
        >>> batch_cosine_similarity([[1,0], [0,1]], [[1,0], [0,1]])
        array([[1., 0.],
               [0., 1.]])
    """
    log.debug(f"Computing batch cosine similarity between {len(vectors_a)} and {len(vectors_b)} vectors")
    
    try:
        a = np.array(vectors_a, dtype=float)
        b = np.array(vectors_b, dtype=float)
        
        # Normalize vectors
        a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
        
        similarities = np.dot(a_norm, b_norm.T)
        log.debug(f"Batch cosine similarity computed. Shape: {similarities.shape}")
        return similarities
        
    except Exception as e:
        log.error(f"Error computing batch cosine similarity: {e}")
        return np.array([])