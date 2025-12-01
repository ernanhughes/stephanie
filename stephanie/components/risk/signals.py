from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Interfaces & dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HallucinationContext:
    question: str
    retrieved_passages: Optional[List[str]] = None
    # Generators / scorers
    sampler: Optional[Callable[[str, int], List[str]]] = None  # prompt->N variants
    embedder: Optional[Callable[[List[str]], np.ndarray]] = None  # texts->(N,D)
    entailment: Optional[Callable[[str, str], float]] = None  # entails(a,b)
    # PowerSampler optional telemetry
    power_acceptance_rate: Optional[float] = None
    power_lp_delta_mean: Optional[float] = None
    power_reject_streak_max: Optional[int] = None
    power_token_multiplier: Optional[float] = None
    # Controls
    n_semantic_samples: int = 6
    max_answer_sentences: int = 64

@dataclass
class CollectedSignals:
    # Scalar signals
    se_mean: Optional[float]
    se_spread: Optional[float]
    se_n_clusters: Optional[int]
    meta_inv_violations: Optional[float]
    meta_span_penalty: Optional[float]
    rag_entailment_avg: Optional[float]
    rag_unsupported_frac: Optional[float]
    # Optional power sampler passthrough
    power_acceptance_rate: Optional[float]
    power_lp_delta_mean: Optional[float]
    power_reject_streak_max: Optional[int]
    power_token_multiplier: Optional[float]
    # VPM channels (1-D arrays)
    vpm_channels: Dict[str, np.ndarray]
    # Debug detail (for dashboards)
    debug: Dict[str, object]

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        # Convert numpy arrays to lists for JSON
        d["vpm_channels"] = {k: v.tolist() for k, v in self.vpm_channels.items()}
        return d


class AttrSink:
    """Abstract sink for writing attributes (connect to EvaluationAttributeORM)."""
    def write_many(self, items: List[Tuple[str, float]]) -> None:
        raise NotImplementedError


class DictAttrSink(AttrSink):
    def __init__(self) -> None:
        self.store: Dict[str, float] = {}
    def write_many(self, items: List[Tuple[str, float]]) -> None:
        for k, v in items:
            if v is None:
                continue
            try:
                self.store[k] = float(v)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Utilities (lightweight similarity, clustering, sentence splitting)
# ---------------------------------------------------------------------------

_word_re = re.compile(r"[A-Za-z0-9_]+")

def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _word_re.findall(s)]

def _lexical_sim(a: str, b: str) -> float:
    """Jaccard similarity over word sets as a cheap fallback."""
    A, B = set(_tokenize(a)), set(_tokenize(b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def _pairwise_cosine(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    n = np.maximum(1.0, np.linalg.norm(X, axis=1, keepdims=True))
    Y = X / n
    S = Y @ Y.T
    return S


def _greedy_cluster(sim: np.ndarray, thresh: float = 0.8) -> List[int]:
    """Simple greedy clustering on a similarity matrix. Returns cluster ids per item."""
    n = sim.shape[0]
    labels = [-1] * n
    cid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = cid
        # assign others above threshold to this cluster
        for j in range(i + 1, n):
            if labels[j] == -1 and sim[i, j] >= thresh:
                labels[j] = cid
        cid += 1
    return labels


def _sentences(text: str, max_sentences: int = 64) -> List[Tuple[int, int, str]]:
    """Very light sentence splitter; returns (start, end, sentence)."""
    sents: List[Tuple[int, int, str]] = []
    start = 0
    for m in re.finditer(r"[^.!?\n]+[.!?\n]", text):
        end = m.end()
        s = text[m.start():end].strip()
        if s:
            sents.append((m.start(), end, s))
        start = end
        if len(sents) >= max_sentences:
            break
    # tail
    if start < len(text) and len(sents) < max_sentences:
        s = text[start:].strip()
        if s:
            sents.append((start, len(text), s))
    return sents


# ---------------------------------------------------------------------------
# 1) Semantic Entropy (reference-free)
# ---------------------------------------------------------------------------

def semantic_entropy(
    base_answer: str,
    ctx: HallucinationContext,
) -> Tuple[Optional[float], Optional[float], Optional[int], Dict[str, object]]:
    """
    Computes semantic entropy using clustering over N answer variants.
    Returns: (se_mean, se_spread, n_clusters, debug)
    """
    N = max(1, ctx.n_semantic_samples)
    if ctx.sampler is None:
        # No sampler – fall back to single sample entropy 0
        return 0.0, 0.0, 1, {"note": "no sampler provided; SE degenerated"}

    samples = ctx.sampler(ctx.question, N)
    # Ensure base answer included for stability
    if base_answer not in samples:
        samples = [base_answer] + samples
    # Embeddings or lexical sim
    if ctx.embedder is not None:
        E = ctx.embedder(samples)
        S = _pairwise_cosine(E)
    else:
        # lexical similarity matrix
        n = len(samples)
        S = np.eye(n, dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                s = _lexical_sim(samples[i], samples[j])
                S[i, j] = S[j, i] = s

    labels = _greedy_cluster(S, thresh=0.8)
    # cluster probabilities
    counts: Dict[int, int] = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    probs = np.array([c / len(labels) for c in counts.values()], dtype=np.float32)
    # Shannon entropy over cluster mass
    se = float(-(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum())
    # Spread via Gini-like dispersion over cluster sizes
    gini = float(1.0 - (probs ** 2).sum())
    debug = {
        "samples": samples,
        "labels": labels,
        "cluster_probs": probs.tolist(),
        "similarity_mean": float((S.sum() - np.trace(S)) / max(1, (S.size - len(samples))))
    }
    return se, gini, len(counts), debug


# ---------------------------------------------------------------------------
# 2) Metamorphic testing (MetaQA-style)
# ---------------------------------------------------------------------------

# Very small synonym list for invariant mutations; extend in production
_SYNONYMS = {
    "largest": ["biggest", "greatest"],
    "capital": ["main city", "seat of government"],
    "year": ["yr"],
}

_DEF_NEGATION = {
    " is ": " is not ",
    " are ": " are not ",
    " was ": " was not ",
    " were ": " were not ",
}

def _mutations(question: str) -> List[Tuple[str, str]]:
    """Return list of (type, mutated_question)."""
    muts: List[Tuple[str, str]] = []
    # Synonym swaps (invariance expected)
    for k, vs in _SYNONYMS.items():
        if k in question.lower():
            for v in vs:
                muts.append(("synonym", re.sub(k, v, question, flags=re.IGNORECASE)))
    # Add trivial paraphrase: add a harmless prefix
    muts.append(("paraphrase", "Briefly: " + question))
    # Negation (anti-invariance): if contains a be-verb, flip one occurrence
    for k, v in _DEF_NEGATION.items():
        if k in question.lower():
            muts.append(("negation", re.sub(k, v, question, count=1, flags=re.IGNORECASE)))
            break
    return muts


def metaqa(
    base_answer: str,
    ctx: HallucinationContext,
) -> Tuple[Optional[float], Optional[float], Dict[str, object]]:
    """
    Compute metamorphic invariance violation rate and span penalty proxy.
    Returns: (violation_rate, span_penalty, debug)
    """
    muts = _mutations(ctx.question)
    if not muts or ctx.sampler is None:
        return None, None, {"note": "no mutations or sampler"}

    # Generate answers for mutated prompts
    answers: List[Tuple[str, str]] = []  # (type, ans)
    for mtype, q2 in muts:
        try:
            a2 = ctx.sampler(q2, 1)[0]
        except Exception:
            continue
        answers.append((mtype, a2))

    if not answers:
        return None, None, {"note": "sampler returned no answers"}

    # Similarity / entailment
    def sim(a: str, b: str) -> float:
        if ctx.entailment is not None:
            # symmetricized entailment
            return 0.5 * (ctx.entailment(a, b) + ctx.entailment(b, a))
        if ctx.embedder is not None:
            E = ctx.embedder([a, b])
            s = float(_pairwise_cosine(E)[0, 1])
            return (s + 1) / 2.0  # map [-1,1] -> [0,1]
        return _lexical_sim(a, b)

    violations = 0
    penalties: List[float] = []
    for mtype, a2 in answers:
        s = sim(base_answer, a2)
        if mtype in ("synonym", "paraphrase"):
            # Should remain semantically close
            violated = s < 0.6
        elif mtype == "negation":
            # Should differ significantly
            violated = s > 0.4
        else:
            violated = False
        violations += int(violated)
        penalties.append(max(0.0, 0.6 - s) if mtype != "negation" else max(0.0, s - 0.4))

    violation_rate = violations / max(1, len(answers))
    span_penalty = float(np.mean(penalties)) if penalties else None
    debug = {
        "mutations": muts,
        "answers": answers,
        "violation_rate": violation_rate,
        "penalties": penalties,
    }
    return violation_rate, span_penalty, debug


# ---------------------------------------------------------------------------
# 3) MetaRAG-style span scoring (evidence alignment)
# ---------------------------------------------------------------------------

def metarag(
    answer: str,
    ctx: HallucinationContext,
    entail_thresh: float = 0.6,
) -> Tuple[Optional[float], Optional[float], Dict[str, object]]:
    """
    Evaluate answer sentences against retrieved passages.
    Returns: (entailment_avg, unsupported_frac, debug)
    """
    passages = ctx.retrieved_passages or []
    if not passages:
        return None, None, {"note": "no retrieved passages provided"}

    sents = _sentences(answer, max_sentences=ctx.max_answer_sentences)

    def entail(a: str, b: str) -> float:
        if ctx.entailment is not None:
            return ctx.entailment(a, b)
        if ctx.embedder is not None:
            E = ctx.embedder([a, b])
            return float((_pairwise_cosine(E)[0, 1] + 1) / 2.0)
        return _lexical_sim(a, b)

    sent_scores: List[float] = []
    unsupported_idx: List[int] = []

    for i, (_, _, s) in enumerate(sents):
        # max entailment over all passages
        mx = 0.0
        for p in passages:
            mx = max(mx, entail(s, p))
        sent_scores.append(mx)
        if mx < entail_thresh:
            unsupported_idx.append(i)

    entailment_avg = float(np.mean(sent_scores)) if sent_scores else None
    unsupported_frac = len(unsupported_idx) / max(1, len(sent_scores)) if sent_scores else None

    debug = {
        "sentences": sents,
        "sent_scores": sent_scores,
        "unsupported_indices": unsupported_idx,
        "entail_thresh": entail_thresh,
    }
    return entailment_avg, unsupported_frac, debug


# ---------------------------------------------------------------------------
# VPM channel helpers
# ---------------------------------------------------------------------------

def _normalize_channel(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.size == 0:
        return x
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _channels_from_debug(debug: Dict[str, object]) -> Dict[str, np.ndarray]:
    channels: Dict[str, np.ndarray] = {}
    # MetaRAG sentence-level channel
    if "sent_scores" in debug:
        channels["rag.sent_entail"] = _normalize_channel(np.array(debug["sent_scores"], dtype=np.float32))
        mask = np.zeros_like(channels["rag.sent_entail"])
        for idx in debug.get("unsupported_indices", []):
            if 0 <= idx < mask.size:
                mask[idx] = 1.0
        channels["rag.unsupported_mask"] = mask
    # MetaQA penalties as a short vector
    if "penalties" in debug:
        channels["meta.penalty"] = _normalize_channel(np.array(debug["penalties"], dtype=np.float32))
    # SE cluster mass (histogram)
    if "cluster_probs" in debug:
        channels["se.cluster_mass"] = _normalize_channel(np.array(debug["cluster_probs"], dtype=np.float32))
    return channels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect(answer: str, ctx: HallucinationContext, sink: Optional[AttrSink] = None) -> CollectedSignals:
    """Compute hallucination-related signals and optionally write to sink."""
    sink = sink or DictAttrSink()

    se_mean, se_spread, se_n, dbg_se = semantic_entropy(answer, ctx)
    meta_viols, meta_pen, dbg_meta = metaqa(answer, ctx)
    rag_avg, rag_unsupported, dbg_rag = metarag(answer, ctx)

    # Merge VPM channels
    vpm_channels: Dict[str, np.ndarray] = {}
    for dbg in (dbg_se, dbg_meta, dbg_rag):
        vpm_channels.update(_channels_from_debug(dbg))

    # Write scalar attributes
    items: List[Tuple[str, float]] = [
        ("se.mean", se_mean if se_mean is not None else float("nan")),
        ("se.spread", se_spread if se_spread is not None else float("nan")),
        ("se.n_clusters", float(se_n) if se_n is not None else float("nan")),
        ("meta.inv_violations", meta_viols if meta_viols is not None else float("nan")),
        ("meta.span_penalty", meta_pen if meta_pen is not None else float("nan")),
        ("rag.entailment_avg", rag_avg if rag_avg is not None else float("nan")),
        ("rag.unsupported_frac", rag_unsupported if rag_unsupported is not None else float("nan")),
    ]

    # Optional power sampler passthroughs
    if ctx.power_acceptance_rate is not None:
        items.append(("power.acceptance_rate", ctx.power_acceptance_rate))
    if ctx.power_lp_delta_mean is not None:
        items.append(("power.lp_delta_mean", ctx.power_lp_delta_mean))
    if ctx.power_reject_streak_max is not None:
        items.append(("power.reject_streak_max", float(ctx.power_reject_streak_max)))
    if ctx.power_token_multiplier is not None:
        items.append(("power.token_multiplier", ctx.power_token_multiplier))

    sink.write_many(items)

    debug = {
        "se": dbg_se,
        "meta": dbg_meta,
        "rag": dbg_rag,
    }

    return CollectedSignals(
        se_mean=se_mean,
        se_spread=se_spread,
        se_n_clusters=se_n,
        meta_inv_violations=meta_viols,
        meta_span_penalty=meta_pen,
        rag_entailment_avg=rag_avg,
        rag_unsupported_frac=rag_unsupported,
        power_acceptance_rate=ctx.power_acceptance_rate,
        power_lp_delta_mean=ctx.power_lp_delta_mean,
        power_reject_streak_max=ctx.power_reject_streak_max,
        power_token_multiplier=ctx.power_token_multiplier,
        vpm_channels=vpm_channels,
        debug=debug,
    )


# ---------------------------------------------------------------------------
# Minimal self-test (run this module directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Toy sampler & embedder for demo purposes only
    def toy_sampler(prompt: str, n: int) -> List[str]:
        base = f"The capital of France is Paris. ({prompt[:10]})"
        alts = [
            base,
            "Paris is the capital of France.",
            "France's capital is Paris.",
            "The main city of France is Paris.",
            "The capital of France is Lyon.",
            "Paris is not the capital of France.",
        ]
        return alts[:max(1, n)]

    def toy_embedder(texts: List[str]) -> np.ndarray:
        # ultra-cheap: TF-like hashed counts over 1-grams
        vocab: Dict[str, int] = {}
        rows: List[List[float]] = []
        for t in texts:
            toks = _tokenize(t)
            rows.append(toks)
            for tok in toks:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, toks in enumerate(rows):
            for tok in toks:
                X[i, vocab[tok]] += 1.0
        # l2 normalize
        X /= np.maximum(1e-6, np.linalg.norm(X, axis=1, keepdims=True))
        return X

    def toy_entail(a: str, b: str) -> float:
        return _lexical_sim(a, b)

    ctx = HallucinationContext(
        question="What is the capital of France?",
        sampler=toy_sampler,
        embedder=toy_embedder,
        entailment=toy_entail,
        retrieved_passages=["Paris is the capital and most populous city of France."],
        power_acceptance_rate=0.42,
    )
    sink = DictAttrSink()
    ans = "The capital of France is Paris. It is a major European city."
    out = collect(ans, ctx, sink)
    print("Signals:", out.as_dict())
    print("Stored attrs:", sink.store)
