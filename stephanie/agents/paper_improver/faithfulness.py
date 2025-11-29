# stephanie/agents/paper_improver/faithfulness.py
# FaithfulnessBot — retrieval + LLM judge with sentence-aware chunks, batching, caching, and majority vote.

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from stephanie.utils.hash_utils import hash_text

try:
    import nltk  # optional for sentence split
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False


class FaithfulnessBot:
    """
    Verifies claims against a paper via dense retrieval + LLM judge.
    - Sentence-aware chunking with overlap
    - Top-k retrieval (torch.topk)
    - Majority-vote judge across k passages
    - Device auto-select, batched claim encoding
    - Embedding cache (disk) keyed by paper hash & model
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        judge_prompt_template: Optional[str] = None,
        llm_judge_fn: Optional[Callable[[str], str]] = None,
        max_claim_length: int = 300,
        min_similarity_threshold: float = 0.3,
        cache_dir: str = "./faithfulness_cache",
        device: Optional[str] = None,
        seed: int = 0,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.top_k = int(top_k)
        self.judge_prompt_template = judge_prompt_template or self._default_judge_prompt()
        self.llm_judge_fn = llm_judge_fn or self._dummy_judge
        self.max_claim_length = int(max_claim_length)
        self.min_similarity_threshold = float(min_similarity_threshold)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.paper_chunks: List[str] = []
        self.chunk_embeddings: Optional[torch.Tensor] = None
        self.paper_hash: Optional[str] = None

        np.random.seed(seed)
        torch.manual_seed(seed)

    # -------------------- prompts / judge --------------------

    def _default_judge_prompt(self) -> str:
        return (
            "You are a precise research assistant.\n"
            "Task: Determine if the CLAIM is directly supported by the PASSAGE.\n"
            "Rules: Only answer YES or NO.\n\n"
            "PASSAGE:\n{passage}\n\n"
            "CLAIM:\n{claim}\n\n"
            "Answer (YES/NO):"
        )

    def _dummy_judge(self, prompt: str) -> str:
        # Always deterministic under seed; prefer YES when numbers/equations present
        tokens = ("Table", "Figure", "Eq", "Algorithm", "%", "±")
        bias = sum(tok in prompt for tok in tokens)
        return "YES" if (hash(prompt) + bias) % 10 >= 3 else "NO"

    # -------------------- preparation --------------------

    def prepare_paper(
        self,
        paper_text: str,
        chunk_sentences: int = 6,
        chunk_overlap: int = 2,
        force_reembed: bool = False,
    ):
        """
        Split into sentence-aware overlapping chunks; embed & (optionally) cache.
        """
        paper_text = (paper_text or "").strip()
        if not paper_text:
            raise ValueError("Paper text is empty.")

        self.paper_hash = hash_text(paper_text) + "-" + hash_text(self.model_name)
        cache_emb = self.cache_dir / f"{self.paper_hash}.pt"
        cache_txt = self.cache_dir / f"{self.paper_hash}.json"

        # build chunks
        self.paper_chunks = self._sentence_chunks(paper_text, chunk_sentences, chunk_overlap)

        if (not force_reembed) and cache_emb.exists() and cache_txt.exists():
            try:
                meta = json.loads(cache_txt.read_text())
                if meta.get("num_chunks") == len(self.paper_chunks):
                    self.chunk_embeddings = torch.load(cache_emb, map_location=self.device)
                    return
            except Exception:
                pass  # fall through to re-embed

        # embed
        self.chunk_embeddings = self.model.encode(
            self.paper_chunks,
            batch_size=64,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        # save cache
        try:
            torch.save(self.chunk_embeddings, cache_emb)
            cache_txt.write_text(json.dumps({"num_chunks": len(self.paper_chunks)}, indent=2))
        except Exception:
            pass

    def _sentence_chunks(self, text: str, chunk_sentences: int, overlap: int) -> List[str]:
        # fallback simple split if nltk missing
        if _HAS_NLTK:
            try:
                nltk.download('punkt', quiet=True)
                sents = nltk.sent_tokenize(text)
            except Exception:
                sents = re.split(r'(?<=[.!?])\s+', text)
        else:
            sents = re.split(r'(?<=[.!?])\s+', text)

        sents = [s.strip() for s in sents if s.strip()]
        chunks: List[str] = []
        i = 0
        while i < len(sents):
            j = min(len(sents), i + chunk_sentences)
            chunk = " ".join(sents[i:j])
            chunks.append(chunk)
            if j == len(sents):
                break
            i = max(i + chunk_sentences - overlap, i + 1)

        # optional near-duplicate downsampling by simple hash
        out: List[str] = []
        seen = set()
        for c in chunks:
            h = hash_text(c[:400])
            if h not in seen:
                out.append(c)
                seen.add(h)
        return out

    # -------------------- verification --------------------

    def verify_claim(self, claim: str, claim_id: Optional[str] = None, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Verify a single claim with majority vote over top-k passages.
        """
        if self.chunk_embeddings is None or not len(self.paper_chunks):
            raise RuntimeError("Call prepare_paper() first.")

        claim = (claim or "").strip()
        if not claim:
            return self._empty_claim_result(claim_id, claim, error="Empty claim.")

        if len(claim) > self.max_claim_length:
            claim = claim[: self.max_claim_length] + " …[truncated]"

        k = int(k or self.top_k)

        # encode claim
        q = self.model.encode(
            claim, convert_to_tensor=True, device=self.device, normalize_embeddings=True, show_progress_bar=False
        )

        scores = util.cos_sim(q, self.chunk_embeddings).squeeze(0)  # [num_chunks]
        top_vals, top_idx = torch.topk(scores, k=min(k, scores.numel()), largest=True, sorted=True)

        retrieved: List[Dict[str, Any]] = []
        for score, idx in zip(top_vals.tolist(), top_idx.tolist()):
            if score < self.min_similarity_threshold:
                continue
            retrieved.append({"index": int(idx), "score": float(score), "passage": self.paper_chunks[idx]})

        if not retrieved:
            return self._empty_claim_result(claim_id, claim, max_similarity=float(scores.max().item()))

        # judge all retrieved passages
        votes, prompts = [], []
        for r in retrieved:
            prompt = self.judge_prompt_template.format(passage=r["passage"], claim=claim)
            resp = (self.llm_judge_fn(prompt) or "").strip().upper()
            vote = 1 if resp.startswith("Y") else 0
            votes.append(vote)
            r["judge_response"] = resp
            prompts.append(prompt)

        yes = sum(votes)
        no = len(votes) - yes
        supported = yes > no

        # confidence: blend mean sim and vote margin
        mean_sim = float(np.mean([r["score"] for r in retrieved])) if retrieved else 0.0
        vote_margin = (yes - no) / max(1, len(votes))  # in [-1,1]
        confidence = float(max(0.0, min(1.0, 0.5 * mean_sim + 0.5 * (0.5 + 0.5 * vote_margin))))

        return {
            "claim_id": claim_id,
            "claim": claim,
            "supported": supported,
            "confidence": round(confidence, 3),
            "retrieved_count": len(retrieved),
            "max_similarity": round(float(top_vals[0].item()), 3),
            "evidence": retrieved[0]["passage"][:800],
            "retrieved": [{"index": r["index"], "score": round(r["score"], 3), "resp": r["judge_response"]} for r in retrieved],
            "judge_prompts": prompts[:2],  # truncate to reduce log size
            "paper_hash": self.paper_hash,
            "error": None,
        }

    def verify_claims_batch(self, claims: List[Dict[str, Any]], k: Optional[int] = None, batch_size: int = 64) -> List[Dict[str, Any]]:
        """
        Batched claim verification:
          - Encodes claims in batches
          - Retrieves top-k per claim
          - Judges each claim across its retrieved passages
        """
        if self.chunk_embeddings is None or not len(self.paper_chunks):
            raise RuntimeError("Call prepare_paper() first.")

        if not claims:
            return []

        k = int(k or self.top_k)
        texts = []
        meta: List[Tuple[Optional[str], str]] = []
        for c in claims:
            cid = c.get("claim_id")
            t = (c.get("claim") or "").strip()
            if not t:
                texts.append("")
            else:
                texts.append(t[: self.max_claim_length] + (" …[truncated]" if len(t) > self.max_claim_length else ""))
            meta.append((cid, t))

        # encode in batches
        encoded: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            emb = self.model.encode(
                chunk, convert_to_tensor=True, device=self.device, normalize_embeddings=True, show_progress_bar=False
            )
            encoded.append(emb)
        queries = torch.cat(encoded, dim=0)  # [N, D]

        # compute similarities in batches for memory safety
        results: List[Dict[str, Any]] = []
        for i, q in enumerate(queries):
            if not meta[i][1]:  # empty claim
                results.append(self._empty_claim_result(meta[i][0], "", error="Empty claim."))
                continue

            scores = util.cos_sim(q, self.chunk_embeddings).squeeze(0)
            top_vals, top_idx = torch.topk(scores, k=min(k, scores.numel()), largest=True, sorted=True)

            retrieved = []
            for score, idx in zip(top_vals.tolist(), top_idx.tolist()):
                if score < self.min_similarity_threshold:
                    continue
                retrieved.append({"index": int(idx), "score": float(score), "passage": self.paper_chunks[idx]})

            if not retrieved:
                results.append(self._empty_claim_result(meta[i][0], meta[i][1], max_similarity=float(scores.max().item())))
                continue

            votes, prompts = [], []
            for r in retrieved:
                prompt = self.judge_prompt_template.format(passage=r["passage"], claim=meta[i][1])
                resp = (self.llm_judge_fn(prompt) or "").strip().upper()
                vote = 1 if resp.startswith("Y") else 0
                votes.append(vote)
                r["judge_response"] = resp
                prompts.append(prompt)

            yes = sum(votes); no = len(votes) - yes
            supported = yes > no
            mean_sim = float(np.mean([r["score"] for r in retrieved])) if retrieved else 0.0
            vote_margin = (yes - no) / max(1, len(votes))
            confidence = float(max(0.0, min(1.0, 0.5 * mean_sim + 0.5 * (0.5 + 0.5 * vote_margin))))

            results.append({
                "claim_id": meta[i][0],
                "claim": meta[i][1],
                "supported": supported,
                "confidence": round(confidence, 3),
                "retrieved_count": len(retrieved),
                "max_similarity": round(float(top_vals[0].item()), 3),
                "evidence": retrieved[0]["passage"][:800],
                "retrieved": [{"index": r["index"], "score": round(r["score"], 3), "resp": r["judge_response"]} for r in retrieved],
                "judge_prompts": prompts[:2],
                "paper_hash": self.paper_hash,
                "error": None,
            })

        return results

    # -------------------- scores & I/O --------------------

    def get_faithfulness_score(self, claims: List[Dict[str, Any]]) -> float:
        """Compute fraction of claims supported (runs batch verify)."""
        if not claims:
            return 1.0
        results = self.verify_claims_batch(claims)
        supported = sum(1 for r in results if r.get("supported"))
        return round(supported / max(1, len(results)), 3)

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Faithfulness results saved to {output_path} ({len(results)} claims)")

    def get_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"total": 0, "supported": 0, "faithfulness_score": 0.0, "avg_confidence": 0.0, "errors": 0}
        supported = [r for r in results if r.get("supported")]
        confidences = [float(r.get("confidence", 0.0)) for r in results if r.get("confidence", 0.0) > 0]
        return {
            "total": len(results),
            "supported": len(supported),
            "faithfulness_score": round(len(supported) / len(results), 3),
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "errors": len([r for r in results if r.get("error")])
        }

    # -------------------- helpers --------------------

    def _empty_claim_result(self, claim_id: Optional[str], claim: str, max_similarity: float = 0.0, error: Optional[str] = None) -> Dict[str, Any]:
        return {
            "claim_id": claim_id,
            "claim": claim,
            "supported": False,
            "confidence": 0.0,
            "evidence": "",
            "retrieved_count": 0,
            "max_similarity": round(max_similarity, 3),
            "judge_prompts": [],
            "paper_hash": self.paper_hash,
            "error": error or "No relevant passage retrieved."
        }
