
# stephanie/agents/paper_improver/faithfulness.py

# faithfulness.py — Automated claim verification against source paper.
# Uses retrieval + LLM judge to score faithfulness. Logs mismatches. Safe, auditable, measurable.

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from sentence_transformers import SentenceTransformer, util
import numpy as np

class FaithfulnessBot:
    """
    Verifies that generated claims are supported by the source paper.
    Steps:
      1. Chunk paper into passages.
      2. For each claim, retrieve top-k most relevant passages.
      3. Ask LLM judge: “Does this passage support this claim? YES/NO”
      4. Return verdict + confidence + evidence snippet.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        judge_prompt_template: Optional[str] = None,
        llm_judge_fn: Optional[Callable] = None,
        max_claim_length: int = 300,
        min_similarity_threshold: float = 0.3
    ):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.judge_prompt_template = judge_prompt_template or self._default_judge_prompt()
        self.llm_judge_fn = llm_judge_fn or self._dummy_judge  # replace with real LLM in prod
        self.max_claim_length = max_claim_length
        self.min_similarity_threshold = min_similarity_threshold
        self.paper_chunks = []
        self.chunk_embeddings = None

    def _default_judge_prompt(self) -> str:
        return """You are a precise research assistant. Your task is to verify whether a claim is directly supported by a given passage from a paper.

Paper Passage:
{passage}

Claim:
{claim}

Answer only YES or NO. Do not explain.
Answer:"""

    def _dummy_judge(self, prompt: str) -> str:
        """Mock LLM judge — replace with real API (e.g., local Phi-2, OpenAI, Claude)."""
        # Simulate 80% accuracy — real one should be deterministic + seeded
        if "Table" in prompt or "Figure" in prompt or "Eq" in prompt:
            return "YES"
        if "never" in prompt or "always" in prompt or "proves" in prompt:
            return "NO"
        return "YES" if hash(prompt) % 10 > 2 else "NO"

    def prepare_paper(self, paper_text: str, chunk_size: int = 200, chunk_overlap: int = 50):
        """
        Preprocess and embed paper text for retrieval.
        Splits into overlapping chunks for dense retrieval.
        """
        if not paper_text.strip():
            raise ValueError("Paper text is empty.")

        # Clean and split
        paper_text = re.sub(r'\s+', ' ', paper_text).strip()
        self.paper_chunks = self._chunk_text(paper_text, chunk_size, chunk_overlap)

        # Embed chunks
        print(f"🔍 Embedding {len(self.paper_chunks)} paper chunks...")
        self.chunk_embeddings = self.model.encode(self.paper_chunks, convert_to_tensor=True)
        print("✅ Paper prepared for claim verification.")

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start = end - overlap
        return chunks

    def verify_claim(self, claim: str, claim_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify a single claim against the prepared paper.
        Returns verdict, confidence, evidence, and retrieval score.
        """
        if not self.chunk_embeddings:
            raise RuntimeError("Call prepare_paper() first.")

        if len(claim) > self.max_claim_length:
            claim = claim[:self.max_claim_length] + "... [truncated]"

        # Embed claim
        claim_embedding = self.model.encode(claim, convert_to_tensor=True)

        # Compute similarities
        cos_scores = util.cos_sim(claim_embedding, self.chunk_embeddings)[0]
        top_results = np.argpartition(-cos_scores.cpu(), range(self.top_k))[:self.top_k]

        # Get top passages
        retrieved = []
        for idx in top_results[0:self.top_k]:
            score = cos_scores[idx].item()
            if score < self.min_similarity_threshold:
                continue
            retrieved.append({
                "passage": self.paper_chunks[idx],
                "score": round(score, 3),
                "index": int(idx)
            })

        if not retrieved:
            return {
                "claim_id": claim_id,
                "claim": claim,
                "supported": False,
                "confidence": 0.0,
                "evidence": "",
                "retrieved_count": 0,
                "max_similarity": 0.0,
                "judge_prompt": "",
                "judge_response": "NO",
                "error": "No relevant passage retrieved."
            }

        # Use top passage for judging
        best_passage = retrieved[0]["passage"]
        prompt = self.judge_prompt_template.format(passage=best_passage, claim=claim)

        # Call LLM judge
        judge_response = self.llm_judge_fn(prompt).strip().upper()
        supported = "YES" in judge_response

        return {
            "claim_id": claim_id,
            "claim": claim,
            "supported": supported,
            "confidence": retrieved[0]["score"],  # use retrieval score as proxy
            "evidence": best_passage[:500],
            "retrieved_count": len(retrieved),
            "max_similarity": retrieved[0]["score"],
            "judge_prompt": prompt,
            "judge_response": judge_response,
            "error": None
        }

    def verify_claims_batch(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify a batch of claims. Returns list of verdicts.
        Input: [{"claim_id": "...", "claim": "..."}, ...]
        """
        results = []
        for c in claims:
            try:
                result = self.verify_claim(c["claim"], c.get("claim_id"))
            except Exception as e:
                result = {
                    "claim_id": c.get("claim_id"),
                    "claim": c["claim"],
                    "supported": False,
                    "confidence": 0.0,
                    "evidence": "",
                    "error": str(e)
                }
            results.append(result)
        return results

    def get_faithfulness_score(self, claims: List[Dict[str, Any]]) -> float:
        """Compute % of claims supported."""
        if not claims:
            return 1.0
        results = self.verify_claims_batch(claims)
        supported = sum(1 for r in results if r["supported"])
        return round(supported / len(claims), 3)

    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save verification results to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Faithfulness results saved to {output_path} ({len(results)} claims)")

    def get_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return summary stats for dashboard."""
        if not results:
            return {"total": 0, "supported": 0, "faithfulness_score": 0.0, "avg_confidence": 0.0}

        supported = [r for r in results if r["supported"]]
        confidences = [r["confidence"] for r in results if r["confidence"] > 0]

        return {
            "total": len(results),
            "supported": len(supported),
            "faithfulness_score": round(len(supported) / len(results), 3),
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "errors": len([r for r in results if r.get("error")])
        }