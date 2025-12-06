# stephanie/tools/recommend_similar_papers_tool.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from stephanie.tools.base_tool import BaseTool
from stephanie.scoring.metrics.common import ScoreTuple


@dataclass
class SimilarPaper:
    """Lightweight carrier for a similar paper recommendation."""
    paper_id: str
    title: str
    similarity: float
    meta: Dict[str, Any]


class RecommendSimilarPapersTool(BaseTool):
    """
    Recommend papers similar to the current Scorable using embeddings.

    Responsibilities:
      - Take a Scorable that represents a paper / document.
      - Get its embedding via memory.embedding (or equivalent).
      - Retrieve a pool of candidate papers from a store/service.
      - Compute cosine similarity and select top-k above a threshold.
      - Attach the ranked list to scorable.meta["similar_papers"].
      - Return a ScoreTuple summarizing the recommendation quality.
    """

    # Tool identity for logging / metrics
    name: str = "recommend_similar_papers"
    dimension: str = "similar_papers"

    def __init__(
        self,
        cfg: Dict[str, Any],
        memory,
        container,
        logger,
        *,
        top_k: int = 5,
        min_similarity: float = 0.25,
        candidate_limit: int = 200,
    ) -> None:
        super().__init__(cfg=cfg, memory=memory, container=container, logger=logger)
        self.top_k = int(top_k)
        self.min_similarity = float(min_similarity)
        self.candidate_limit = int(candidate_limit)

        # Assumptions about available services:
        # - memory.embedding: your EmbeddingStore-like service.
        # - container.get("paper_store"): something that can enumerate papers.
        self.embedding_store = getattr(memory, "embedding", None)
        if self.embedding_store is None:
            self.logger.warning(
                "RecommendSimilarPapersTool: memory.embedding not found; "
                "you must inject an embedding store manually."
            )

        self.paper_store = None
        try:
            self.paper_store = container.get("paper_store")
        except Exception:
            # Optional – you can wire this later
            self.logger.info(
                "RecommendSimilarPapersTool: no 'paper_store' in container; "
                "you must provide candidates via context['candidate_papers']."
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def apply(self, scorable, context: Optional[Dict[str, Any]] = None) -> List[ScoreTuple]:
        """
        Main entrypoint.

        Expected:
          - `scorable` has `.id` and `.text` (or `.content`) attributes.
          - Either:
              * paper_store provides candidate papers, OR
              * caller passes context["candidate_papers"] as iterable of objects
                with .id and .text (and optionally .title / .meta).

        Side-effects:
          - Adds `similar_papers` list to scorable.meta (or context["similar_papers"]).
        """
        context = context or {}

        # 1) Extract query text
        query_text = self._get_text_from_scorable(scorable)
        if not query_text:
            self.logger.warning(
                "RecommendSimilarPapersTool: scorable has no text/content; "
                "skipping similarity search."
            )
            return []

        # 2) Get query embedding
        query_vec = self._embed_text(query_text)
        if query_vec is None:
            self.logger.warning(
                "RecommendSimilarPapersTool: failed to embed query text; "
                "skipping similarity search."
            )
            return []

        # 3) Get candidate papers
        candidates = self._get_candidate_papers(scorable, context)
        if not candidates:
            self.logger.info("RecommendSimilarPapersTool: no candidate papers available.")
            return []

        candidate_vecs: List[Tuple[Any, np.ndarray]] = []
        for cand in candidates:
            cand_text = self._get_text_from_candidate(cand)
            if not cand_text:
                continue
            vec = self._embed_text(cand_text)
            if vec is None:
                continue
            candidate_vecs.append((cand, vec))

        if not candidate_vecs:
            self.logger.info("RecommendSimilarPapersTool: no candidates with embeddings.")
            return []

        # 4) Compute cosine similarities
        ranked: List[SimilarPaper] = self._rank_by_similarity(
            query_vec, candidate_vecs
        )

        # 5) Filter by threshold and top_k
        kept = [
            sp for sp in ranked
            if sp.similarity >= self.min_similarity
        ][: self.top_k]

        # 6) Attach to scorable / context for downstream use
        sim_payload = [
            {
                "paper_id": sp.paper_id,
                "title": sp.title,
                "similarity": float(sp.similarity),
                "meta": sp.meta,
            }
            for sp in kept
        ]

        # Prefer scorable.meta if present, else dump into context
        meta_target = getattr(scorable, "meta", None)
        if isinstance(meta_target, dict):
            meta_target.setdefault("similar_papers", sim_payload)
        else:
            context["similar_papers"] = sim_payload

        # 7) Summarize as a single ScoreTuple for metrics
        best_sim = max((sp.similarity for sp in kept), default=0.0)
        score_tuple = ScoreTuple(
            dimension=self.dimension,
            score=float(best_sim),
            sub_dimension=None,
            meta={
                "count": len(kept),
                "top_k": self.top_k,
                "min_similarity": self.min_similarity,
                "papers": sim_payload,
            },
        )

        return [score_tuple]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_text_from_scorable(self, scorable) -> str:
        """Try common attributes for the scorable text."""
        if scorable is None:
            return ""
        for attr in ("text", "content", "body"):
            val = getattr(scorable, attr, None)
            if isinstance(val, str) and val.strip():
                return val
        # Fallback: maybe the scorable wraps a document object
        doc = getattr(scorable, "document", None)
        if doc is not None:
            for attr in ("text", "content", "body", "raw_text"):
                val = getattr(doc, attr, None)
                if isinstance(val, str) and val.strip():
                    return val
        return ""

    def _get_text_from_candidate(self, cand) -> str:
        """Same as scorable, but for candidate paper objects."""
        if cand is None:
            return ""
        for attr in ("text", "content", "abstract", "body"):
            val = getattr(cand, attr, None)
            if isinstance(val, str) and val.strip():
                return val
        return ""

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Thin wrapper around your embedding store."""
        if self.embedding_store is None or not text:
            return None

        try:
            # Adapt this to your actual embedding API:
            # e.g. embedding_store.get_embedding(text, cfg=self.cfg["embedding"])
            vec = self.embedding_store.get_embedding(text, cfg=self.cfg.get("embedding"))
            if vec is None:
                return None
            arr = np.asarray(vec, dtype=np.float32)
            if arr.ndim == 1:
                return arr
            # If 2D (e.g. [1, D]), squeeze
            return arr.squeeze()
        except Exception as e:
            self.logger.error(f"RecommendSimilarPapersTool: embedding failed: {e}")
            return None

    def _get_candidate_papers(self, scorable, context: Dict[str, Any]) -> Iterable[Any]:
        """
        Source of candidate papers.

        Priority:
          1) context["candidate_papers"] if provided.
          2) self.paper_store.iter_all_papers() or a similar API
             (you can adapt this to CaseBookStore / MemCube / your own store).
        """
        # 1) Explicit candidates in context
        cand = context.get("candidate_papers")
        if cand is not None:
            return cand

        # 2) Try paper_store (if wired)
        if self.paper_store is not None:
            # You’ll adapt this to whatever API you have:
            # e.g. paper_store.list_recent_papers(limit=self.candidate_limit)
            try:
                return self.paper_store.list_papers(limit=self.candidate_limit)
            except Exception as e:
                self.logger.error(
                    f"RecommendSimilarPapersTool: paper_store.list_papers failed: {e}"
                )

        # 3) No candidates
        return []

    def _rank_by_similarity(
        self,
        query_vec: np.ndarray,
        candidates: List[Tuple[Any, np.ndarray]],
    ) -> List[SimilarPaper]:
        """Cosine similarity against all candidate vectors."""
        q_norm = float(np.linalg.norm(query_vec) + 1e-8)
        ranked: List[SimilarPaper] = []

        for cand, vec in candidates:
            v_norm = float(np.linalg.norm(vec) + 1e-8)
            sim = float(np.dot(query_vec, vec) / (q_norm * v_norm))

            paper_id = str(getattr(cand, "id", getattr(cand, "paper_id", "unknown")))
            title = (
                getattr(cand, "title", None)
                or getattr(cand, "name", None)
                or paper_id
            )
            meta = getattr(cand, "meta", {}) or {}

            ranked.append(
                SimilarPaper(
                    paper_id=paper_id,
                    title=title,
                    similarity=sim,
                    meta=meta,
                )
            )

        # Sort descending by similarity
        ranked.sort(key=lambda sp: sp.similarity, reverse=True)
        return ranked
