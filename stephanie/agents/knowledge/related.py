# stephanie/agents/knowledge/goal_document_search.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


@dataclass
class DocHit:
    id: str
    title: str
    score: float
    text: Optional[str] = None
    metadata: Optional[dict] = None


class RelatedAgent(BaseAgent):
    """
    Retrieve documents related to the current goal using the project's embedding index.
    Also writes a compact `knowledge` list suitable for seeding ATS/PromptCompiler.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.document_type: str = cfg.get("document_type", "document")
        self.embedding_name: str = getattr(self.memory.embedding, "name", "unknown")
        self.top_k: int = int(cfg.get("top_k", 50))
        self.min_score: Optional[float] = cfg.get("min_score")  # e.g., 0.25
        self.namespace: Optional[str] = cfg.get("namespace")    # optional multi-index routing
        self.include_text: bool = bool(cfg.get("include_text", True))
        self.llm_rerank: bool = bool(cfg.get("llm_rerank", False))
        self.rerank_k: int = int(cfg.get("rerank_k", min(100, self.top_k)))
        self.output_key: str = cfg.get("output_key", "related_documents")
        self.knowledge_key: str = cfg.get("knowledge_key", "knowledge")

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal = context.get(GOAL) or {}
        goal_text = (goal.get("goal_text") or "").strip()
        if not goal_text:
            raise ValueError("goal_document_search: missing goal.goal_text")

        # 1) Retrieve from embedding store
        raw_hits = self.memory.embedding.search_related_scorables(
            query=goal_text,
            top_k=self.top_k,
            target_type=self.document_type,
        )

        # 2) Normalize → DocHit, filter/dedupe
        hits: List[DocHit] = []
        seen = set()
        for r in raw_hits or []:
            rid = str(r.get("id"))
            if not rid or rid in seen:
                continue
            seen.add(rid)
            score = float(r.get("score", 0.0))
            if self.min_score is not None and score < self.min_score:
                continue
            hits.append(
                DocHit(
                    id=rid,
                    title=str(r.get("title") or r.get("name") or "")[:256],
                    score=score,
                    text=(r.get("text") or r.get("content")) if self.include_text else None,
                    metadata=r.get("metadata") or {},
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)

        # 3) Optional LLM re-rank for the top N (cheap pairwise relevance)
        if self.llm_rerank and hits:
            reranked = await self._llm_rerank(goal_text, hits[: self.rerank_k])
            # keep original tail unchanged
            hits = reranked + hits[self.rerank_k:]

        # 4) Write outputs
        context[self.output_key] = [h.__dict__ for h in hits]

        # Also append compact “knowledge” lines ATS can consume directly
        knowledge_lines = [
            self._to_knowledge_line(h) for h in hits[: min(20, len(hits))]
        ]
        # Merge without clobbering caller-provided knowledge
        existing = context.get(self.knowledge_key) or []
        context[self.knowledge_key] = list(dict.fromkeys(existing + knowledge_lines))  # dedupe while preserving order

        # 5) Telemetry
        self.report({
            "event": "goal_document_search",
            "document_type": self.document_type,
            "embedding_type": self.embedding_name,
            "namespace": self.namespace,
            "top_k": self.top_k,
            "rerank": self.llm_rerank,
            "returned": len(hits),
            "examples": [
                {"id": h.id, "title": h.title, "score": round(h.score, 4)}
                for h in hits[:3]
            ],
        })

        return context

    async def _llm_rerank(self, query: str, hits: List[DocHit]) -> List[DocHit]:
        """
        Lightweight LLM re-ranker: asks the model to score each candidate's relevance (0–1).
        Uses a single prompt for all candidates to minimize calls.
        """
        # Construct a single scoring prompt
        items = "\n".join(
            f"- [{i}] title: {h.title}\n  snippet: {(h.text or '')[:400]}"
            for i, h in enumerate(hits)
        )
        prompt = f"""
Score relevance of each item to the query on a 0..1 scale.

Query: {query}

Items:
{items}

Return strictly JSON with a list of objects: {{"idx": int, "score": float}}.
"""
        try:
            raw = await self.llm(prompt)  # BaseAgent.llm
            # naive JSON extract (be tolerant to model formatting)
            import json
            import re
            m = re.search(r"\{.*\}", raw, re.S) or re.search(r"\[.*\]", raw, re.S)
            obj = json.loads(m.group(0)) if m else []
            scores = {int(d.get("idx")): float(d.get("score")) for d in obj if "idx" in d and "score" in d}
            # attach rerank score as tie-breaker
            paired = [(scores.get(i, 0.0), h) for i, h in enumerate(hits)]
            paired.sort(key=lambda t: t[0], reverse=True)
            return [h for _, h in paired]
        except Exception:
            # fall back to original hits if LLM parsing fails
            return hits

    def _to_knowledge_line(self, h: DocHit) -> str:
        """A single-line hint the planner can use. Keep it concise."""
        snippet = (h.text or h.metadata.get("summary") or "").strip().replace("\n", " ")
        if snippet:
            snippet = snippet[:200]
            return f"{h.title}: {snippet}"
        return f"{h.title}"
