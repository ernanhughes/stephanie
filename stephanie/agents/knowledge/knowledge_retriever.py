# stephanie/agents/knowledge/knowledge_retriever.py
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Dict, List, Optional

from stephanie.agents.base_agent import BaseAgent
from stephanie.utils.hash_utils import hash_text


class KnowledgeRetriever(BaseAgent):
    """
    Thin wrapper over KnowledgeGraphService that builds a knowledge_tree for the
    provided text and returns a compact 'relevant_knowledge' bundle.

    Design choices:
    - **Single code path**: assumes a KnowledgeGraphService with `.build_tree(...)`.
    - **No fancy branching**: if the service is missing, returns a tiny heuristic bundle.
    - **PlanTrace-friendly**: always returns JSON-serializable dicts.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        # one dependency: the KG service
        try:
            self.kg = container.get("knowledge_graph")
        except Exception:
            self.kg = None

        # knobs
        self.top_k_claims = int(cfg.get("knowledge_retriever", {}).get("top_k_claims", 8))
        self.top_k_insights = int(cfg.get("knowledge_retriever", {}).get("top_k_insights", 8))
        self.include_vpm = bool(cfg.get("knowledge_retriever", {}).get("include_vpm_payload", True))

    # ---------- public API ----------

    def retrieve(
        self,
        *,
        query: str,
        context: str,
        chat_corpus: Optional[List[Dict[str, Any]]] = None,
        trajectories: Optional[List[Dict[str, Any]]] = None,
        domains: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build and return relevant knowledge for a (query, context) pair.

        Returns:
            {
              "query": str,
              "paper_id": str,
              "knowledge_tree": {...},            # from KG service
              "highlights": { "claims": [...], "insights": [...], "entities": [...] },
              "vpm_payload": {...} | None,
              "stats": {...}
            }
        """
        t0 = time.time()
        chat_corpus = chat_corpus or []
        trajectories = trajectories or []
        domains = domains or []

        paper_id = _stable_id(query or context)

        # Preferred path: rely *only* on the service's build_tree
        if self.kg and hasattr(self.kg, "build_tree"):
            self.report({"event": "KRStart", "paper_id": paper_id, "len_context": len(context)})
            tree = self.kg.build_tree(
                paper_text=context or (query or ""),
                paper_id=paper_id,
                chat_corpus=chat_corpus,
                trajectories=trajectories,
                domains=domains,
            ) or {}

            # Small, useful highlights for PlanTrace viewing
            claims = [c.get("text", "") for c in (tree.get("claims") or [])][: self.top_k_claims]
            insights = [i.get("text", "") for i in (tree.get("insights") or [])][: self.top_k_insights]

            # `entities` shape can vary; normalize to strings when possible
            ents_raw = tree.get("entities") or []
            if ents_raw and isinstance(ents_raw[0], dict):
                entities = [e.get("text", "") for e in ents_raw if e.get("text")]
            else:
                # service may return a list[str]
                entities = [str(e) for e in ents_raw]

            vpm_payload = None
            if self.include_vpm and hasattr(self.kg, "export_for_vpm"):
                try:
                    vpm_payload = self.kg.export_for_vpm(tree)
                except Exception:
                    vpm_payload = None

            bundle = {
                "query": query,
                "paper_id": paper_id,
                "knowledge_tree": tree,
                "highlights": {
                    "claims": claims,
                    "insights": insights,
                    "entities": entities[: self.top_k_claims],
                },
                "vpm_payload": vpm_payload,
                "stats": {
                    "build_ms": int((time.time() - t0) * 1000),
                    "counts": {
                        "claims": len(tree.get("claims") or []),
                        "insights": len(tree.get("insights") or []),
                        "entities": len(tree.get("entities") or []),
                        "relationships": len(tree.get("relationships") or []),
                    },
                },
            }
            self.report({"event": "KRComplete", "paper_id": paper_id, **bundle["stats"]["counts"]})
            return bundle

        # Fallback (service missing): a tiny heuristic bundle
        self.report({"event": "KRServiceMissing", "paper_id": paper_id})
        claims = _heuristic_claims(context, k=self.top_k_claims)
        entities = _heuristic_entities(context)
        bundle = {
            "query": query,
            "paper_id": paper_id,
            "knowledge_tree": {
                "paper_id": paper_id,
                "section_name": "Full Paper",
                "claims": [{"id": f"c{i+1}", "text": c, "confidence": 0.6} for i, c in enumerate(claims)],
                "insights": [],
                "entities": [{"id": f"e{i+1}", "text": e, "type": "TERM"} for i, e in enumerate(entities)],
                "relationships": [],
                "claim_coverage": 0.0,
                "evidence_strength": 0.0,
            },
            "highlights": {"claims": claims, "insights": [], "entities": entities},
            "vpm_payload": None,
            "stats": {"build_ms": int((time.time() - t0) * 1000), "counts": {"claims": len(claims), "insights": 0, "entities": len(entities), "relationships": 0}},
        }
        return bundle

    # ---------- optional: health probe ----------
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "ok" if (self.kg and hasattr(self.kg, "build_tree")) else "degraded",
            "has_service": bool(self.kg),
        }


# ---------- small helpers (no external deps) ----------

def _stable_id(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "kr:empty"
    return "kr:" + hash_text(s)[:12]

def _heuristic_claims(text: str, k: int = 8) -> List[str]:
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    picks: List[str] = []
    for s in sents:
        sl = s.lower()
        if any(w in sl for w in ("we ", "our ", "this paper", "results", "achiev", "improv", "conclud", "show", "demonstrat")):
            picks.append(s.strip())
        if len(picks) >= k:
            break
    return picks or [x.strip() for x in sents[:k]]

def _heuristic_entities(text: str) -> List[str]:
    # crude: capitalized tokens & acronyms
    if not text:
        return []
    toks = re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", text)
    uniq = []
    seen = set()
    for t in toks:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq[:12]
