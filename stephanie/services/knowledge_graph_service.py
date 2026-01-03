# stephanie/services/knowledge_graph_service.py
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stephanie.scoring.scorable import Scorable
from stephanie.services.knowledge_graph.entity_canonicalizer import \
    EntityCanonicalizer
from stephanie.services.knowledge_graph.graph_indexer import GraphIndexer
from stephanie.services.knowledge_graph.subgraphs.edge_index import \
    JSONLEdgeIndex
from stephanie.services.knowledge_graph.subgraphs.seed_finder import \
    EmbeddingSeedFinder
from stephanie.services.knowledge_graph.subgraphs.subgraph_builder import (
    SubgraphBuilder, SubgraphConfig)
from stephanie.services.service_protocol import Service
from stephanie.tools.ner_tool import NerTool

log = logging.getLogger(__name__)


class KnowledgeGraphService(Service):
    """
    Thin coordinator service that delegates to modular KG components.

    Responsibilities:
      - Lifecycle (initialize/shutdown)
      - Event subscription (index_request → GraphIndexer)
      - Nexus + embedding store mirroring (upsert_node/edge)
      - Public APIs (search, query subgraph, tree building)
      - Stats/health
    """

    def __init__(self, cfg: Dict, memory: Any, container: Any, logger: Any):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        # Subsystems (lazy-initialized)
        self._initialized = False
        self._rel_path = None
        self.nexus_store = None
        self.embedding_store = None
        self._edge_index = None
        self._seed_finder = None
        self._subgraph_builder = None
        self._graph_indexer = None
        self._canonicalizer = EntityCanonicalizer()
        self._entity_tool = NerTool(self.cfg, self.memory, self.container, self.logger)

        # Paths
        kg_cfg = cfg.get("knowledge_graph", {})
        data_dir = kg_cfg.get("data_dir") or "./data/kg"
        os.makedirs(data_dir, exist_ok=True)
        self._rel_path = Path(data_dir) / "relationships.jsonl"

        # Stats
        self._stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "node_types": {},
            "edge_types": {},
        }

    @property
    def name(self):
        return "knowledge-graph"

    # --- Collaborator factories (lazy init) ----------------------------------

    @property
    def edge_index(self) -> JSONLEdgeIndex:
        if self._edge_index is None:
            self._edge_index = JSONLEdgeIndex(rel_path=self._rel_path, logger=self.logger)
        return self._edge_index

    @property
    def seed_finder(self) -> EmbeddingSeedFinder:
        if self._seed_finder is None:
            self._seed_finder = EmbeddingSeedFinder(
                search_entities_fn=self.search_entities,
                detect_entities_fn=self.detect_entities,
                logger=self.logger,
            )
        return self._seed_finder

    @property
    def subgraph_builder(self) -> SubgraphBuilder:
        if self._subgraph_builder is None:
            self._subgraph_builder = SubgraphBuilder(
                seed_finder=self.seed_finder,
                edge_index=self.edge_index,
                nexus_store=self.nexus_store,
                logger=self.logger,
            )
        return self._subgraph_builder

    @property
    def graph_indexer(self) -> GraphIndexer:
        if self._graph_indexer is None:
            self._graph_indexer = GraphIndexer(
                kg_service=self,
                detect_entities_fn=self.detect_entities,
                fetch_text_fn=self.fetch_text_for_indexing,
                logger=self.logger,
            )
        return self._graph_indexer

    def _canonical_entity_id(self, ent: Dict[str, Any]) -> str:
        return EntityCanonicalizer.canonical_id(ent.get("type"), ent.get("text"))

    # --- Public KG APIs -----------------------------------------------------

    def upsert_node(
        self,
        *,
        node_id: str,
        properties: Optional[Dict[str, Any]] = None,
        refresh: bool = False,
    ) -> bool:
        """Upsert node into Nexus + embedding store (if embeddings present)."""
        try:
            properties = properties or {}
            self.nexus_store.upsert_node(node_id, properties=properties)
            # Embed if text fields present
            text = properties.get("text") or properties.get("name") or properties.get("title")
            if text and self.embedding_store:
                emb = self.embedding_store.encode(text)
                if emb is not None:
                    meta = {
                        "node_id": node_id,
                        "type": properties.get("type", "node"),
                        **{k: v for k, v in properties.items() if isinstance(v, (str, int, float))},
                    }
                    self.embedding_store.add([text], [emb], [meta])
            self._stats["total_nodes"] = int(self._stats.get("total_nodes", 0)) + 1
            node_type = properties.get("type", "UNKNOWN")
            self._stats["node_types"][node_type] = int(self._stats["node_types"].get(node_type, 0)) + 1
            return True
        except Exception as e:
            log.warning("KG: failed to upsert node %s: %s", node_id, e)
            return False

    def upsert_edge(
        self,
        *,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Upsert edge into Nexus + persist to JSONL."""
        try:
            properties = properties or {}
            # Persist to JSONL
            edge_record = {
                "source": source_id,
                "target": target_id,
                "type": rel_type,
                "ts": datetime.now(timezone.utc).isoformat(),
                "confidence": float(properties.get("confidence", 1.0)),
                **properties,
            }
            with self._rel_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(edge_record, ensure_ascii=False) + "\n")

            # keep index hot
            try:
                self.edge_index.append_edge(edge_record)
            except Exception:
                log.debug("KG: edge_index append failed (non-fatal)", exc_info=True)

            # Upsert to Nexus
            self.nexus_store.upsert_edge(source_id, target_id, rel_type, properties=properties)
            self._stats["total_edges"] = int(self._stats.get("total_edges", 0)) + 1
            edge_type = rel_type or "UNKNOWN"
            self._stats["edge_types"][edge_type] = int(self._stats["edge_types"].get(edge_type, 0)) + 1
            return True
        except Exception as e:
            log.warning("KG: failed to upsert edge %s→%s: %s", source_id, target_id, e)
            return False

    def search_entities(self, query: str, k: int = 10) -> List[Tuple[str, float, Dict]]:
        if not self.embedding_store:
            return []
        try:
            results = self.embedding_store.search(query, k=k)
            return [(r["node_id"], r["score"], r) for r in results]
        except Exception as e:
            log.warning("KG: search_entities failed: %s", e)
            return []

    def detect_entities(self, text: str) -> List[Dict]:
        # Fallback: empty (requires external NER)
        return self._entity_tool.detect(text=text)

    def build_query_subgraph(self, *, query: str, cfg: SubgraphConfig | None = None) -> dict:
        cfg = cfg or SubgraphConfig()
        return self.subgraph_builder.build(query=query, cfg=cfg)

    def build_tree(
        self,
        *,
        claim: str,
        scorable_id: str,
        max_depth: int = 3,
        min_confidence: float = 0.5,
    ) -> dict:
        """Build verification tree for a claim (simplified)."""
        nodes = [{"id": "root", "text": claim, "type": "claim"}]
        edges = []
        # Placeholder: integrate with future InspectorAgent
        return {"nodes": nodes, "edges": edges, "meta": {"claim": claim}}

    def _plan_to_query(self, plan: Dict[str, Any], k: int = 10) -> str:
        """Create a compact KG query string from a fused plan.

        We intentionally avoid dumping full paper/chat text into the query. Instead we
        use: section title, top domains, a few claims, and top entity surfaces.
        """
        parts: List[str] = []

        section_title = (
            plan.get("section_title")
            or plan.get("section_name")
            or plan.get("title")
            or ""
        )
        if isinstance(section_title, str) and section_title.strip():
            parts.append(section_title.strip())

        # domains: may be list[str] or list[dict]
        domains = plan.get("domains") or []
        dom_names: List[str] = []
        if isinstance(domains, list):
            for d in domains:
                if isinstance(d, str):
                    dom_names.append(d)
                elif isinstance(d, dict):
                    dom_names.append(d.get("domain") or d.get("name") or d.get("label") or "")
        dom_names = [d.strip() for d in dom_names if isinstance(d, str) and d.strip()]
        if dom_names:
            parts.append("domains: " + ", ".join(dom_names[:5]))

        # include a couple of claims (short)
        units = plan.get("units") or []
        if isinstance(units, list):
            for u in units[:3]:
                if not isinstance(u, dict):
                    continue
                claim = u.get("claim") or u.get("text") or ""
                if isinstance(claim, str) and claim.strip():
                    parts.append("claim: " + claim.strip()[:200])

        # entities: plan.get("entities") tends to be dict[type -> (dict|list)]
        ents = plan.get("entities") or {}
        surfaces: List[str] = []
        if isinstance(ents, dict):
            for _, ev in ents.items():
                if isinstance(ev, dict):
                    # {surface: count} or {surface: {...}}
                    for s in ev.keys():
                        surfaces.append(str(s))
                elif isinstance(ev, list):
                    for item in ev:
                        if isinstance(item, str):
                            surfaces.append(item)
                        elif isinstance(item, dict):
                            surfaces.append(item.get("text") or item.get("entity") or "")

        surfaces = [s.strip() for s in surfaces if isinstance(s, str) and s.strip()]
        # de-dupe preserving order
        seen = set()
        uniq: List[str] = []
        for s in surfaces:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            # keep surfaces reasonably small
            if len(s) <= 80:
                uniq.append(s)

        if uniq:
            parts.append("entities: " + ", ".join(uniq[: max(1, int(k)) ]))

        q = " | ".join(parts).strip()
        # clamp query length
        return q[:1000] if q else "plan context"

    def build_context_for_plan(
        self,
        plan_scorable_id: Any,
        query: str = "",
        k: int = 10,
    ) -> Dict[str, Any]:
        """Build context for plan generation using a KG subgraph.

        Backwards compatible:
        - build_context_for_plan(plan_scorable_id: str, query="")
        - build_context_for_plan(plan: dict, k=5, query="")

        Notes:
        - `k` is used as a *query seed budget* (how many entity surfaces to include),
            not as a hard retrieval limit inside SubgraphBuilder.
        """
        plan_id = None
        derived_query = query

        if isinstance(plan_scorable_id, dict):
            plan = plan_scorable_id
            plan_id = (
                plan.get("plan_id")
                or plan.get("id")
                or plan.get("meta", {}).get("knowledge_hash")
                or plan.get("meta", {}).get("plan_scorable_id")
                or ""
            )
            if not derived_query:
                derived_query = self._plan_to_query(plan, k=k)
        else:
            plan_id = str(plan_scorable_id)
            if not derived_query:
                derived_query = f"What supports plan {plan_id}?"

        log.info("KG: build_context_for_plan start | plan_id=%s | k=%s | q_len=%s", plan_id, k, len(derived_query or ""))

        sg = self.build_query_subgraph(
            query=derived_query,
            cfg=SubgraphConfig(max_hops=2, max_nodes=150),
        )
        nodes = [n["id"] for n in sg.get("nodes", [])]
        edges = [
            f"{e['source']} --[{e['type']}]--> {e['target']}"
            for e in sg.get("edges", [])
        ]

        out = {
            "plan_id": plan_id,
            "subgraph_nodes": nodes,
            "subgraph_edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "evidence_rate": (sg.get("meta", {}) or {}).get("stats", {}).get("evidence_rate"),
            "query": derived_query,
        }
        log.info("KG: build_context_for_plan end | plan_id=%s | nodes=%s | edges=%s", plan_id, out["node_count"], out["edge_count"])
        return out

    # --- Lifecycle ----------------------------------------------------------

    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            from stephanie.memory.embedding_store import EmbeddingStore
            from stephanie.memory.nexus_store import NexusStore

            self.nexus_store: NexusStore = self.memory.nexus
            self.embedding_store: EmbeddingStore = self.memory.embedding

            # Warm up edge index (loads file once)
            _ = self.edge_index

            await self._setup_event_listeners()
            self._initialized = True
            log.info("KnowledgeGraphService initialized")
        except Exception as e:
            log.error("KG init failed: %s", e, exc_info=True)
            raise

    async def _setup_event_listeners(self):
        await self.subscribe(
            "knowledge_graph.index_request",
            lambda payload: asyncio.create_task(self.graph_indexer.handle_index_request(payload))
        )
        log.info("KnowledgeGraphService listening for index requests")

    async def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._initialized else "unhealthy",
            "stats": self._stats.copy(),
        }

    async def shutdown(self) -> None:
        log.info("KnowledgeGraphService shutdown complete")

    def _normalize_entity_surface(self, s: str) -> str:
        return EntityCanonicalizer.normalize_surface(s)

    def _add_entity_node(
        self,
        *,
        node_id: str,
        entity: Dict[str, Any],
        domains: List[Dict[str, Any]],
        scorable_id: str,
        scorable_type: str,
        meta: Dict[str, Any],
    ) -> None:
        props = {
            "type": entity.get("type", "ENTITY"),
            "text": entity.get("text", ""),
            "scorable_id": scorable_id,
            "scorable_type": scorable_type,
            "domains": domains,
            "meta": meta,
        }
        self.upsert_node(node_id=node_id, properties=props)

    def _add_relationship(
        self,
        *,
        source_id: str,
        target_id: str,
        rel_type: str,
        confidence: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        properties = properties or {}
        properties.setdefault("confidence", float(confidence))
        self.upsert_edge(
            source_id=source_id,
            target_id=target_id,
            rel_type=rel_type,
            properties=properties,
        )

    async def fetch_text_for_indexing(
        self,
        scorable_id: str,
        inline_text: Optional[str],
        payload: Dict[str, Any],
    ) -> Optional[str]:
        # 1) inline text wins
        if inline_text and inline_text.strip():
            return inline_text

        # 2) attempt memory fetch (adapt to your Scorable store)
        try:
            scorable = self.memory.scorables.get(scorable_id)  # adjust
            if scorable and getattr(scorable, "text", None):
                return scorable.text
        except Exception:
            log.debug("KG: failed to fetch scorable text", exc_info=True)

        # 3) attempt payload fallbacks
        for k in ("content", "document", "body"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v

        return None
