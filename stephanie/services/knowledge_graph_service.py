# stephanie/services/knowledge_graph_service.py
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stephanie.services.service_protocol import Service
from stephanie.services.knowledge_graph.subgraphs.edge_index import JSONLEdgeIndex
from stephanie.services.knowledge_graph.subgraphs.seed_finder import EmbeddingSeedFinder
from stephanie.services.knowledge_graph.subgraphs.subgraph_builder import SubgraphBuilder, SubgraphConfig
from stephanie.services.knowledge_graph.graph_indexer import GraphIndexer
from stephanie.services.knowledge_graph.entity_canonicalizer import EntityCanonicalizer


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
        super().__init__(cfg)
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

    # --- Collaborator factories (lazy init) ----------------------------------

    @property
    def edge_index(self) -> JSONLEdgeIndex:
        if self._edge_index is None:
            self._edge_index = JSONLEdgeIndex(str(self._rel_path), self.logger)
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
            self.logger.warning("KG: failed to upsert node %s: %s", node_id, e)
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
                **properties,
            }
            with self._rel_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(edge_record, ensure_ascii=False) + "\n")

            # Refresh edge index so new edges are queryable immediately
            if self._edge_index is not None:
                self._edge_index.invalidate()

            # Upsert to Nexus
            self.nexus_store.upsert_edge(source_id, target_id, rel_type, properties=properties)
            self._stats["total_edges"] = int(self._stats.get("total_edges", 0)) + 1
            edge_type = rel_type or "UNKNOWN"
            self._stats["edge_types"][edge_type] = int(self._stats["edge_types"].get(edge_type, 0)) + 1
            return True
        except Exception as e:
            self.logger.warning("KG: failed to upsert edge %s→%s: %s", source_id, target_id, e)
            return False

    def search_entities(self, query: str, k: int = 10) -> List[Tuple[str, float, Dict]]:
        if not self.embedding_store:
            return []
        try:
            results = self.embedding_store.search(query, k=k)
            return [(r["node_id"], r["score"], r) for r in results]
        except Exception as e:
            self.logger.warning("KG: search_entities failed: %s", e)
            return []

    def detect_entities(self, text: str) -> List[Dict]:
        # Fallback: empty (requires external NER)
        return []

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

    def build_context_for_plan(
        self, plan_scorable_id: str, query: str = ""
    ) -> Dict[str, Any]:
        """Build context for plan generation using KG subgraph."""
        sg = self.build_query_subgraph(
            query=query or f"What supports plan {plan_scorable_id}?",
            cfg=SubgraphConfig(max_hops=2, max_nodes=150),
        )
        nodes = [n["id"] for n in sg["nodes"]]
        edges = [
            f"{e['source']} --[{e['type']}]--> {e['target']}"
            for e in sg["edges"]
        ]
        return {
            "subgraph_nodes": nodes,
            "subgraph_edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "evidence_rate": sg["meta"]["stats"]["evidence_rate"],
            "query": query,
        }

    # --- Lifecycle ----------------------------------------------------------

    async def initialize(self) -> None:
        if self._initialized:
            return
        try:
            from stephanie.memory.nexus_store import NexusStore
            from stephanie.memory.embedding_store import EmbeddingStore

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
        if not self._initialized:
            return
        try:
            if self.nexus_store:
                await self.nexus_store.shutdown()
            if self.embedding_store:
                await self.embedding_store.shutdown()
        except Exception as e:
            self.logger.error("KG shutdown error: %s", e)
        self._initialized = False
        log.info("KnowledgeGraphService shutdown complete")