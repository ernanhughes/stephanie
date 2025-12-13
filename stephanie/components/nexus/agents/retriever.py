from __future__ import annotations

import json
import logging
import math
import statistics as stats
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from stephanie.agents.base_agent import BaseAgent
from stephanie.components.nexus.blossom.runner import BlossomRunnerAgent
from stephanie.scoring.scorable import Scorable
from stephanie.services.scoring_service import ScoringService
from stephanie.utils.progress_mixin import ProgressMixin


class NexusLightRAGRetriever(BaseAgent):
    """
    Graph-first retrieval over Nexus, LightRAG-style.

    - Low-level keywords -> Nexus nodes (index_keys, name, etc.)
    - High-level keywords -> Nexus edges (index_keys, value_summary)
    - One-hop expansion around those hits
    - Returns a text bundle built from node/edge value_summary fields.
    """

    def __init__(
        self,
        nexus_store: NexusStore,
        embedding_store: EmbeddingStore,
        keyword_extractor: NexusLightRAGKeywordExtractor,
        config: Optional[NexusLightRAGConfig] = None,
    ):
        super().__init__()
        self.nexus_store = nexus_store
        self.embedding_store = embedding_store
        self.keyword_extractor = keyword_extractor
        self.config = config or NexusLightRAGConfig()

    async def retrieve(
        self,
        query: str,
        goal: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> NexusLightRAGContext:
        # 1. Extract low/high-level keywords
        kw = await self.keyword_extractor.extract(query=query, goal=goal)
        low_kws = kw.get("low_level_keywords", [])
        high_kws = kw.get("high_level_keywords", [])

        # 2. Search nodes for low-level keywords
        node_hits = await self._search_nodes(low_kws)

        # 3. Search edges for high-level keywords
        edge_hits = await self._search_edges(high_kws)

        # 4. Expand 1-hop neighborhood in Nexus
        expanded_nodes, expanded_edges = await self._expand_neighborhood(
            node_hits, edge_hits, hops=self.config.expand_hops
        )

        # 5. Build text context from value_summary
        text_context, truncated = self._build_text_context(expanded_nodes, expanded_edges)

        # 6. Assemble result
        return NexusLightRAGContext(
            query=query,
            goal=goal,
            node_ids=[n.id for n in expanded_nodes],
            edge_ids=[e.id for e in expanded_edges],
            text_context=text_context,
            nodes=expanded_nodes,
            edges=expanded_edges,
            metadata={
                "low_level_keywords": low_kws,
                "high_level_keywords": high_kws,
                "truncated": truncated,
                "extra_context": extra_context or {},
            },
        )

    # --- internal helpers -------------------------------------------------

    async def _search_nodes(self, keywords: List[str]) -> List[NexusNodeORM]:
        if not keywords:
            return []

        # Example: use an embedding/semantic search over (name + index_keys + value_summary)
        # You can swap this for MRQ-based search or your custom NexusStore query.
        query_texts = keywords
        candidates: List[NexusNodeORM] = []

        for kw in query_texts:
            # EmbeddingStore is assumed to have a method to search over a collection
            # You may instead call nexus_store.search_nodes_by_text(kw, top_k=...)
            hits = await self.embedding_store.search_nexus_nodes(
                text=kw,
                top_k=self.config.top_k_nodes,
            )
            candidates.extend(hits)

        # Deduplicate by id, keep best scores if you have them
        unique: Dict[str, NexusNodeORM] = {}
        for node in candidates:
            if node.id not in unique:
                unique[node.id] = node
        return list(unique.values())

    async def _search_edges(self, keywords: List[str]) -> List[NexusEdgeORM]:
        if not keywords:
            return []

        candidates: List[NexusEdgeORM] = []

        for kw in keywords:
            # Same idea, but over edges
            hits = await self.embedding_store.search_nexus_edges(
                text=kw,
                top_k=self.config.top_k_edges,
            )
            candidates.extend(hits)

        unique: Dict[str, NexusEdgeORM] = {}
        for edge in candidates:
            if edge.id not in unique:
                unique[edge.id] = edge
        return list(unique.values())

    async def _expand_neighborhood(
        self,
        nodes: List[NexusNodeORM],
        edges: List[NexusEdgeORM],
        hops: int = 1,
    ) -> Tuple[List[NexusNodeORM], List[NexusEdgeORM]]:
        """
        Expand to neighbors around the initial node/edge hits.
        For now: only 1-hop. You can later generalize to k-hop BFS.
        """
        if hops <= 0:
            return nodes, edges

        node_ids = {n.id for n in nodes}
        edge_ids = {e.id for e in edges}

        # 1-hop expansion:
        # - For each node: add all incident edges + their opposite nodes.
        # - For each edge: add head/tail nodes + their incident edges.
        for node in list(nodes):
            incident_edges = await self.nexus_store.get_edges_for_node(node.id)
            for e in incident_edges:
                if e.id not in edge_ids:
                    edge_ids.add(e.id)
                    edges.append(e)
                # add opposite node
                head_id, tail_id = e.head_id, e.tail_id
                for nid in (head_id, tail_id):
                    if nid not in node_ids:
                        neighbor = await self.nexus_store.get_node(nid)
                        if neighbor:
                            node_ids.add(neighbor.id)
                            nodes.append(neighbor)

        for edge in list(edges):
            # ensure head/tail nodes are included
            for nid in (edge.head_id, edge.tail_id):
                if nid not in node_ids:
                    neighbor = await self.nexus_store.get_node(nid)
                    if neighbor:
                        node_ids.add(neighbor.id)
                        nodes.append(neighbor)

        return nodes, edges

    def _build_text_context(
        self,
        nodes: List[NexusNodeORM],
        edges: List[NexusEdgeORM],
    ) -> Tuple[str, bool]:
        """
        Concatenate value_summary of nodes and edges into a context string.
        Optionally truncate by approximate token budget.
        """
        parts: List[str] = []

        # Nodes first
        for n in nodes:
            if n.value_summary:
                parts.append(f"[NODE: {getattr(n, 'name', n.id)}]\n{n.value_summary}")

        # Then edges
        for e in edges:
            label = getattr(e, "relation_type", e.id)
            if e.value_summary:
                parts.append(f"[EDGE: {label}]\n{e.value_summary}")

        context = "\n\n".join(parts)

        # Simple character-based truncation as a proxy for tokens
        # You can replace this with real token counting.
        max_chars = self.config.max_context_tokens * 4  # ~4 chars per token rough guess
        truncated = False
        if len(context) > max_chars:
            context = context[:max_chars]
            truncated = True

        return context, truncated
