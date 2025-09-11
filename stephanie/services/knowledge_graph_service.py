# stephanie/services/knowledge_graph_service.py
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import traceback
import torch
import os
import json
import hashlib

from stephanie.services.service_protocol import Service
from stephanie.models.hnsw_index import HNSWIndex
from stephanie.models.ner_retriever import EntityDetector, NERRetrieverEmbedder
from stephanie.analysis.scorable_classifier import ScorableClassifier


class KnowledgeGraphService(Service):
    """
    Knowledge Graph Service - The "contextual glue" that connects entities across knowledge.
    """

    def __init__(self, cfg: Dict, memory: Any, logger: Any):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.enabled = cfg.get("knowledge_graph", {}).get("enabled", True)
        self._initialized = False
        self._graph = None
        self._entity_detector = None
        self._retriever = None
        self._classifier = None

        # Paths for persistence
        self._rel_path = self.cfg.get("relationship_store", "data/knowledge_graph/relationships.jsonl")
        os.makedirs(os.path.dirname(self._rel_path), exist_ok=True)

        # Config params
        self.entity_threshold = cfg.get("knowledge_graph", {}).get("entity_threshold", 0.65)
        self.relationship_threshold = cfg.get("knowledge_graph", {}).get("relationship_threshold", 0.75)
        self.max_hops = cfg.get("knowledge_graph", {}).get("max_hops", 3)
        self.domain_aware = cfg.get("knowledge_graph", {}).get("domain_aware", True)

        # Stats
        self._stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "node_types": {},
            "edge_types": {},
            "last_update": None,
            "queries": 0,
            "query_time": 0.0,
            "build_time": 0.0
        }

    @property
    def name(self) -> str:
        return "knowledge-graph"

    def initialize(self, **kwargs) -> None:
        if self._initialized or not self.enabled:
            return

        start_time = time.time()
        try:
            self._entity_detector = EntityDetector(
                device=self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            )
            self._retriever = NERRetrieverEmbedder(
                model_name=self.cfg.get("ner_model", "meta-llama/Llama-3-8b"),
                layer=self.cfg.get("ner_layer", 16),
                device=self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                embedding_dim=self.cfg.get("ner_dim", 2048),
                index_path=self.cfg.get("index_path", "data/knowledge_graph/index"),
                logger=self.logger,
                memory=self.memory,
                cfg=self.cfg
            )
            self._classifier = ScorableClassifier(
                memory=self.memory,
                logger=self.logger,
                config_path=self.cfg.get("domain_config", "config/domain/seeds.yaml"),
                metric=self.cfg.get("domain_metric", "cosine")
            )

            self._graph = HNSWIndex(
                dim=self.cfg.get("ner_dim", 2048),
                index_path=self.cfg.get("index_path", "data/knowledge_graph/index"),
                space=self.cfg.get("index_space", "cosine"),
                persistent=True
            )

            self._load_graph()
            self._initialized = True
            self._stats["last_update"] = datetime.now(timezone.utc).isoformat()
            self._stats["build_time"] = time.time() - start_time

            self.logger.log("KnowledgeGraphInitialized", {
                "node_count": self._stats["total_nodes"],
                "edge_count": self._stats["total_edges"],
                "duration": self._stats["build_time"]
            })

        except Exception as e:
            self.logger.log("KnowledgeGraphInitError", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            raise RuntimeError(f"KnowledgeGraph failed to initialize: {e}")

    # -------------------------
    # Entity & Node Management
    # -------------------------
    def _create_entity_node_id(self, entity: Dict[str, Any], scorable_id: str) -> str:
        """
        Stable node ID format: {scorable_id}:{entity_type}:{start}-{end}
        Example: "47:METHOD:123-132"
        """
        return f"{scorable_id}:{entity['type']}:{entity['start']}-{entity['end']}"

    def _add_entity_node(self, node_id: str, entity: Dict[str, Any],
                         domains: List[Dict[str, float]], scorable_id: str,
                         scorable_type: str):
        try:
            metadata = {
                "node_id": node_id,
                "text": entity["text"],
                "type": entity["type"],
                "domains": [d["domain"] for d in domains],
                "domain_scores": {d["domain"]: d["score"] for d in domains},
                "sources": [{
                    "scorable_id": scorable_id,
                    "scorable_type": scorable_type,
                    "start": entity["start"],
                    "end": entity["end"],
                    "source_text": entity["source_text"]
                }],
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            # Use retriever embedding API
            embedding = self._retriever.embed_type_query(entity["text"])
            self._graph.add(embedding, metadata)

            self.logger.log("KnowledgeGraphNodeAdded", {
                "node_id": node_id,
                "text": entity["text"][:50],
                "type": entity["type"],
                "scorable_id": scorable_id,
                "scorable_type": scorable_type
            })

        except Exception as e:
            self.logger.log("KnowledgeGraphNodeAddError", {
                "node_id": node_id,
                "error": str(e)
            })

    # -------------------------
    # Relationships
    # -------------------------
    def _add_relationship(self, source_id: str, target_id: str,
                          rel_type: str, confidence: float):
        """Persist relationship to JSONL + log event."""
        rel = {
            "source": source_id,
            "target": target_id,
            "type": rel_type,
            "confidence": confidence,
            "ts": datetime.now(timezone.utc).isoformat()
        }
        try:
            with open(self._rel_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rel) + "\n")

            self.logger.log("KnowledgeGraphRelationshipAdded", rel)
            self._stats["total_edges"] += 1
            self._track_edge_stats(rel_type)

        except Exception as e:
            self.logger.log("KnowledgeGraphRelationshipAddError", {
                "source": source_id,
                "target": target_id,
                "error": str(e)
            })

    # -------------------------
    # Relationship Builder
    # -------------------------
    def _build_relationships(self, entities: List[Dict[str, Any]],
                             domains: List[Dict[str, float]],
                             scorable_id: str) -> List[Dict[str, Any]]:
        """Build relationships using proximity + heuristics."""
        relationships = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                distance = abs(e1["end"] - e2["start"])
                if distance < 100:
                    rel_type = self._infer_relationship_type(e1, e2)
                    confidence = self._calculate_relationship_confidence(e1, e2, distance, domains)
                    if confidence >= self.relationship_threshold:
                        relationships.append({
                            "source": self._create_entity_node_id(e1, scorable_id),
                            "target": self._create_entity_node_id(e2, scorable_id),
                            "type": rel_type,
                            "confidence": confidence,
                            "evidence": "positional"
                        })
        return relationships

    # -------------------------
    # Path Traversal (with loop guard)
    # -------------------------
    def _find_relationship_path(self, start_node: str,
                                query_entity: Dict[str, Any],
                                max_hops: int) -> List[Dict[str, Any]]:
        """Traverse graph with loop prevention."""
        path = []
        visited = {start_node}
        current = start_node

        for _ in range(max_hops):
            relationships = self.get_relationships(current)
            if not relationships:
                break
            best_rel = max(relationships, key=lambda r: r["confidence"])
            if best_rel["target"] in visited:
                break
            path.append(best_rel)
            visited.add(best_rel["target"])
            current = best_rel["target"]

        return path
