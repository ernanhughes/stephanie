# stephanie/services/knowledge_graph_service.py
from __future__ import annotations

import os
import json
import time
import torch
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from stephanie.services.service_protocol import Service
from stephanie.models.hnsw_index import HNSWIndex
from stephanie.models.ner_retriever import EntityDetector, NERRetrieverEmbedder
from stephanie.analysis.scorable_classifier import ScorableClassifier


class KnowledgeGraphService(Service):
    """
    Knowledge Graph Service
    -----------------------
    The "contextual glue" connecting entities across knowledge.
    
    Modes:
      - sync     → blocks on add
      - deferred → queues adds for batch flush
      - evented  → publishes to KnowledgeBus for async processing
    """

    MODES = {"sync", "deferred", "evented"}

    def __init__(self, cfg: Dict[str, Any], memory: Any, logger: Any):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.enabled = cfg.get("knowledge_graph", {}).get("enabled", True)
        self.mode = cfg.get("knowledge_graph", {}).get("mode", "sync")
        if self.mode not in self.MODES:
            raise ValueError(f"Invalid knowledge_graph.mode: {self.mode}, must be one of {self.MODES}")

        # Components
        self._graph: Optional[HNSWIndex] = None
        self._entity_detector: Optional[EntityDetector] = None
        self._retriever: Optional[NERRetrieverEmbedder] = None
        self._classifier: Optional[ScorableClassifier] = None

        # State
        self._initialized = False
        self._rel_path = cfg.get("relationship_store", "data/knowledge_graph/relationships.jsonl")
        os.makedirs(os.path.dirname(self._rel_path), exist_ok=True)

        # Parameters
        self.entity_threshold = cfg.get("knowledge_graph", {}).get("entity_threshold", 0.65)
        self.relationship_threshold = cfg.get("knowledge_graph", {}).get("relationship_threshold", 0.75)
        self.max_hops = cfg.get("knowledge_graph", {}).get("max_hops", 3)
        self.domain_aware = cfg.get("knowledge_graph", {}).get("domain_aware", True)

        # Deferred mode queues
        self._pending_nodes: List[Tuple[Dict[str, Any], Dict[str, float], str, str]] = []
        self._pending_relationships: List[Dict[str, Any]] = []

        # Stats
        self._stats: Dict[str, Any] = {
            "total_nodes": 0,
            "total_edges": 0,
            "node_types": {},
            "edge_types": {},
            "last_update": None,
            "queries": 0,
            "query_time": 0.0,
            "build_time": 0.0,
            "deferred_node_count": 0,
            "events_published": 0,
            "initialized_at": None,
        }

    # === Service Protocol ===
    def initialize(self, **kwargs) -> None:
        if self._initialized or not self.enabled:
            return
        start_time = time.time()
        try:
            device = self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            self._entity_detector = EntityDetector(device=device)
            self._retriever = NERRetrieverEmbedder(
                model_name=self.cfg.get("ner_model", "meta-llama/Llama-3.2-1B-Instruct"),
                layer=self.cfg.get("ner_layer", 16),
                device=device,
                embedding_dim=self.cfg.get("ner_dim", 2048),
                index_path=self.cfg.get("index_path", "data/knowledge_graph/index"),
                logger=self.logger,
                memory=self.memory,
                cfg=self.cfg,
            )
            self._classifier = ScorableClassifier(
                memory=self.memory,
                logger=self.logger,
                config_path=self.cfg.get("domain_config", "config/domain/seeds.yaml"),
                metric=self.cfg.get("domain_metric", "cosine"),
            )
            self._graph = HNSWIndex(
                dim=self.cfg.get("ner_dim", 2048),
                index_path=self.cfg.get("index_path", "data/knowledge_graph/index"),
                space=self.cfg.get("index_space", "cosine"),
                persistent=True,
            )

            self._load_graph()
            self._initialized = True
            self._stats["last_update"] = datetime.now(timezone.utc).isoformat()
            self._stats["build_time"] = time.time() - start_time
            self._stats["initialized_at"] = datetime.now(timezone.utc).isoformat()

            self.logger.log("KnowledgeGraphInitialized", {
                "mode": self.mode,
                "node_count": self._stats["total_nodes"],
                "edge_count": self._stats["total_edges"],
                "duration": self._stats["build_time"],
            })
        except Exception as e:
            self.logger.log("KnowledgeGraphInitError", {"error": str(e), "traceback": traceback.format_exc()})
            raise RuntimeError(f"KnowledgeGraph failed to initialize: {e}")

    def health_check(self) -> Dict[str, Any]:
        status = "healthy" if self._initialized else "unhealthy"
        return {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                **self._stats,
                "mode": self.mode,
                "deferred_queue": len(self._pending_nodes),
            },
            "dependencies": {
                "embedding_index": "ready" if self._graph else "missing",
                "entity_detector": "ready" if self._entity_detector else "missing",
                "classifier": "ready" if self._classifier else "missing",
            },
        }

    def shutdown(self) -> None:
        """Clean shutdown, flush deferred, clear state."""
        if self.mode == "deferred":
            try:
                self.flush()
            except Exception:
                pass
        self._initialized = False
        self._entity_detector = None
        self._retriever = None
        self._classifier = None
        self._graph = None
        self._pending_nodes.clear()
        self._pending_relationships.clear()
        self.logger.log("KnowledgeGraphShutdown", {"status": "stopped"})

    @property
    def name(self) -> str:
        return "knowledge-graph-v1"

    # -------------------------
    # Public API — Use This!
    # -------------------------
    def ingest_scorable(
        self,
        scorable_id: str,
        text: str,
        scorable_type: str = "document"
    ) -> None:
        """
        Main entry point from agents.
        Does NOT block in deferred/evented mode.
        """
        if not self._initialized or not self.enabled:
            return

        try:
            # Detect entities
            entities = self._entity_detector.extract_entities(text)
            filtered_ents = [e for e in entities if e.get("score", 0) >= self.entity_threshold]
            if not filtered_ents:
                return

            # Classify domains
            domains = self._classifier.classify(scorable_id, text)

            # Build relationships
            relationships = self._build_relationships(filtered_ents, domains, scorable_id)

            # Handle based on mode
            if self.mode == "sync":
                self._add_entities_sync(filtered_ents, domains, scorable_id, scorable_type, text)
                self._add_relationships_sync(relationships)
            elif self.mode == "deferred":
                self._queue_for_later(filtered_ents, domains, scorable_id, scorable_type, relationships)
            elif self.mode == "evented":
                self._publish_to_bus(scorable_id, text, filtered_ents, domains, relationships)

        except Exception as e:
            self.logger.log("KnowledgeGraphIngestError", {
                "scorable_id": scorable_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            })

    # -------------------------
    # Sync Mode — Immediate Add
    # -------------------------
    def _add_entities_sync(self, entities: List[Dict], domains: List[Dict], scorable_id: str, scorable_type: str, text: str):
        for ent in entities:
            ent["source_text"] = text[ent["start"]:ent["end"]]
            node_id = self._create_entity_node_id(ent, scorable_id)
            self._add_entity_node(node_id, ent, domains, scorable_id, scorable_type)
            self._track_node_stats(ent["type"])

    def _add_relationships_sync(self, relationships: List[Dict]):
        for rel in relationships:
            self._add_relationship(rel["source"], rel["target"], rel["type"], rel["confidence"])

    # -------------------------
    # Deferred Mode — Queue Locally
    # -------------------------
    def _queue_for_later(
        self,
        entities: List[Dict],
        domains: List[Dict],
        scorable_id: str,
        scorable_type: str,
        relationships: List[Dict]
    ):
        self._pending_nodes.extend([
            (ent, domains, scorable_id, scorable_type)
            for ent in entities
        ])
        self._pending_relationships.extend(relationships)
        self._stats["deferred_node_count"] += len(entities)

    def flush(self) -> None:
        """Call this at end of batch to process all queued items."""
        if not self._pending_nodes and not self._pending_relationships:
            return

        start_time = time.time()
        added = 0
        try:
            for args in self._pending_nodes:
                ent, domains, sid, stype = args
                ent["source_text"] = ent.get("source_text", "")
                node_id = self._create_entity_node_id(ent, sid)
                self._add_entity_node(node_id, ent, domains, sid, stype)
                added += 1

            for rel in self._pending_relationships:
                self._add_relationship(rel["source"], rel["target"], rel["type"], rel["confidence"])

            # Clear queue
            self._pending_nodes.clear()
            self._pending_relationships.clear()

            self.logger.log("KnowledgeGraphFlushed", {
                "node_count": added,
                "relationship_count": len(self._pending_relationships),
                "duration": time.time() - start_time
            })

        except Exception as e:
            self.logger.log("KnowledgeGraphFlushError", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })

    # -------------------------
    # Evented Mode — Publish to Bus
    # -------------------------
    def _publish_to_bus(
        self,
        scorable_id: str,
        text: str,
        entities: List[Dict],
        domains: List[Dict],
        relationships: List[Dict]
    ):
        event = {
            "event_type": "knowledge_graph.index_request",
            "payload": {
                "scorable_id": scorable_id,
                "text": text,
                "entities": entities,
                "domains": domains,
                "relationships": relationships,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        try:
            self.memory.bus.publish(event)
            self._stats["events_published"] += 1
            self.logger.log("KnowledgeGraphIndexEventPublished", {
                "scorable_id": scorable_id,
                "entity_count": len(entities),
                "event_type": "index_request"
            })
        except Exception as e:
            self.logger.log("KnowledgeGraphPublishError", {
                "error": str(e),
                "scorable_id": scorable_id
            })

    # -------------------------
    # Entity & Node Management
    # -------------------------
    def _create_entity_node_id(self, entity: Dict[str, Any], scorable_id: str) -> str:
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
                    "source_text": entity.get("source_text", "")
                }],
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            embedding = self._retriever.embed_type_query(entity["text"])
            self._graph.add(embedding, metadata)

            self.logger.log("KnowledgeGraphNodeAdded", {
                "node_id": node_id,
                "text": entity["text"][:50],
                "type": entity["type"],
                "scorable_id": scorable_id,
                "scorable_type": scorable_type
            })
            self._stats["total_nodes"] += 1
            self._track_node_stats(entity["type"])

        except Exception as e:
            self.logger.log("KnowledgeGraphNodeAddError", {
                "node_id": node_id,
                "error": str(e)
            })

    # -------------------------
    # Relationships
    # -------------------------
    def _add_relationship(self, source_id: str, target_id: str, rel_type: str, confidence: float):
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
    # Path Traversal
    # -------------------------
    def _find_relationship_path(self, start_node: str,
                                query_entity: Dict[str, Any],
                                max_hops: int) -> List[Dict[str, Any]]:
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

    # -------------------------
    # Helpers
    # -------------------------
    def _load_graph(self) -> None:
        """Load existing nodes and relationships into stats counters."""
        try:
            # Count nodes if available
            self._stats["total_nodes"] = (
                self._graph.ntotal if hasattr(self._graph, "ntotal") else 0
            )

            # Count edges from relationships file
            if os.path.exists(self._rel_path):
                with open(self._rel_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            rel = json.loads(line.strip())
                            self._stats["total_edges"] += 1
                            rel_type = rel.get("type", "unknown")
                            self._stats["edge_types"][rel_type] = (
                                self._stats["edge_types"].get(rel_type, 0) + 1
                            )
                        except Exception:
                            continue

            self.logger.log("KnowledgeGraphLoaded", {
                "nodes": self._stats["total_nodes"],
                "edges": self._stats["total_edges"],
            })
        except Exception as e:
            self.logger.log("KnowledgeGraphLoadError", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })

    def _track_edge_stats(self, rel_type: str):
        self._stats["edge_types"][rel_type] = self._stats["edge_types"].get(rel_type, 0) + 1

    def _track_node_stats(self, node_type: str):
        self._stats["node_types"][node_type] = self._stats["node_types"].get(node_type, 0) + 1

    def get_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """Retrieve all outgoing relationships."""
        rels = []
        if not os.path.exists(self._rel_path):
            return rels
        with open(self._rel_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rel = json.loads(line.strip())
                    if rel["source"] == node_id:
                        rels.append(rel)
                except:
                    continue
        return sorted(rels, key=lambda x: -x["confidence"])

    def _infer_relationship_type(self, e1: Dict, e2: Dict) -> str:
        ordered = e1["end"] < e2["start"]
        first, second = (e1, e2) if ordered else (e2, e1)
        type_pairs = {
            ("METHOD", "DATASET"): "evaluates",
            ("DATASET", "METRIC"): "measured_by",
            ("MODEL", "TASK"): "performs",
            ("AUTHOR", "PAPER"): "wrote",
            ("PAPER", "METHOD"): "introduces"
        }
        return type_pairs.get((first["type"], second["type"]), "related_to")

    def _calculate_relationship_confidence(self, e1: Dict, e2: Dict, distance: int, domains: List[Dict]) -> float:
        base_score = 1.0 - (distance / 100)
        domain_bonus = 0.1 if any(d["domain"] in {"ml", "nlp"} for d in domains) else 0.0
        proximity_bonus = 0.1 if distance < 20 else 0.0
        return max(min(base_score + domain_bonus + proximity_bonus, 1.0), 0.0)