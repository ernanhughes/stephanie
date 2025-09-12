# stephanie/services/knowledge_graph_service.py
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import traceback
import torch
import os
import json
import hashlib
import numpy as np

from stephanie.services.service_protocol import Service
from stephanie.models.hnsw_index import HNSWIndex
from stephanie.models.ner_retriever import EntityDetector, NERRetrieverEmbedder
from stephanie.analysis.scorable_classifier import ScorableClassifier


class KnowledgeGraphService(Service):
    """
    Knowledge Graph Service - The "contextual glue" that connects entities across knowledge.
    Now includes:
      - Consistent embedding space (NER retriever)
      - Deduplicated node insertion
      - Properly enriched search results
      - Accurate stats tracking
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
            "build_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
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
                model_name=self.cfg.get("ner_model", "meta-llama/Llama-3.2-1B-Instruct"),
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

    # -------------------------
    # Public Search API (Fixed)
    # -------------------------
    def search_entities(self, text: str, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search KG using the SAME embedder used during indexing.
        Results cached via entity_cache (keyed by raw text).
        Returns: list of (node_id, score, metadata)
        """
        if not self._initialized or not self.enabled:
            return []

        start_time = time.time()

        try:
            # ⚡ 1. Check cache first
            if hasattr(self.memory, "entity_cache"):
                cached = self.memory.entity_cache.get_by_embedding(text)
                if cached and cached.results_json:
                    self._stats["cache_hits"] += 1
                    self._stats["queries"] += 1
                    self._stats["query_time"] += time.time() - start_time
                    return cached.results_json

            # 🧠 2. Embed text (must match indexing model)
            embedding_vec = self._retriever.embed_type_query(text)

            # 🔍 3. Search HNSW index
            raw_results = self._graph.search(embedding_vec, k=k)

            # 🎯 4. Enrich results
            enriched = [
                (
                    node_id,
                    float(score),
                    {**meta, "node_id": node_id, "score": float(score)},
                )
                for node_id, score, meta in raw_results
            ]

            self.memory.entity_cache.upsert(
                text,    
                enriched,
                embedding_type="ner",
                scorable_type="query",
            )
            self._stats["cache_misses"] += 1

            # 📊 6. Stats
            self._stats["queries"] += 1
            self._stats["query_time"] += time.time() - start_time
            return enriched

        except Exception as e:
            self.logger.log("KnowledgeGraphSearchError", {
                "text": text,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return []


    def build_context_for_plan(self, plan: dict, k: int = 5) -> dict:
        """Return KG neighbors per claim_id for drafting/scoring."""
        out = {"neighbors": {}}
        for u in plan.get("units", []):
            q = (u.get("claim") or u.get("evidence") or "").strip()
            cid = u.get("claim_id")
            if not q or not cid:
                continue
            hits = self.search_entities(q, k=k)  # [(node_id, score, meta), ...]
            out["neighbors"][cid] = [
                {
                    "node_id": nid,
                    "text": meta.get("text", ""),
                    "type": meta.get("type", ""),
                    "score": float(score),
                    "sources": meta.get("sources", []),
                    "domains": meta.get("domains", []),
                }
                for (nid, score, meta) in hits
            ]
        return out

    # -------------------------
    # Entity & Node Management
    # -------------------------
    def _create_entity_node_id(self, entity: Dict[str, Any], scorable_id: str) -> str:
        return f"{scorable_id}:{entity['type']}:{entity['start']}-{entity['end']}"

    def _add_entity_node(self, node_id: str, entity: Dict[str, Any],
                         domains: List[Dict[str, float]], scorable_id: str,
                         scorable_type: str):
        try:
            # Avoid duplicates
            if self._graph.has_metadata(node_id):  # assuming HNSWIndex supports this
                return  # skip duplicate

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

            # Use same retriever → consistent embedding space
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
    def _load_graph(self):
        """Load existing nodes and relationships."""
        self._stats["total_nodes"] = self._graph.ntotal if hasattr(self._graph, 'ntotal') else 0

        # Rebuild edge stats
        if os.path.exists(self._rel_path):
            with open(self._rel_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rel = json.loads(line.strip())
                        self._stats["total_edges"] += 1
                        self._track_edge_stats(rel["type"])
                    except Exception as e:
                        self.logger.log("KnowledgeGraphLoadError", {"error": str(e)})

    def _track_edge_stats(self, rel_type: str):
        self._stats["edge_types"][rel_type] = self._stats["edge_types"].get(rel_type, 0) + 1

    def _track_node_stats(self, node_type: str):
        self._stats["node_types"][node_type] = self._stats["node_types"].get(node_type, 0) + 1

    def get_relationships(self, node_id: str) -> List[Dict[str, Any]]:
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
        base_score = np.exp(-distance / 50.0)  # exponential decay
        domain_bonus = 0.1 if any(d["domain"] in {"ml", "nlp"} for d in domains) else 0.0
        proximity_bonus = 0.1 if distance < 20 else 0.0
        return max(min(base_score + domain_bonus + proximity_bonus, 1.0), 0.0)
    
    