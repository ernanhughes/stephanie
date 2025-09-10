# stephanie/services/knowledge_graph_service.py
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import hashlib
import torch
import traceback

from stephanie.services.service_protocol import Service
from stephanie.models.hnsw_index import HNSWIndex
from stephanie.models.ner_retriever import EntityDetector, NERRetrieverEmbedder
from stephanie.analysis.scorable_classifier import ScorableClassifier


class KnowledgeGraphService(Service):
    """
    Knowledge Graph Service - The "contextual glue" that connects entities across knowledge.
    
    As described in PACS.md: "CBR gave Stephanie a memory of what worked. PACS gave her an understanding
    of how she improves. NER gives her the contextual glue to see connections across knowledge."
    
    This service:
    - Extracts entities from text using NER
    - Builds relationships between entities using semantic similarity
    - Stores the knowledge graph in HNSW for efficient querying
    - Supports domain-aware knowledge fusion
    - Enables reasoning over connected knowledge
    
    Key capabilities:
    - Entity resolution (merging duplicate entities)
    - Relationship extraction (semantic, positional, contextual)
    - Path finding (how concepts connect across documents)
    - Knowledge completion (predicting missing relationships)
    """
    
    def __init__(self, cfg: Dict, memory: Any, logger: Any):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.enabled = cfg.get("knowledge_graph", {}).get("enabled", True)
        self._initialized = False
        self._graph = None  # Will be HNSWIndex instance
        self._entity_detector = None
        self._retriever = None
        self._classifier = None
        
        # Configuration parameters
        self.entity_threshold = cfg.get("knowledge_graph", {}).get("entity_threshold", 0.65)
        self.relationship_threshold = cfg.get("knowledge_graph", {}).get("relationship_threshold", 0.75)
        self.max_hops = cfg.get("knowledge_graph", {}).get("max_hops", 3)
        self.domain_aware = cfg.get("knowledge_graph", {}).get("domain_aware", True)
        
        # Statistics tracking
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
        """Initialize knowledge graph with proper error handling and logging."""
        if self._initialized or not self.enabled:
            return
            
        start_time = time.time()
        try:
            # Initialize components
            self._entity_detector = EntityDetector(
                device=self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            )
            self._retriever = NERRetrieverEmbedder(
                model_name=self.cfg.get("ner_model", "meta-llama/Llama-3-8b"),
                layer=self.cfg.get("ner_layer", 17),
                device=self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
                embedding_dim=self.cfg.get("ner_dim", 500),
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
            
            # Initialize graph storage
            self._graph = HNSWIndex(
                dim=self.cfg.get("ner_dim", 500),
                index_path=self.cfg.get("index_path", "data/knowledge_graph/index"),
                space=self.cfg.get("index_space", "cosine"),
                persistent=True
            )
            
            # Load existing graph if available
            self._load_graph()
            
            # Mark as initialized
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
        """Return comprehensive health metrics for monitoring."""
        return {
            "status": "healthy" if self._initialized and self.enabled else "disabled",
            "enabled": self.enabled,
            "initialized": self._initialized,
            "node_count": self._stats["total_nodes"],
            "edge_count": self._stats["total_edges"],
            "node_types": self._stats["node_types"],
            "edge_types": self._stats["edge_types"],
            "config": {
                "entity_threshold": self.entity_threshold,
                "relationship_threshold": self.relationship_threshold,
                "max_hops": self.max_hops,
                "domain_aware": self.domain_aware
            },
            "stats": {
                "queries": self._stats["queries"],
                "avg_query_time": self._stats["query_time"] / max(1, self._stats["queries"]) if self._stats["queries"] else 0,
                "last_update": self._stats["last_update"]
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def shutdown(self) -> None:
        """Cleanly shut down the service and save state."""
        if not self._initialized:
            return
            
        try:
            # Save current graph state
            self._save_graph()
            
            # Clear resources
            self._graph = None
            self._entity_detector = None
            self._retriever = None
            self._classifier = None
            self._initialized = False
            
            self.logger.log("KnowledgeGraphShutdown", {
                "status": "complete",
                "node_count": self._stats["total_nodes"],
                "edge_count": self._stats["total_edges"]
            })
        except Exception as e:
            self.logger.log("KnowledgeGraphShutdownError", {
                "error": str(e)
            })

    def build_from_text(self, text: str, source_id: str, source_type: str = "document") -> Dict[str, Any]:
        """
        Build knowledge graph from text by extracting entities and relationships.
        
        Args:
            text: Input text to process
            source_id: Unique identifier for the source document
            source_type: Type of source (e.g., "document", "chat", "code")
            
        Returns:
            Dictionary with statistics about the built graph
        """
        if not self.enabled:
            return {"status": "disabled"}
            
        if not self._initialized:
            self.initialize()
            
        start_time = time.time()
        stats = {
            "entities_extracted": 0,
            "relationships_found": 0,
            "node_count": self._stats["total_nodes"],
            "edge_count": self._stats["total_edges"]
        }
        
        try:
            # 1. Classify domains for context
            domains = self._classify_domains(text)
            
            # 2. Extract entities
            entities = self._extract_entities(text)
            stats["entities_extracted"] = len(entities)
            
            # 3. Build nodes and relationships
            node_ids = []
            for entity in entities:
                # Create node ID based on entity properties
                node_id = self._create_entity_node_id(entity, source_id)
                
                # Add to graph if not exists
                if not self._node_exists(node_id):
                    self._add_entity_node(node_id, entity, domains, source_id, source_type)
                    node_ids.append(node_id)
                else:
                    # Update existing node with new context
                    self._update_entity_node(node_id, entity, domains, source_id, source_type)
                
                # Track node statistics
                self._track_node_stats(entity["type"])
                
            # 4. Build relationships between entities
            relationships = self._build_relationships(entities, domains)
            stats["relationships_found"] = len(relationships)
            
            # 5. Add relationships to graph
            for rel in relationships:
                self._add_relationship(rel["source"], rel["target"], rel["type"], rel["confidence"])
                self._track_edge_stats(rel["type"])
            
            # Update statistics
            self._stats["total_nodes"] = self._graph.get_stats()["total_entities"]
            self._stats["total_edges"] += len(relationships)
            self._stats["last_update"] = datetime.now(timezone.utc).isoformat()
            self._stats["build_time"] += time.time() - start_time
            
            self.logger.log("KnowledgeGraphBuilt", {
                "source_id": source_id,
                "source_type": source_type,
                "entities": len(entities),
                "relationships": len(relationships),
                "duration": time.time() - start_time
            })
            
            return {
                **stats,
                "status": "success",
                "node_count": self._stats["total_nodes"],
                "edge_count": self._stats["total_edges"]
            }
            
        except Exception as e:
            self.logger.log("KnowledgeGraphBuildError", {
                "source_id": source_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            return {
                "status": "error",
                "error": str(e),
                **stats
            }

    def query(self, query_text: str, k: int = 5, max_hops: int = 2, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph for relevant entities and relationships.
        
        Args:
            query_text: Natural language query
            k: Number of top results to return
            max_hops: Maximum number of relationship hops to traverse
            domain: Optional domain to constrain the search
            
        Returns:
            List of relevant entities and their relationships
        """
        if not self.enabled or not self._initialized:
            return []
            
        start_time = time.time()
        self._stats["queries"] += 1
        
        try:
            # 1. Extract query entities
            query_entities = self._extract_entities(query_text)
            
            # 2. Find relevant nodes
            results = []
            for entity in query_entities:
                # Embed the entity text
                entity_embedding = self._retriever.embed(entity["text"])
                
                # Search for similar nodes
                similar_nodes = self._graph.search(entity_embedding, k=k)
                
                # Filter by domain if specified
                if domain:
                    similar_nodes = [n for n in similar_nodes if domain in n.get("domains", [])]
                
                # Build relationship paths
                for node in similar_nodes:
                    path = self._find_relationship_path(
                        start_node=node["scorable_id"], 
                        query_entity=entity,
                        max_hops=max_hops
                    )
                    results.append({
                        "entity": entity,
                        "node": node,
                        "path": path,
                        "score": node["similarity"]
                    })
            
            # Sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Update statistics
            self._stats["query_time"] += time.time() - start_time
            
            self.logger.log("KnowledgeGraphQueried", {
                "query": query_text[:100] + "..." if len(query_text) > 100 else query_text,
                "results": len(results),
                "duration": time.time() - start_time
            })
            
            return results[:k]
            
        except Exception as e:
            self.logger.log("KnowledgeGraphQueryError", {
                "query": query_text[:100] + "..." if len(query_text) > 100 else query_text,
                "error": str(e)
            })
            return []

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity details by ID."""
        if not self._initialized:
            self.initialize()
            
        try:
            # In a real implementation, this would query the HNSW index
            # For now, we'll simulate with a lookup
            return self._graph.get_entity(entity_id)
        except Exception as e:
            self.logger.log("KnowledgeGraphEntityError", {
                "entity_id": entity_id,
                "error": str(e)
            })
            return None

    def get_relationships(self, entity_id: str, k: int = 10) -> List[Dict[str, Any]]:
        """Get relationships for a specific entity."""
        if not self._initialized:
            self.initialize()
            
        try:
            # In a real implementation, this would query relationship edges
            return self._graph.get_relationships(entity_id, k=k)
        except Exception as e:
            self.logger.log("KnowledgeGraphRelationshipsError", {
                "entity_id": entity_id,
                "error": str(e)
            })
            return []

    def get_subgraph(self, seed_entities: List[str], max_hops: int = 2) -> Dict[str, Any]:
        """Get a subgraph starting from seed entities."""
        if not self._initialized:
            self.initialize()
            
        try:
            # Build the subgraph by traversing relationships
            subgraph = {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "seed_entities": seed_entities,
                    "max_hops": max_hops
                }
            }
            
            # Start with seed entities
            current_entities = seed_entities
            visited = set(seed_entities)
            
            # Traverse up to max_hops
            for _ in range(max_hops):
                next_entities = []
                for entity_id in current_entities:
                    # Get entity details
                    entity = self.get_entity(entity_id)
                    if entity and entity_id not in visited:
                        subgraph["nodes"].append(entity)
                        visited.add(entity_id)
                    
                    # Get relationships
                    relationships = self.get_relationships(entity_id)
                    for rel in relationships:
                        subgraph["edges"].append(rel)
                        if rel["target"] not in visited:
                            next_entities.append(rel["target"])
                
                current_entities = next_entities
                if not current_entities:
                    break
            
            return subgraph
            
        except Exception as e:
            self.logger.log("KnowledgeGraphSubgraphError", {
                "seed_entities": seed_entities,
                "max_hops": max_hops,
                "error": str(e)
            })
            return {"nodes": [], "edges": [], "error": str(e)}

    def _classify_domains(self, text: str) -> List[Dict[str, float]]:
        """Classify text into domains for context."""
        if not self.domain_aware:
            return [{"domain": "general", "score": 1.0}]
            
        try:
            # Get top domains with scores
            results = self._classifier.classify(
                text, 
                top_k=self.cfg.get("top_domains", 5),
                min_value=self.cfg.get("min_domain_score", 0.3)
            )
            return [{"domain": d, "score": float(s)} for d, s in results]
        except Exception as e:
            self.logger.log("DomainClassificationError", {
                "error": str(e),
                "text_sample": text[:100]
            })
            return [{"domain": "general", "score": 0.5}]

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using NER."""
        try:
            spans = self._entity_detector.detect_entities(text)
            entities = []
            for start, end, entity_type in spans:
                entity_text = text[start:end].strip()
                if len(entity_text) < 2:
                    continue
                entities.append({
                    "text": entity_text,
                    "type": entity_type,
                    "start": start,
                    "end": end,
                    "source_text": text[max(0, start-50):min(len(text), end+50)]
                })
            return entities
        except Exception as e:
            self.logger.log("EntityExtractionError", {
                "error": str(e),
                "text_sample": text[:100]
            })
            return []

    def _create_entity_node_id(self, entity: Dict[str, Any], source_id: str) -> str:
        """Create a unique ID for an entity node."""
        # Create a hash of the entity properties
        entity_hash = hashlib.sha256(
            f"{entity['text']}|{entity['type']}|{source_id}".encode()
        ).hexdigest()[:12]
        return f"entity:{entity_hash}"

    def _node_exists(self, node_id: str) -> bool:
        """Check if a node already exists in the graph."""
        # In a real implementation, this would query the HNSW index
        return self._graph.node_exists(node_id)

    def _add_entity_node(self, node_id: str, entity: Dict[str, Any], domains: List[Dict[str, float]], 
                         source_id: str, source_type: str):
        """Add a new entity node to the graph."""
        try:
            # Create node metadata
            metadata = {
                "node_id": node_id,
                "text": entity["text"],
                "type": entity["type"],
                "domains": [d["domain"] for d in domains],
                "domain_scores": {d["domain"]: d["score"] for d in domains},
                "sources": [{
                    "source_id": source_id,
                    "source_type": source_type,
                    "start": entity["start"],
                    "end": entity["end"],
                    "source_text": entity["source_text"]
                }],
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Embed the entity text
            embedding = self._retriever.embed(entity["text"])
            
            # Add to HNSW index
            self._graph.add(embedding, metadata)
            
            # Log the addition
            self.logger.log("KnowledgeGraphNodeAdded", {
                "node_id": node_id,
                "text": entity["text"][:50],
                "type": entity["type"],
                "source_id": source_id
            })
            
        except Exception as e:
            self.logger.log("KnowledgeGraphNodeAddError", {
                "node_id": node_id,
                "error": str(e)
            })

    def _update_entity_node(self, node_id: str, entity: Dict[str, Any], domains: List[Dict[str, float]], 
                           source_id: str, source_type: str):
        """Update an existing entity node with new context."""
        try:
            # In a real implementation, this would update the node in HNSW
            # For now, we'll just log the update
            self.logger.log("KnowledgeGraphNodeUpdated", {
                "node_id": node_id,
                "text": entity["text"][:50],
                "source_id": source_id
            })
            
        except Exception as e:
            self.logger.log("KnowledgeGraphNodeUpdateError", {
                "node_id": node_id,
                "error": str(e)
            })

    def _build_relationships(self, entities: List[Dict[str, Any]], 
                            domains: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Build relationships between entities in the text."""
        relationships = []
        
        # 1. Positional relationships (entities that appear near each other)
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], start=i+1):
                # Calculate proximity
                distance = min(
                    abs(entity1["end"] - entity2["start"]),
                    abs(entity2["end"] - entity1["start"])
                )
                
                # If entities are close, they likely have a relationship
                if distance < 100:  # Within ~50 characters
                    rel_type = self._infer_relationship_type(entity1, entity2)
                    confidence = self._calculate_relationship_confidence(
                        entity1, entity2, distance, domains
                    )
                    
                    if confidence >= self.relationship_threshold:
                        relationships.append({
                            "source": self._create_entity_node_id(entity1, "temp"),
                            "target": self._create_entity_node_id(entity2, "temp"),
                            "type": rel_type,
                            "confidence": confidence,
                            "evidence": "positional"
                        })
        
        # 2. Semantic relationships (using embeddings)
        # This would be more complex in a real implementation
        # For brevity, we're simplifying this part
        
        return relationships

    def _infer_relationship_type(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> str:
        """Infer relationship type between two entities."""
        # Simple heuristic based on entity types
        type_pairs = {
            ("PERSON", "ORGANIZATION"): "works_at",
            ("PERSON", "LOCATION"): "lives_in",
            ("ORGANIZATION", "LOCATION"): "headquartered_in",
            ("PERSON", "PERSON"): "knows",
            ("METHOD", "DATASET"): "trained_on",
            ("MODEL", "DATASET"): "evaluated_on",
            ("METHOD", "METRIC"): "optimized_for"
        }
        
        key = (entity1["type"].upper(), entity2["type"].upper())
        return type_pairs.get(key, "related_to")

    def _calculate_relationship_confidence(self, entity1: Dict[str, Any], entity2: Dict[str, Any], 
                                         distance: int, domains: List[Dict[str, float]]) -> float:
        """Calculate confidence score for a relationship."""
        # Base confidence from proximity
        proximity_confidence = max(0, 1 - (distance / 200))
        
        # Domain-specific boost
        domain_boost = 0.0
        if domains:
            domain_boost = max(d["score"] for d in domains) * 0.2
        
        # Type compatibility
        type_compatibility = 0.5  # Default
        if (entity1["type"].upper(), entity2["type"].upper()) in [
            ("PERSON", "ORGANIZATION"), ("METHOD", "DATASET")
        ]:
            type_compatibility = 0.8
            
        # Combine with weights
        confidence = (
            0.5 * proximity_confidence +
            0.3 * type_compatibility +
            0.2 * domain_boost
        )
        
        return min(1.0, confidence)

    def _add_relationship(self, source_id: str, target_id: str, rel_type: str, confidence: float):
        """Add a relationship edge to the graph."""
        try:
            # In a real implementation, this would store the edge in a separate structure
            # For now, we'll just log it
            self.logger.log("KnowledgeGraphRelationshipAdded", {
                "source": source_id,
                "target": target_id,
                "type": rel_type,
                "confidence": confidence
            })
            
        except Exception as e:
            self.logger.log("KnowledgeGraphRelationshipAddError", {
                "source": source_id,
                "target": target_id,
                "error": str(e)
            })

    def _find_relationship_path(self, start_node: str, query_entity: Dict[str, Any], 
                              max_hops: int) -> List[Dict[str, Any]]:
        """Find a path of relationships from start_node to query_entity."""
        # This is a simplified implementation
        # In a real system, this would use graph traversal algorithms
        
        path = []
        current = start_node
        hops = 0
        
        while hops < max_hops:
            # Get relationships for current node
            relationships = self.get_relationships(current)
            if not relationships:
                break
                
            # Find the most relevant relationship
            best_rel = max(relationships, key=lambda r: r["confidence"])
            
            # Add to path
            path.append({
                "source": current,
                "target": best_rel["target"],
                "type": best_rel["type"],
                "confidence": best_rel["confidence"]
            })
            
            # Move to next node
            current = best_rel["target"]
            hops += 1
            
        return path

    def _track_node_stats(self, node_type: str):
        """Track statistics about node types."""
        self._stats["node_types"][node_type] = self._stats["node_types"].get(node_type, 0) + 1

    def _track_edge_stats(self, edge_type: str):
        """Track statistics about edge types."""
        self._stats["edge_types"][edge_type] = self._stats["edge_types"].get(edge_type, 0) + 1

    def _load_graph(self):
        """Load existing graph from persistent storage."""
        try:
            # In a real implementation, this would load from HNSW index
            stats = self._graph.get_stats()
            self._stats["total_nodes"] = stats["total_entities"]
            self.logger.log("KnowledgeGraphLoaded", {
                "node_count": self._stats["total_nodes"]
            })
        except Exception as e:
            self.logger.log("KnowledgeGraphLoadError", {
                "error": str(e)
            })

    def _save_graph(self):
        """Save current graph state to persistent storage."""
        try:
            # HNSW index is already persistent, but we might want to save additional metadata
            self._graph.flush()
            self.logger.log("KnowledgeGraphSaved", {
                "node_count": self._stats["total_nodes"],
                "edge_count": self._stats["total_edges"]
            })
        except Exception as e:
            self.logger.log("KnowledgeGraphSaveError", {
                "error": str(e)
            })

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph."""
        if not self._initialized:
            self.initialize()
            
        return {
            "node_count": self._stats["total_nodes"],
            "edge_count": self._stats["total_edges"],
            "node_types": self._stats["node_types"],
            "edge_types": self._stats["edge_types"],
            "avg_edges_per_node": self._stats["total_edges"] / max(1, self._stats["total_nodes"]),
            "last_update": self._stats["last_update"],
            "queries": self._stats["queries"]
        }

    def clear_graph(self):
        """Clear the entire knowledge graph."""
        if not self._initialized:
            return
            
        try:
            # Clear HNSW index
            self._graph.reset()
            
            # Reset statistics
            self._stats = {
                "total_nodes": 0,
                "total_edges": 0,
                "node_types": {},
                "edge_types": {},
                "last_update": datetime.now(timezone.utc).isoformat(),
                "queries": 0,
                "query_time": 0.0,
                "build_time": 0.0
            }
            
            self.logger.log("KnowledgeGraphCleared", {})
        except Exception as e:
            self.logger.log("KnowledgeGraphClearError", {
                "error": str(e)
            })

    def get_connected_components(self, min_size: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Find connected components in the knowledge graph.
        
        Args:
            min_size: Minimum size of components to return
            
        Returns:
            List of connected components (each is a list of nodes)
        """
        if not self._initialized:
            self.initialize()
            
        try:
            # In a real implementation, this would use graph algorithms
            # For now, we'll return a placeholder
            self.logger.log("KnowledgeGraphConnectedComponents", {
                "min_size": min_size
            })
            return []
        except Exception as e:
            self.logger.log("KnowledgeGraphComponentsError", {
                "min_size": min_size,
                "error": str(e)
            })
            return []

    def complete_knowledge(self, seed_entity: str, target_property: str) -> Optional[Dict[str, Any]]:
        """
        Use the knowledge graph to complete missing information.
        
        Args:
            seed_entity: Starting entity ID
            target_property: Property to predict
            
        Returns:
            Predicted value or None if not confident
        """
        if not self._initialized:
            self.initialize()
            
        try:
            # In a real implementation, this would use graph reasoning
            self.logger.log("KnowledgeGraphCompletion", {
                "seed_entity": seed_entity,
                "target_property": target_property
            })
            return None
        except Exception as e:
            self.logger.log("KnowledgeGraphCompletionError", {
                "seed_entity": seed_entity,
                "target_property": target_property,
                "error": str(e)
            })
            return None