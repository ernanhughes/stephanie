# stephanie/agents/knowledge/knowledge_tree_builder.py
"""
KnowledgeTreeBuilderAgent (Enhanced)
----------------------------
Builds a verifiable knowledge tree for paper sections from:
- Paper claims with entity grounding
- Critical conversation messages with confidence scores
- Entity-based connections with calibrated similarity
- Domain-aware relationship types

Unlike the basic implementation, this version:
- Uses your full NER system from KnowledgeFusionAgent
- Incorporates calibrated similarity scores
- Adds domain-aware relationship types
- Includes trajectory evidence from conversation paths
- Structures the tree for optimal verification

Input context:
  - paper_section: { section_name, section_text, paper_id, domain? }
  - critical_messages: [ { text, score, reason, ... } ]
  - conversation_trajectories: [ { start_idx, end_idx, messages, score, goal_achieved } ]
  - domains: [ { domain: str, score: float } ]  # From KnowledgeFusion

Output context:
  - knowledge_tree: {
        root: str,
        paper_id: str,
        section_name: str,
        section_text: str,
        domains: [ { domain: str, score: float } ],
        claims: [ { id, text, source, confidence, entities: [entity_ids] } ],
        insights: [ { id, text, confidence, timestamp, entities: [entity_ids], trajectory_id? } ],
        entities: [ { id, text, type, source, calibrated_similarity } ],
        relationships: [ 
          { 
            id: str,
            from: str,  # claim_id or insight_id
            to: str,    # claim_id or insight_id or entity_id
            type: str,  # "supports", "contradicts", "extends", etc.
            confidence: float,
            evidence: [ { section_span, trajectory_span, strength } ]
          }
        ],
        trajectory_paths: [ { id, messages: [insight_ids], confidence } ]
    }
"""

from __future__ import annotations
import re
import uuid
import json
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from dataclasses import asdict

import numpy as np
from stephanie.agents.base_agent import BaseAgent
from stephanie.knowledge.casebook_store import CaseBookStore
from stephanie.scoring.scorable_factory import TargetType
from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.models.hnsw_index import HNSWIndex
from stephanie.models.ner_retriever import EntityDetector, NERRetrieverEmbedder
import time
import torch

@dataclass
class KTConfig:
    """Configuration for KnowledgeTreeBuilderAgent."""
    # Entity extraction
    ner_k: int = 12
    ner_min_sim: float = 0.60
    ner_min_calibrated_sim: float = 0.45
    
    # Claim extraction
    min_claim_length: int = 40
    claim_keywords: List[str] = None
    max_claims: int = 8
    
    # Relationship building
    entity_overlap_threshold: float = 0.08
    relationship_threshold: float = 0.75
    max_hops: int = 3
    
    # Domain awareness
    domain_aware: bool = True
    domain_boost: float = 0.15
    
    # Output
    include_trajectory_paths: bool = True
    include_evidence_spans: bool = True
    max_evidence_spans: int = 3
    
    def __post_init__(self):
        if self.claim_keywords is None:
            self.claim_keywords = [
                "show", "demonstrat", "result", "achiev", 
                "improv", "evidence", "prove", "find", "conclude"
            ]


class KnowledgeTreeBuilderAgent(BaseAgent):
    """
    Enhanced KnowledgeTreeBuilderAgent that creates a verifiable knowledge tree
    with entity-based connections, calibrated similarity, and domain awareness.
    
    This agent:
    - Uses your existing NER system for precise entity extraction
    - Builds connections based on entity overlap and calibrated similarity
    - Incorporates domain information into relationship types
    - Structures the tree for optimal verification by downstream agents
    
    Designed to run after ConversationFilterAgent and before VerifiedSectionGenerator.
    """
    
    def __init__(self, cfg: Dict[str, Any], memory: Any, logger: logging.Logger):
        super().__init__(cfg, memory, logger)
        
        # Configuration
        self.kt_cfg = KTConfig(**cfg.get("knowledge_tree", {}))
        
        # Components
        self.casebooks: CaseBookStore = cfg.get("casebooks") or CaseBookStore()
        self.classifier: Optional[ScorableClassifier] = cfg.get("classifier")
        self.bus = getattr(self.memory, "bus", None)
        
        # Entity extraction components (reuse from KnowledgeFusion)
        self._entity_detector = None
        self._retriever = None
        
        # Stats
        self.stats = {
            "total_sections": 0,
            "total_claims": 0,
            "total_insights": 0,
            "total_entities": 0,
            "total_relationships": 0,
            "processing_time": 0.0,
            "domain_aware_connections": 0
        }
        
        self.logger.info("KnowledgeTreeBuilderAgent initialized", {
            "config": {k: v for k, v in asdict(self.kt_cfg).items() if k != "claim_keywords"},
            "domain_aware": self.kt_cfg.domain_aware,
            "message": "Ready to build knowledge trees"
        })
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Builds a knowledge tree from paper section and critical conversation messages.
        
        Expected context:
          - paper_section: { section_name, section_text, paper_id, domain? }
          - critical_messages: [ { text, score, reason, ... } ]
          - conversation_trajectories: [ { ... } ] (optional)
          - domains: [ { domain: str, score: float } ] (from KnowledgeFusion)
          
        Returns context with:
          - knowledge_tree: structured knowledge tree for verification
        """
        start_time = time.time()
        
        paper_section = context.get("paper_section")
        critical_messages = context.get("critical_messages", [])
        conversation_trajectories = context.get("conversation_trajectories", [])
        domains = context.get("domains", [])
        
        if not paper_section:
            self.logger.log("KnowledgeTreeBuilderSkipped", {
                "reason": "missing_section",
                "has_section": bool(paper_section),
                "has_messages": bool(critical_messages)
            })
            return context
            
        # Clean and prepare data
        section_text = paper_section.get("section_text", "")
        section_name = paper_section.get("section_name", "Unknown")
        paper_id = paper_section.get("paper_id")
        
        if not section_text or len(section_text.strip()) < 10:
            self.logger.log("KnowledgeTreeBuilderSkipped", {
                "reason": "empty_section",
                "section": section_name
            })
            return context
            
        self.stats["total_sections"] += 1
        
        try:
            # Initialize entity extraction if needed
            if not self._entity_detector or not self._retriever:
                self._init_entity_extraction()
            
            # Build the knowledge tree
            tree = self._build_knowledge_tree(
                paper_section, 
                critical_messages,
                conversation_trajectories,
                domains
            )
            
            # Update context
            context["knowledge_tree"] = tree
            
            # Log results
            processing_time = time.time() - start_time
            self.stats["processing_time"] = processing_time
            
            self.logger.log("KnowledgeTreeBuilt", {
                "section": section_name,
                "paper_id": paper_id,
                "claims": len(tree["claims"]),
                "insights": len(tree["insights"]),
                "entities": len(tree["entities"]),
                "relationships": len(tree["relationships"]),
                "processing_time": f"{processing_time:.2f}s"
            })
            
            # Optional: Publish tree to bus for debugging/monitoring
            if self.bus and context.get("publish_tree", True):
                self._publish_tree(tree)
                
            return context
            
        except Exception as e:
            self.logger.log("KnowledgeTreeBuildError", {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "section": section_name,
                "paper_id": paper_id
            })
            context["tree_build_error"] = str(e)
            return context
    
    def _init_entity_extraction(self):
        """Initialize entity extraction components if not already set up."""
        try:
            # Reuse the same components as KnowledgeFusionAgent
            self._entity_detector = EntityDetector(
                device=self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            )
            self._retriever = NERRetrieverEmbedder(
                model_name=self.cfg.get("ner_model", "dslim/bert-base-NER"),
                device=self.cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            )
            self.logger.info("Entity extraction components initialized")
        except Exception as e:
            self.logger.warning("EntityExtractionInitFailed", {
                "error": str(e),
                "message": "Falling back to heuristic entity extraction"
            })
            # Will use heuristic fallback in _extract_entities
    
    def _build_knowledge_tree(self,
                             paper_section: Dict[str, Any],
                             critical_messages: List[Dict[str, Any]],
                             conversation_trajectories: List[Dict[str, Any]],
                             domains: List[Dict[str, float]]) -> Dict[str, Any]:
        """Build the complete knowledge tree structure."""
        # 1. Extract claims from paper section
        claims = self._extract_claims(paper_section["section_text"])
        
        # 2. Extract entities from paper section
        paper_entities = self._extract_entities(paper_section["section_text"])
        
        # 3. Process critical messages into insights
        insights = self._process_insights(critical_messages, conversation_trajectories)
        
        # 4. Extract entities from insights
        insight_entities = []
        for insight in insights:
            entities = self._extract_entities(insight["text"])
            for entity in entities:
                # Add insight reference to entity
                entity["source_insights"] = entity.get("source_insights", []) + [insight["id"]]
            insight_entities.extend(entities)
        
        # 5. Build combined entity set (deduplicated)
        all_entities = self._merge_entities(paper_entities, insight_entities)
        
        # 6. Build relationships between claims, insights, and entities
        relationships = self._build_relationships(
            claims, 
            insights, 
            all_entities, 
            domains,
            paper_section["section_text"]
        )
        
        # 7. Build trajectory paths if enabled
        trajectory_paths = []
        if self.kt_cfg.include_trajectory_paths and conversation_trajectories:
            trajectory_paths = self._build_trajectory_paths(
                conversation_trajectories, 
                insights
            )
        
        # 8. Structure the final tree
        return {
            "root": paper_section.get("section_name", "section"),
            "paper_id": paper_section.get("paper_id"),
            "section_name": paper_section.get("section_name", "Unknown"),
            "section_text": paper_section["section_text"][:500] + "..." 
                if len(paper_section["section_text"]) > 500 
                else paper_section["section_text"],
            "domains": domains,
            "claims": claims,
            "insights": insights,
            "entities": all_entities,
            "relationships": relationships,
            "trajectory_paths": trajectory_paths,
            "metadata": {
                "build_time": datetime.now(timezone.utc).isoformat(),
                "domain_aware": self.kt_cfg.domain_aware,
                "relationship_threshold": self.kt_cfg.relationship_threshold
            }
        }
    
    def _extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract claims from section text with entity grounding."""
        # Split into sentences
        sentences = self._sentences(text)
        self.stats["total_claims"] += len(sentences)
        
        # Filter for "claimy" sentences
        claims = []
        for i, sent in enumerate(sentences):
            if len(sent.strip()) < self.kt_cfg.min_claim_length:
                continue
                
            # Check for claim keywords
            is_claim = any(
                re.search(rf"\b{kw}\b", sent, re.I) 
                for kw in self.kt_cfg.claim_keywords
            )
            
            if not is_claim and i >= len(sentences) - 3:  # Last few sentences are often conclusions
                is_claim = True
                
            if is_claim:
                claims.append({
                    "id": f"claim_{i+1}",
                    "text": sent,
                    "source": "paper",
                    "confidence": 1.0,
                    "position": i / max(1, len(sentences))
                })
        
        # Limit to max claims
        claims = claims[:self.kt_cfg.max_claims]
        
        # Add entity grounding to claims
        for claim in claims:
            entities = self._extract_entities(claim["text"])
            claim["entities"] = [e["id"] for e in entities]
            
        return claims
    
    def _process_insights(self, 
                         critical_messages: List[Dict[str, Any]],
                         conversation_trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process critical messages into structured insights."""
        insights = []
        trajectory_map = {
            traj["trajectory_id"]: traj 
            for traj in conversation_trajectories
        } if conversation_trajectories else {}
        
        for i, msg in enumerate(critical_messages):
            # Create insight structure
            insight = {
                "id": f"ins_{uuid.uuid4().hex[:8]}",
                "text": msg["text"],
                "confidence": float(msg["score"]),
                "timestamp": msg.get("timestamp"),
                "vpm_dims": msg.get("vpm_dims", {}),
                "similarity": msg.get("similarity", 0.0),
                "reason": msg.get("reason", "Relevance-based filtering"),
                "source": "conversation"
            }
            
            # Add trajectory information if available
            if msg.get("trajectory_id") and msg["trajectory_id"] in trajectory_map:
                traj = trajectory_map[msg["trajectory_id"]]
                insight["trajectory_id"] = traj["trajectory_id"]
                insight["trajectory_score"] = traj["score"]
                insight["in_trajectory"] = True
                
                # Add evidence spans if available
                if self.kt_cfg.include_evidence_spans and traj.get("supporting_evidence"):
                    insight["evidence_spans"] = traj["supporting_evidence"][:self.kt_cfg.max_evidence_spans]
            
            insights.append(insight)
            
        self.stats["total_insights"] += len(insights)
        return insights
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using NER system or heuristic fallback."""
        entities = []
        
        try:
            # Try NER system first
            if self._entity_detector and self._retriever:
                # Detect entities
                raw_entities = self._entity_detector.detect_entities(text)
                
                # Get embeddings and similarities
                for entity in raw_entities:
                    try:
                        emb = self._retriever.embed_entity(entity["text"], entity["type"])
                        similarity = self._retriever.get_similarity(emb)
                        
                        entities.append({
                            "id": f"ent_{uuid.uuid4().hex[:6]}",
                            "text": entity["text"],
                            "type": entity["type"],
                            "start": entity["start"],
                            "end": entity["end"],
                            "similarity": similarity,
                            "calibrated_similarity": max(
                                self.kt_cfg.ner_min_calibrated_sim,
                                min(1.0, similarity * 1.2)
                            )
                        })
                    except Exception as e:
                        self.logger.debug("EntityEmbeddingFailed", {
                            "text": entity["text"],
                            "error": str(e)
                        })
                        # Add without embedding data
                        entities.append({
                            "id": f"ent_{uuid.uuid4().hex[:6]}",
                            "text": entity["text"],
                            "type": entity["type"],
                            "start": entity["start"],
                            "end": entity["end"],
                            "similarity": 0.5,
                            "calibrated_similarity": 0.5
                        })
            else:
                # Heuristic fallback
                entities = self._heuristic_entity_extraction(text)
                
        except Exception as e:
            self.logger.warning("EntityExtractionFailed", {
                "error": str(e),
                "text_length": len(text)
            })
            entities = self._heuristic_entity_extraction(text)
            
        self.stats["total_entities"] += len(entities)
        return entities
    
    def _heuristic_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Heuristic entity extraction as fallback when NER system fails."""
        entities = []
        
        # Extract potential entities using regex
        patterns = [
            (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", "PERSON"),  # Proper names
            (r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)+\b", "MISC"),      # Hyphenated terms
            (r"\b(?:[A-Z][a-z]*\s+){2,}[A-Z][a-z]*\b", "ORG"),  # Organizations
            (r"\b\d+(?:\.\d+)?%?\b", "NUMBER"),                 # Numbers
            (r"\b[A-Z]{2,}\b", "ACRONYM"),                      # Acronyms
            (r"\b(?:Fig|Figure|Table)\s+\d+\b", "REFERENCE")    # References
        ]
        
        for pattern, entity_type in patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    "id": f"ent_{uuid.uuid4().hex[:6]}",
                    "text": match.group(),
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "similarity": 0.6,
                    "calibrated_similarity": 0.6
                })
                
        return entities
    
    def _merge_entities(self, 
                       paper_entities: List[Dict[str, Any]], 
                       insight_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge and deduplicate entities from paper and insights."""
        # Create mapping from normalized text to entity
        entity_map = {}
        
        # Process paper entities first (higher priority)
        for entity in paper_entities:
            norm_text = self._normalize_entity(entity["text"])
            entity_map[norm_text] = {
                **entity,
                "source": "paper",
                "paper_occurrences": 1,
                "insight_occurrences": 0
            }
        
        # Process insight entities, merging with existing ones
        for entity in insight_entities:
            norm_text = self._normalize_entity(entity["text"])
            
            if norm_text in entity_map:
                # Update existing entity
                existing = entity_map[norm_text]
                existing["insight_occurrences"] = existing.get("insight_occurrences", 0) + 1
                # Update similarity if new one is higher
                if entity["calibrated_similarity"] > existing["calibrated_similarity"]:
                    existing["calibrated_similarity"] = entity["calibrated_similarity"]
            else:
                # New entity
                entity_map[norm_text] = {
                    **entity,
                    "source": "insight",
                    "paper_occurrences": 0,
                    "insight_occurrences": 1
                }
        
        # Convert back to list
        entities = list(entity_map.values())
        
        # Calculate overall confidence
        for entity in entities:
            total_occurrences = entity["paper_occurrences"] + entity["insight_occurrences"]
            paper_weight = 0.7  # Paper entities are more trustworthy
            
            confidence = (
                paper_weight * entity["paper_occurrences"] + 
                (1 - paper_weight) * entity["insight_occurrences"]
            ) / max(1, total_occurrences)
            
            entity["confidence"] = min(1.0, confidence * entity["calibrated_similarity"] * 1.2)
            
        return entities
    
    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text for deduplication."""
        # Convert to lowercase
        text = text.lower()
        # Remove articles and common stopwords
        text = re.sub(r"\b(the|a|an|this|that|these|those)\b", "", text)
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def _build_relationships(self,
                            claims: List[Dict[str, Any]],
                            insights: List[Dict[str, Any]],
                            entities: List[Dict[str, Any]],
                            domains: List[Dict[str, float]],
                            section_text: str) -> List[Dict[str, Any]]:
        """Build relationships between claims, insights, and entities."""
        relationships = []
        
        # 1. Claim-Entity relationships
        for claim in claims:
            for entity_id in claim.get("entities", []):
                # Find the entity object
                entity = next((e for e in entities if e["id"] == entity_id), None)
                if not entity:
                    continue
                    
                # Calculate relationship confidence
                confidence = self._calculate_relationship_confidence(
                    claim["text"], 
                    entity["text"], 
                    0,  # Distance is 0 for direct mentions
                    domains
                )
                
                if confidence >= self.kt_cfg.relationship_threshold:
                    relationships.append({
                        "id": f"rel_{uuid.uuid4().hex[:6]}",
                        "from": claim["id"],
                        "to": entity["id"],
                        "type": "mentions",
                        "confidence": confidence,
                        "evidence": self._extract_evidence_spans(claim["text"], entity["text"], section_text)
                    })
        
        # 2. Insight-Entity relationships
        for insight in insights:
            # Extract entities from insight text
            insight_entities = self._extract_entities(insight["text"])
            for entity in insight_entities:
                # Find matching entity in our entity list
                norm_text = self._normalize_entity(entity["text"])
                matched_entity = next(
                    (e for e in entities if self._normalize_entity(e["text"]) == norm_text), 
                    None
                )
                
                if not matched_entity:
                    continue
                    
                # Calculate relationship confidence
                confidence = self._calculate_relationship_confidence(
                    insight["text"],
                    matched_entity["text"],
                    0,
                    domains
                )
                
                if confidence >= self.kt_cfg.relationship_threshold:
                    relationships.append({
                        "id": f"rel_{uuid.uuid4().hex[:6]}",
                        "from": insight["id"],
                        "to": matched_entity["id"],
                        "type": "references",
                        "confidence": confidence,
                        "evidence": self._extract_evidence_spans(insight["text"], matched_entity["text"], section_text)
                    })
        
        # 3. Claim-Insight relationships (via shared entities)
        for claim in claims:
            claim_entities = set(claim.get("entities", []))
            for insight in insights:
                # Extract entities from insight
                insight_entities = {e["id"] for e in self._extract_entities(insight["text"])}
                
                # Find shared entities
                shared_entities = claim_entities & insight_entities
                if not shared_entities:
                    continue
                    
                # Calculate relationship confidence based on shared entities
                confidence = 0.5
                for entity_id in shared_entities:
                    entity = next((e for e in entities if e["id"] == entity_id), None)
                    if entity:
                        confidence = max(confidence, entity["confidence"])
                
                if confidence >= self.kt_cfg.relationship_threshold:
                    # Determine relationship type based on insight content
                    rel_type = self._determine_claim_insight_relationship(claim, insight)
                    
                    relationships.append({
                        "id": f"rel_{uuid.uuid4().hex[:6]}",
                        "from": claim["id"],
                        "to": insight["id"],
                        "type": rel_type,
                        "confidence": confidence,
                        "shared_entities": list(shared_entities),
                        "evidence": self._extract_evidence_spans(claim["text"], insight["text"], section_text)
                    })
        
        # 4. Entity-Entity relationships (co-occurrence)
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Check if they co-occur in section text
                if self._entities_co_occur(entity1["text"], entity2["text"], section_text):
                    # Calculate relationship confidence
                    confidence = self._calculate_entity_relationship_confidence(
                        entity1, 
                        entity2, 
                        section_text,
                        domains
                    )
                    
                    if confidence >= self.kt_cfg.relationship_threshold:
                        rel_type = self._determine_entity_relationship_type(entity1, entity2)
                        
                        relationships.append({
                            "id": f"rel_{uuid.uuid4().hex[:6]}",
                            "from": entity1["id"],
                            "to": entity2["id"],
                            "type": rel_type,
                            "confidence": confidence,
                            "evidence": self._extract_entity_evidence_spans(entity1, entity2, section_text)
                        })
        
        self.stats["total_relationships"] += len(relationships)
        return relationships
    
    def _calculate_relationship_confidence(self,
                                         text1: str,
                                         text2: str,
                                         distance: int,
                                         domains: List[Dict[str, float]]) -> float:
        """Calculate confidence score for a relationship."""
        # Base score based on lexical overlap
        base_score = self._lexical_overlap(text1, text2)
        
        # Distance penalty
        distance_penalty = min(1.0, distance / 100)
        base_score = base_score * (1 - distance_penalty * 0.3)
        
        # Domain boost if relevant domains are present
        domain_boost = 0.0
        if self.kt_cfg.domain_aware:
            domain_names = {d["domain"] for d in domains}
            # ML/NLP domains get higher boost
            if any(d in domain_names for d in {"ml", "nlp", "machine learning", "natural language processing"}):
                domain_boost = self.kt_cfg.domain_boost
                
        # Proximity bonus for close entities
        proximity_bonus = 0.1 if distance < 20 else 0.0
        
        # Final confidence
        confidence = min(1.0, base_score + domain_boost + proximity_bonus)
        return max(0.0, confidence)
    
    def _calculate_entity_relationship_confidence(self,
                                                entity1: Dict[str, Any],
                                                entity2: Dict[str, Any],
                                                section_text: str,
                                                domains: List[Dict[str, float]]) -> float:
        """Calculate confidence for entity-entity relationship."""
        # Base on entity similarities
        base_score = min(entity1["confidence"], entity2["confidence"])
        
        # Co-occurrence strength
        co_occurrence = self._entity_co_occurrence_strength(
            entity1["text"], 
            entity2["text"], 
            section_text
        )
        base_score = (base_score * 0.7) + (co_occurrence * 0.3)
        
        # Domain boost
        domain_boost = 0.0
        if self.kt_cfg.domain_aware:
            domain_names = {d["domain"] for d in domains}
            if any(d in domain_names for d in {"ml", "nlp"}):
                domain_boost = self.kt_cfg.domain_boost * 0.5
                
        return min(1.0, base_score + domain_boost)
    
    def _determine_claim_insight_relationship(self, claim: Dict[str, Any], insight: Dict[str, Any]) -> str:
        """Determine the type of relationship between a claim and insight."""
        insight_text = insight["text"].lower()
        
        # Check for supporting language
        if any(kw in insight_text for kw in ["supports", "confirms", "validates", "evidence for"]):
            return "supports"
        if any(kw in insight_text for kw in ["extends", "builds on", "enhances"]):
            return "extends"
        if any(kw in insight_text for kw in ["contradicts", "challenges", "disproves"]):
            return "contradicts"
        if any(kw in insight_text for kw in ["clarifies", "explains", "elaborates"]):
            return "clarifies"
            
        # Default relationship type based on confidence
        if insight["confidence"] > 0.8:
            return "supports"
        elif insight["confidence"] < 0.5:
            return "questionable"
        else:
            return "relates_to"
    
    def _determine_entity_relationship_type(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> str:
        """Determine the type of relationship between two entities."""
        # Special cases for common entity type pairs
        type_pairs = {
            ("METHOD", "RESULT"): "produces",
            ("MODEL", "DATASET"): "trained_on",
            ("ALGORITHM", "PROBLEM"): "solves",
            ("TECHNIQUE", "CHALLENGE"): "addresses",
            ("ARCHITECTURE", "TASK"): "performs",
            ("PARAMETER", "METRIC"): "affects",
            ("HYPERPARAMETER", "PERFORMANCE"): "influences",
            ("LOSS", "OPTIMIZATION"): "minimized_by",
            ("FEATURE", "PREDICTION"): "contributes_to",
            ("INPUT", "OUTPUT"): "maps_to"
        }
        
        key = (entity1["type"].upper(), entity2["type"].upper())
        if key in type_pairs:
            return type_pairs[key]
            
        # Default relationship
        return "co_occurs_with"
    
    def _build_trajectory_paths(self,
                               trajectories: List[Dict[str, Any]],
                               insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build structured trajectory paths from conversation trajectories."""
        paths = []
        
        # Map insight IDs for quick lookup
        insight_map = {ins["id"]: ins for ins in insights}
        
        for i, traj in enumerate(trajectories):
            # Get insight IDs in this trajectory
            trajectory_insights = []
            for msg in traj["messages"]:
                # Find matching insight
                insight = next(
                    (ins for ins in insights if ins["text"] == msg["text"]), 
                    None
                )
                if insight:
                    trajectory_insights.append(insight["id"])
            
            if not trajectory_insights:
                continue
                
            # Calculate trajectory confidence (average of insight confidences)
            confidences = [
                insight_map[ins_id]["confidence"] 
                for ins_id in trajectory_insights 
                if ins_id in insight_map
            ]
            trajectory_confidence = sum(confidences) / len(confidences) if confidences else 0.5
            
            paths.append({
                "id": f"path_{i+1}",
                "trajectory_id": traj.get("trajectory_id", f"traj_{i+1}"),
                "insight_ids": trajectory_insights,
                "confidence": trajectory_confidence,
                "length": len(trajectory_insights),
                "goal_achieved": traj.get("goal_achieved", False),
                "start_idx": traj.get("start_idx", 0),
                "end_idx": traj.get("end_idx", 0)
            })
            
        return paths
    
    def _extract_evidence_spans(self, 
                              text1: str, 
                              text2: str, 
                              full_text: str) -> List[Dict[str, Any]]:
        """Extract evidence spans showing connection between two texts."""
        if not self.kt_cfg.include_evidence_spans:
            return []
            
        evidence = []
        
        # Find where text2 appears in full_text
        text2_pos = full_text.find(text2)
        if text2_pos >= 0:
            # Get surrounding context
            start = max(0, text2_pos - 50)
            end = min(len(full_text), text2_pos + len(text2) + 50)
            section_span = full_text[start:end]
            
            # Find similar span in text1
            best_match = self._find_best_match_span(text1, section_span)
            if best_match and best_match["strength"] > 0.3:
                evidence.append({
                    "section_span": section_span,
                    "trajectory_span": best_match["span"],
                    "strength": best_match["strength"]
                })
                
        return evidence[:self.kt_cfg.max_evidence_spans]
    
    def _extract_entity_evidence_spans(self, 
                                     entity1: Dict[str, Any], 
                                     entity2: Dict[str, Any], 
                                     section_text: str) -> List[Dict[str, Any]]:
        """Extract evidence spans for entity-entity relationships."""
        if not self.kt_cfg.include_evidence_spans:
            return []
            
        evidence = []
        
        # Find sentences where both entities appear
        sentences = self._sentences(section_text)
        for sent in sentences:
            if entity1["text"].lower() in sent.lower() and entity2["text"].lower() in sent.lower():
                # Get surrounding context
                start = max(0, section_text.find(sent) - 30)
                end = min(len(section_text), section_text.find(sent) + len(sent) + 30)
                context = section_text[start:end]
                
                evidence.append({
                    "context": context,
                    "sentence": sent,
                    "entities": [entity1["text"], entity2["text"]],
                    "strength": 0.8  # High confidence for direct co-occurrence
                })
                
        return evidence[:self.kt_cfg.max_evidence_spans]
    
    def _find_best_match_span(self, source: str, target: str) -> Dict[str, Any]:
        """Find the best matching span in source for the target text."""
        source_sents = self._sentences(source)
        best_score = 0.0
        best_span = ""
        
        for sent in source_sents:
            score = self._lexical_overlap(sent, target)
            if score > best_score:
                best_score = score
                best_span = sent
                
        return {
            "span": best_span,
            "strength": best_score
        }
    
    def _lexical_overlap(self, a: str, b: str) -> float:
        """Calculate lexical overlap between two texts."""
        if not a or not b:
            return 0.0
            
        a_words = set(re.findall(r"\b\w+\b", a.lower()))
        b_words = set(re.findall(r"\b\w+\b", b.lower()))
        
        if not a_words:
            return 0.0
            
        return len(a_words & b_words) / len(a_words)
    
    def _entities_co_occur(self, entity1: str, entity2: str, text: str) -> bool:
        """Check if two entities co-occur in the text."""
        return (entity1.lower() in text.lower() and 
                entity2.lower() in text.lower() and
                abs(text.lower().find(entity1.lower()) - text.lower().find(entity2.lower())) < 200)
    
    def _entity_co_occurrence_strength(self, entity1: str, entity2: str, text: str) -> float:
        """Calculate strength of entity co-occurrence."""
        if not self._entities_co_occur(entity1, entity2, text):
            return 0.0
            
        # Find positions
        pos1 = text.lower().find(entity1.lower())
        pos2 = text.lower().find(entity2.lower())
        
        # Calculate distance-based strength
        distance = abs(pos1 - pos2)
        max_distance = 200  # Consider entities within 200 chars as co-occurring
        return max(0.0, 1.0 - (distance / max_distance))
    
    def _sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if len(p.strip()) > 2]
    
    def _publish_tree(self, tree: Dict[str, Any]):
        """Publish knowledge tree to the bus for monitoring/debugging."""
        try:
            event = {
                "event_type": "knowledge_tree.built",
                "payload": {
                    "tree_id": f"tree_{uuid.uuid4().hex[:8]}",
                    "section_name": tree["section_name"],
                    "paper_id": tree["paper_id"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": {
                        "claims": len(tree["claims"]),
                        "insights": len(tree["insights"]),
                        "entities": len(tree["entities"]),
                        "relationships": len(tree["relationships"]),
                        "trajectory_paths": len(tree["trajectory_paths"])
                    }
                }
            }
            self.bus.publish("knowledge_tree.built", event)
        except Exception as e:
            self.logger.warning("TreePublishFailed", {
                "error": str(e),
                "section": tree.get("section_name")
            })
    
    def health_check(self) -> Dict[str, Any]:
        """Return health status and metrics for the tree builder agent."""
        return {
            "status": "healthy",
            "stats": self.stats,
            "config": {k: v for k, v in asdict(self.kt_cfg).items() if k != "claim_keywords"},
            "message": "Knowledge tree builder operational"
        }