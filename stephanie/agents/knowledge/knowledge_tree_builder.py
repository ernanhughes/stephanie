# stephanie/agents/knowledge/knowledge_tree_builder.py
"""
KnowledgeTreeBuilderAgent (Integrated)
----------------------------
Builds a verifiable knowledge tree for paper sections from:
- Paper claims with entity grounding
- Critical conversation messages with confidence scores
- Entity-based connections with calibrated similarity
- Domain-aware relationship types
- Trajectory evidence from conversation paths

This integrated implementation:
- Uses your full NER system from KnowledgeFusionAgent
- Incorporates calibrated similarity scores
- Adds domain-aware relationship types
- Includes trajectory evidence from conversation paths
- Structures the tree for optimal verification
- Supports distributed verification via NATS/KV

Input context:
  - paper_section: { section_name, section_text, paper_id, domain? }
  - critical_messages: [ { text, score, reason, ... } ]
  - conversation_trajectories: [ { start_idx, end_idx, messages, score, goal_achieved } ]
  - domains: [ { domain: str, score: float } ]  # From KnowledgeFusion
  - fusion_entities: { surface_entities, expanded_entities }  # Optional from KnowledgeFusion

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

import hashlib
import json
import logging
import re
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.casebook_store import CaseBookStore
from stephanie.models.ner_retriever import EntityDetector, NERRetrieverEmbedder
from stephanie.tools.scorable_classifier import ScorableClassifier

log = logging.getLogger(__name__)

# Constants from the new analysis
_MIN_INSIGHT_SCORE = 0.70
_MAX_CLAIMS = 10
_MIN_CONN_OVERLAP = 0.06

def _sentences(t: str, max_sents: int = 80) -> List[str]:
    """Split text into sentences with optional max limit."""
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t.strip())
    sentences = [p.strip() for p in parts if len(p.strip()) > 2]
    return sentences[:max_sents] if max_sents and len(sentences) > max_sents else sentences

def _extract_claim_sents(text: str, max_claims: int = _MAX_CLAIMS) -> List[str]:
    """Extract claim sentences with improved heuristic."""
    sents = _sentences(text)
    claimish = [
        s for s in sents 
        if len(s) > 40 and re.search(
            r"(show|demonstrat|result|achiev|improv|evidence|increase|decrease|prove|find|conclude)", 
            s, 
            re.I
        )
    ]
    return (claimish or sents)[:max_claims]

def _norm_token_set(t: str) -> Set[str]:
    """Normalize text to token set for Jaccard similarity."""
    return set(re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", (t or "").lower()))

def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Calculate Jaccard similarity between two token sets."""
    if not a or not b: 
        return 0.0
    i = len(a & b)
    u = len(a | b)
    return i / max(1, u)

def _soft_dedup(items: List[Dict[str, Any]], key="text", thr=0.8) -> List[Dict[str, Any]]:
    """Soft deduplication based on Jaccard similarity."""
    out: List[Dict[str, Any]] = []
    for it in items:
        tw = _norm_token_set(it.get(key, ""))
        if any(_jaccard(tw, _norm_token_set(o.get(key,""))) >= thr for o in out):
            continue
        out.append(it)
    return out

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
    max_claims: int = _MAX_CLAIMS
    
    # Relationship building
    entity_overlap_threshold: float = _MIN_CONN_OVERLAP
    relationship_threshold: float = 0.75
    max_hops: int = 3
    
    # Domain awareness
    domain_aware: bool = True
    domain_boost: float = 0.15
    
    # Output
    include_trajectory_paths: bool = True
    include_evidence_spans: bool = True
    max_evidence_spans: int = 3
    max_connections: int = 200
    
    def __post_init__(self):
        if self.claim_keywords is None:
            self.claim_keywords = [
                "show", "demonstrat", "result", "achiev", 
                "improv", "evidence", "prove", "find", "conclude"
            ]


class KnowledgeTreeBuilderAgent(BaseAgent):
    """
    Integrated KnowledgeTreeBuilderAgent that creates a verifiable knowledge tree
    with entity-based connections, calibrated similarity, and domain awareness.
    
    This agent:
    - Uses your existing NER system for precise entity extraction
    - Builds connections based on entity overlap and calibrated similarity
    - Incorporates domain information into relationship types
    - Structures the tree for optimal verification by downstream agents
    - Supports distributed verification via NATS/KV
    
    Designed to run after ConversationFilterAgent and before VerifiedSectionGenerator.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container: Any, logger: logging.Logger):
        super().__init__(cfg, memory, container=container, logger=logger)
        
        # Configuration
        self.kt_cfg = KTConfig(**cfg.get("knowledge_graph", {}))

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
            "domain_aware_connections": 0,
            "processing_time": 0.0
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
          - fusion_entities: { surface_entities, expanded_entities } (optional from KnowledgeFusion)
          
        Returns context with:
          - knowledge_tree: structured knowledge tree for verification
        """
        start_time = time.time()
        
        paper_section = context.get("paper_section")
        critical_messages = context.get("critical_messages", [])
        conversation_trajectories = context.get("conversation_trajectories", [])
        domains = context.get("domains", [])
        fusion_entities = context.get("fusion_entities", {}) or {}
        
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
            knowledge_graph = self._build_knowledge_graph(
                paper_section, 
                critical_messages,
                conversation_trajectories,
                domains,
                fusion_entities
            )
            
            # Update context
            context["knowledge_graph"] = knowledge_graph
            
            # Log results
            processing_time = time.time() - start_time
            self.stats["processing_time"] = processing_time
            
            self.logger.log("KnowledgeTreeBuilt", {
                "section": section_name,
                "paper_id": paper_id,
                "claims": len(knowledge_graph["claims"]),
                "insights": len(knowledge_graph["insights"]),
                "entities": len(knowledge_graph["entities"]),
                "relationships": len(knowledge_graph["relationships"]),
                "processing_time": f"{processing_time:.2f}s"
            })
            
            # Optional: Publish tree to bus for verification
            if self.bus and context.get("publish_tree", True):
                await self._publish_tree(knowledge_graph)
                
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
            log.warning("EntityExtractionInitFailed", {
                "error": str(e),
                "message": "Falling back to heuristic entity extraction"
            })
            # Will use heuristic fallback in _extract_entities
    
    def _build_knowledge_graph(self,
                             paper_section: Dict[str, Any],
                             critical_messages: List[Dict[str, Any]],
                             conversation_trajectories: List[Dict[str, Any]],
                             domains: List[Dict[str, float]],
                             fusion_entities: Dict[str, Any]) -> Dict[str, Any]:
        """Build the complete knowledge tree structure."""
        section_text = paper_section["section_text"]
        
        # 1. Extract claims from paper section (using improved heuristic)
        claims = self._extract_claims(section_text)
        
        # 2. Process critical messages into insights (with soft deduplication)
        insights = self._process_insights(critical_messages, conversation_trajectories)
        
        # 3. Use fusion entities if available, otherwise extract our own
        paper_entities, insight_entities = self._get_entities(
            section_text, 
            insights, 
            fusion_entities
        )
        
        # 4. Build combined entity set (deduplicated)
        all_entities = self._merge_entities(paper_entities, insight_entities)
        
        # 5. Build relationships between claims, insights, and entities
        relationships = self._build_relationships(
            claims, 
            insights, 
            all_entities, 
            domains,
            section_text
        )
        
        # 6. Build trajectory paths if enabled
        trajectory_paths = []
        if self.kt_cfg.include_trajectory_paths and conversation_trajectories:
            trajectory_paths = self._build_trajectory_paths(
                conversation_trajectories, 
                insights
            )
        
        # 7. Structure the final tree
        return {
            "root": paper_section.get("section_name", "section"),
            "paper_id": paper_section.get("paper_id"),
            "section_name": paper_section.get("section_name", "Unknown"),
            "section_text": section_text[:500] + "..." 
                if len(section_text) > 500 
                else section_text,
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
    

    def _merge_entities(
        self, 
        paper_entities: List[Dict[str, Any]], 
        insight_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate paper + insight entities into one unified set.
        Deduplication is done by normalized text match (case/punct stripped).
        """
        all_entities: List[Dict[str, Any]] = []
        seen: Dict[str, str] = {}  # normalized_text -> entity_id

        for ent in paper_entities + insight_entities:
            norm_text = self._normalize_entity(ent.get("text", ""))
            if not norm_text:
                continue

            if norm_text in seen:
                # merge meta into the already-existing entity
                existing = next((e for e in all_entities if e["id"] == seen[norm_text]), None)
                if existing:
                    # optional: merge sources or add provenance
                    existing_sources = set(existing.get("source", "").split(","))
                    new_source = ent.get("source", "")
                    if new_source and new_source not in existing_sources:
                        existing["source"] = ",".join(existing_sources | {new_source})

                    # merge calibrated similarity (take max)
                    existing["calibrated_similarity"] = max(
                        existing.get("calibrated_similarity", 0.0),
                        ent.get("calibrated_similarity", 0.0)
                    )

                    # merge embeddings if both exist (average them)
                    if existing.get("embedding") is not None and ent.get("embedding") is not None:
                        try:
                            import numpy as np
                            e1 = np.array(existing["embedding"])
                            e2 = np.array(ent["embedding"])
                            existing["embedding"] = ((e1 + e2) / 2.0).tolist()
                        except Exception:
                            pass
            else:
                eid = ent.get("id") or f"ent_{uuid.uuid4().hex[:8]}"
                ent["id"] = eid
                all_entities.append(ent)
                seen[norm_text] = eid

        self.stats["total_entities"] += len(all_entities)
        return all_entities

    def _get_entities(self,
                     section_text: str,
                     insights: List[Dict[str, Any]],
                     fusion_entities: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Get entities from fusion entities if available, otherwise extract."""
        paper_entities = []
        insight_entities = []
        
        # Try to use fusion entities first
        if fusion_entities:
            surface_entities = fusion_entities.get("surface_entities", [])
            expanded_entities = fusion_entities.get("expanded_entities", [])
            
            # Convert surface entities to our format
            for i, ent in enumerate(surface_entities):
                paper_entities.append({
                    "id": f"ent_{i}_{uuid.uuid4().hex[:4]}",
                    "text": ent["text"],
                    "type": ent["type"],
                    "start": ent.get("start", 0),
                    "end": ent.get("end", 0),
                    "source": "paper",
                    "similarity": ent.get("similarity", 0.6),
                    "calibrated_similarity": ent.get("calibrated_similarity", 0.6)
                })
            
            # Convert expanded entities to our format
            for i, ent in enumerate(expanded_entities):
                insight_entities.append({
                    "id": f"ent_exp_{i}_{uuid.uuid4().hex[:4]}",
                    "text": ent["text"],
                    "type": ent["type"],
                    "start": ent.get("start", 0),
                    "end": ent.get("end", 0),
                    "source": "insight",
                    "similarity": ent.get("similarity", 0.6),
                    "calibrated_similarity": ent.get("calibrated_similarity", 0.6)
                })
                
            return paper_entities, insight_entities
        
        # Fall back to our own extraction
        paper_entities = self._extract_entities(section_text)
        
        for insight in insights:
            entities = self._extract_entities(insight["text"])
            for entity in entities:
                # Add insight reference to entity
                entity["source_insights"] = entity.get("source_insights", []) + [insight["id"]]
            insight_entities.extend(entities)
            
        return paper_entities, insight_entities
    
    def _extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract claims from section text with entity grounding."""
        # Use improved claim extraction from the new analysis
        claim_texts = _extract_claim_sents(text)
        self.stats["total_claims"] += len(claim_texts)
        
        claims = []
        for i, claim_text in enumerate(claim_texts):
            claims.append({
                "id": f"claim_{i+1}",
                "text": claim_text,
                "source": "paper",
                "confidence": 1.0,
                "position": i / max(1, len(claim_texts))
            })
        
        # Add entity grounding to claims
        for claim in claims:
            entities = self._extract_entities(claim["text"])
            claim["entities"] = [e["id"] for e in entities]
            
        return claims
    
    def _process_insights(self, 
                         critical_messages: List[Dict[str, Any]],
                         conversation_trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process critical messages into structured insights with soft deduplication."""
        # Convert to our insight format
        raw_insights = []
        trajectory_map = {
            traj["trajectory_id"]: traj 
            for traj in conversation_trajectories
        } if conversation_trajectories else {}
        
        for i, msg in enumerate(critical_messages):
            # Only include high-score insights
            if float(msg["score"]) < _MIN_INSIGHT_SCORE:
                continue
                
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
            
            raw_insights.append(insight)
        
        # Soft deduplicate insights
        insights = _soft_dedup(raw_insights, key="text", thr=0.8)
        self.stats["total_insights"] += len(insights)
        
        return insights
    
    def _build_relationships(self,
                            claims: List[Dict[str, Any]],
                            insights: List[Dict[str, Any]],
                            entities: List[Dict[str, Any]],
                            domains: List[Dict[str, float]],
                            section_text: str) -> List[Dict[str, Any]]:
        """Build relationships between claims, insights, and entities."""
        relationships = []
        ent_lexicon = {e["text"].lower() for e in entities}
        
        # 1. Claim-Insight relationships (using Jaccard + entity ping)
        for claim in claims:
            cw = _norm_token_set(claim["text"])
            for insight in insights:
                iw = _norm_token_set(insight["text"])
                # lexical connection
                lex = _jaccard(cw, iw)
                # entity ping: any known entity inside insight?
                ent_hit = 1 if any(ent in insight["text"].lower() for ent in ent_lexicon) else 0
                # simple blend
                strength = min(1.0, 0.7*lex + 0.3*ent_hit)
                
                if strength >= self.kt_cfg.entity_overlap_threshold:
                    # Determine relationship type
                    rel_type = self._determine_claim_insight_relationship(claim, insight)
                    
                    relationships.append({
                        "id": f"rel_{uuid.uuid4().hex[:6]}",
                        "from": claim["id"],
                        "to": insight["id"],
                        "type": rel_type,
                        "confidence": strength,
                        "strength": strength,
                        "lex": float(lex),
                        "ent": int(ent_hit),
                        "evidence": self._extract_evidence_spans(claim["text"], insight["text"], section_text)
                    })
        
        # 2. Claim-Entity relationships (if we have entity grounding)
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
        
        # 3. Insight-Entity relationships
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
        
        # Sort by strength and limit connections
        relationships.sort(key=lambda x: (-x.get("strength", 0), x["from"]))
        return relationships[:self.kt_cfg.max_connections]
    
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
    

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using the configured detector/retriever.
        Falls back to regex-based heuristic if components not initialized.
        """
        entities: List[Dict[str, Any]] = []
        if not text or not text.strip():
            return entities

        try:
            # Ensure detector/retriever initialized
            if not self._entity_detector or not self._retriever:
                self._init_entity_extraction()

            # Detect entities
            dets = self._entity_detector.detect_entities(text)
            for i, d in enumerate(dets):
                span = (d.get("start", 0), d.get("end", len(d["text"])))

                # Embed returns a tensor (vector), not a dict
                emb = self._retriever.embed_entity(d["text"], span)
                if hasattr(emb, "detach"):  # torch.Tensor
                    emb = emb.detach().cpu().numpy()

                # Simple calibration heuristic: normalize length of vector
                calibrated_sim = float(d.get("score", 0.0))
                if emb is not None and len(emb) > 0:
                    norm = float(np.linalg.norm(emb))
                    if norm > 0:
                        calibrated_sim = min(1.0, calibrated_sim + (1.0 / norm))

                entities.append({
                    "id": f"ent_{uuid.uuid4().hex[:8]}",
                    "text": d["text"],
                    "type": d.get("label", "UNK"),
                    "start": d.get("start", -1),
                    "end": d.get("end", -1),
                    "source": "auto",
                    "similarity": float(d.get("score", 0.0)),
                    "calibrated_similarity": calibrated_sim,
                    "embedding": emb.tolist() if emb is not None else None
                })
        except Exception as e:
            # fallback: regex heuristic for proper nouns/acronyms
            log.warning("error: %s", str(e))
            matches = re.findall(r"\b([A-Z][A-Za-z0-9\-]{2,})\b", text)
            for m in set(matches):
                entities.append({
                    "id": f"ent_{uuid.uuid4().hex[:8]}",
                    "text": m,
                    "type": "Heuristic",
                    "source": "fallback",
                    "similarity": 0.5,
                    "calibrated_similarity": 0.5
                })

        return entities

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
    
    def _find_best_match_span(self, source: str, target: str) -> Dict[str, Any]:
        """Find the best matching span in source for the target text."""
        source_sents = _sentences(source)
        best_score = 0.0
        best_span = ""
        
        for sent in source_sents:
            score = _jaccard(_norm_token_set(sent), _norm_token_set(target))
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
    
    def _sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if len(p.strip()) > 2]
    
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
    
    async def _publish_tree(self, tree: Dict[str, Any]):
        """Publish knowledge tree to the bus for verification."""
        try:
            # Create a job ID based on tree content
            tree_hash = hashlib.sha256(json.dumps(tree).encode("utf-8")).hexdigest()[:16]
            
            # Create a verification job
            job = {
                "job_id": tree_hash,
                "tree": tree,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Store the full job in KV (to avoid MaxPayloadError)
            if hasattr(self.bus, "get_kv"):
                try:
                    kv = self.bus.get_kv(
                        bucket="vsec.jobs", 
                        description="Verification jobs",
                        max_age_seconds=3600
                    )
                    kv.put(tree_hash, json.dumps(job).encode("utf-8"))
                except Exception as e:
                    log.error("TreeKVStoreFailed error:%s job_id:%s", 
                         str(e),
                         tree_hash
                    )
            
            # Publish a small job envelope
            envelope = {
                "job_id": tree_hash,
                "section": tree["section_name"],
                "paper_id": tree["paper_id"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.bus.publish("verify.section", envelope)
            
            self.logger.log("VerificationJobEnqueued", {
                "job_id": tree_hash,
                "section": tree["section_name"],
                "paper_id": tree["paper_id"]
            })
        except Exception as e:
            log.warning("TreePublishFailed", {
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