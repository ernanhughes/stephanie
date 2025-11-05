# stephanie/agents/knowledge/scorable_annotate.py
"""
Scorable Annotation Agent

This agent enriches scorables (context items) with domain classification and named entity recognition (NER)
to support knowledge extraction and learning from context items. Unlike the ChatAnnotateAgent which processes
database-stored conversations, this agent works directly on scorables passed through the pipeline context.

Key Features:
- Domain classification using seed-based and goal-aware classifiers
- Named Entity Recognition with optional Knowledge Graph integration
- Works directly on scorables in context (no database roundtrip)
- Preserves all original scorable metadata while adding annotations
- Idempotent operation (skips already annotated fields by default)
- Comprehensive logging and reporting
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.scorable_classifier import ScorableClassifier

log = logging.getLogger(__name__)

class ScorableAnnotateAgent(BaseAgent):
    """
    Agent that enriches scorables with domain and NER annotations.
    
    This agent processes scorables passed through the context, adding:
    1. Domain classifications (what the scorable content is about)
    2. Named Entity Recognition (people, places, concepts mentioned)
    
    The annotations are added directly to the scorable metadata, enabling
    improved retrieval, scoring, and knowledge extraction downstream.
    """
    
    def __init__(self, cfg, memory, container, logger):
        # Initialize parent class with configuration, memory, container and logger
        super().__init__(cfg, memory, container, logger)
        
        # Initialize domain classifiers with configuration paths
        self.seed_classifier = ScorableClassifier(
            memory, logger, config_path=cfg.get("seed_config", "config/domain/seeds.yaml")
        )
        self.goal_classifier = ScorableClassifier(
            memory, logger, config_path=cfg.get("goal_config", "config/domain/goal_prompt.yaml")
        )

        # Domain classification settings
        self.max_k = int(cfg.get("max_domains_per_source", 3))  # Max domains per scorable
        self.min_conf = float(cfg.get("min_confidence", 0.10))  # Minimum confidence threshold

        # Processing controls
        self.only_missing = bool(cfg.get("only_missing", True))  # Skip already annotated fields
        self.force = bool(cfg.get("force", False))  # Force re-annotation of all scorables
        self.progress_enabled = bool(cfg.get("progress", True))  # Enable/disable progress bars
        self.filter_role = cfg.get("filter_role", False)  # Role filter elabled
        self.scorable_role = cfg.get("scorable_role", "candidate")  # Role to filter scorables

    async def run(self, context: dict) -> dict:
        """
        Execute the annotation process on scorables in the context.
        
        Args:
            context: Execution context dictionary containing scorables
            
        Returns:
            Updated context with annotated scorables
        """
        # Get scorables from context
        scorables = context.get("scorables", [])
        if not scorables:
            log.debug("No scorables found in context to annotate")
            return context
        
        # Filter scorables by role if specified
        if self.scorable_role and self.filter_role:
            original_count = len(scorables)
            scorables = [s for s in scorables if s.get("role") == self.scorable_role]
            log.debug(f"Filtered {original_count} → {len(scorables)} scorables by role='{self.scorable_role}'")
            
            if not scorables:
                log.warning("No scorables with role='%s' found in context", self.scorable_role)
                return context

        # Pre-count scorables that need processing
        total_scorables = len(scorables)
        to_annotate = 0
        
        for scorable in scorables:
            domains = self.memory.scorable_domains.get_domains(scorable.get("id"), scorable.get("scorable_type"))
            log.debug(f"Scorable {scorable.get('id')} domains: {domains}")
            ner = self.memory.scorable_entities.get_by_scorable(scorable.get("id"), scorable.get("scorable_type"))
            log.debug(f"Scorable {scorable.get('id')} NER: {ner}")
            if self.only_missing and not self.force: 
                if not domains or not ner:
                    to_annotate += 1
            else:
                to_annotate += 1

        # Log start of annotation process
        self.logger.log("ScorableAnnotateStart", {
            "scorables_total": total_scorables,
            "scorables_to_annotate": to_annotate,
            "role": self.scorable_role,
            "only_missing": self.only_missing,
            "force": self.force
        })

        # Initialize Knowledge Graph if available
        kg = self.container.get("knowledge_graph") 
        if kg:
            kg.initialize()

        # Create progress bar
        pbar = tqdm(
            total=to_annotate,
            desc=f"Annotating {self.scorable_role} scorables",
            disable=not self.progress_enabled,
        )

        # Initialize statistics counters
        stats = {
            "scorables_processed": 0,
            "domains_annotated": 0,
            "ner_annotated": 0,
            "skipped": 0
        }

        # Process each scorable
        annotated_scorables = []
        for scorable in scorables:
            # Get text and goal from scorable
            text = scorable.get("text", "")
            if not text:
                log.debug(f"Skipping scorable {scorable.get('id')} - empty text")
                annotated_scorables.append(scorable)
                continue
            goal_text = scorable.get("goal_text", context.get("goal", {}).get("goal_text", ""))
            scorable_id = scorable.get("id", "unknown")
            scorable_type = scorable.get("scorable_type", "unknown")
            # Initialize meta if not present
            if "meta" not in scorable:
                scorable["meta"] = {}

            # Check if already annotated (if only_missing is True)
            domains = self.memory.scorable_domains.get_domains(scorable_id, scorable_type)
            if self.only_missing and not self.force and domains:
                log.debug(f"Skipping scorable {scorable_id} - domains already annotated")
                stats["skipped"] += 1
            else:
                # Annotate domains
                domains = self._annotate_domains(text, goal_text)
                if domains:
                    scorable["meta"]["domains"] = domains
                    stats["domains_annotated"] += 1


            # Annotate NER
            ner = self.memory.scorable_entities.get_by_scorable(scorable_id, scorable_type)
            if self.only_missing and not self.force and domains and ner:
                stats["skipped"] += 1
                annotated_scorables.append(scorable)
            else:
                ner = self._annotate_ner(text, kg)
                if ner:
                    scorable["meta"]["ner"] = ner

                    # persist to DB + (optional) immediate index
                    saved = await self._persist_ner_entities(
                        scorable_id=scorable_id,
                        scorable_type=scorable_type,
                        text=text,
                        entities=ner,
                        kg=kg,  # pass through so we can also publish an index request
                        immediate_index=True,        # set False if you only want KG to index
                        publish_to_kg=True           # set False if you don't want to publish
                    )
                    stats["ner_annotated"] += saved
            
            # Update scorable with annotations
            annotated_scorables.append(scorable)
            stats["scorables_processed"] += 1
            pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
        # Update context with annotated scorables
        context["scorables"] = annotated_scorables
        
        # Log completion
        self.logger.log("ScorableAnnotateDone", {
            **stats,
            "role": self.scorable_role,
            "only_missing": self.only_missing,
            "force": self.force
        })
        
        # Add summary to context for downstream processing
        context["scorable_annotation_summary"] = {
            "scorables_total": total_scorables,
            **stats,
            "role": self.scorable_role
        }
        
        self.report({
            "event": "scorables_annotated",
            "count": stats["scorables_processed"],
            "role": self.scorable_role
        })
        
        return context
    
    def _annotate_domains(self, text: str, goal_text: str) -> List[Dict[str, Any]]:
        """Annotate text with domain classifications using both seed and goal classifiers"""
        if not text:
            return []
        
        # Use seed classifier (domain-agnostic)
        seed_domains = self.seed_classifier.classify(
            text=text,
            max_k=self.max_k,
            min_conf=self.min_conf
        )
        
        # Use goal classifier if goal_text is provided
        goal_domains = []
        if goal_text:
            goal_domains = self.goal_classifier.classify(
                text=text,
                goal=goal_text,
                max_k=self.max_k,
                min_conf=self.min_conf
            )
        
        # Combine and deduplicate domains
        domain_map = {}
        for d in seed_domains + goal_domains:
            domain = d["domain"]
            if domain not in domain_map or d["score"] > domain_map[domain]["score"]:
                domain_map[domain] = {
                    "domain": domain,
                    "score": d["score"],
                    "source": d["source"]
                }
        
        return list(domain_map.values())
    
    def _annotate_ner(self, text: str, kg: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Annotate text with named entities using Knowledge Graph if available"""
        if not text:
            return []
        
        # Try to use Knowledge Graph for NER
        if kg:
            try:
                entities = kg.detect_entities(text)
                # Convert to standard format
                return [{
                    "text": e["text"],
                    "type": e["type"],
                    "start": e.get("start", 0),
                    "end": e.get("end", 0),
                    "role": "assistant"  # Default role for scorables
                } for e in entities]
            except Exception as e:
                log.warning("Knowledge Graph NER failed: %s", str(e))
        
        # Fallback to simple entity extraction if KG not available
        # This is a simplified version - in production you'd use a proper NER model
        entities = []
        words = text.split()
        
        # Simple pattern matching for demonstration
        for i, word in enumerate(words):
            if word.istitle() and len(word) > 2 and i > 0:  # Likely a proper noun
                entities.append({
                    "text": word,
                    "type": "PERSON" if word in ["Dr", "Mr", "Ms", "Professor"] else "ORG",
                    "start": text.find(word),
                    "end": text.find(word) + len(word),
                    "role": "assistant"
                })
        
        return entities[:10]  # Limit to top 10 entities
    

    async def _persist_ner_entities(
        self,
        *,
        scorable_id: str,
        scorable_type: str,
        text: str,
        entities: list[dict],
        kg=None,
        immediate_index: bool = True,
        publish_to_kg: bool = True,
    ) -> int:
        """
        Upsert entities into ScorableEntityStore, and optionally:
        - immediately index into the shared HNSW for retrieval
        - publish a KG index_request so nodes/edges are created
        Returns: number of entities saved.
        """
        if not entities:
            return 0

        saved = 0

        # --- 1) Save to the DB store (upsert)
        try:
            store = getattr(self.memory, "scorable_entities", None)
            for e in entities:
                if not e.get("text"):
                    continue
                payload = {
                    "scorable_id": str(scorable_id),
                    "scorable_type": scorable_type,
                    "entity_text": e["text"],
                    "entity_type": e.get("type", "UNKNOWN"),
                    "start": int(e.get("start", -1) or -1),
                    "end": int(e.get("end", -1) or -1),
                    "similarity": float(e.get("score", 0.0) or 0.0),
                    "source_text": text[:100] + "...",
                }
                if store:
                    store.insert(payload)
                    saved += 1
        except Exception as ex:
            log.error("ScorableAnnotateAgent.persist_ner_entities: DB insert failed: %s", str(ex))

        # --- 2) Immediate ANN indexing (optional, same embedder space as retriever)
        if immediate_index:
            try:
                retr = store.retriever if store else None
                if retr:
                    # batch embed all spans
                    spans = [(int(e.get("start", -1) or -1), int(e.get("end", -1) or -1)) for e in entities]
                    # keep only valid spans
                    valid_pairs = [(e, s) for e, s in zip(entities, spans) if s[0] >= 0 and s[1] > s[0] and s[1] <= len(text)]
                    if valid_pairs:
                        ents_valid, spans_valid = zip(*valid_pairs)
                        vecs_nested = retr.embed_entities_for_batch([text], [list(spans_valid)])
                        vecs = vecs_nested[0] if vecs_nested else []

                        new_meta = []
                        new_embs = []
                        for e, emb, (cs, ce) in zip(ents_valid, vecs, spans_valid):
                            meta = {
                                "scorable_id": str(scorable_id),
                                "scorable_type": scorable_type,
                                "entity_text": e["text"],
                                "start": cs,
                                "end": ce,
                                "entity_type": e.get("type", "UNKNOWN"),
                                "source_text": text[:100] + "...",
                            }  
                            new_meta.append(meta)
                            new_embs.append(emb)
                        if new_embs:
                            retr.index.add(np.asarray(new_embs, dtype=np.float32), new_meta, save=True)
            except Exception as ex:
                log.warning("ScorableAnnotateAgent.persist_ner_entities: immediate index skipped: %s", str(ex))

        # --- 3) Publish to KG so it adds nodes/edges (optional)
        if publish_to_kg:
            try:
                bus = getattr(self.memory, "bus", None)
                if bus and hasattr(bus, "publish"):
                    await bus.publish("knowledge_graph.index_request", {
                        "scorable_id": str(scorable_id),
                        "scorable_type": scorable_type,
                        "text": text,
                        "entities": entities,
                        # if you stored domains in scorable["meta"]["domains"], include them:
                        "domains": (getattr(self, "last_domains", None) or
                                    (scorable.get("meta", {}).get("domains", []) if isinstance(scorable := locals().get("scorable", {}), dict) else []))
                    })
            except Exception as ex:
                log.warning("ScorableAnnotateAgent.persist_ner_entities: KG publish failed: %s", str(ex))

        return saved
    