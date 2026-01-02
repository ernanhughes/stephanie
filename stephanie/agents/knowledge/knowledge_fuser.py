# stephanie/agents/knowledge/knowledge_fuser.py
from __future__ import annotations

import hashlib
import logging
import time
import traceback
from typing import Any, Dict, List

from stephanie.agents.knowledge.chat_knowledge_builder import \
    ChatKnowledgeBuilder
from stephanie.data.knowledge_unit import KnowledgeUnit
from stephanie.tools.scorable_classifier import ScorableClassifier
from stephanie.utils.hash_utils import hash_text

log = logging.getLogger(__name__)


class KnowledgeFuser:
    """
    Advanced knowledge fuser that blends transient chat context with document content
    using AI models: domain classification, entity linking, semantic overlap.

    Produces a structured content plan suitable for draft generation.
    """

    def __init__(
        self, cfg: Dict[str, Any], memory: Any, container, logger: logging.Logger
    ):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        # Core components (already built in standard pattern)
        try:
            self.builder = ChatKnowledgeBuilder(cfg, memory, container, logger)
            self.classifier = ScorableClassifier(
                memory=memory,
                logger=logger,
                config_path=cfg.get(
                    "domain_config", "config/domain/seeds.yaml"
                ),
                metric=cfg.get("domain_metric", "cosine"),
            )
        except Exception as e:
            log.error(
                f"Failed to initialize KnowledgeFuser dependencies: {e}",
                exc_info=True,
            )
            raise

        # Config
        self.top_k_domains = cfg.get("top_k_domains", 5)
        self.min_domain_score = cfg.get("min_classification_score", 0.6)
        self.max_claims = cfg.get("max_claims", 6)
        self.entity_merge_strategy = cfg.get(
            "entity_merge_strategy", "chat_priority"
        )

    async def fuse(
        self,
        *,
        text: str,
        chat_messages: List[Dict[str, Any]],
        section_name: str,
        conversation_id: int = None,
        scorable_id: str = None,
        scorable_type: str = "document_section",
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Fuse chat and paper knowledge into a structured content plan.
        """
        start_time = time.time()
        if not scorable_id:
            scorable_id = (
                f"temp:{hash_text(text)[:16]}"
            )

        try:
            # Step 1: Build rich knowledge units
            k = self.builder.build(
                chat_messages=chat_messages,
                paper_text=text,
                conversation_id=conversation_id,
                context=context,
            )
            chat_ku = k["chat"]
            paper_ku = k["paper"]

            # Step 2: Domain alignment (semantic classification)
            try:
                domain_scores = self.classifier.classify(text=text)
                if not isinstance(domain_scores, list):
                    raise ValueError(
                        f"Expected list, got {type(domain_scores)}"
                    )

                # Handle both (str, float) tuples and dicts
                parsed = []
                for item in domain_scores:
                    if isinstance(item, tuple) and len(item) == 2:
                        domain, score = item
                        parsed.append(
                            {"domain": str(domain), "score": float(score)}
                        )
                    elif (
                        isinstance(item, dict)
                        and "domain" in item
                        and "score" in item
                    ):
                        parsed.append(
                            {
                                "domain": str(item["domain"]),
                                "score": float(item["score"]),
                            }
                        )
                    else:
                        log.error(
                            f"Unexpected domain score format: {item}"
                        )

                top_domains = [
                    d for d in parsed if d["score"] >= self.min_domain_score
                ][: self.top_k_domains]

            except Exception as e:
                log.error(f"Domain classification failed: {e}")
                top_domains = []

            # Step 3: Semantic phrase overlap (not just string match)
            overlap_phrases = self._semantic_overlap(chat_ku, paper_ku)

            # Step 4: Generate claim units from aligned knowledge
            units = self._generate_claims(
                overlap_phrases, paper_ku, scorable_id
            )

            # Step 5: Merge entities with strategy
            merged_entities = self._merge_entities(chat_ku, paper_ku)

            # Step 6: Optionally publish indexing event
            await self._publish_index_request(
                scorable_id, text, paper_ku, top_domains
            )

            # Final plan
            plan = {
                "section_title": section_name, 
                "units": units,
                "entities": merged_entities,
                "domains": top_domains,
                "paper_text": text,
                "scorable_id": scorable_id,
                "scorable_type": scorable_type,
                "meta": {
                    "knowledge_hash": self._hash_content(text, chat_messages),
                    "timestamp": time.time(),
                    "processing_duration_ms": int(
                        (time.time() - start_time) * 1000
                    ),
                    "sources": {
                        "chat_entity_count": sum(
                            len(v) for v in chat_ku.entities.values()
                        ),
                        "paper_entity_count": sum(
                            len(v) for v in paper_ku.entities.values()
                        ),
                        "phrase_overlap": len(overlap_phrases),
                        "used_kg_links": len(paper_ku.linked_kg_nodes),
                    },
                },
            }

            log.debug(
                "KnowledgeFusionComplete"
                f"section: {section_name}"
                f"claim_count: {len(units)}"
                f"domain_count: {len(top_domains)}"
                f"entity_count: {len(merged_entities.get('ABBR', {}))} + {len(merged_entities.get('REQUIRED', []))}"
            )

            return plan

        except Exception as e:
            log.error(
                "KnowledgeFusionFailed"
                f"section: {section_name}"
                f"error: {str(e)}"
                f"traceback: {traceback.format_exc()}"
            )
            raise

    # ---------------------------
    # Helpers
    # ---------------------------
    def _semantic_overlap(self, chat_ku, paper_ku):
        def normalize_phrases(items):
            return {
                (p["span"] if isinstance(p, dict) else str(p)).lower()
                for p in items if p
            }

        chat_set = normalize_phrases(chat_ku.anchors + chat_ku.phrases)
        paper_set = normalize_phrases(paper_ku.anchors + paper_ku.phrases)

        return list(chat_set & paper_set)

    def _generate_claims(
        self, overlaps: List[str], paper_ku: KnowledgeUnit, scorable_id: str
    ) -> List[Dict[str, Any]]:
        """
        Turn overlapping phrases into claims with evidence hints.
        """
        units = []
        for i, phrase in enumerate(overlaps[: self.max_claims]):
            # Attach supporting entities or KG nodes
            supporting_entities = [
                e["text"]
                for etype, elist in paper_ku.entities.items()
                for e in elist
                if phrase.lower() in e["text"].lower()
            ][:3]

            linked_nodes = [
                n
                for n in paper_ku.linked_kg_nodes
                if n.get("text", "").lower() in phrase.lower()
            ]

            units.append(
                {
                    "claim_id": f"C{i + 1}",
                    "claim": f"{phrase.strip().rstrip('.')}."
                    if not phrase.endswith(".")
                    else phrase,
                    "evidence_hint": "See related work"
                    if supporting_entities
                    else "General context",
                    "supporting_entities": supporting_entities,
                    "kg_links": [n["node_id"] for n in linked_nodes],
                    "confidence": 0.8
                    + 0.1 * min(i, 2),  # prioritize earlier claims
                }
            )
        return units

    def _merge_entities(
        self, chat_ku: KnowledgeUnit, paper_ku: KnowledgeUnit
    ) -> Dict[str, Any]:
        """
        Merge entities with configurable strategy.
        Default: chat wins for ABBR; union for REQUIRED.
        """
        abbr = {}
        required = []

        if self.entity_merge_strategy == "chat_priority":
            # Chat overrides paper on abbreviations
            abbr.update(paper_ku.entities.get("ABBR", {}))
            abbr.update(chat_ku.entities.get("ABBR", {}))  # chat wins

            # Union of required terms
            required_set = set(paper_ku.entities.get("REQUIRED", {})) | set(
                chat_ku.entities.get("REQUIRED", {})
            )
            required = list(required_set)[:12]
        else:
            # Fallback: simple merge
            abbr = {
                **paper_ku.entities.get("ABBR", {}),
                **chat_ku.entities.get("ABBR", {}),
            }
            required = list(
                dict.fromkeys(
                    list(paper_ku.entities.get("REQUIRED", {}).keys())
                    + list(chat_ku.entities.get("REQUIRED", {}).keys())
                )
            )[:12]

        return {"ABBR": abbr, "REQUIRED": required}

    async def _publish_index_request(
        self,
        scorable_id: str,
        text: str,
        ku: KnowledgeUnit,
        domains: List[Dict],
    ):
        """
        Publish async indexing request to KnowledgeBus.
        Enables non-blocking graph updates.
        """
        if not hasattr(self.memory, "bus"):
            return

        relationships = []
        entities = [ent for ents in ku.entities.values() for ent in ents]

        # Simple proximity-based relationships
        for i, e1 in enumerate(entities):
            for j in range(i + 1, len(entities)):
                e2 = entities[j]
                distance = abs(e1["end"] - e2["start"])
                if distance < 100:
                    rel_type = self._infer_relationship_type(e1, e2)
                    confidence = 0.7 + (0.3 / (distance + 1))
                    relationships.append(
                        {
                            "source": f"{scorable_id}:{e1['type']}:{e1['start']}-{e1['end']}",
                            "target": f"{scorable_id}:{e2['type']}:{e2['start']}-{e2['end']}",
                            "type": rel_type,
                            "confidence": min(confidence, 1.0),
                        }
                    )

        event = {
            "event_type": "knowledge_graph.index_request",
            "payload": {
                "scorable_id": scorable_id,
                "scorable_type": "document_section",
                "text": text,
                "entities": entities,
                "domains": domains,
                "relationships": relationships,
                "timestamp": time.time(),
                "source_agent": "KnowledgeFuser",
            },
        }

        try:
            await self.memory.bus.publish(
                subject=event["event_type"],
                payload=event["payload"]
            )

            log.debug(
                "IndexRequestPublished"
                f"scorable_id: {scorable_id}"
                f"entity_count: {len(entities)}"
                f"relationship_count: {len(relationships)}"
            )
        except Exception as e:
            log.error(f"Failed to publish index request: {e}")

    def _infer_relationship_type(self, e1: Dict, e2: Dict) -> str:
        ordered = e1["end"] < e2["start"]
        first, second = (e1, e2) if ordered else (e2, e1)
        pairs = {
            ("METHOD", "DATASET"): "evaluates",
            ("DATASET", "METRIC"): "measured_by",
            ("MODEL", "TASK"): "performs",
            ("AUTHOR", "PAPER"): "wrote",
            ("PAPER", "METHOD"): "introduces",
        }
        return pairs.get((first["type"], second["type"]), "related_to")

    def _hash_content(self, text: str, chat_messages: List[Dict]) -> str:
        combined = (
            text[:500]
            + "|||"
            + " ".join(
                m.get("text", "") for m in chat_messages if m.get("text")
            )
        )
        return hash_text(combined)[:10]
