# stephanie/agents/knowledge/chat_knowledge_builder.py
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from stephanie.data.knowledge_unit import KnowledgeUnit
from stephanie.memory.chat_store import ChatStore
from stephanie.models.ner_retriever import EntityDetector
from stephanie.scoring.scorable import Scorable, ScorableFactory, ScorableType
from stephanie.tools.scorable_classifier import ScorableClassifier
from stephanie.utils.hash_utils import hash_text

log = logging.getLogger(__name__)

class ChatKnowledgeBuilder:
    """
    AI-powered builder of structured knowledge units from conversations and documents.

    Integrates:
      - Domain classification (ScorableClassifier)
      - Named Entity Recognition (EntityDetector)
      - Knowledge Graph linking (KnowledgeGraphService)
      - Conversation history (ChatStore)

    Follows Stephanie agent pattern: initialized with cfg, memory, container, logger.
    Gracefully degrades when subsystems fail.
    """

    def __init__(self, cfg: Dict[str, Any], memory: Any, container, logger: Optional[logging.Logger] = None):
        self.cfg = cfg
        self.memory = memory
        self.container = container
        self.logger = logger

        # Lazily initialize components only if needed
        try:
            self.classifier = ScorableClassifier(
                memory=memory,
                logger=self.logger,
                config_path=cfg.get("domain_config", "config/domain/seeds.yaml"),
                metric=cfg.get("domain_metric", "cosine")
            )
            self.logger.info("Domain classifier loaded.")
        except Exception as e:
            self.classifier = None
            log.error(f"Failed to initialize ScorableClassifier: {e}")

        try:
            self.entity_detector = EntityDetector()
            self.logger.info("NER detector loaded.")
        except Exception as e:
            self.entity_detector = None
            log.error(f"Failed to initialize EntityDetector: {e}")

        try:
            self.kg_service = self.container.get("knowledge_graph")
            self.logger.info("KnowledgeGraphService connected.")
        except Exception as e:
            self.kg_service = None
            log.error(f"Failed to initialize KnowledgeGraphService: {e}")

        try:
            self.chat_store = ChatStore(memory.session, logger=self.logger)
            self.logger.info("ChatStore connected.")
        except Exception as e:
            self.chat_store = None
            log.error(f"Failed to initialize ChatStore: {e}")

    # ---------------------------
    # Public API
    # ---------------------------
    def build(
        self,
        chat_messages: List[Dict[str, Any]],
        paper_text: str,
        conversation_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, KnowledgeUnit]:
        """
        Build aligned knowledge units from user chat and paper text.
        Optionally enrich with prior conversation context.
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Extract plain text from messages
            chat_text = " ".join(
                m["text"] for m in chat_messages
                if isinstance(m.get("text"), str) and m["text"].strip()
            ).strip()

            # Generate stable IDs
            chat_hash = hash_text(chat_text)[:16]
            paper_hash = hash_text(paper_text)[:16]


            chat_scorable = self.memory.dynamic_scorables.add(
                pipeline_run_id=context.get("pipeline_run_id"),
                scorable_type=ScorableType.DYNAMIC,
                source="chat_knowledge_builder",
                text=chat_text,
                meta={"hash": chat_hash, "kind": "chat"}
            ) 

            paper_scorable = self.memory.dynamic_scorables.add(
                pipeline_run_id=context.get("pipeline_run_id"),
                scorable_type=ScorableType.DYNAMIC,
                source="chat_knowledge_builder",
                text=paper_text,
                meta={"hash": paper_hash, "kind": "paper"}
            ) 
            chat_ku = self._process_with_ai(ScorableFactory.from_orm(chat_scorable), source="chat")
            paper_ku = self._process_with_ai(ScorableFactory.from_orm(paper_scorable), source="paper")


            # Enrich with historical context
            if conversation_id and self.chat_store:
                ctx_ku = self._build_contextual_knowledge(conversation_id)
                if ctx_ku.stats.get("error"):
                    log.debug(f"Context enrichment skipped: {ctx_ku.stats['error']}")
                else:
                    chat_ku.provenance["context_from_conversation"] = ctx_ku.to_dict()

            # Final stats
            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.info(f"Built knowledge pair in {duration:.0f}ms", extra={
                "chat_entities": sum(len(v) for v in chat_ku.entities.values()),
                "paper_entities": sum(len(v) for v in paper_ku.entities.values()),
                "domains_matched": len(set(chat_ku.domains) & set(paper_ku.domains))
            })

            return {"chat": chat_ku, "paper": paper_ku}

        except Exception:
            self.logger.error("ChatKnowledgeBuilder.build failed", exc_info=True)
            raise

    # ---------------------------
    # Core Processing Pipeline
    # ---------------------------
    def _process_with_ai(self, scorable: Scorable, source: str) -> KnowledgeUnit:
        text = scorable.text
        scorable_id = scorable.id
        if not text.strip():
            return KnowledgeUnit(text="", stats={"empty": True, "source": source})

        start_time = datetime.now(timezone.utc)

        domains, raw_entities, entities_by_type, kg_nodes = {}, [], {}, []
        phrases, anchors = [], []

        # 1. Domain Classification
        if self.classifier:
            try:
                domain_matches = self.classifier.classify(text=text, top_k=5)
                domain_scores = [{"domain": d, "score": float(s)} for d, s in domain_matches]

                domains = {d["domain"]: float(d["score"]) for d in domain_scores if d.get("score", 0) > 0.01}
            except Exception as e:
                log.error(f"[{source}] Domain classification failed: {e}")
        else:
            domains = {}

        # 2. Entity Detection
        if self.entity_detector:
            try:
                raw_entities = self.entity_detector.detect_entities(text)
                for ent in raw_entities:
                    entities_by_type.setdefault(ent["type"], []).append(ent)
            except Exception as e:
                log.error(f"[{source}] NER extraction failed: {e}")
        else:
            entities_by_type = {}

        # 3. Phrase Extraction
        phrases = self._extract_salient_phrases(text, domains, entities_by_type)

        # 4. Link to Knowledge Graph
        if self.kg_service and self.kg_service._initialized:
            try:
                for ent in raw_entities:
                    # Unique local provenance for this entity
                    node_id = f"{scorable_id}:{ent['type']}:{ent['start']}-{ent['end']}"
                    matched_nodes = []

                    # Fallback: search by embedding of entity text
                    query_vec = self.kg_service._retriever.embed_type_query(ent["text"])
                    results = self.kg_service._graph.search(query_vec, k=3)

                    for _, score, meta in results:
                        if meta.get("text", "").lower() == ent["text"].lower():
                            matched_nodes.append({
                                **meta,                       # KG-provided metadata
                                "score": float(score),        # similarity score
                                "node_id": node_id,           # our stable local ID
                                "scorable_id": scorable_id,   # provenance back to source scorable
                                "entity_text": ent["text"],   # original surface form
                                "entity_type": ent["type"],   # PERSON, ORG, etc
                                "span": f"{ent['start']}-{ent['end']}",
                                "source": source              # chat | paper | context
                            })

                    kg_nodes.extend(matched_nodes)
            except Exception as e:
                log.error(f"[{source}] KG linking failed: {e}")
        else:
            kg_nodes = []

        # 5. Select Anchors
        anchors = self._select_anchors(phrases, domains, entities_by_type)

        return KnowledgeUnit(
            text=text,
            domains=domains,
            phrases=phrases,
            anchors=anchors,
            entities=entities_by_type,
            linked_kg_nodes=kg_nodes,
            provenance={
                "source": source,
                "scorable_id": scorable_id,
                "used_classifier": bool(self.classifier and domains),
                "used_ner": bool(self.entity_detector and raw_entities),
                "used_kg": bool(self.kg_service and kg_nodes),
            },
            stats={
                "char_length": len(text),
                "word_count": len(text.split()),
                "phrase_count": len(phrases),
                "entity_count": sum(len(v) for v in entities_by_type.values()),
                "kg_link_count": len(kg_nodes),
                "processing_duration_ms": int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000),
                "timestamp": start_time.isoformat(),
            },
        )

    # ---------------------------
    # Helpers
    # ---------------------------
    def _extract_salient_phrases(self, text: str, domains: Dict[str, float], entities: Dict[str, List[Dict]]) -> List[str]:
        import re
        sentences = [s.strip() for s in re.split(r"[.;!?]", text) if len(s.strip().split()) >= 2]
        domain_terms = {d for d in domains if domains[d] > 0.2}
        entity_texts = {e["text"].lower() for ents in entities.values() for e in ents}

        scored = []
        for sent in sentences:
            words = sent.lower().split()
            score = len(words)
            if any(d.lower() in words for d in domain_terms):
                score += 2.0
            if any(e in words for e in entity_texts):
                score += 1.5
            scored.append((sent, score))

        scored.sort(key=lambda x: -x[1])
        return [s for s, _ in scored[:50]]

    def _select_anchors(self, phrases: List[str], domains: Dict[str, float], entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        domain_terms = {d.lower() for d in domains if domains[d] > 0.2}
        entity_texts = {e["text"].lower() for ents in entities.values() for e in ents}

        anchors = []
        for p in phrases[:20]:
            words = p.lower().split()
            score = len(words)
            if any(d in words for d in domain_terms):
                score += 1.5
            if any(e in words for e in entity_texts):
                score += 1.0
            anchors.append({
                "span": p,
                "score": float(score),
                "length": len(words),
                "contains_entity": any(e in words for e in entity_texts),
                "overlaps_domain": any(d in words for d in domain_terms)
            })

        anchors.sort(key=lambda x: -x["score"])
        return anchors[:10]

    def _build_contextual_knowledge(self, conversation_id: int) -> KnowledgeUnit:
        try:
            turns = self.memory.chats.list_turns_for_conversation_with_texts(conversation_id)
            snippets = []
            for turn in turns:
                u = (turn.user_message.text or "").strip()
                a = (turn.assistant_message.text or "").strip()
                if u or a:
                    snippets.append(f"USER: {u}\nASSISTANT: {a}")

            combined = "\n\n".join(snippets).strip()
            if not combined:
                return KnowledgeUnit(text="", stats={"error": "no_content_in_context"})

            scorable = Scorable(text=combined, id=f"context:{conversation_id}")
            return self._process_with_ai(
                source=f"context:{conversation_id}",
                scorable=scorable
            )
        except Exception as e:
            log.error(f"Context enrichment failed for conv={conversation_id}: {e}")
            return KnowledgeUnit(text="", stats={"error": str(e)}) 