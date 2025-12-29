# stephanie/agents/knowledge/knowledge_fusion.py
"""
KnowledgeFusionAgent
--------------------
Fuses domains (seed-centroid classifier), entities (NER retriever), and
recent chat interactions into a transient "knowledge plan" per section.

Inputs (context):
- goal: { id, goal_text, ... }   (optional, used to bias domains)
- paper: { id, title, ... }       (optional, for metadata)
- sections: [ { section_name, section_text, paper_id? }, ... ]  <-- required
- chat_corpus: [ { role, text, ts? }, ... ]                     <-- optional; if absent tries memory
- top_domains: int (default=20)
- ner_k: int (default=12)
- ner_min_sim: float (default=0.60)

Outputs (context):
- knowledge_plans: List[dict]  # one per section, transient only
  Each plan contains:
    {
      "section_title": str,
      "paper_id": ...,
      "domains": [{"domain": str, "score": float}, ...],  # top_k (no DB writes)
      "entities": [{"text": str, "type": str, "similar": [..], "source": "paper|chat"} ...],
      "chat_support": [{"snippet": str, "overlap_entities": [...], "sim": float}, ...],
      "claims": [{"claim_id": str, "claim": str, "grounded_entities": [str, ...]}],
      "tags": list[str],  # quick, normalized tags (domains + key entities)
    }

Notes:
- This agent is "read-only" for DB models: it does not write domains/entities
  into persistent stores. Instead, it builds ephemeral plans that downstream
  agents can use for:
    - targeted verification
    - VPM construction for this session
    - blog post generation / outline expansion
"""

from __future__ import annotations

import asyncio
import logging
import re
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import time
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from stephanie.agents.base_agent import BaseAgent
from stephanie.orm.ner_retriever import EntityDetector, NERRetrieverEmbedder
from stephanie.scoring.calibration_manager import CalibrationManager
from stephanie.scoring.scorable import Scorable, ScorableFactory
from stephanie.tools.scorable_classifier import ScorableClassifier
from stephanie.utils.hash_utils import hash_text
from stephanie.utils.text_utils import sentences

log = logging.getLogger(__name__)



def _truncate(text: str, max_len: int = 220) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _uniq(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@dataclass
class KFConfig:
    top_domains: int = 20
    min_domain_score: float = 0.0
    ner_k: int = 12
    ner_min_sim: float = 0.60
    ner_min_calibrated_sim: float = (
        0.45  # lower threshold, since calibration should shrink false positives
    )
    ephemeral_index_dir: str = "/tmp"
    max_chat_snippets: int = 12
    max_chunk_size: int = 5000
    entity_detection_fallback: bool = True
    enable_chunking: bool = False


class KnowledgeFusionAgent(BaseAgent):
    """
    Fuse (domains ⨁ entities ⨁ chat overlap) into a transient knowledge plan.
    """

    def __init__(self, cfg, memory, container, logger):
        super().__init__(cfg, memory, container, logger)
        self.kfc = KFConfig(
            top_domains=cfg.get("top_domains", 20),
            min_domain_score=cfg.get("min_domain_score", 0.0),
            ner_k=cfg.get("ner_k", 12),
            ner_min_sim=cfg.get("ner_min_sim", 0.60),
            ner_min_calibrated_sim=cfg.get("ner_min_calibrated_sim", 0.45),
            ephemeral_index_dir=cfg.get("ephemeral_index_dir", "/tmp"),
            max_chat_snippets=cfg.get("max_chat_snippets", 12),
            max_chunk_size=cfg.get("max_chunk_size", 5000),
            entity_detection_fallback=cfg.get(
                "entity_detection_fallback", True
            ),
            enable_chunking=cfg.get("enable_chunking", False),
        )
        # Domain backbone (no DB writes of domain tags)
        self.domain_clf = ScorableClassifier(
            memory=self.memory,
            logger=self.logger,
            config_path=cfg.get("domain_config", "config/domain/seeds.yaml"),
            metric=cfg.get("domain_metric", "cosine"),
        )

        # Entity layer (ANN over session-only content)
        self.entity_detector = EntityDetector(
            device=cfg.get("device", "cuda")
        )
        self.ner = NERRetrieverEmbedder(
            model_name=cfg.get("ner_model", "dslim/bert-base-NER"),
            layer=cfg.get(
                "ner_layer", 16
            ),  # in paper we seee 17 here the llm has on y 16 layers
            device=cfg.get("device", "cpu"),
            embedding_dim=cfg.get("ner_dim", 1024),
            index_path="data/ner_retriever/index",  # persistent path
            projection_enabled=cfg.get("ner_projection", False),
            projection_dim=cfg.get("ner_projection_dim", 1024),
            projection_dropout=cfg.get("ner_projection_dropout", 0.1),
            logger=self.logger,
            memory=self.memory,
            cfg=cfg,
        )

        self.calibration = CalibrationManager(
            cfg=cfg.get("calibration", {}),
            memory=self.memory,
            logger=self.logger,
        )

        # Periodic calibration trainer
        self.calibration_trainer = CalibrationTrainer(
            cfg=cfg.get("calibration", {}),
            memory=self.memory,
            logger=self.logger,
            calibration_manager=self.calibration,
        )

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entrypoint: build knowledge plans per section and stash in context.
        """
        sections = context.get("sections") or []
        if not sections:
            log.info("KnowledgeFusionAgent: no sections; skipping")
            return context

        goal = context.get("goal") or {}
        paper = context.get("paper") or {}
        chat_corpus = context.get("chat_corpus") or []

        # Build scorables (sections + chat)
        scorables = self._build_session_scorables(sections, chat_corpus)

        # Optionally index entities into KG asynchronously via bus
        if self.kfc.enable_chunking:
            await self._index_session_entities_with_chunking(scorables)
        else:
            await self._index_session_entities(scorables)

        # Domain classification per section (cheap, reused by entities)
        section_domains = self._score_section_domains(sections, goal)

        # Extract entities per section (via NER retriever)
        section_entities = self._extract_section_entities(sections, section_domains)

        # Collect relevant chat snippets
        chat_snippets = self._select_chat_snippets(chat_corpus, section_entities)

        # Generate lightweight claims per section
        section_claims = self._generate_claims(sections, section_entities)

        # Build knowledge plans
        plans = []
        for sec in sections:
            sec_id = sec.get("id")
            sec_title = sec.get("section_name") or "Unnamed section"
            paper_id = sec.get("paper_id") or paper.get("id") or "unknown"

            domains = section_domains.get(sec_id, [])
            entities = section_entities.get(sec_id, [])
            claims = section_claims.get(sec_id, [])
            chat_support = chat_snippets.get(sec_id, [])

            tags = self._generate_tags(domains, entities)

            plans.append(
                {
                    "section_id": sec_id,
                    "section_title": sec_title,
                    "paper_id": paper_id,
                    "domains": domains,
                    "entities": entities,
                    "claims": claims,
                    "chat_support": chat_support,
                    "tags": tags,
                }
            )

        context["knowledge_plans"] = plans

        # Trigger calibration training opportunistically
        try:
            self.calibration_trainer.maybe_train()
        except Exception:
            log.warning(
                "KnowledgeFusionAgent: calibration training failed",
                exc_info=True,
            )

        return context

    # ------------------------------------------------------------------
    # Scorable construction (sections + chat)
    # ------------------------------------------------------------------

    def _build_session_scorables(
        self,
        sections: List[Dict[str, Any]],
        chat: List[Dict[str, str]],
    ) -> List[Scorable]:
        """Build scorables using proper DB IDs (not fabricated composite IDs)."""
        scorables: List[Scorable] = []

        # Paper sections as scorables (entity-bearing units)
        for sec in sections:
            text = sec.get("section_text", "") or ""
            if not text:
                continue

            # Use actual DB ID (as string) and proper target type
            scorable_id = str(sec.get("id", f"temp_{uuid.uuid4().hex}"))
            target_type = sec.get("target_type", "document_section")

            scorables.append(
                Scorable(
                    id=scorable_id,
                    text=text,
                    target_type=target_type,
                )
            )

        # Recent chat messages as scorables (so entities from chat are retrievable)
        for item in chat or []:
            conv = item[0] if isinstance(item, (tuple, list)) else item
            scorable = ScorableFactory.from_orm(conv)
            scorables.append(scorable)

        return scorables

    # ------------------------------------------------------------------
    # Entity indexing via bus (no DB writes)
    # ------------------------------------------------------------------

    async def _index_session_entities(self, scorables: List[Scorable]) -> None:
        """
        Instead of indexing directly, publish to KnowledgeBus for async processing.
        Offload heavy sync work (classification + NER) to threads so we don't
        block the event loop.
        """
        total_queued = 0
        events_published = 0

        log.info(
            "KnowledgeFusionIndexStart scorables=%d ner_min_sim=%.3f",
            len(scorables),
            self.kfc.ner_min_sim,
        )

        for idx, scorable in enumerate(
            tqdm(scorables, desc="KnowledgeFusion Indexing", unit="scorable")
        ):
            text = (scorable.text or "").strip()
            if len(text) < 100:
                continue

            try:
                # Periodic heartbeat
                if idx % 5 == 0:
                    log.info(
                        "KnowledgeFusionIndexProgress processed=%d remaining=%d scorable_id=%s",
                        idx,
                        len(scorables) - idx,
                        str(scorable.id),
                    )

                # 1) Domain classification (offload to thread)
                domain_matches = await asyncio.to_thread(
                    self.domain_clf.classify,
                    text,
                    self.kfc.top_domains,
                    self.kfc.min_domain_score,
                )
                domains = [
                    {"domain": d, "score": float(s)}
                    for d, s in domain_matches
                ]

                # 2) Entity detection (offload to thread)
                results = await asyncio.to_thread(
                    self.entity_detector.detect_entities,
                    text,
                )
                entities = self._format_entities(results, text, source="paper")
                filtered_ents = [
                    e
                    for e in entities
                    if e.get(
                        "calibrated_similarity",
                        e.get("similarity", 0.0),
                    )
                    >= self.kfc.ner_min_sim
                ]

                # 3) Local relationships (cheap heuristic)
                relationships: List[Dict[str, Any]] = []
                for i, e1 in enumerate(filtered_ents):
                    for j in range(i + 1, len(filtered_ents)):
                        e2 = filtered_ents[j]
                        distance = abs(e1["end"] - e2["start"])
                        if distance < 100:
                            rel_type = self._infer_relationship_type(e1, e2)
                            confidence = self._calculate_relationship_confidence(
                                e1, e2, distance, domains
                            )
                            if confidence >= 0.75:
                                relationships.append(
                                    {
                                        "source": f"{scorable.id}:{e1['type']}:{e1['start']}-{e1['end']}",
                                        "target": f"{scorable.id}:{e2['type']}:{e2['start']}-{e2['end']}",
                                        "type": rel_type,
                                        "confidence": confidence,
                                    }
                                )

                # 4) Publish indexing job
                event = {
                    "event_type": "knowledge_graph.index_request",
                    "payload": {
                        "scorable_id": scorable.id,
                        "scorable_type": scorable.target_type,
                        "text": text,
                        "entities": filtered_ents,
                        "domains": domains,
                        "relationships": relationships,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source_agent": "KnowledgeFusionAgent",
                    },
                }

                await self.memory.bus.publish(
                    subject=event["event_type"],
                    payload=event["payload"],
                )

                total_queued += len(filtered_ents)
                events_published += 1

            except Exception as e:
                log.warning(
                    "KnowledgeFusionIndexEventFailed scorable_id=%s error=%s",
                    str(scorable.id),
                    str(e),
                )
                log.debug(
                    "KnowledgeFusionIndexEventFailed traceback=%s",
                    traceback.format_exc(),
                )

        log.info(
            "KnowledgeFusionIndexEventsPublished events_published=%d entities_queued=%d",
            events_published,
            total_queued,
        )

    async def _index_session_entities_with_chunking(
        self, scorables: List[Scorable]
    ) -> None:
        """
        Chunk large texts and publish async indexing requests.
        """
        total_queued = 0
        events_published = 0

        log.info(
            "KnowledgeFusionChunkIndexStart scorables=%d max_chunk_size=%d",
            len(scorables),
            self.kfc.max_chunk_size,
        )

        for s_idx, scorable in enumerate(
            tqdm(
                scorables,
                desc="KnowledgeFusion Chunked Indexing",
                unit="scorable",
            )
        ):
            text = (scorable.text or "").strip()
            if not text:
                continue

            # Build chunks with offsets
            chunks: List[Dict[str, Any]] = []
            if len(text) > self.kfc.max_chunk_size:
                for i in range(0, len(text), self.kfc.max_chunk_size):
                    chunk_text = text[i : i + self.kfc.max_chunk_size]
                    chunks.append(
                        {
                            "id": f"{scorable.id}_chunk_{i}",
                            "text": chunk_text,
                            "offset": i,
                        }
                    )
            else:
                chunks = [{"id": scorable.id, "text": text, "offset": 0}]

            for c_idx, chunk in enumerate(chunks):
                try:
                    if c_idx % 5 == 0:
                        log.info(
                            "KnowledgeFusionChunkIndexProgress scorable_id=%s chunk_idx=%d/%d",
                            str(scorable.id),
                            c_idx,
                            len(chunks),
                        )

                    # 1) Domain classification (thread)
                    domain_matches = await asyncio.to_thread(
                        self.domain_clf.classify,
                        chunk["text"],
                        self.kfc.top_domains,
                        self.kfc.min_domain_score,
                    )
                    domains = [
                        {"domain": d, "score": float(s)}
                        for d, s in domain_matches
                    ]

                    # 2) Entity detection (thread) on chunk text
                    results = await asyncio.to_thread(
                        self.entity_detector.detect_entities,
                        chunk["text"],
                    )
                    entities = self._format_entities(
                        results, chunk["text"], source="paper"
                    )

                    filtered_ents = [
                        e
                        for e in entities
                        if e.get(
                            "calibrated_similarity",
                            e.get("similarity", 0.0),
                        )
                        >= self.kfc.ner_min_sim
                    ]

                    # 3) Adjust spans relative to full text offset
                    offset = chunk.get("offset", 0)
                    for e in filtered_ents:
                        e["start"] += offset
                        e["end"] += offset

                    # 4) Relationships within chunk
                    relationships: List[Dict[str, Any]] = []
                    for i, e1 in enumerate(filtered_ents):
                        for j in range(i + 1, len(filtered_ents)):
                            e2 = filtered_ents[j]
                            distance = abs(e1["end"] - e2["start"])
                            if distance < 100:
                                rel_type = self._infer_relationship_type(e1, e2)
                                confidence = (
                                    self._calculate_relationship_confidence(
                                        e1, e2, distance, domains
                                    )
                                )
                                if confidence >= 0.75:
                                    relationships.append(
                                        {
                                            "source": f"{chunk['id']}:{e1['type']}:{e1['start']}-{e1['end']}",
                                            "target": f"{chunk['id']}:{e2['type']}:{e2['start']}-{e2['end']}",
                                            "type": rel_type,
                                            "confidence": confidence,
                                        }
                                    )

                    # 5) Publish event
                    event = {
                        "event_type": "knowledge_graph.index_request",
                        "payload": {
                            "scorable_id": chunk["id"],
                            "scorable_type": f"{scorable.target_type}_chunk",
                            "text": chunk["text"],
                            "entities": filtered_ents,
                            "domains": domains,
                            "relationships": relationships,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "source_agent": "KnowledgeFusionAgent",
                            "is_chunk": True,
                            "original_scorable_id": scorable.id,
                        },
                    }

                    await self.memory.bus.publish(
                        subject=event["event_type"],
                        payload=event["payload"],
                    )

                    total_queued += len(filtered_ents)
                    events_published += 1

                except Exception as e:
                    log.warning(
                        "KnowledgeFusionChunkIndexEventFailed chunk_id=%s error=%s",
                        chunk.get("id"),
                        str(e),
                    )
                    log.debug(
                        "KnowledgeFusionChunkIndexEventFailed traceback=%s",
                        traceback.format_exc(),
                    )

        log.info(
            "KnowledgeFusionChunkIndexEventsPublished events_published=%d entities_queued=%d",
            events_published,
            total_queued,
        )

    # ------------------------------------------------------------------
    # Domain scoring
    # ------------------------------------------------------------------

    def _score_section_domains(
        self, sections: List[Dict[str, Any]], goal: Dict[str, Any]
    ) -> Dict[Any, List[Dict[str, float]]]:
        """
        Score top domains for each section using seed-centroid classifier.
        """
        section_domains: Dict[Any, List[Dict[str, float]]] = {}

        # Optionally bias domains by goal text
        goal_text = (goal.get("goal_text") or "").strip()
        bias_tokens = goal_text.split()[:12] if goal_text else []

        for sec in sections:
            text = (sec.get("section_text") or "").strip()
            sec_id = sec.get("id")
            if not text:
                continue

            # Compose classification text
            clf_text = text
            if bias_tokens:
                clf_text = " ".join(bias_tokens) + "\n\n" + text

            matches = self.domain_clf.classify(
                text=clf_text,
                top_k=self.kfc.top_domains,
                min_value=self.kfc.min_domain_score,
            )
            section_domains[sec_id] = [
                {"domain": d, "score": float(s)} for d, s in matches
            ]

        return section_domains

    # ------------------------------------------------------------------
    # Entity extraction per section
    # ------------------------------------------------------------------

    def _extract_section_entities(
        self,
        sections: List[Dict[str, Any]],
        section_domains: Dict[Any, List[Dict[str, float]]],
    ) -> Dict[Any, List[Dict[str, Any]]]:
        """
        Use EntityDetector (surface-level) + NERRetrieverEmbedder + calibration
        to build entities per section, including 'similar' cluster info.
        """
        section_entities: Dict[Any, List[Dict[str, Any]]] = {}

        for sec in sections:
            sec_id = sec.get("id")
            text = (sec.get("section_text") or "").strip()
            if not text:
                continue

            # Run surface NER (entities as spans)
            try:
                ner_results = self.entity_detector.detect_entities(text)
            except Exception:
                if not self.kfc.entity_detection_fallback:
                    log.warning(
                        "KnowledgeFusion: NER failed for section_id=%s; skipping",
                        sec_id,
                        exc_info=True,
                    )
                    continue
                # Fallback: no NER; treat as having no entities
                ner_results = []

            surface_entities = self._format_entities(
                ner_results, text, source="paper"
            )

            # Expand with domain-aware retrieval + calibration
            domains = section_domains.get(sec_id, [])
            expanded = self._expand_entities(surface_entities, domains)

            section_entities[sec_id] = expanded

        return section_entities

    def _expand_entities(
        self,
        surface_entities: List[Dict[str, Any]],
        section_domains: List[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Expand entities with domain-aware retrieval and calibrated similarity"""
        expanded = []
        # Get primary domain for calibration (highest scoring)
        primary_domain = (
            section_domains[0]["domain"] if section_domains else None
        )

        for ent in surface_entities:
            # Query with domain calibration
            try:
                # Critical: Use domain for calibration
                similar, raw_sim = self.ner.retrieve_similar(
                    query=ent["text"],
                    k=self.kfc.ner_k,
                    domain=primary_domain,
                    return_raw_similarity=True,
                )
                calibrated_prob = self.calibration.calibrate_probability(
                    domain=primary_domain, raw_sim=raw_sim
                )

                # Calculate confidence in this calibration
                calibration_confidence = self.calibration.get_confidence(
                    domain=primary_domain, query=ent["text"]
                )

                # Apply confidence-weighted threshold
                effective_threshold = (
                    self.kfc.ner_min_calibrated_sim
                    * (0.8 + 0.4 * calibration_confidence)
                )
                if calibrated_prob >= effective_threshold:
                    similar.append(
                        {
                            "entity_text": s.get("entity_text", ""),
                            "similarity": float(s.get("similarity", 0.0)),
                            "calibrated_similarity": float(
                                calibrated_prob
                            ),
                        }
                        for s in similar
                    )

                ent["similar"] = similar
                ent["calibrated_similarity"] = float(calibrated_prob)

            except Exception as e:
                log.warning(
                    "KnowledgeFusion: entity expansion failed for %s: %s",
                    ent.get("text"),
                    str(e),
                )
                ent["similar"] = []
                ent["calibrated_similarity"] = ent.get("similarity", 0.0)

            expanded.append(ent)

        return expanded

    # ------------------------------------------------------------------
    # Chat snippet selection
    # ------------------------------------------------------------------

    def _select_chat_snippets(
        self,
        chat_corpus: List[Dict[str, Any]],
        section_entities: Dict[Any, List[Dict[str, Any]]],
    ) -> Dict[Any, List[Dict[str, Any]]]:
        """
        For each section, pick up to max_chat_snippets chat snippets that share
        entities with that section's entities.
        """
        section_to_snippets: Dict[Any, List[Dict[str, Any]]] = {
            k: [] for k in section_entities.keys()
        }

        # Flatten section entities into text set for quick overlap
        section_ent_texts: Dict[Any, set] = {}
        for sec_id, ents in section_entities.items():
            section_ent_texts[sec_id] = {
                e["text"].lower() for e in ents if e.get("text")
            }

        # For each chat message, compute entity overlap with each section
        for msg in chat_corpus or []:
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            msg_ents = {
                e["text"].lower()
                for e in self._format_entities(
                    self.entity_detector.detect_entities(text),
                    text,
                    source="chat",
                )
                if e.get("text")
            }

            for sec_id, sec_ents in section_ent_texts.items():
                overlap = sec_ents.intersection(msg_ents)
                if not overlap:
                    continue
                # crude similarity = overlap size
                sim = float(len(overlap))
                section_to_snippets[sec_id].append(
                    {
                        "snippet": _truncate(text, 260),
                        "overlap_entities": sorted(list(overlap)),
                        "sim": sim,
                    }
                )

        # Keep top-k per section
        for sec_id, snippets in section_to_snippets.items():
            snippets.sort(key=lambda x: x["sim"], reverse=True)
            section_to_snippets[sec_id] = snippets[: self.kfc.max_chat_snippets]

        return section_to_snippets

    # ------------------------------------------------------------------
    # Claims + metrics + tags
    # ------------------------------------------------------------------

    def _get_parent_domain(self, domain: str) -> Optional[str]:
        """Get parent domain from hierarchy configuration."""
        domain_hierarchy = self.cfg.get("domain_hierarchy", {})
        return domain_hierarchy.get(domain)

    def _infer_relationship_type(self, e1: Dict, e2: Dict) -> str:
        ordered = e1["end"] < e2["start"]
        first, second = (e1, e2) if ordered else (e2, e1)
        type_pairs = {
            ("METHOD", "DATASET"): "applied_to",
            ("METHOD", "METRIC"): "evaluated_by",
            ("MODEL", "DATASET"): "trained_on",
        }
        key = (first.get("type"), second.get("type"))
        return type_pairs.get(key, "RELATED")

    def _calculate_relationship_confidence(
        self,
        e1: Dict[str, Any],
        e2: Dict[str, Any],
        distance: int,
        domains: List[Dict[str, Any]],
    ) -> float:
        """Heuristic confidence combining distance + domain prominence."""
        base = 1.0 if distance < 50 else 0.8 if distance < 100 else 0.6
        domain_boost = 0.0
        for d in domains[:3]:
            if d.get("score", 0.0) > 0.5:
                domain_boost += 0.05
        return min(1.0, base + domain_boost)

    def _format_entities(
        self,
        ner_results: List[Dict[str, Any]],
        text: str,
        source: str = "paper",
    ) -> List[Dict[str, Any]]:
        """Normalize raw NER tuples or dicts into a unified entity schema."""
        entities = []
        for r in ner_results or []:
            if isinstance(r, dict):
                ent_text = (r.get("text") or "").strip()
                ent_type = r.get("type") or r.get("label") or "MISC"
                start = int(r.get("start", 0))
                end = int(r.get("end", start + len(ent_text)))
            else:
                # tuple fallback: (text, type, start, end)
                try:
                    ent_text, ent_type, start, end = r
                    ent_text = (ent_text or "").strip()
                except Exception:
                    continue

            if not ent_text:
                continue

            snippet = _truncate(text[start:end], 120)
            entities.append(
                {
                    "text": ent_text,
                    "type": ent_type,
                    "start": start,
                    "end": end,
                    "source": source,
                    "snippet": snippet,
                }
            )

        return entities

    def _generate_claims(
        self,
        sections: List[Dict[str, Any]],
        section_entities: Dict[Any, List[Dict[str, Any]]],
    ) -> Dict[Any, List[Dict[str, Any]]]:
        """
        Simple claim extractor: treat each sufficiently long sentence as a claim,
        and mark which entities it references.
        """
        section_claims: Dict[Any, List[Dict[str, Any]]] = {}

        for sec in sections:
            sec_id = sec.get("id")
            text = (sec.get("section_text") or "").strip()
            if not text:
                continue

            texts = sentences(text)
            ents = section_entities.get(sec_id, [])

            claims = []
            for s in texts:
                if len(s) < 40:
                    continue
                # find which entities appear in this sentence
                grounded = [
                    e["text"]
                    for e in ents
                    if e["text"].lower() in s.lower()
                ]
                if not grounded:
                    continue
                claim_id = hash_text(f"{sec_id}:{s}")
                claims.append(
                    {
                        "claim_id": claim_id,
                        "claim": s,
                        "grounded_entities": _uniq(grounded),
                    }
                )

            section_claims[sec_id] = claims

        return section_claims

    def _generate_tags(
        self, domains: List[Dict[str, float]], entities: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate tags with domain weighting as per PACS.md."""
        # Top domains (weighted higher)
        domain_tags = [
            d["domain"]
            for d in sorted(domains, key=lambda x: x["score"], reverse=True)[
                :3
            ]
        ]
        # Top entities (by length; naive proxy for salience)
        ent_texts = [e["text"] for e in entities]
        ent_texts.sort(key=len, reverse=True)
        ent_tags = ent_texts[:5]

        tags = [t.lower().replace(" ", "_") for t in domain_tags + ent_tags]
        return _uniq(tags)


class CalibrationTrainer:
    """Handles periodic training of calibration models from collected data."""

    def __init__(
        self, cfg: Dict, memory, logger: Any, calibration_manager: "CalibrationManager"
    ):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.calibration = calibration_manager
        self.last_train = 0
        self.train_interval = cfg.get(
            "calibration_train_interval", 3600
        )  # Default: 1 hour
        self.lookback_hours = cfg.get("calibration_lookback_hours", 24)

    def maybe_train(self) -> bool:
        now = time()
        if now - self.last_train < self.train_interval:
            return False

        trained_any = False
        for domain in self._get_domains_to_train():
            pos, neg, total = self.calibration.domain_counts(
                domain
            )  # implement below
            self.logger.info(
                "CalibrationTrainer: label mix",
                extra={
                    "domain": domain,
                    "pos": pos,
                    "neg": neg,
                    "total": total,
                },
            )

            MIN_POS = self.cfg.get("min_pos", 10)
            MIN_NEG = self.cfg.get("min_neg", 10)

            if pos == 0 and neg == 0:
                self.logger.info(
                    "CalibrationTrainer: no samples yet",
                    extra={"domain": domain},
                )
                continue

            # Train with fallback if single class; require balance for full model
            if pos < 1 or neg < 1:
                log.warning(
                    "CalibrationTrainer: one-class data — using fallback",
                    extra={"domain": domain, "pos": pos, "neg": neg},
                )
                trained_any |= self.calibration.train_model(
                    domain, allow_fallback=True
                )
                continue

            if pos < MIN_POS or neg < MIN_NEG:
                log.warning(
                    "CalibrationTrainer: skipping — insufficient class balance",
                    extra={
                        "domain": domain,
                        "pos": pos,
                        "neg": neg,
                        "need_pos": MIN_POS,
                        "need_neg": MIN_NEG,
                    },
                )
                continue

            trained_any |= self.calibration.train_model(
                domain, allow_fallback=False
            )

        if trained_any:
            self.last_train = now
        return trained_any

    def _get_domains_to_train(self) -> List[str]:
        """Get domains that need retraining."""
        # Configured domains
        configured = self.cfg.get("domains", ["general"])

        # Recently active domains
        recent = self.memory.calibration_events.get_recent_domains(self.lookback_hours)

        # Deduplicate + preserve order
        return list(dict.fromkeys(configured + recent))

