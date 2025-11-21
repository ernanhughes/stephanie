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

Designed to be piped directly into DraftGeneratorAgent (as 'section plan').
"""

from __future__ import annotations

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
from stephanie.models.ner_retriever import EntityDetector, NERRetrieverEmbedder
from stephanie.scoring.calibration_manager import CalibrationManager
from stephanie.scoring.scorable import Scorable, ScorableFactory
from stephanie.tools.scorable_classifier import ScorableClassifier

log = logging.getLogger(__name__)


def _sentences(text: str) -> List[str]:
    if not text:
        return []
    # light sentence split
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


def _unique_keep_order(xs: List[str]) -> List[str]:
    seen, out = set(), []
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
        0.45  # Critical: use calibrated similarity threshold
    )
    ephemeral_index_dir: str = "/tmp"  # index writes go to tmp; nothing to DB
    max_chat_snippets: int = 12
    max_chunk_size: int = 5000  # For indexing large texts
    entity_detection_fallback: bool = (
        True  # Use heuristic fallback if BERT-NER fails
    )
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
            enable_chunking=cfg.get("enable_chunking", False)
        )
        # Domain backbone (no DB writes of domain tags)
        self.domain_clf = ScorableClassifier(
            memory=self.memory,
            logger=self.logger,
            config_path=cfg.get("domain_config", "config/domain/seeds.yaml"),
            metric=cfg.get("domain_metric", "cosine"),
        )
        # Entity layer (ANN over session-only content)
        self.entity_detector = EntityDetector(device=cfg.get("device", "cuda"))
        self.ner = NERRetrieverEmbedder(
            model_name=cfg.get(
                "ner_model", "dslim/bert-base-NER"
            ),
            layer=cfg.get("ner_layer", 16), # in paper we seee 17 here the llm has on y 16 layers
            device=cfg.get("device", "cpu"),
            embedding_dim=cfg.get("ner_dim", 2048),
            index_path="data/ner_retriever/index",   # persistent path
            projection_enabled=cfg.get("ner_projection", False),
            projection_dim=cfg.get("ner_projection_dim", 2048),
            projection_dropout=cfg.get("ner_projection_dropout", 0.1),
            logger=self.logger,
            memory=self.memory,
            cfg=cfg,
        )

        from stephanie.scoring.calibration_manager import CalibrationManager
        self.calibration = CalibrationManager(
            cfg=cfg.get("calibration", {}),
            memory=self.memory,
            logger=self.logger
        )
    
        # ADD THIS: Periodic calibration trainer
        self.calibration_trainer = CalibrationTrainer(
            cfg=cfg.get("calibration", {}),
            memory=self.memory,
            logger=self.logger,
            calibration_manager=self.calibration,
        )


    async def run(self, context: dict) -> dict:
        goal = context.get("goal", {})

        chat = self.memory.chats.get_top_conversations(limit=10)
        documents = context.get("documents", []) or []
        for paper in tqdm(
            documents, desc="KnowledgeFusion Papers", unit="paper"
        ):
            sections = (
                self.memory.document_sections.get_by_document(paper.get("id"))
                or []
            )
            sections = [
                s.to_dict()
                for s in sections
                if s.section_text and len(s.section_text) > 20
            ]
            self.calibration_trainer.maybe_train()

            self.report(
                {
                    "event": "start",
                    "step": "KnowledgeFusion",
                    "details": f"Sections: {len(sections)}, Chat msgs: {len(chat)}",
                    "paper_title": paper.get("title"),
                }
            )

            # Build session index
            scorables = self._build_session_scorables(sections, chat)
            if self.kfc.enable_chunking:
                await self._index_session_entities_with_chunking(scorables)
            else:
                await self._index_session_entities(scorables)

            # Progress bar for sections
            plans: List[Dict[str, Any]] = []
            for sec in tqdm(
                sections,
                desc=f"Sections of {paper.get('title', 'paper')}",
                unit="sec",
            ):
                try:
                    chat_dicts = [msg[0].to_dict() for msg in chat] if chat else []
                    plan = self._plan_for_section(sec, goal, chat_dicts)
                    plans.append(plan)
                except Exception as e:
                    self.logger.log(
                        "KnowledgeFusionSectionError",{
                            "section": sec.get("section_name"),
                            "error": str(e),
                            "traceback": traceback.format_exc(),
                        },
                    )

                # Structured log progress %
                pct = round(100 * len(plans) / max(1, len(sections)), 2)
                self.logger.log(
                    "KnowledgeFusionProgress",
                    {
                        "paper": paper.get("title"),
                        "sections_done": len(plans),
                        "sections_total": len(sections),
                        "pct_complete": pct,
                    },
                )

            context[self.output_key] = plans
            context["knowledge_plans"] = plans

            self.report(
                {
                    "event": "end",
                    "step": "KnowledgeFusion",
                    "details": f"Produced {len(plans)} plans (transient)",
                }
            )
        return context

    # ----------------------------
    # Internals
    # ----------------------------
    def _plan_for_section(self, section: dict, goal: dict, chat: List[Dict[str, str]]) -> Dict[str, Any]:
        sec_name = section.get("section_name", "section")
        text = section.get("section_text", "") or ""
        # 🔴 ADD: Fail fast if no content
        if not text.strip():
            self.logger.log("SectionTextMissing", {
                "section_name": sec_name,
                "paper_id": section.get("paper_id"),
                "warning": "Skipping plan generation due to empty section_text"
            })
            return {
                "section_title": sec_name,
                "paper_id": section.get("paper_id"),
                "error": "empty_section_text",
                "domains": [],
                "entities": [],
                "claims": [],
                "tags": [],
                "meta": {"skipped": True}
            }
        paper_id = section.get("paper_id")
        
        # ✅ CORRECT: Get actual scorable ID and target type
        scorable_id = str(section.get("id", f"temp_{uuid.uuid4().hex}"))
        scorable_type = section.get("target_type", "document_section")
        
        # A) Domains (top_k = 20; no DB writes) - WITH PROPER GOAL CONTEXT
        goal_context = self._build_goal_context(goal)
        dom_matches = self.domain_clf.classify(
            text=text,
            top_k=self.kfc.top_domains,
            min_value=self.kfc.min_domain_score,
            context=goal_context,
        )
        domains = [{"domain": d, "score": float(s)} for d, s in dom_matches]
        
        # B) Entities in the section (surface) + expand w/ nearest neighbors (from session index)
        surface_entities = self._detect_entities(text)
        # Critical: Pass domains to entity expansion for calibration
        expanded_entities = self._expand_entities(surface_entities, domains)
        
        # C) Chat overlap: find chat snippets that share entities or are semantically close
        chat_support = self._chat_overlap(chat, surface_entities, expanded_entities, limit=self.kfc.max_chat_snippets)
        
        # D) Claims with entity grounding
        claims = self._extract_claims_with_entities(text, expanded_entities)
        
        # E) Quick tags for the improver: domains + top entity lemmas
        tags = self._generate_tags(domains, expanded_entities)
        
        plan = {
            "section_title": sec_name,
            "section_name": sec_name,
            "paper_id": paper_id,
            "scorable_id": scorable_id,  
            "scorable_type": scorable_type,  
            "paper_text": text,
            "domains": domains,
            "entities": expanded_entities,
            "chat_support": chat_support,
            "claims": claims,
            "tags": tags,
            "goal_template": self.cfg.get("goal_template", "academic_summary"),
            "generation_style": self.cfg.get("generation_style", "grounded_explanatory"),
            "meta": {
                "knowledge_hash": self._compute_hash(paper_id, text, chat),
                "domain_confidence": self._get_domain_confidence(domains)
            }
        }
        return plan

    def _build_goal_context(self, goal: dict) -> Dict[str, Any]:
        """Build proper goal context for domain classification as per PACS.md"""
        return {
            "goal_text": goal.get("goal_text", ""),
            "goal_id": goal.get("id", ""),
            "strategy": goal.get("strategy", ""),
            "focus_area": goal.get("focus_area", ""),
            "goal_type": goal.get("type", "blog_generation"),
            "audience": goal.get("audience", "academic"),
            "intent": goal.get("intent", "explanation"),
        }

    def _detect_entities(
        self, text: str, source: str = "paper"
    ) -> List[Dict[str, Any]]:
        """Full NER pipeline with BERT-NER + heuristic fallback as per PACS.md"""
        # Primary: BERT-NER
        try:
            results = self.entity_detector.detect_entities(text)
            if results:
                return self._format_entities(results, text, source)
        except Exception as e:
            self.logger.log(
                "NERFallback", {"error": str(e), "method": "bert-ner"}
            )

        # Fallback: Heuristic rules if configured
        if self.kfc.entity_detection_fallback:
            return self._heuristic_entity_detection(text, source)
        return []

    def _format_entities(
        self, results: List[Dict[str, Any]], text: str, source: str
    ) -> List[Dict[str, Any]]:
        """Format entity detector results into standardized structure with calibrated similarity."""
        entities = []
        type_map = {
            "PER": "PERSON",
            "ORG": "ORGANIZATION",
            "LOC": "LOCATION",
            "MISC": "MISC",
            "DATE": "DATE",
            "TIME": "TIME",
            "MONEY": "MONEY",
            "PERCENT": "PERCENT",
            "FAC": "FACILITY",
            "GPE": "GPE",
            "METHOD": "METHOD",
            "METRIC": "METRIC",
            "ACRONYM": "ACRONYM",
        }

        for ent in results:
            start = ent.get("start", 0)
            end = ent.get("end", 0)
            etype = ent.get("type", "UNKNOWN")
            std_type = type_map.get(etype, etype)

            entities.append(
                {
                    "text": ent.get("text", text[start:end]),
                    "type": std_type,
                    "start": start,
                    "end": end,
                    "source": source,
                    "similarity": ent.get("score", 0.9),
                    "calibrated_similarity": ent.get("score", 0.9),
                }
            )

        return entities

    def _heuristic_entity_detection(
        self, text: str, source: str
    ) -> List[Dict[str, Any]]:
        """Heuristic entity detection as fallback per PACS.md."""
        entities = []

        # 1. Acronyms (all-caps words > 2 chars)
        for match in re.finditer(r"\b([A-Z]{2,})\b", text):
            entities.append(
                {
                    "text": match.group(1),
                    "type": "ACRONYM",
                    "start": match.start(),
                    "end": match.end(),
                    "source": source,
                    "similarity": 0.7,
                    "calibrated_similarity": 0.7,
                }
            )

        # 2. Methods (common ML terms)
        method_terms = [
            "MCTS",
            "Chain-of-Thought",
            "RAG",
            "Transformer",
            "AlphaZero",
            "L2Norm",
            "Backprop",
            "Gradient",
            "Attention",
            "Embedding",
        ]
        for term in method_terms:
            for match in re.finditer(
                r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE
            ):
                entities.append(
                    {
                        "text": term,
                        "type": "METHOD",
                        "start": match.start(),
                        "end": match.end(),
                        "source": source,
                        "similarity": 0.8,
                        "calibrated_similarity": 0.8,
                    }
                )

        # 3. Metrics (common ML metrics)
        metric_terms = [
            "accuracy",
            "precision",
            "recall",
            "F1",
            "AUC",
            "RMSE",
            "MAE",
            "BLEU",
            "ROUGE",
            "perplexity",
        ]
        for term in metric_terms:
            for match in re.finditer(
                r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE
            ):
                entities.append(
                    {
                        "text": term,
                        "type": "METRIC",
                        "start": match.start(),
                        "end": match.end(),
                        "source": source,
                        "similarity": 0.75,
                        "calibrated_similarity": 0.75,
                    }
                )

        return entities

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
                sims = self.ner.retrieve_entities(
                    query=ent["text"],
                    k=self.kfc.ner_k,
                    min_similarity=self.kfc.ner_min_sim,
                    domain=primary_domain,
                )
            except Exception as e:
                self.logger.log(
                    "NERQueryError", {"entity": ent["text"], "error": str(e)}
                )
                sims = []

            similar = []
            for s in sims:
                # Always prefer calibrated similarity if available
                raw_sim = s.get("similarity", 0.0)
                calibrated_prob = self.calibration.get_calibrated_probability(
                    domain=primary_domain,
                    raw_sim=raw_sim
                )

                
                # Calculate confidence in this calibration
                calibration_confidence = self.calibration.get_confidence(
                    domain=primary_domain,
                    query=ent["text"]
                )
                
                # Apply confidence-weighted threshold
                effective_threshold = self.kfc.ner_min_calibrated_sim * (0.8 + 0.4 * calibration_confidence)

                effective_threshold = self.kfc.ner_min_calibrated_sim
                if calibrated_prob >= effective_threshold:
                    similar.append(
                        {
                            "entity_text": s.get("entity_text", ""),
                            "similarity": float(raw_sim),
                            "calibrated_similarity": float(calibrated_prob),
                            "calibration_confidence": calibration_confidence,
                            "entity_type": s.get("entity_type", "UNKNOWN"),
                            "source_text": s.get("source_text", "")[:200],
                            "scorable_id": s.get("scorable_id", ""),
                            "scorable_type": s.get("scorable_type", ""),
                            "domain": s.get("domain", primary_domain),
                        }
                    )

                # Log calibration event
                self.calibration.log_event(
                    domain=primary_domain,
                    query=ent["text"],
                    raw_sim=raw_sim,
                    is_relevant=calibrated_prob >= self.kfc.ner_min_calibrated_sim,
                    scorable_id=ent.get("scorable_id", "unknown"),
                    scorable_type=ent.get("scorable_type", "unknown"),
                    entity_type=ent.get("type", None)
                )

           # Only include if we have meaningful similar entities
            if similar or ent.get("calibrated_similarity", 0.0) >= self.kfc.ner_min_calibrated_sim:
                expanded.append(
                    {
                        "text": ent["text"],
                        "type": ent["type"],
                        "source": ent["source"],
                        "similar": similar,
                        "similarity": ent.get("similarity", 0.0),
                        "calibrated_similarity": ent.get("calibrated_similarity", 0.0),
                    }
                )

        # Dedup by head text, prioritize those with better calibrated similarity
        return self._dedup_entities(expanded)

    def _dedup_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Deduplicate entities while preserving those with highest calibrated similarity"""
        by_head = {}
        for e in entities:
            key = e["text"].lower()
            # If we have calibrated similarity, use that for comparison
            current_sim = by_head.get(key, {}).get(
                "calibrated_similarity", 0.0
            )
            new_sim = e.get("calibrated_similarity", 0.0)

            if key not in by_head or new_sim > current_sim:
                by_head[key] = e

        return list(by_head.values())

    def _extract_claims_with_entities(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract claims with entity grounding as per PACS.md."""
        sents = [s for s in _sentences(text) if len(s) > 40]
        claims = [
            {"claim_id": f"C{i + 1}", "claim": s}
            for i, s in enumerate(sents[:5])
        ]

        # Map entities to claims
        entity_map = {e["text"].lower(): e for e in entities}
        for claim in claims:
            claim["grounded_entities"] = []
            for word in claim["claim"].split():
                word_lower = word.lower().strip(".,;:")
                if word_lower in entity_map:
                    claim["grounded_entities"].append(
                        entity_map[word_lower]["text"]
                    )

        return claims

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

        # Top entities (weighted lower)
        entity_tags = []
        for entity in entities:
            # Prefer entities with high calibrated similarity
            for similar in sorted(
                entity.get("similar", []),
                key=lambda x: x.get("calibrated_similarity", 0),
                reverse=True,
            ):
                if (
                    similar["calibrated_similarity"] > 0.7
                    and similar["entity_text"] not in entity_tags
                ):
                    entity_tags.append(similar["entity_text"])
            if len(entity_tags) >= 5:
                break

        # Combine with domain weighting
        return _unique_keep_order(domain_tags + entity_tags)

    def _chat_overlap(
        self,
        chat: List[Dict[str, str]],
        surface_entities: List[Dict[str, Any]],
        expanded_entities: List[Dict[str, Any]],
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Rank chat snippets by entity overlap + embedding similarity to section entities.
        No DB writes; uses memory.embedding if available for transient scoring.
        """
        # entity vocabulary from section
        entity_terms = set([e["text"].lower() for e in surface_entities])
        for e in expanded_entities:
            for s in e.get("similar", []):
                t = s.get("entity_text", "").lower()
                if t:
                    entity_terms.add(t)

        results = []
        if not chat:
            return results

        # Prepare an aggregate entity string as query anchor
        query_anchor = (
            ", ".join(sorted(list(entity_terms))[:24]) or "section entities"
        )
        try:
            q_emb = self.memory.embedding.get_or_create(query_anchor)
        except Exception:
            q_emb = None

        for item in chat:
            text = item.get("text", "") or ""
            if not text:
                continue
            # quick overlap
            overlap = [t for t in entity_terms if t in text.lower()]
            overlap_score = (
                min(1.0, len(overlap) / max(1, len(entity_terms)))
                if entity_terms
                else 0.0
            )

            # semantic proximity (optional if embedding infra present)
            sim_score = 0.0
            if q_emb is not None:
                try:
                    c_emb = self.memory.embedding.get_or_create(text[:2000])
                    # cosine similarity
                    num = float((q_emb * c_emb).sum())
                    den = (float((q_emb * q_emb).sum()) ** 0.5) * (
                        float((c_emb * c_emb).sum()) ** 0.5
                    ) + 1e-8
                    sim_score = num / den
                except Exception as e:
                    self.logger.log(
                        "ChatOverlapEmbeddingError",
                        {"error": str(e), "snippet_length": len(text)},
                    )
                    sim_score = 0.0

            score = 0.6 * overlap_score + 0.4 * sim_score
            if score > 0.05:  # keep useful ones
                results.append(
                    {
                        "snippet": text[:500],
                        "overlap_entities": overlap[:8],
                        "sim": round(float(score), 4),
                        "role": item.get("role"),
                        "ts": item.get("ts"),
                    }
                )

        # sort by score desc, truncate
        results.sort(key=lambda r: r["sim"], reverse=True)
        return results[:limit]

    # ----------------------------
    # Session indexing (ephemeral)
    # ----------------------------
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
                
            # ✅ CORRECT: Use actual DB ID (as string) and proper target type
            scorable_id = str(sec.get("id", f"temp_{uuid.uuid4().hex}"))
            target_type = sec.get("target_type", "document_section")
            
            scorables.append(Scorable(
                id=scorable_id,
                text=text,
                target_type=target_type
            ))

        # Recent chat messages as scorables (so entities from chat are retrievable)
        for msg in chat or []:
            scorable = ScorableFactory.from_orm(msg[0])
            scorables.append(scorable)

        return scorables

    async def _index_session_entities(self, scorables: List[Scorable]) -> None:
        """
        Instead of indexing directly, publish to KnowledgeBus for async processing.
        No DB writes; no blocking.
        """
        total_queued = 0
        events_published = 0

        for scorable in scorables:
            text = scorable.text.strip()
            if len(text) < 100:
                continue

            try:
                # Classify domains for this scorable
                domain_matches = self.domain_clf.classify(
                    text=text,
                    top_k=self.kfc.top_domains,
                    min_value=self.kfc.min_domain_score
                )
                domains = [{"domain": d, "score": float(s)} for d, s in domain_matches]
                # Detect entities
                results = self.entity_detector.detect_entities(text)  # raw tuples
                entities = self._format_entities(results, text, source="paper")  # normalize to dicts
                filtered_ents = [
                    e for e in entities
                    if e.get("calibrated_similarity", e.get("similarity", 0)) >= self.kfc.ner_min_sim
                ]

                # Build relationships (local heuristic)
                relationships = []
                for i, e1 in enumerate(filtered_ents):
                    for j in range(i + 1, len(filtered_ents)):
                        e2 = filtered_ents[j]
                        distance = abs(e1["end"] - e2["start"])
                        if distance < 100:
                            rel_type = self._infer_relationship_type(e1, e2)
                            confidence = self._calculate_relationship_confidence(e1, e2, distance, domains)
                            if confidence >= 0.75:
                                relationships.append({
                                    "source": f"{scorable.id}:{e1['type']}:{e1['start']}-{e1['end']}",
                                    "target": f"{scorable.id}:{e2['type']}:{e2['start']}-{e2['end']}",
                                    "type": rel_type,
                                    "confidence": confidence
                                })

                # Publish indexing job
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
                        "source_agent": "KnowledgeFusionAgent"
                    }
                }

                await self.memory.bus.publish(
                    subject=event["event_type"],
                    payload=event["payload"]
                )

                total_queued += len(filtered_ents)
                events_published += 1

            except Exception as e:
                self.logger.log("KnowledgeFusionIndexEventFailed", {
                    "scorable_id": scorable.id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })

        self.logger.log("KnowledgeFusionIndexEventsPublished", {
            "events_published": events_published,
            "entities_queued": total_queued
        })

    async def _index_session_entities_with_chunking(self, scorables: List[Scorable]) -> None:
        """
        Chunk large texts and publish async indexing requests.
        """
        total_queued = 0
        events_published = 0

        for scorable in scorables:
            text = scorable.text.strip()
            if not text:
                continue

            chunks = []

            if len(text) > self.kfc.max_chunk_size:
                for i in range(0, len(text), self.kfc.max_chunk_size):
                    chunk_text = text[i:i + self.kfc.max_chunk_size]
                    if len(chunk_text.strip()) > 50:
                        chunk_id = f"{scorable.id}_chunk_{i // self.kfc.max_chunk_size}"
                        chunks.append({
                            "id": chunk_id,
                            "text": chunk_text,
                            "parent_id": scorable.id,
                            "offset": i
                        })
            else:
                chunks = [{"id": scorable.id, "text": text}]

            for chunk in chunks:
                try:
                    # Reuse domain classification logic
                    domain_matches = self.domain_clf.classify(
                        text=chunk["text"],
                        top_k=self.kfc.top_domains,
                        min_value=self.kfc.min_domain_score
                    )
                    domains = [{"domain": d, "score": float(s)} for d, s in domain_matches]

                    results = self.entity_detector.detect_entities(text)  # raw tuples
                    entities = self._format_entities(results, text, source="paper")  # normalize to dicts
                    filtered_ents = [
                        e for e in entities
                        if e.get("calibrated_similarity", e.get("similarity", 0)) >= self.kfc.ner_min_sim
                    ]

                    # Adjust entity spans relative to chunk offset
                    offset = chunk.get("offset", 0)
                    for e in filtered_ents:
                        e["start"] += offset
                        e["end"] += offset

                    # Relationships within chunk
                    relationships = []
                    for i, e1 in enumerate(filtered_ents):
                        for j in range(i + 1, len(filtered_ents)):
                            e2 = filtered_ents[j]
                            distance = abs(e1["end"] - e2["start"])
                            if distance < 100:
                                rel_type = self._infer_relationship_type(e1, e2)
                                confidence = self._calculate_relationship_confidence(e1, e2, distance, domains)
                                if confidence >= 0.75:
                                    relationships.append({
                                        "source": f"{chunk['id']}:{e1['type']}:{e1['start']}-{e1['end']}",
                                        "target": f"{chunk['id']}:{e2['type']}:{e2['start']}-{e2['end']}",
                                        "type": rel_type,
                                        "confidence": confidence
                                    })

                    # Publish event
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
                            "original_scorable_id": scorable.id
                        }
                    }

                    await self.memory.bus.publish(
                        subject=event["event_type"],
                        payload=event["payload"]
                    )
                    total_queued += len(filtered_ents)
                    events_published += 1

                except Exception as e:
                    self.logger.log("KnowledgeFusionChunkIndexEventFailed", {
                        "chunk_id": chunk.get("id"),
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })

        self.logger.log("KnowledgeFusionChunkedIndexEventsPublished", {
            "chunks_queued": events_published,
            "entities_queued": total_queued
        })

    # ----------------------------
    # Utility methods
    # ----------------------------
    def _compute_hash(
        self, paper_id: Optional[Any], text: str, chat: List[Dict]
    ) -> str:
        """Compute hash of knowledge state for versioning"""
        import hashlib

        combined = f"{paper_id or 'unknown'}|||{text[:1000]}|||{len(chat)}"
        return hashlib.sha256(combined.encode()).hexdigest()[:8]

    def _get_domain_confidence(self, domains: List[Dict[str, float]]) -> float:
        """Estimate confidence in domain classification"""
        if not domains:
            return 0.0
        # Weighted average of scores
        total = sum(d["score"] for d in domains)
        return min(1.0, total / len(domains))

    def _fallback_chat_corpus(self, context: dict) -> List[Dict[str, str]]:
        """
        Best-effort: if the caller didn't pass chat_corpus, try to pull recent
        conversational text from memory (safe, read-only). If nothing exists,
        return an empty list.
        """
        try:
            # If you maintain conversations/casebooks for chats, adapt here:
            # e.g., self.memory.conversations.latest(n=200)
            return context.get("recent_messages", [])
        except Exception as e:
            self.logger.log("ChatFallbackError", {"error": str(e)})
            return []

    def _get_domain_with_fallbacks(self, domain: str) -> str:
        """Get domain with hierarchical fallbacks (specific → parent → general → identity)."""
        # Try specific domain first
        if self.calibration.has_calibration(domain):
            return domain
            
        # Try parent domain (e.g., "computer_vision" → "ai")
        parent_domain = self._get_parent_domain(domain)
        if parent_domain and self.calibration.has_calibration(parent_domain):
            return parent_domain
            
        # Try general domain
        if self.calibration.has_calibration("general"):
            return "general"
            
        # No calibration available - use identity function
        return "identity"

    def _get_parent_domain(self, domain: str) -> Optional[str]:
        """Get parent domain from hierarchy configuration."""
        domain_hierarchy = self.cfg.get("domain_hierarchy", {})
        return domain_hierarchy.get(domain)
    
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


class CalibrationTrainer:
    """Handles periodic training of calibration models from collected data."""
    
    def __init__(self, cfg: Dict, memory, logger: Any, calibration_manager: 'CalibrationManager'):
        self.cfg = cfg
        self.memory = memory
        self.logger = logger
        self.calibration = calibration_manager
        self.last_train = 0
        self.train_interval = cfg.get("calibration_train_interval", 3600)  # Default: 1 hour
        self.lookback_hours = cfg.get("calibration_lookback_hours", 24)

    def maybe_train(self) -> bool:
        now = time()
        if now - self.last_train < self.train_interval:
            return False

        trained_any = False
        for domain in self._get_domains_to_train():
            pos, neg, total = self.calibration.domain_counts(domain)   # implement below
            self.logger.info("CalibrationTrainer: label mix",
                            extra={"domain": domain, "pos": pos, "neg": neg, "total": total})

            MIN_POS = self.cfg.get("calibration", {}).get("min_pos", 10)
            MIN_NEG = self.cfg.get("calibration", {}).get("min_neg", 10)

            if pos == 0 and neg == 0:
                self.logger.info("CalibrationTrainer: no samples yet", extra={"domain": domain})
                continue

            # Train with fallback if single class; require balance for full model
            if pos < 1 or neg < 1:
                log.warning("CalibrationTrainer: one-class data — using fallback",
                                    extra={"domain": domain, "pos": pos, "neg": neg})
                trained_any |= self.calibration.train_model(domain, allow_fallback=True)
                continue

            if pos < MIN_POS or neg < MIN_NEG:
                log.warning("CalibrationTrainer: skipping — insufficient class balance",
                                    extra={"domain": domain, "pos": pos, "neg": neg,
                                        "need_pos": MIN_POS, "need_neg": MIN_NEG})
                continue

            trained_any |= self.calibration.train_model(domain, allow_fallback=False)

        if trained_any:
            self.last_train = now
        return trained_any
    
    def _get_domains_to_train(self) -> List[str]:
        """Get domains that need retraining."""
        # Configured domains
        configured = self.cfg.get("domains", ["general"])
        
        # Recently active domains
        recent = self._get_recent_domains(self.lookback_hours)
        
        # Deduplicate + preserve order
        return list(dict.fromkeys(configured + recent))

    def _get_recent_domains(self, hours: int = 24) -> List[str]:
        """Get domains with recent calibration activity (fallback: general)."""
        try:
            # If CalibrationManager has a store:
            if hasattr(self.calibration, "memory") and hasattr(self.calibration.memory, "calibration_events"):
                since = datetime.now() - timedelta(hours=hours)
                return self.memory.calibration_events.get_recent_domains(since=since)
        except Exception as e:
            log.warning(f"CalibrationTrainer: failed to fetch recent domains: {e}")

        return ["general"]
