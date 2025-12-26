# stephanie/services/knowledge_graph_service.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from stephanie.memory.nexus_store import NexusStore
from stephanie.models.hnsw_index import HNSWIndex
from stephanie.models.ner_retriever import EntityDetector, NERRetrieverEmbedder
from stephanie.services.service_protocol import Service
from stephanie.tools.scorable_classifier import ScorableClassifier
from stephanie.utils.hash_utils import hash_text
from stephanie.utils.text_utils import safe_slice, sentences
from stephanie.services.subgraphs.edge_index import JSONLEdgeIndex
from stephanie.services.subgraphs.seed_finder import EmbeddingSeedFinder
from stephanie.services.subgraphs.subgraph_builder import SubgraphBuilder, SubgraphConfig


log = logging.getLogger(__name__)


def _normalize_text(t: str) -> str:
    # Minimal normalization; extend with NFC, ZWSP stripping as needed
    return (t or "").replace("\u200b", "").strip()


def _locate_sentence_ix(
    sent_spans: List[Tuple[int, int]], start: int, end: int
) -> int:
    for i, (s, e) in enumerate(sent_spans):
        # entity ranges that overlap a sentence count as in that sentence
        if not (end <= s or start >= e):
            return i
    return -1


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
        self.nexus_store: NexusStore = memory.nexus
        self.enabled = cfg.get("knowledge_graph", {}).get("enabled", True)
        self._initialized = False
        self._graph = None
        self._entity_detector = None
        self._retriever = None
        self._classifier = None

        self._edge_index = None
        self._seed_finder = None
        self._subgraph_builder = None

        # Paths for persistence
        self._rel_path = self.cfg.get(
            "relationship_store", "data/knowledge_graph/relationships.jsonl"
        )
        os.makedirs(os.path.dirname(self._rel_path), exist_ok=True)

        # Config params
        self.entity_threshold = cfg.get("knowledge_graph", {}).get(
            "entity_threshold", 0.65
        )
        self.relationship_threshold = cfg.get("knowledge_graph", {}).get(
            "relationship_threshold", 0.75
        )
        self.max_hops = cfg.get("knowledge_graph", {}).get("max_hops", 3)
        self.domain_aware = cfg.get("knowledge_graph", {}).get(
            "domain_aware", True
        )

        # also used by the verification/knowledge-tree path (separate from indexing threshold)
        self.verification_threshold = self.cfg.get("knowledge_graph", {}).get(
            "verification_threshold", 0.90
        )

        kg_cfg = cfg.get("knowledge_graph") or {}

        # hard limits / guards used by build_tree & helpers
        self.max_nodes = int(kg_cfg.get("max_nodes", 200))
        self.max_relationships = int(kg_cfg.get("max_relationships", 1000))
        self.max_rels_per_node = int(kg_cfg.get("max_rels_per_node", 16))
        self.max_entities_per_section = int(
            kg_cfg.get("max_entities_per_section", 128)
        )
        self.nexus_run_id = kg_cfg.get("nexus_run_id", "kg_live")

        # (if you didn’t add these earlier, include them now too)
        self.min_claim_len = int(kg_cfg.get("min_claim_len", 30))
        self.max_claim_len = int(kg_cfg.get("max_claim_len", 420))

        # service mode + deferred buffers (health_check/shutdown reference these)
        self.mode = kg_cfg.get("mode", "online")  # "online" | "deferred"
        self._pending_nodes: list = []
        self._pending_relationships: list = []

        # how we split into candidate claims (regex on sentence boundaries)
        self.claim_split_regex = kg_cfg.get("claim_split_regex", r"[.!?]\s+")

        # verification gate you might use elsewhere
        self.verification_threshold = float(
            kg_cfg.get("verification_threshold", 0.90)
        )

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
            "cache_misses": 0,
        }


    @property
    def name(self) -> str:
        return "knowledge-graph"

    @property
    def edge_index(self) -> JSONLEdgeIndex:
        if self._edge_index is None:
            self._edge_index = JSONLEdgeIndex(rel_path=self._rel_path, logger=self.logger)
        return self._edge_index

    @property
    def seed_finder(self) -> EmbeddingSeedFinder:
        if self._seed_finder is None:
            self._seed_finder = EmbeddingSeedFinder(
                search_entities_fn=self.search_entities,
                detect_entities_fn=self.detect_entities,
                logger=self.logger,
            )
        return self._seed_finder

    @property
    def subgraph_builder(self) -> SubgraphBuilder:
        if self._subgraph_builder is None:
            self._subgraph_builder = SubgraphBuilder(
                seed_finder=self.seed_finder,
                edge_index=self.edge_index,
                nexus_store=self.nexus_store,
                logger=self.logger,
            )
        return self._subgraph_builder

    def build_query_subgraph(self, *, query: str, cfg: SubgraphConfig | None = None) -> dict:
        cfg = cfg or SubgraphConfig()
        sg = self.subgraph_builder.build(query=query, cfg=cfg)
        return sg

    def initialize(self, **kwargs) -> None:
        if self._initialized or not self.enabled:
            return

        start_time = time.time()
        try:
            self._entity_detector = EntityDetector(
                device=self.cfg.get(
                    "device", "cuda" if torch.cuda.is_available() else "cpu"
                )
            )
            self._retriever = NERRetrieverEmbedder(
                model_name=self.cfg.get("ner_model", "dslim/bert-base-NER"),
                layer=self.cfg.get("ner_layer", 16),
                device=self.cfg.get(
                    "device", "cuda" if torch.cuda.is_available() else "cpu"
                ),
                embedding_dim=self.cfg.get("ner_dim", 1024),
                index_path=self.cfg.get(
                    "index_path", "data/knowledge_graph/index"
                ),
                logger=self.logger,
                memory=self.memory,
                cfg=self.cfg,
            )
            self._classifier = ScorableClassifier(
                memory=self.memory,
                logger=self.logger,
                config_path=self.cfg.get(
                    "domain_config", "config/domain/seeds.yaml"
                ),
                metric=self.cfg.get("domain_metric", "cosine"),
            )

            self._graph = HNSWIndex(
                dim=self.cfg.get("ner_dim", 1024),
                index_path=self.cfg.get(
                    "index_path", "data/knowledge_graph/index"
                ),
                space=self.cfg.get("index_space", "cosine"),
                persistent=True,
            )

            self._load_graph()
            self._initialized = True
            self._stats["last_update"] = datetime.now(timezone.utc).isoformat()
            self._stats["build_time"] = time.time() - start_time

            asyncio.create_task(self._setup_event_listeners())

            log.info(
                "KnowledgeGraphInitialized in %.2f seconds nodes=%d edges=%d",
                self._stats["build_time"],
                self._stats["total_nodes"],
                self._stats["total_edges"],
            )

        except Exception as e:
            log.error(
                "KnowledgeGraphInitError: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise RuntimeError(f"KnowledgeGraph failed to initialize: {e}")

    # in KnowledgeGraphService
    def detect_entities(self, text: str) -> List[Dict[str, Any]]:
        if not self._entity_detector:
            raise RuntimeError("KG not initialized")
        return self._entity_detector.detect_entities(text or "")

    async def _setup_event_listeners(self):
        """Subscribe to indexing events via the bus."""
        await self.subscribe(
            "knowledge_graph.index_request", self._handle_index_request
        )
        log.info("KnowledgeGraphService listening for index requests")

    # --- main handler --------------------------------------------------------

    async def _handle_index_request(self, payload: Dict[str, Any]):
        """Process an indexing request from the event bus (text-aware)."""
        try:
            scorable_id = payload["scorable_id"]
            scorable_type = payload["scorable_type"]
            entities = payload.get("entities") or []
            domains = payload.get("domains") or []

            # 1) Acquire text (prefer store fetch; fallback to payload["text"])
            text = None
            try:
                # If you have a DynamicScorable store, fetch by scorable_id
                # Example: orm = self.scorable_store.get_by_scorable_id(scorable_id)
                # text = getattr(orm, "text", None)
                if hasattr(self, "scorable_store"):
                    orm = getattr(
                        self.scorable_store, "get_by_scorable_id", None
                    )
                    if callable(orm):
                        row = self.scorable_store.get_by_scorable_id(
                            scorable_id
                        )  # implement if missing
                        text = getattr(row, "text", None)
            except Exception as e:
                log.warning(f"KG: fetch text by scorable_id failed: {e}")

            if not text:
                text = payload.get("text")

            if not text:
                log.error(
                    "KG: no text available for scorable_id=%s", scorable_id
                )
                await self.publish(
                    "knowledge_graph.index_failed",
                    {
                        "scorable_id": scorable_id,
                        "error": "Missing text for indexing",
                    },
                )
                return

            text = _normalize_text(text)
            doc_hash = hash_text(text)

            # 2) Sentence segmentation (cheap)
            sent_spans = sentences(text)

            # 3) Validate/repair entities against text; add surface/context/sentence_ix
            fixed_entities = []
            for ent in entities:
                start = int(ent.get("start", -1))
                end = int(ent.get("end", -1))
                etype = ent.get("type") or "UNKNOWN"

                # Ensure offsets sane; if not, try to recover by searching the declared string
                surface = ent.get("text")
                if start < 0 or end < 0 or end <= start or end > len(text):
                    # Try to locate surface if provided
                    if surface:
                        pos = text.find(surface)
                        if pos >= 0:
                            start, end = pos, pos + len(surface)
                            log.debug(
                                "KG: repaired offsets for %s via surface search (%d-%d)",
                                etype,
                                start,
                                end,
                            )
                    else:
                        # If we can’t recover, skip this entity (or you can run a lightweight NER here)
                        log.warning(
                            "KG: skipping entity with invalid offsets and no surface: %s",
                            ent,
                        )
                        continue

                # If surface missing or mismatched, derive from offsets
                derived = safe_slice(text, start, end)
                if not surface or surface != derived:
                    surface = derived

                sentence_ix = _locate_sentence_ix(sent_spans, start, end)
                # Context: sentence window (or +/- 60 chars if sentence not found)
                if 0 <= sentence_ix < len(sent_spans):
                    sstart, send = sent_spans[sentence_ix]
                    context = text[sstart:send]
                else:
                    context = safe_slice(
                        text, max(0, start - 60), min(len(text), end + 60)
                    )

                fixed = {
                    **ent,
                    "start": start,
                    "end": end,
                    "type": etype,
                    "text": surface,
                    "sentence_ix": sentence_ix,
                    "context": context,
                    "doc_hash": doc_hash,
                }
                fixed_entities.append(fixed)

            # 4) Add nodes with context
            for ent in fixed_entities:
                node_id = (
                    f"{scorable_id}:{ent['type']}:{ent['start']}-{ent['end']}"
                )
                self._add_entity_node(
                    node_id=node_id,
                    entity=ent,
                    domains=domains,
                    scorable_id=scorable_id,
                    scorable_type=scorable_type,
                    # You may store large fields in meta inside your node implementation
                    meta={
                        "surface": ent["text"],
                        "sentence_ix": ent["sentence_ix"],
                        "context": ent["context"],
                        "doc_hash": ent["doc_hash"],
                    },
                )

            # 5) Build relationships (existing) + sentence co-occurrence edges (new)
            relationships = (
                self._build_relationships(fixed_entities, domains, scorable_id)
                or []
            )

            # Sentence co-occurrence (cheap signal)
            by_sent = {}
            for e in fixed_entities:
                by_sent.setdefault(e["sentence_ix"], []).append(e)
            for s_ix, ents in by_sent.items():
                if s_ix < 0 or len(ents) < 2:
                    continue
                n = len(ents)
                for i in range(n):
                    for j in range(i + 1, n):
                        a, b = ents[i], ents[j]
                        relationships.append(
                            {
                                "source_id": f"{scorable_id}:{a['type']}:{a['start']}-{a['end']}",
                                "target_id": f"{scorable_id}:{b['type']}:{b['start']}-{b['end']}",
                                "rel_type": "CO_OCCURS_IN_SENTENCE",
                                "confidence": 1.0,  # match _add_relationship signature
                                # if you want to persist 'meta', extend _add_relationship to accept it or store it elsewhere
                            }
                        )
            # 6) Add relationships
            for rel in relationships:
                self._add_relationship(
                    source_id=rel["source"],
                    target_id=rel["target"],
                    rel_type=rel["type"],
                    confidence=rel["confidence"],
                )

            # 7) Publish completion
            self.publish(
                "knowledge_graph.index_complete",
                {
                    "scorable_id": scorable_id,
                    "node_count": len(fixed_entities),
                    "relationship_count": len(relationships),
                    "doc_hash": doc_hash,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            log.info(
                "KG: indexed scorable_id=%s nodes=%d rels=%d",
                scorable_id,
                len(fixed_entities),
                len(relationships),
            )

        except Exception as e:
            log.error(f"Indexing failed: {str(e)}", exc_info=True)
            self.publish(
                "knowledge_graph.index_failed",
                {"scorable_id": payload.get("scorable_id"), "error": str(e)},
            )

    def health_check(self) -> Dict[str, Any]:
        status = "healthy" if self._initialized else "unhealthy"
        return {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                **self._stats,
            },
            "dependencies": {
                "embedding_index": "ready" if self._graph else "missing",
                "entity_detector": "ready"
                if self._entity_detector
                else "missing",
                "classifier": "ready" if self._classifier else "missing",
            },
        }

    def shutdown(self) -> None:
        """Clean shutdown, clear state."""
        self._initialized = False
        self._entity_detector = None
        self._retriever = None
        self._classifier = None
        self._graph = None
        self._pending_nodes.clear()
        self._pending_relationships.clear()
        log.info("KnowledgeGraphShutdown: stopped")

    def build_tree(
        self,
        *,
        paper_text: str,
        paper_id: str = "",
        chat_corpus: List[Dict[str, Any]] = None,
        trajectories: List[Dict[str, Any]] = None,
        domains: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a lightweight knowledge_tree for verification use.
        Does NOT do any image work.
        """
        chat_corpus = chat_corpus or []
        trajectories = trajectories or []
        domains = domains or []

        # extract
        entities = self._extract_entities_safe(paper_text)
        claims = self._extract_claims_simple(
            paper_text, min_len=self.min_claim_len
        )
        insights = [
            m["text"]
            for m in chat_corpus
            if isinstance(m, dict) and m.get("text")
        ]

        # relationships (claim↔insight) filtered by verification_threshold
        relationships: List[Dict[str, Any]] = []
        strength_accum, count = 0.0, 0
        for c in claims:
            smax = 0.0
            for ins in insights:
                s = self._rel_strength_claim_insight(c, ins, entities, domains)
                if s >= float(self.verification_threshold):
                    relationships.append(
                        {
                            "source": self._node_id("claim", c),
                            "target": self._node_id("insight", ins),
                            "type": "supports",
                            "confidence": float(s),
                        }
                    )
                smax = max(smax, s)
            if smax > 0:
                strength_accum += smax
                count += 1

        temporal = self._temporal_edges(trajectories)
        # keep only strong temporal edges if you want, else include all; here we include all
        relationships.extend(temporal)

        # nodes (truncate to avoid blow-up)
        nodes: List[Dict[str, Any]] = []
        for e in entities[: self.max_nodes]:
            nodes.append(
                {
                    "id": self._node_id("ent", e["text"]),
                    "text": e["text"],
                    "type": "entity",
                }
            )
        for c in claims[: self.max_nodes - len(nodes)]:
            nodes.append(
                {"id": self._node_id("claim", c), "text": c, "type": "claim"}
            )
        for ins in insights[: self.max_nodes - len(nodes)]:
            nodes.append(
                {
                    "id": self._node_id("insight", ins),
                    "text": ins,
                    "type": "insight",
                }
            )

        # metrics
        claim_coverage = self._estimate_claim_coverage(claims, insights)
        evidence_strength = (strength_accum / max(1, count)) if count else 0.0
        temporal_coherence = self._temporal_coherence_metric(temporal)
        domain_alignment = self._domain_alignment_metric(
            domains, claims, entities
        )

        # gaps
        gaps: List[Dict[str, Any]] = []
        for i, c in enumerate(claims):
            addressed = any(self._contains_concept(ins, c) for ins in insights)
            if not addressed:
                gaps.append(
                    {
                        "claim_id": f"gap_{i + 1}",
                        "claim_text": c,
                        "gap_type": "missing_explanation",
                        "severity": self._gap_severity(c),
                    }
                )

        return {
            "nodes": nodes,
            "relationships": relationships,
            "claims": [
                {"id": self._node_id("claim", c), "text": c} for c in claims
            ],
            "entities": [e["text"] for e in entities],
            "claim_coverage": float(self._clamp01(claim_coverage)),
            "evidence_strength": float(self._clamp01(evidence_strength)),
            "temporal_coherence": float(self._clamp01(temporal_coherence)),
            "domain_alignment": float(self._clamp01(domain_alignment)),
            "knowledge_gaps": gaps,
            "meta": {
                "paper_id": paper_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "verification_threshold": float(self.verification_threshold),
            },
        }

    def search_entities(
        self, text: str, k: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search KG using the SAME embedder used during indexing.
        Results cached via entity_cache keyed by scorable_embeddings.id (int).
        Returns: list of (node_id, score, metadata)
        """
        if not self._initialized or not self.enabled:
            return []

        start_time = time.time()

        try:
            # 1) Ensure we have a canonical embedding and its integer id
            #    (use the *same* embedder/model you index with)
            embedding_vec = self._retriever.embed_type_query(text)
            emb_id = self.memory.embedding.get_id_for_text(
                text=text,
            )

            # 2) Try cache → embedding_ref is an INT (FK to scorable_embeddings.id)
            if emb_id and hasattr(self.memory, "entity_cache"):
                cached_row = self.memory.entity_cache.get_by_embedding(emb_id)
                if cached_row and cached_row.results_json:
                    self._stats["cache_hits"] += 1
                    self._stats["queries"] += 1
                    self._stats["query_time"] += time.time() - start_time
                    return cached_row.results_json

            # 3) Miss → do the actual HNSW search
            # 3) Miss → do the actual HNSW search
            # Prefer tuple adapter if available
            raw_results = self._graph.search_tuples(embedding_vec, k=k)

            # 4) Normalize to JSON-safe tuples (node_id, score, metadata)
            enriched = []
            for node_id, score, meta in raw_results:
                enriched.append(
                    (
                        str(node_id),
                        float(score),
                        {
                            **(meta or {}),
                            "node_id": str(node_id),
                            "score": float(score),
                        },
                    )
                )

            # 5) Upsert cache
            if emb_id and hasattr(self.memory, "entity_cache"):
                self.memory.entity_cache.upsert(emb_id, enriched)
                self._stats["cache_misses"] += 1

            self._stats["queries"] += 1
            self._stats["query_time"] += time.time() - start_time
            return enriched

        except Exception as e:
            log.error(
                "KnowledgeGraphSearchError",
                {
                    "text": text[:200],
                    "error": str(e),
                },
            )
            return []

    def build_context_for_plan(self, plan: dict, k: int = 5) -> dict:
        """Return KG neighbors per claim_id for drafting/scoring."""
        out = {"neighbors": {}}
        for u in plan.get("units", []):
            q = (u.get("claim") or u.get("evidence") or "").strip()
            cid = u.get("claim_id")
            if not q or not cid:
                continue
            hits = self.search_entities(
                q, k=k
            )  # [(node_id, score, meta), ...]
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

    def export_for_vpm(self, knowledge_tree: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a minimal VPM payload (no images). ZeroModel will own all image work.
        metrics per node are broadcast from global tree metrics; you can specialize later.
        """
        nodes = knowledge_tree.get("nodes", [])
        entities = [n.get("text", "") for n in nodes]
        metrics = ["coverage", "evidence", "temporal", "domain"]

        cov = float(knowledge_tree.get("claim_coverage", 0.0))
        evd = float(knowledge_tree.get("evidence_strength", 0.0))
        tmp = float(knowledge_tree.get("temporal_coherence", 0.0))
        dom = float(knowledge_tree.get("domain_alignment", 0.0))

        matrix = [[cov, evd, tmp, dom] for _ in nodes]

        rels = []
        for r in knowledge_tree.get("relationships", []):
            src = r.get("source")
            tgt = r.get("target")
            src_i = next(
                (i for i, n in enumerate(nodes) if n.get("id") == src), None
            )
            tgt_i = next(
                (i for i, n in enumerate(nodes) if n.get("id") == tgt), None
            )
            if src_i is not None and tgt_i is not None:
                rels.append(
                    {
                        "source_idx": src_i,
                        "target_idx": tgt_i,
                        "type": r.get("type", "rel"),
                        "confidence": float(r.get("confidence", 0.0)),
                    }
                )

        return {
            "entities": entities,
            "metrics": metrics,
            "matrix": matrix,
            "relationships": rels,
        }

    def _estimate_claim_coverage(
        self, claims: List[str], insights: List[str]
    ) -> float:
        if not claims:
            return 0.0
        covered = 0
        for c in claims:
            if any(self._contains_concept(i, c) for i in insights):
                covered += 1
        return covered / max(1, len(claims))

    def _temporal_coherence_metric(
        self, temporal_rels: List[Dict[str, Any]]
    ) -> float:
        if not temporal_rels:
            return 0.0
        ok, tot = 0, 0
        for r in temporal_rels:
            if "time_delta" in r:
                tot += 1
                if (r.get("time_delta") or 0) >= 0:
                    ok += 1
        return (ok / tot) if tot else 0.8  # neutral default

    def _domain_alignment_metric(
        self,
        domains: List[Dict[str, Any]],
        claims: List[str],
        ents: List[Dict[str, Any]],
    ) -> float:
        if not domains:
            return 0.7
        names = [
            str(d.get("domain", "")).lower()
            for d in domains
            if isinstance(d, dict)
        ]
        text = " ".join(claims + [e.get("text", "") for e in ents])
        score = sum(1 for d in names if d and d in text.lower())
        return self._clamp01(score / max(1, len(names)))

    def _gap_severity(self, claim: str) -> float:
        sev = 0.50
        if re.search(
            r"\b(critical|essential|must|key|fundamental|core)\b", claim, re.I
        ):
            sev += 0.30
        if re.search(
            r"\b\d+(\.\d+)?\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b",
            claim,
            re.I,
        ):
            sev += 0.20
        return self._clamp01(sev)

    def _temporal_edges(
        self, trajectories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        rels: List[Dict[str, Any]] = []
        for traj in trajectories or []:
            msgs = traj.get("messages") or []
            for i in range(1, len(msgs)):
                a, b = msgs[i - 1], msgs[i]
                a_id = self._node_id("msg", a.get("text", ""))
                b_id = self._node_id("msg", b.get("text", ""))
                conf = 0.90
                rtype = "temporal_sequence"
                if self._contains_causal_language(b.get("text", "")):
                    rtype = "causal_sequence"
                    conf = 0.95
                rels.append(
                    {
                        "source": a_id,
                        "target": b_id,
                        "type": rtype,
                        "confidence": conf,
                        "timestamp": b.get("timestamp"),
                        "time_delta": (b.get("timestamp", 0) or 0)
                        - (a.get("timestamp", 0) or 0),
                    }
                )
        return rels

    def _rel_strength_claim_insight(
        self,
        claim: str,
        insight: str,
        entities: List[Dict[str, Any]],
        domains: List[Dict[str, Any]],
    ) -> float:
        j = self._jaccard(self._token_set(claim), self._token_set(insight))

        ent_texts = [e.get("text", "") for e in entities]
        ce = {t.lower() for t in ent_texts if t and t.lower() in claim.lower()}
        ie = {
            t.lower() for t in ent_texts if t and t.lower() in insight.lower()
        }
        overlap = (len(ce & ie) / max(1, len(ce | ie))) if (ce or ie) else 0.0
        entity_bonus = 0.30 * overlap

        domain_boost = 0.0
        if self.domain_aware:
            names = {
                str(d.get("domain", "")).lower()
                for d in (domains or [])
                if isinstance(d, dict)
            }
            if any(x in names for x in {"ml", "machine learning", "nlp"}):
                domain_boost = 0.15

        return self._clamp01(j + entity_bonus + domain_boost)

    # -------------------------
    # Entity & Node Management
    # -------------------------
    def upsert_node(self, node_id: str, properties: Dict[str, Any]) -> None:
        """
        Public, generic node upsert used by higher-level components
        (InformationGraphBuilder, future wiki/encyclopedia builders, etc.).

        Semantics (for now):
        - If a node with this node_id already exists in the backing index
            (as per metadata), do nothing.
        - Otherwise, embed a representative text field and add a new node.

        `properties` is a loose bag of metadata; we store it under
        `properties` in the index metadata so you can evolve this over time.
        """
        try:
            if self._graph.has_metadata(node_id):
                # Even if the HNSW node exists, we still want Nexus to be up-to-date,
                # so don't early-return until after we mirror below if needed.
                pass

            # Choose a representative text field for embedding
            text = (
                properties.get("summary")
                or properties.get("title")
                or properties.get("text")
                or node_id
            )

            base_meta: Dict[str, Any] = {
                "node_id": node_id,
                "text": text,
                "type": properties.get("type", "Node"),
                "created_at": properties.get(
                    "created_at",
                    datetime.now(timezone.utc).isoformat(),
                ),
                # Keep the full property bag for later graph / viz tooling
                "properties": dict(properties),
            }

            # Optional normalized fields if present
            if "domains" in properties:
                base_meta["domains"] = properties["domains"]
            if "domain_scores" in properties:
                base_meta["domain_scores"] = properties["domain_scores"]

            # Same retriever → same embedding space as entity nodes
            embedding = self.memory.embedding.get_or_create(base_meta["text"])
            self._graph.add(embedding, base_meta)

            log.info(
                "KnowledgeGraphNodeUpserted node_id %s type %s text %s",
                node_id,
                base_meta["type"],
                base_meta["text"][:80],
            )

            self._stats["total_nodes"] += 1
            self._track_node_stats(base_meta["type"])

            # --- NEW: mirror this node into Nexus via the store ---
            try:
                # Node identity / type
                node_type = base_meta["type"]
                name = text

                # Domains & entities if present
                domains = base_meta.get("domains") or []
                entities = base_meta.get("entities") or []

                # Meta: keep the full property bag
                payload = {
                    "domains": domains,
                    "entities": entities,
                    "meta": {
                        "properties": properties,
                    },
                }

                # Allow callers to tell us which scorables this node came from
                source_ids: List[str] = []
                sid = properties.get("scorable_id")
                if sid:
                    source_ids.append(str(sid))
                for sid in properties.get("source_ids", []):
                    s_str = str(sid)
                    if s_str not in source_ids:
                        source_ids.append(s_str)

                self.nexus_store.create_or_update_node(
                    node_id=node_id,
                    name=name,
                    node_type=node_type,
                    payload=payload,
                    source_ids=source_ids,
                )
            except Exception:
                log.exception("KnowledgeGraphNodeUpsertError (nexus mirror) node id %s", node_id)

        except Exception as e:
            log.warning(
                "KnowledgeGraphNodeUpsertError node id %s error: %s",
                node_id,
                str(e),
            )

    def upsert_concept(
        self,
        concept_id: str,
        name: str,
        summary: str,
        domains: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Upsert a canonical concept both in the encyclopedia DB and
        as a node in the knowledge graph.
        """
        # 1) Ensure encyclopedia row exists / is refreshed
        try:
            wiki_url = extra.get("wiki_url") if extra else None
            sections = extra.get("sections") if extra else None

            self.memory.encyclopedia.ensure_concept(
                concept_id=concept_id,
                name=name,
                summary=summary or "",
                wiki_url=wiki_url,
                domains=domains,
                sections=sections,
            )
        except Exception as e:
            log.warning(
                "KG upsert_concept: failed to persist %s to encyclopedia: %s",
                concept_id,
                e,
            )

        # 2) Upsert the KG node as before
        properties = {
            "type": "concept",
            "name": name,
            "summary": summary,
            "domains": domains or [],
            "source": "encyclopedia",
        }
        if extra:
            properties.update(extra)

        node_id = f"concept:{concept_id}"
        self.upsert_node(node_id=node_id, properties=properties)

    def upsert_paper(
        self,
        paper_id: str,
        title: str,
        abstract: str,
        year: Optional[int] = None,
        domains: Optional[List[str]] = None,
    ) -> None:
        """
        Upsert a paper node, so we can connect ideas and concepts back to sources.
        """
        properties = {
            "type": "paper",
            "title": title,
            "summary": abstract,
            "year": year,
            "domains": domains or [],
            "source": "paper",
        }
        node_id = f"paper:{paper_id}"
        self.upsert_node(node_id=node_id, properties=properties)

    def upsert_gap(
        self,
        gap_id: str,
        claim_text: str,
        severity: float,
        paper_id: Optional[str] = None,
        concept_ids: Optional[List[str]] = None,
    ) -> str:
        """
        Store a knowledge gap as a first-class node in the graph.
        Returns the node_id.
        """
        properties = {
            "type": "gap",
            "text": claim_text,
            "severity": float(severity),
            "source": "knowledge_tree",
        }
        node_id = f"gap:{gap_id}"
        self.upsert_node(node_id=node_id, properties=properties)

        if paper_id:
            self.upsert_edge(
                source_id=node_id,
                target_id=f"paper:{paper_id}",
                rel_type="GAP_IN_PAPER",
                properties={"confidence": 0.9},
            )

        for cid in concept_ids or []:
            self.upsert_edge(
                source_id=node_id,
                target_id=f"concept:{cid}",
                rel_type="GAP_IN_CONCEPT",
                properties={"confidence": 0.9},
            )

        return node_id

    def link_entity_to_concept(
        self,
        entity_node_id: str,
        concept_id: str,
        confidence: float = 0.9,
    ) -> None:
        """
        Link a low-level entity node to a canonical concept.
        """
        concept_node_id = f"concept:{concept_id}"
        self.upsert_edge(
            source_id=entity_node_id,
            target_id=concept_node_id,
            rel_type="MENTIONS_CONCEPT",
            properties={"confidence": confidence},
        )

    def link_paper_to_concept(
        self,
        paper_id: str,
        concept_id: str,
        rel_type: str = "DISCUSSES",
        confidence: float = 0.9,
    ) -> None:
        self.upsert_edge(
            source_id=f"paper:{paper_id}",
            target_id=f"concept:{concept_id}",
            rel_type=rel_type,
            properties={"confidence": confidence},
        )

    def _extract_entities_safe(self, text: str) -> List[Dict[str, Any]]:
        try:
            if not text:
                return []
                # and fix _extract_entities_safe
            ents = self._entity_detector.detect_entities(text)

            if ents:
                return [
                    e for e in ents if isinstance(e, dict) and e.get("text")
                ]
        except Exception as e:
            log.warning("KGEntityExtractionError error: %s", str(e))

        # heuristic fallback (capitalized tokens)
        tokens = re.findall(r"\b[A-Z][A-Za-z0-9\-\_]{2,}\b", text or "")
        uniq = {}
        for t in tokens:
            key = self._normalize_entity(t)
            if key not in uniq:
                uniq[key] = {
                    "text": t,
                    "type": "TERM",
                    "start": -1,
                    "end": -1,
                    "source_text": t,
                }
        return list(uniq.values())

    def _extract_claims_simple(
        self, text: str, min_len: int = None
    ) -> List[str]:
        min_len = min_len or self.min_claim_len
        cand = re.split(r"(?<=[\.\!\?])\s+", text or "")
        out: List[str] = []
        for s in cand:
            s = (s or "").strip()
            if len(s) < min_len:
                continue
            if re.search(
                r"\b(we|our|this paper|results|achieve|improve|demonstrate|propose)\b",
                s,
                re.I,
            ):
                out.append(s)
            elif re.search(
                r"\b\d+(\.\d+)?\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b",
                s,
                re.I,
            ):
                out.append(s)
        return out[:256]

    def find_innovation_candidates(
        self,
        novelty_min: float = 0.4,
        novelty_max: float = 0.8,
        limit_pairs: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Return candidate concept pairs that live in the 'innovation frontier' band.
        """
        concept_nodes: List[Dict[str, Any]] = self._get_all_concept_nodes()
        n = len(concept_nodes)
        candidates: List[Dict[str, Any]] = []

        for i in range(n):
            for j in range(i + 1, n):
                a = concept_nodes[i]
                b = concept_nodes[j]
                novelty = self._estimate_novelty_between_concepts(a, b)
                if novelty_min <= novelty <= novelty_max:
                    candidates.append(
                        {
                            "concept_a": a["node_id"],
                            "concept_b": b["node_id"],
                            "novelty": novelty,
                        }
                    )
                if len(candidates) >= limit_pairs:
                    break
            if len(candidates) >= limit_pairs:
                break

        # sort by descending novelty (or middle-peaked if you prefer)
        candidates.sort(key=lambda x: -x["novelty"])
        return candidates

    def _estimate_novelty_between_concepts(
        self,
        concept_a_meta: Dict[str, Any],
        concept_b_meta: Dict[str, Any],
        max_cooccurrence: int = 20,
    ) -> float:
        """
        Heuristic novelty score in [0,1].
        Higher = more novel (but maybe less plausible).
        Components:
        - semantic distance (1 - similarity)
        - graph co-occurrence penalty (if heavily connected, novelty is low)
        """
        # 1) semantic similarity via embeddings if available
        sim = 0.0
        try:
            text_a = (
                concept_a_meta.get("text")
                or concept_a_meta.get("summary")
                or ""
            )
            text_b = (
                concept_b_meta.get("text")
                or concept_b_meta.get("summary")
                or ""
            )
            emb_a = self.memory.embedding.get_or_create(text_a)
            emb_b = self.memory.embedding.get_or_create(text_b)
            # cosine similarity
            sim = float(
                np.dot(emb_a, emb_b)
                / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
            )
        except Exception:
            sim = 0.0

        semantic_distance = 1.0 - self._clamp01(sim)

        # 2) graph co-occurrence: how many edges connect them (directly or via shared papers)
        co_count = self._cooccurrence_count(
            concept_a_meta["node_id"], concept_b_meta["node_id"]
        )
        co_penalty = min(co_count / max_cooccurrence, 1.0)

        # If they co-occur a lot, novelty is low, regardless of embedding distance
        novelty = self._clamp01(
            0.7 * semantic_distance + 0.3 * (1.0 - co_penalty)
        )
        return novelty

    def _cooccurrence_count(self, node_a: str, node_b: str) -> int:
        """
        Approximate co-occurrence by counting how many papers connect to both concepts.
        """
        papers_with_a = set()
        papers_with_b = set()

        if not os.path.exists(self._rel_path):
            return 0

        with open(self._rel_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rel = json.loads(line.strip())
                except Exception:
                    continue
                s, t = rel.get("source"), rel.get("target")
                rtype = rel.get("type")

                # concept <-DISCUSSES- paper
                if rtype == "DISCUSSES":
                    if t == node_a:
                        papers_with_a.add(s)
                    if t == node_b:
                        papers_with_b.add(s)

        return len(papers_with_a & papers_with_b)

    def _create_entity_node_id(
        self, entity: Dict[str, Any], scorable_id: str
    ) -> str:
        return (
            f"{scorable_id}:{entity['type']}:{entity['start']}-{entity['end']}"
        )


    def _add_entity_node(
        self,
        node_id: str,
        entity: Dict[str, Any],
        domains: List[Dict[str, Any]],
        scorable_id: str,
        scorable_type: str,
        meta: Dict[str, Any],
    ):
        # 1) existing behavior (HNSW index, relationships.jsonl, etc.)
        self._legacy_add_entity_node(node_id, entity, domains, scorable_id, scorable_type, meta)

        # 2) NEW: mirror into Nexus
        try:
            name = entity["text"]
            node_type = entity.get("type", "ENTITY")

            payload = {
                "scorable_id": scorable_id,
                "scorable_type": scorable_type,
                "entity_type": node_type,
                "meta": meta,
                "domains": domains,
            }

            self.nexus_store.create_or_update_node(
                node_id=node_id,
                name=name,
                node_type=node_type,
                payload=payload,
                source_ids=[scorable_id],
            )
        except Exception:
            log.exception("Failed to mirror entity node into Nexus node_id %s", node_id)


    def _legacy_add_entity_node(
        self,
        node_id: str,
        entity: Dict[str, Any],
        domains: List[Dict[str, float]],
        scorable_id: str,
        scorable_type: str,
        meta: Optional[Dict[str, Any]] = None,
    ):
        try:
            # Avoid duplicates (only if the index supports it)
            try:
                if hasattr(
                    self._graph, "has_metadata"
                ) and self._graph.has_metadata(node_id):
                    return
            except Exception:
                # If the backend doesn’t support duplicate checking, continue and let the index de-dupe or accept dupes
                pass

            # Build base metadata with guards (avoid KeyError on missing fields)
            base_meta = {
                "node_id": node_id,
                "text": entity.get("text", ""),
                "type": entity.get("type", "UNKNOWN"),
                "domains": [
                    d.get("domain")
                    for d in (domains or [])
                    if isinstance(d, dict) and d.get("domain")
                ],
                "domain_scores": {
                    d.get("domain"): d.get("score")
                    for d in (domains or [])
                    if isinstance(d, dict) and d.get("domain") is not None
                },
                "sources": [
                    {
                        "scorable_id": scorable_id,
                        "scorable_type": scorable_type,
                        "start": int(entity.get("start", -1)),
                        "end": int(entity.get("end", -1)),
                        "source_text": entity.get(
                            "source_text", entity.get("text", "")
                        ),
                    }
                ],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            # Merge caller-provided meta last so it can add context/sentence_ix/doc_hash
            if meta:
                base_meta.update(meta)

            # Use same retriever → consistent embedding space
            embedding = self._retriever.embed_type_query(base_meta["text"])
            self._graph.add(embedding, base_meta)

            log.info(
                "KnowledgeGraphNodeAdded node_id %s text %s type %s scorable_id %s scorable_type %s",
                node_id,
                base_meta["text"][:50],
                base_meta["type"],
                scorable_id,
                scorable_type,
            )
            self._stats["total_nodes"] += 1
            self._track_node_stats(base_meta["type"])

        except Exception as e:
            log.error(
                "KnowledgeGraphNodeAddError node_id %s error %s",
                node_id,
                str(e),
            )

    # -------------------------
    # Public Edge Upsert (for builders)
    # -------------------------
    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Public, generic edge upsert used by higher-level components
        (InformationGraphBuilder, future encyclopedia/wiki builders, etc.).

        Semantics (for now):
        - Append an edge record to the relationships JSONL, same as
            _add_relationship, but allow arbitrary metadata via `properties`.
        - If `properties` includes `confidence`, it is used; otherwise
            we default to 0.9.

        This does NOT currently deduplicate edges; if you call it multiple
        times with the same (source, target, type), you'll get multiple
        rows. That’s acceptable for now and can be tightened later.
        """
        properties = properties or {}

        # derive confidence, default to "fairly strong"
        confidence = float(properties.get("confidence", 0.9))

        rel = {
            "source": source_id,
            "target": target_id,
            "type": rel_type,
            "confidence": confidence,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        # Merge any extra properties (topic, source_type, etc.)
        for k, v in properties.items():
            if k not in rel:
                rel[k] = v

        try:
            # --- Legacy JSONL persistence (keeps old tooling working) ---
            with open(self._rel_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rel) + "\n")

            log.info("KnowledgeGraphEdgeUpserted: %s", rel)

            self._stats["total_edges"] += 1
            self._track_edge_stats(rel_type)

            # --- NEW: Mirror into Nexus edges via the store ---
            try:
                if self.nexus_store:
                    # All non-confidence properties become 'channels' on the edge
                    channels = dict(properties)
                    channels.pop("confidence", None)

                    self.nexus_store.write_edges(
                        run_id=self.nexus_run_id,
                        edges=[
                            {
                                "src": source_id,
                                "dst": target_id,
                                "type": rel_type,
                                "weight": confidence,
                                "channels": channels,
                            }
                        ],
                    )
            except Exception:
                log.exception(
                    "KnowledgeGraphEdgeUpsertError (nexus mirror) source_id %s target_id %s",
                    source_id,
                    target_id,
                )

        except Exception as e:
            log.error(
                "KnowledgeGraphEdgeUpsertError source_id %s target_id %s error %s",
                source_id,
                target_id,
                str(e),
            )

    async def query_frontier_concepts(
        self,
        novelty_range: Tuple[float, float] = (0.4, 0.8),
        quiz_accuracy_range: Tuple[float, float] = (0.3, 0.7),
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Return concepts in the 'Goldilocks' band for ideation.

        Primary source:
        - AIEncyclopediaStore.get_frontier_concepts() (quiz-based frontier band)

        Cold start:
        - If there are no frontier concepts yet, fall back to
            a simple list of concepts from the encyclopedia.
        """

        # 1) Preferred path: use PretrainZero-style frontier band
        frontier_concepts = self.memory.encyclopedia.get_frontier_concepts(
            min_accuracy=quiz_accuracy_range[0],
            max_accuracy=quiz_accuracy_range[1],
            novelty_min=novelty_range[0],
            novelty_max=novelty_range[1],
            limit=limit * 2,
        )

        # 2) Cold start fallback: just grab some concepts so callers don't get stuck
        if not frontier_concepts:
            frontier_concepts = self.memory.encyclopedia.list_concepts(
                limit=limit
            )

        results: List[Dict[str, Any]] = []
        for c in frontier_concepts:
            # We don't currently store novelty per concept; you can later
            # replace this with a graph-based estimate.
            novelty = 0.5

            results.append(
                {
                    "id": c.concept_id,  # use stable concept slug
                    "name": c.name,
                    "summary": c.summary or "",
                    "domains": c.domains or [],
                    "quiz_stats": {
                        "accuracy": c.quiz_accuracy
                        if c.quiz_accuracy is not None
                        else 0.5,
                        "novelty": novelty,
                    },
                }
            )
            if len(results) >= limit:
                break

        return results

    async def get_concept(self, concept_id: str) -> Dict[str, Any]:
        """
        Async get_concept - use this everywhere (no sync fallback needed).
        Falls back to minimal dict if concept missing.
        """
        try:
            concept = await self.memory.encyclopedia.get_concept(concept_id)
            return {
                "id": concept.id,
                "name": concept.name,
                "summary": concept.summary or "",
                "domains": concept.domains or [],
                "quiz_stats": {
                    "accuracy": concept.quiz_accuracy or 0.5,
                    "novelty": concept.novelty_score or 0.5,
                },
            }
        except Exception as e:
            log.warning(
                "KG get_concept failed for %s: %s", concept_id, str(e)
            )
            return {"id": concept_id, "name": concept_id, "summary": ""}

    # -------------------------
    # Relationships
    # -------------------------
    def _add_relationship(
        self, source_id: str, target_id: str, rel_type: str, confidence: float
    ):
        """Persist relationship to JSONL + log event."""
        rel = {
            "source": source_id,
            "target": target_id,
            "type": rel_type,
            "confidence": confidence,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self._rel_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rel) + "\n")

            log.info("KnowledgeGraphRelationshipAdded: %s", rel)
            self._stats["total_edges"] += 1
            self._track_edge_stats(rel_type)

        except Exception as e:
            log.error(
                "KnowledgeGraphRelationshipAddError source_id %s target_id %s error %s",
                source_id,
                target_id,
                str(e),
            )

    # -------------------------
    # Relationship Builder
    # -------------------------
    def _build_relationships(
        self,
        entities: List[Dict[str, Any]],
        domains: List[Dict[str, float]],
        scorable_id: str,
    ) -> List[Dict[str, Any]]:
        """Build relationships using proximity + heuristics."""
        relationships = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                distance = abs(e1["end"] - e2["start"])
                if distance < 100:
                    rel_type = self._infer_relationship_type(e1, e2)
                    confidence = self._calculate_relationship_confidence(
                        e1, e2, distance, domains
                    )
                    if confidence >= self.relationship_threshold:
                        relationships.append(
                            {
                                "source": self._create_entity_node_id(
                                    e1, scorable_id
                                ),
                                "target": self._create_entity_node_id(
                                    e2, scorable_id
                                ),
                                "type": rel_type,
                                "confidence": confidence,
                                "evidence": "positional",
                            }
                        )
        return relationships

    # -------------------------
    # Path Traversal
    # -------------------------
    def _find_relationship_path(
        self, start_node: str, query_entity: Dict[str, Any], max_hops: int
    ) -> List[Dict[str, Any]]:
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
        self._stats["total_nodes"] = (
            self._graph.ntotal if hasattr(self._graph, "ntotal") else 0
        )

        # Rebuild edge stats
        if os.path.exists(self._rel_path):
            with open(self._rel_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rel = json.loads(line.strip())
                        self._stats["total_edges"] += 1
                        self._track_edge_stats(rel["type"])
                    except Exception as e:
                        log.error(
                            "KnowledgeGraphLoadError: %s", str(e)
                        )

    def _track_edge_stats(self, rel_type: str):
        self._stats["edge_types"][rel_type] = (
            self._stats["edge_types"].get(rel_type, 0) + 1
        )

    def _track_node_stats(self, node_type: str):
        self._stats["node_types"][node_type] = (
            self._stats["node_types"].get(node_type, 0) + 1
        )

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
                except Exception:
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
            ("PAPER", "METHOD"): "introduces",
        }
        return type_pairs.get((first["type"], second["type"]), "related_to")

    def _calculate_relationship_confidence(
        self, e1: Dict, e2: Dict, distance: int, domains: List[Dict]
    ) -> float:
        base_score = np.exp(-distance / 50.0)  # exponential decay
        domain_bonus = (
            0.1 if any(d["domain"] in {"ml", "nlp"} for d in domains) else 0.0
        )
        proximity_bonus = 0.1 if distance < 20 else 0.0
        return max(min(base_score + domain_bonus + proximity_bonus, 1.0), 0.0)


    # ------------------------------------------------------------------ Debug / local tree

    def log_local_tree(
        self,
        scorable_id: str,
        *,
        max_entities: int = 12,
        max_edges_per_entity: int = 8,
    ) -> None:
        """
        Log a compact ASCII-style tree around one scorable's nodes + edges in Nexus.

        Root:    the scorable_id (virtual root)
        Level 1: entity nodes (mirrored via create_or_update_node)
        Level 2: edges from each node to its neighbors in this Nexus run

        This is purely for debugging so you can *see* the graph growing
        as you index papers / sections.
        """
        try:
            nodes = self.nexus_store.list_nodes_for_scorable(
                scorable_id, limit=max_entities
            )
        except Exception as e:
            log.warning(
                "KG log_local_tree: failed to list nodes for scorable_id=%s: %s",
                scorable_id,
                e,
            )
            return

        if not nodes:
            log.info(
                "KG log_local_tree: no Nexus nodes found for scorable_id=%s",
                scorable_id,
            )
            return

        run_id = getattr(self, "nexus_run_id", "unknown")
        lines: list[str] = []
        lines.append(
            f"KG Local Tree — scorable_id={scorable_id} run_id={run_id} "
            f"(nodes={len(nodes)})"
        )

        def _fmt_node(node) -> str:
            meta = getattr(node, "meta", None) or {}
            node_type = meta.get("node_type") or node.target_type or "UNKNOWN"
            name = (
                meta.get("node_name")
                or node.text
                or meta.get("text")
                or node.id
                or ""
            )

            if name and len(name) > 80:
                name = name[:77] + "..."

            return f"{node.id} [{node_type}] {name}"

        for idx, node in enumerate(nodes):
            is_last_entity = idx == len(nodes) - 1
            entity_prefix = "└" if is_last_entity else "├"
            lines.append(f"{entity_prefix}─ {_fmt_node(node)}")

            # Fetch edges touching this node in this Nexus run
            try:
                edges = self.nexus_store.list_edges_for_node(
                    run_id=run_id,
                    node_id=node.id,
                    limit=max_edges_per_entity,
                )
            except Exception as e:
                log.warning(
                    "KG log_local_tree: failed to list edges for node_id=%s: %s",
                    node.id,
                    e,
                )
                continue

            if not edges:
                continue

            # Child prefix depends on whether this entity itself is last
            child_prefix = "   " if is_last_entity else "│  "

            for j, edge in enumerate(edges):
                is_last_edge = j == len(edges) - 1
                edge_branch = "└" if is_last_edge else "├"

                other_id = edge.dst if edge.src == node.id else edge.src
                neighbor = self.nexus_store.get_node(other_id)
                if neighbor:
                    neighbor_label = _fmt_node(neighbor)
                else:
                    neighbor_label = other_id

                edge_label = edge.type or "rel"
                if edge.weight is not None:
                    try:
                        edge_label += f" (w={float(edge.weight):.2f})"
                    except Exception:
                        pass

                lines.append(
                    f"{child_prefix}{edge_branch}─ {edge_label} → {neighbor_label}"
                )

        tree_str = "\n".join(lines)
        log.info("KG_LOCAL_TREE\n%s", tree_str)

    # ---------- tiny helpers ----------

    @staticmethod
    def _token_set(s: str) -> set:
        return set(re.findall(r"[a-z0-9]+", (s or "").lower()))

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        return len(a & b) / max(1, len(a | b))

    @staticmethod
    def _clamp01(x: float) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.0

    @staticmethod
    def _normalize_entity(text: str) -> str:
        return re.sub(r"[^\w\s]", "", (text or "").lower()).strip()

    @staticmethod
    def _contains_causal_language(text: str) -> bool:
        t = (text or "").lower()
        pats = [
            r"therefore",
            r"as a result",
            r"because",
            r"consequently",
            r"thus",
            r"hence",
            r"led to",
            r"so we",
        ]
        return any(re.search(p, t) for p in pats)

    @staticmethod
    def _contains_concept(a: str, b: str) -> bool:
        # conservative concept overlap
        A = KnowledgeGraphService._token_set(a)
        B = KnowledgeGraphService._token_set(b)
        return KnowledgeGraphService._jaccard(A, B) >= 0.18

    @staticmethod
    def _node_id(kind: str, text: str) -> str:
        h = hash_text(kind + "|" + (text or ""))[:16]
        return f"{kind}:{h}"

