# stephanie/services/knowledge_graph_service.py
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from stephanie.analysis.scorable_classifier import ScorableClassifier
from stephanie.models.hnsw_index import HNSWIndex
from stephanie.models.ner_retriever import EntityDetector, NERRetrieverEmbedder
from stephanie.services.service_protocol import Service


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

        # also used by the verification/knowledge-tree path (separate from indexing threshold)
        self.verification_threshold = self.cfg.get("knowledge_graph", {}).get("verification_threshold", 0.90)

        kg_cfg = (cfg.get("knowledge_graph") or {})

        # hard limits / guards used by build_tree & helpers
        self.max_nodes = int(kg_cfg.get("max_nodes", 200))
        self.max_relationships = int(kg_cfg.get("max_relationships", 1000))
        self.max_rels_per_node = int(kg_cfg.get("max_rels_per_node", 16))
        self.max_entities_per_section = int(kg_cfg.get("max_entities_per_section", 128))

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
        self.verification_threshold = float(kg_cfg.get("verification_threshold", 0.90))


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

            asyncio.create_task(self._setup_event_listeners())


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

    async def _setup_event_listeners(self):
        """Subscribe to indexing events via the bus."""
        await self.subscribe(
            "knowledge_graph.index_request",
            self._handle_index_request
        )
        self.logger.info("KnowledgeGraphService listening for index requests")


    async def _handle_index_request(self, payload: Dict[str, Any]):
        """Process an indexing request from the event bus."""
        try:
            scorable_id = payload["scorable_id"]
            scorable_type = payload["scorable_type"]
            text = payload["text"]
            entities = payload["entities"]
            domains = payload["domains"]
            
            # Index entities
            for entity in entities:
                self._add_entity_node(
                    node_id=f"{scorable_id}:{entity['type']}:{entity['start']}-{entity['end']}",
                    entity=entity,
                    domains=domains,
                    scorable_id=scorable_id,
                    scorable_type=scorable_type
                )
                
            # Build relationships
            relationships = self._build_relationships(entities, domains, scorable_id)
            for rel in relationships:
                self._add_relationship(**rel)
                
            # Publish completion event
            self.publish("knowledge_graph.index_complete", {
                "scorable_id": scorable_id,
                "node_count": len(entities),
                "relationship_count": len(relationships),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Indexing failed: {str(e)}", exc_info=True)
            # Publish failure event
            self.publish("knowledge_graph.index_failed", {
                "scorable_id": payload.get("scorable_id"),
                "error": str(e)
            })

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
                "entity_detector": "ready" if self._entity_detector else "missing",
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
        self.logger.log("KnowledgeGraphShutdown", {"status": "stopped"})


    def build_tree(self, *, paper_text: str, paper_id: str = "", chat_corpus: List[Dict[str, Any]] = None,
                   trajectories: List[Dict[str, Any]] = None, domains: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build a lightweight knowledge_tree for verification use.
        Does NOT do any image work.
        """
        chat_corpus = chat_corpus or []
        trajectories = trajectories or []
        domains = domains or []

        # extract
        entities = self._extract_entities_safe(paper_text)
        claims = self._extract_claims_simple(paper_text, min_len=self.min_claim_len)
        insights = [m["text"] for m in chat_corpus if isinstance(m, dict) and m.get("text")]

        # relationships (claim↔insight) filtered by verification_threshold
        relationships: List[Dict[str, Any]] = []
        strength_accum, count = 0.0, 0
        for c in claims:
            smax = 0.0
            for ins in insights:
                s = self._rel_strength_claim_insight(c, ins, entities, domains)
                if s >= float(self.verification_threshold):
                    relationships.append({
                        "source": self._node_id("claim", c),
                        "target": self._node_id("insight", ins),
                        "type": "supports",
                        "confidence": float(s),
                    })
                smax = max(smax, s)
            if smax > 0:
                strength_accum += smax; count += 1

        temporal = self._temporal_edges(trajectories)
        # keep only strong temporal edges if you want, else include all; here we include all
        relationships.extend(temporal)

        # nodes (truncate to avoid blow-up)
        nodes: List[Dict[str, Any]] = []
        for e in entities[: self.max_nodes]:
            nodes.append({"id": self._node_id("ent", e["text"]), "text": e["text"], "type": "entity"})
        for c in claims[: self.max_nodes - len(nodes)]:
            nodes.append({"id": self._node_id("claim", c), "text": c, "type": "claim"})
        for ins in insights[: self.max_nodes - len(nodes)]:
            nodes.append({"id": self._node_id("insight", ins), "text": ins, "type": "insight"})

        # metrics
        claim_coverage = self._estimate_claim_coverage(claims, insights)
        evidence_strength = (strength_accum / max(1, count)) if count else 0.0
        temporal_coherence = self._temporal_coherence_metric(temporal)
        domain_alignment = self._domain_alignment_metric(domains, claims, entities)

        # gaps
        gaps: List[Dict[str, Any]] = []
        for i, c in enumerate(claims):
            addressed = any(self._contains_concept(ins, c) for ins in insights)
            if not addressed:
                gaps.append({
                    "claim_id": f"gap_{i+1}",
                    "claim_text": c,
                    "gap_type": "missing_explanation",
                    "severity": self._gap_severity(c),
                })

        return {
            "nodes": nodes,
            "relationships": relationships,
            "claims": [{"id": self._node_id("claim", c), "text": c} for c in claims],
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

    def search_entities(self, text: str, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
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

            # Persist (or find) this embedding in your embedding store,
            # tag it with type="ner" so we stay in the same space
            if hasattr(self.memory, "embedding"):
                # ensure returns an id for this exact text & type
                emb_id = self.memory.embedding.ensure_query_embedding(
                    text=text,
                    embedding_vec=embedding_vec,
                    embedding_type="ner"
                )
            else:
                # fallback: try an id lookup; as last resort, fail open (no cache)
                emb_id = getattr(self.memory, "get_id_for_text", lambda _t: None)(text)

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
                enriched.append((
                    str(node_id),
                    float(score),
                    {**(meta or {}), "node_id": str(node_id), "score": float(score)}
                ))

            # 5) Upsert cache
            if emb_id and hasattr(self.memory, "entity_cache"):
                self.memory.entity_cache.upsert(emb_id, enriched)
                self._stats["cache_misses"] += 1

            self._stats["queries"] += 1
            self._stats["query_time"] += time.time() - start_time
            return enriched

        except Exception as e:
            self.logger.log("KnowledgeGraphSearchError", {
                "text": text[:200],
                "error": str(e),
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
            src = r.get("source"); tgt = r.get("target")
            src_i = next((i for i, n in enumerate(nodes) if n.get("id") == src), None)
            tgt_i = next((i for i, n in enumerate(nodes) if n.get("id") == tgt), None)
            if src_i is not None and tgt_i is not None:
                rels.append({
                    "source_idx": src_i,
                    "target_idx": tgt_i,
                    "type": r.get("type", "rel"),
                    "confidence": float(r.get("confidence", 0.0)),
                })

        return {
            "entities": entities,
            "metrics": metrics,
            "matrix": matrix,
            "relationships": rels,
        }

    def _estimate_claim_coverage(self, claims: List[str], insights: List[str]) -> float:
        if not claims:
            return 0.0
        covered = 0
        for c in claims:
            if any(self._contains_concept(i, c) for i in insights):
                covered += 1
        return covered / max(1, len(claims))

    def _temporal_coherence_metric(self, temporal_rels: List[Dict[str, Any]]) -> float:
        if not temporal_rels:
            return 0.0
        ok, tot = 0, 0
        for r in temporal_rels:
            if "time_delta" in r:
                tot += 1
                if (r.get("time_delta") or 0) >= 0:
                    ok += 1
        return (ok / tot) if tot else 0.8  # neutral default

    def _domain_alignment_metric(self, domains: List[Dict[str, Any]], claims: List[str], ents: List[Dict[str, Any]]) -> float:
        if not domains:
            return 0.7
        names = [str(d.get("domain", "")).lower() for d in domains if isinstance(d, dict)]
        text = " ".join(claims + [e.get("text","") for e in ents])
        score = sum(1 for d in names if d and d in text.lower())
        return self._clamp01(score / max(1, len(names)))

    def _gap_severity(self, claim: str) -> float:
        sev = 0.50
        if re.search(r"\b(critical|essential|must|key|fundamental|core)\b", claim, re.I):
            sev += 0.30
        if re.search(r"\b\d+(\.\d+)?\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b", claim, re.I):
            sev += 0.20
        return self._clamp01(sev)

    def _temporal_edges(self, trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                    rtype = "causal_sequence"; conf = 0.95
                rels.append({
                    "source": a_id, "target": b_id, "type": rtype, "confidence": conf,
                    "timestamp": b.get("timestamp"),
                    "time_delta": (b.get("timestamp", 0) or 0) - (a.get("timestamp", 0) or 0),
                })
        return rels


    def _rel_strength_claim_insight(
        self, claim: str, insight: str, entities: List[Dict[str, Any]], domains: List[Dict[str, Any]]
    ) -> float:
        j = self._jaccard(self._token_set(claim), self._token_set(insight))

        ent_texts = [e.get("text", "") for e in entities]
        ce = {t.lower() for t in ent_texts if t and t.lower() in claim.lower()}
        ie = {t.lower() for t in ent_texts if t and t.lower() in insight.lower()}
        overlap = (len(ce & ie) / max(1, len(ce | ie))) if (ce or ie) else 0.0
        entity_bonus = 0.30 * overlap

        domain_boost = 0.0
        if self.domain_aware:
            names = {str(d.get("domain", "")).lower() for d in (domains or []) if isinstance(d, dict)}
            if any(x in names for x in {"ml", "machine learning", "nlp"}):
                domain_boost = 0.15

        return self._clamp01(j + entity_bonus + domain_boost)

    # -------------------------
    # Entity & Node Management
    # -------------------------

    def _extract_entities_safe(self, text: str) -> List[Dict[str, Any]]:
        try:
            if not text:
                return []
            ents = self._entity_detector.detect(text) if hasattr(self._entity_detector, "detect") else []
            if ents:
                return [e for e in ents if isinstance(e, dict) and e.get("text")]
        except Exception as e:
            self.logger and self.logger.log("KGEntityExtractionError", {"error": str(e)})

        # heuristic fallback (capitalized tokens)
        tokens = re.findall(r"\b[A-Z][A-Za-z0-9\-\_]{2,}\b", text or "")
        uniq = {}
        for t in tokens:
            key = self._normalize_entity(t)
            if key not in uniq:
                uniq[key] = {"text": t, "type": "TERM", "start": -1, "end": -1, "source_text": t}
        return list(uniq.values())

    def _extract_claims_simple(self, text: str, min_len: int = None) -> List[str]:
        min_len = min_len or self.min_claim_len
        cand = re.split(r"(?<=[\.\!\?])\s+", text or "")
        out: List[str] = []
        for s in cand:
            s = (s or "").strip()
            if len(s) < min_len:
                continue
            if re.search(r"\b(we|our|this paper|results|achieve|improve|demonstrate|propose)\b", s, re.I):
                out.append(s)
            elif re.search(r"\b\d+(\.\d+)?\s*(%|percent|accuracy|precision|recall|f1|auc|rmse|mae|bleu)\b", s, re.I):
                out.append(s)
        return out[:256]

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
        pats = [r"therefore", r"as a result", r"because", r"consequently", r"thus", r"hence", r"led to", r"so we"]
        return any(re.search(p, t) for p in pats)

    @staticmethod
    def _contains_concept(a: str, b: str) -> bool:
        # conservative concept overlap
        A = KnowledgeGraphService._token_set(a)
        B = KnowledgeGraphService._token_set(b)
        return KnowledgeGraphService._jaccard(A, B) >= 0.18

    @staticmethod
    def _node_id(kind: str, text: str) -> str:
        import hashlib
        h = hashlib.sha1((kind + "|" + (text or "")).encode("utf-8")).hexdigest()[:16]
        return f"{kind}:{h}"
