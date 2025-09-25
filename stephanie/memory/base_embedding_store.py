# stephanie/memory/base_embedding_store.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.utils.lru_cache import SimpleLRUCache

_logger = logging.getLogger(__name__)


class BaseEmbeddingStore(BaseSQLAlchemyStore):
    """
    Hybrid semantic + (optional) NER retrieval over scorables.

    Design goals:
    - Keep semantic path unchanged and stable.
    - NER is gated and hardened: query sanitization, part-based fallback, optional lexical fallback,
      and vector guards live ONLY in the NER path.
    - Deterministic, branch-aware combiner to avoid confusing "SemanticFailure" when both paths return 0.
    """

    def __init__(
        self,
        cfg,
        memory,
        table: str,
        name: str,
        embed_fn,
        logger,
        cache_size: int = 10000,
    ):
        super().__init__(memory.session, logger)
        self.cfg = cfg or {}
        self.memory = memory
        self.conn = memory.conn
        self.dim = self.cfg.get("dim", 1024)
        self.hdim = self.dim // 2
        self.table = table
        self.name = name  # store name label
        self.type = self.cfg.get("type", name)  # e.g. "hnet", "hf"
        self.embed_fn = embed_fn

        # Cache: {hash -> (embedding_id, embedding_vector)}
        self._cache = SimpleLRUCache(max_size=cache_size)

        # NER retriever init (optional)
        self.ner_retriever = None
        self.ner_enabled = bool(self.cfg.get("enable_ner", True))
        self.ner_index_initialized = False

        # Retrieval knobs
        self.min_sem_hits_to_skip_ner = int(self.cfg.get("min_sem_hits_to_skip_ner", 3))
        self.min_sem_top_score_to_skip = float(self.cfg.get("min_sem_top_score_to_skip", 0.65))
        self.ner_hybrid_mode = self.cfg.get("ner_hybrid_mode", "merge")  # merge|semantic_only|ner_only|separate

        # Calibration default path
        self.calib_dir = self.cfg.get("calibration_dir", "data/calibration")

        if self.ner_enabled:
            try:
                from stephanie.models.ner_retriever import NERRetrieverEmbedder

                self.ner_retriever = NERRetrieverEmbedder(
                    model_name=self.cfg.get("ner_model", "meta-llama/Llama-3.2-1B-Instruct"),
                    layer=self.cfg.get("ner_layer", 17),
                    device=self.cfg.get("ner_device", "cuda" if torch.cuda.is_available() else "cpu"),
                    embedding_dim=self.cfg.get("ner_dim", 2048),
                    index_path=self.cfg.get("ner_index_path", "data/ner_retriever/index"),
                    logger=self.logger,
                    memory=self.memory,
                )
                self.logger.log(
                    "NERRetrieverInitialized",
                    {
                        "model": self.cfg.get("ner_model", "meta-llama/Llama-3.2-1B-Instruct"),
                        "layer": self.cfg.get("ner_layer", 17),
                        "dim": self.cfg.get("ner_dim", 2048),
                        "index_path": self.cfg.get("ner_index_path", "data/ner_retriever/index"),
                    },
                )
                self.ner_index_initialized = True
            except Exception as e:
                self.logger.log(
                    "NERRetrieverInitFailed",
                    {"error": str(e), "message": "Failed to initialize NER Retriever, disabling"},
                )
                self.ner_retriever = None
                self.ner_enabled = False

    def name(self) -> str:
        return self.name

    def __repr__(self):
        return f"<{self.__class__.__name__} table={self.table} type={self.type}>"

    def get_text_hash(self, text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    # --------------- embedding storage ---------------

    def get_or_create(self, text: str):
        """Return embedding vector for text, caching both id + embedding."""
        text_hash = self.get_text_hash(text)
        cached = self._cache.get(text_hash)
        if cached:
            return cached[1]  # embedding vector

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, embedding FROM {self.table} WHERE text_hash = %s",
                    (text_hash,),
                )
                row = cur.fetchone()
                if row:
                    embedding_id, embedding = row
                    self._cache.set(text_hash, (embedding_id, embedding))
                    return embedding
        except Exception as e:
            self.logger and self.logger.log("EmbeddingFetchFailed", {"error": str(e)})

        # Not found → create
        embedding = self.embed_fn(text, self.cfg)
        embedding_id = None
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.table} (text, text_hash, embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (text_hash) DO NOTHING
                    RETURNING id;
                    """,
                    (text, text_hash, embedding),
                )
                row = cur.fetchone()
                if row:
                    embedding_id = row[0]
            self.conn.commit()
        except Exception as e:
            _logger.error(f"EmbeddingInsertFailed error: {str(e)}")

        # Fall back: lookup id if INSERT didn't return
        if embedding_id is None:
            embedding_id = self.get_id_for_text(text)

        self._cache.set(text_hash, (embedding_id, embedding))
        self.logger and self.logger.log(
            "TextEmbeddingCreated",
            {"len": len(text or ""), "embed_shape": len(embedding), "text": f"{text[:30]}..."},
        )
        return embedding

    def get_id_for_text(self, text: str) -> Optional[int]:
        """Return embedding id for a text, cached if available."""
        text_hash = self.get_text_hash(text)
        cached = self._cache.get(text_hash)
        if cached:
            return cached[0]  # embedding_id

        try:
            with self.conn.cursor() as cur:
                cur.execute(f"SELECT id FROM {self.table} WHERE text_hash = %s", (text_hash,))
                row = cur.fetchone()
                if row:
                    embedding_id = row[0]
                    self._cache.set(text_hash, (embedding_id, None))  # no vector
                    return embedding_id
        except Exception as e:
            self.logger and self.logger.log("EmbeddingIdFetchFailed", {"error": str(e)})
        return None

    # --------------- public retrieval API ---------------

    def search_related_scorables(
        self,
        query: str,
        target_type: str = "document",
        top_k: int = 10,
        with_metadata: bool = True,
        include_ner: bool = True,
        domain: Optional[str] = None,
    ):
        """
        Search for scorables with configurable NER + semantic integration.
        hybrid mode: merge | separate | ner_only | semantic_only
        """
        hybrid_mode = self.ner_hybrid_mode

        semantic_results: List[Dict] = []
        ner_results: List[Dict] = []

        # Semantic first (stable path)
        if hybrid_mode in ["merge", "semantic_only", "separate"]:
            semantic_results = self._get_semantic_results(query, target_type, top_k, with_metadata)

        # NER next (only if enabled and not obviously unnecessary)
        if include_ner and self.ner_enabled and hybrid_mode in ["merge", "ner_only", "separate"]:
            if hybrid_mode == "merge" and self._should_skip_ner(semantic_results):
                ner_results = []
                self.logger.log("NERSkipped", {
                    "reason": "semantic_sufficient",
                    "sem_count": len(semantic_results or []),
                    "sem_top": float(semantic_results[0]["score"]) if semantic_results else 0.0
                })
            else:
                ner_results = self._get_ner_results_hardened(query, top_k)

        if hybrid_mode == "merge":
            combined = self._combine_semantic_and_ner(semantic_results, ner_results, query=query, domain=domain)
            return combined[:top_k]

        if hybrid_mode == "separate":
            return {
                "semantic": semantic_results[:top_k],
                "ner": ner_results[:top_k],
                "combined": self._combine_semantic_and_ner(semantic_results, ner_results, query=query, domain=domain)[:top_k],
            }

        if hybrid_mode == "ner_only":
            return ner_results[:top_k]

        # semantic_only
        return semantic_results[:top_k]

    # --------------- semantic path (keep stable) ---------------

    def _get_semantic_results(self, query: str, target_type: str, top_k: int, with_metadata: bool) -> List[Dict]:
        """Standard semantic search over scorable embeddings."""
        base_sql = """
            SELECT se.scorable_id, se.scorable_type, se.embedding_id,
                   1 - (e.embedding <-> %s::vector) AS score
        """
        join_sql = f"FROM scorable_embeddings se JOIN {self.table} e ON se.embedding_id = e.id"

        if with_metadata and target_type == "document":
            base_sql += ", d.title, d.summary, d.text"
            join_sql += " JOIN documents d ON se.scorable_id::int = d.id"

        sql = f"""
            {base_sql}
            {join_sql}
            WHERE se.scorable_type = %s
            ORDER BY e.embedding <-> %s::vector
            LIMIT %s;
        """

        try:
            query_emb = self.get_or_create(query)
            with self.conn.cursor() as cur:
                cur.execute(sql, (query_emb, target_type, query_emb, top_k))
                rows = cur.fetchall()

            results: List[Dict] = []
            for row in rows:
                base = {
                    "id": row[0],
                    "scorable_id": row[0],
                    "scorable_type": row[1],
                    "embedding_id": row[2],
                    "score": float(row[3]),
                    "retrieval_type": "semantic",
                }
                if with_metadata and target_type == "document":
                    base.update({"title": row[4], "summary": row[5], "text": row[6]})
                results.append(base)
            return results
        except Exception as e:
            self.logger and self.logger.log("ScorableSearchFailed", {"error": str(e), "query": query})
            return []

    # --------------- NER path (hardened) ---------------

    # Query & vector guards used only in NER flow
    @staticmethod
    def _sanitize_query(q: str) -> str:
        q = (q or "").replace("\n", " ")
        q = re.sub(r"\s+", " ", q).strip()
        return q[:512]

    @staticmethod
    def _keywordize(q: str, limit: int = 8) -> List[str]:
        toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\_]{2,}", q or "")
        return [t.strip("-_") for t in toks][:limit]

    def _extract_candidate_entities(self, q: str) -> List[str]:
        """Heuristic entity candidates (capitalized tokens / hyphenated terms)."""
        ents = re.findall(r"\b[A-Z][A-Za-z0-9\-\_]{2,}\b", q or "")
        return list(dict.fromkeys(ents))  # dedupe, preserve order

    def _get_ner_results_hardened(self, query: str, top_k: int) -> List[Dict]:
        """Hardened NER retrieval with fallbacks, confined to NER branch."""
        if not self.ner_enabled or not self.ner_retriever:
            return []

        q = self._sanitize_query(query)
        results: List[Dict] = []

        try:
            # Primary NER retrieval
            primary = self.ner_retriever.retrieve_entities(
                query=q,
                top_k=top_k,
                min_similarity=self.cfg.get("ner_min_similarity", 0.6),
            ) or []

            # Standardize and tag
            for r in primary:
                r.setdefault("retrieval_type", "ner_entity")
            results = primary

            # Part-based fallback
            if not results:
                parts = self._extract_candidate_entities(q)[:5] or self._keywordize(q)
                best_by_id: Dict[str, Dict] = {}
                for p in parts[:5]:
                    try:
                        pr = self.ner_retriever.retrieve_entities(query=p, top_k=max(3, top_k // 2), min_similarity=self.cfg.get("ner_min_similarity", 0.55)) or []
                        for rr in pr:
                            key = str(rr.get("scorable_id") or rr.get("entity_text") or rr.get("node_id") or id(rr))
                            if key not in best_by_id or float(rr.get("similarity", 0.0)) > float(best_by_id[key].get("similarity", 0.0)):
                                rr.setdefault("retrieval_type", "ner_entity_part")
                                best_by_id[key] = rr
                    except Exception:
                        continue
                results = list(best_by_id.values())

            # Optional lexical fallback (best-effort, no hard dependency)
            if not results:
                try:
                    idx = getattr(self.ner_retriever, "index", None)
                    if idx and hasattr(idx, "all_metadata"):
                        metas = [m for m in (idx.all_metadata() or []) if m.get("text")]
                        if metas:
                            try:
                                from rapidfuzz import fuzz, process
                                choices = [m.get("text") for m in metas]
                                top = process.extract(q, choices, scorer=fuzz.token_set_ratio, limit=min(5, max(1, top_k)))
                                mapped = []
                                for (txt, score, idx_i) in top:
                                    meta = metas[int(idx_i)]
                                    mapped.append({
                                        "node_id": str(meta.get("node_id") or meta.get("id") or txt),
                                        "entity_text": txt,
                                        "similarity": float(score) / 100.0,
                                        "type": meta.get("type"),
                                        "domains": meta.get("domains", []),
                                        "sources": meta.get("sources", []),
                                        "retrieval_type": "ner_lexical",
                                    })
                                results = mapped
                                self.logger.log("NERLexicalFallbackUsed", {"query": q, "hits": len(results)})
                            except Exception as e:
                                self.logger.log("NERLexicalFallbackError", {"error": str(e)})
                except Exception as e:
                    self.logger.log("NERIndexAccessError", {"error": str(e)})

            # Normalize schema minimally for downstream combiner
            for r in results:
                r.setdefault("similarity", float(r.get("similarity", r.get("score", 0.0))))
                r.setdefault("entity_text", r.get("entity_text") or r.get("text") or "")
                r.setdefault("scorable_id", r.get("scorable_id"))
                r.setdefault("retrieval_type", r.get("retrieval_type", "ner_entity"))

            self.logger.log("NERSearchResults", {
                "query": q[:120],
                "results_count": len(results),
                "top_similarity": float(results[0].get("similarity", 0.0)) if results else 0.0
            })
            return results

        except Exception as e:
            self.logger and self.logger.log("NERSearchFailed", {"error": str(e), "query": q[:160]})
            return []

    # --------------- dynamic weighting / calibration ---------------

    def _should_skip_ner(self, sem: List[dict]) -> bool:
        if not self.ner_enabled:
            return True
        if not sem:
            return False
        if len(sem) >= self.min_sem_hits_to_skip_ner:
            return True
        top = float(sem[0].get("score", sem[0].get("norm_score", 0.0)))
        return top >= self.min_sem_top_score_to_skip

    def _determine_dynamic_weights(self, query: str, domain: Optional[str] = None) -> Tuple[float, float]:
        """
        Return (ner_weight, semantic_weight) in [0,1].
        """
        ql = (query or "").lower()

        entity_keywords = ["who", "what", "where", "when", "which", "name", "list", "find"]
        conceptual_keywords = ["how", "why", "explain", "describe", "understand", "relationship"]

        entity_count = sum(1 for kw in entity_keywords if kw in ql)
        conceptual_count = sum(1 for kw in conceptual_keywords if kw in ql)

        if entity_count > conceptual_count:
            ner_weight, semantic_weight = 0.7, 0.3
        elif conceptual_count > entity_count:
            ner_weight, semantic_weight = 0.3, 0.7
        else:
            ner_weight, semantic_weight = 0.5, 0.5

        domain_weights = {
            "legal": (0.6, 0.4),
            "scientific": (0.55, 0.45),
            "creative": (0.4, 0.6),
        }
        if domain and domain in domain_weights:
            ner_weight, semantic_weight = domain_weights[domain]

        ner_weight = max(0.0, min(1.0, ner_weight))
        semantic_weight = max(0.0, min(1.0, semantic_weight))

        self.logger.log("DynamicWeighting", {
            "query": (query or "")[:60],
            "ner_weight": ner_weight,
            "semantic_weight": semantic_weight,
            "domain": domain or "unknown",
        })
        return ner_weight, semantic_weight

    def _normalize_scores(self, results: List[Dict], score_key: str = "score") -> None:
        """
        Distribution-aware normalization.

        - semantic scores (score): mean/std to [0,1] via 3σ window
        - ner similarity (similarity): percentile/IQR mapping to [0,1]
        """
        if not results:
            return

        vals = [r.get(score_key, 0.0) for r in results if r.get(score_key) is not None]
        if not vals:
            return

        if score_key == "score":
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            denom = 6 * std + 1e-8
            lower = mean - 3 * std
            for r in results:
                raw = float(r.get(score_key, 0.0))
                norm = (raw - lower) / denom
                r["norm_score"] = float(min(1.0, max(0.0, norm)))
        elif score_key == "similarity":
            p25 = float(np.percentile(vals, 25))
            p75 = float(np.percentile(vals, 75))
            iqr = max(p75 - p25, 1e-8)
            upper = p75 + 1.5 * iqr
            for r in results:
                raw = float(r.get(score_key, 0.0))
                if raw >= p75:
                    norm = 0.5 + 0.5 * min(1.0, (raw - p75) / (upper - p75 + 1e-8))
                else:
                    norm = 0.5 * (raw - p25) / (p75 - p25 + 1e-8)
                r["norm_similarity"] = float(min(1.0, max(0.0, norm)))

    def _standardize_result_schema(self, results: List[Dict]) -> List[Dict]:
        """Ensure all results have consistent schema."""
        if not results:
            return []
        standard_keys = {
            "id", "scorable_id", "scorable_type", "embedding_id",
            "score", "norm_score", "combined_score",
            "retrieval_type", "entity_text", "entity_type",
            "start", "end", "full_text", "title", "summary", "similarity", "norm_similarity"
        }
        for r in results:
            for k in standard_keys:
                r.setdefault(k, None)
            # ensure id
            if r.get("id") is None and r.get("scorable_id") is not None:
                r["id"] = r["scorable_id"]
        return results

    def _calibrate_confidence(self, semantic_results: List[Dict], ner_results: List[Dict], query: str) -> None:
        """Calibrate confidence between systems based on historical performance."""
        domain = self._get_current_domain(query)
        calib = self._load_calibration_data(domain)

        for r in semantic_results:
            base = float(r.get("norm_score", r.get("score", 0.0)))
            r["calibrated_score"] = self._apply_calibration(base, calib.get("semantic", {}))
        for r in ner_results:
            base = float(r.get("norm_similarity", r.get("similarity", 0.0)))
            r["calibrated_similarity"] = self._apply_calibration(base, calib.get("ner", {}))

    def _apply_calibration(self, score: float, calibration: Dict) -> float:
        if not calibration or "coefficients" not in calibration:
            return float(score)
        poly = np.poly1d(calibration["coefficients"])
        return float(poly(score))

    def _get_current_domain(self, query: str) -> str:
        if not hasattr(self, "_domain_classifier"):
            from stephanie.analysis.scorable_classifier import \
                ScorableClassifier
            self._domain_classifier = ScorableClassifier(
                memory=self.memory,
                logger=self.logger,
                config_path=self.cfg.get("domain_config", "config/domain/seeds.yaml"),
            )
        return self._domain_classifier.classify(query)

    def _load_calibration_data(self, domain: str) -> Dict:
        os.makedirs(self.calib_dir, exist_ok=True)
        path = os.path.join(self.calib_dir, f"{domain}_calibration.json")
        if not os.path.exists(path):
            path = os.path.join(self.calib_dir, "general_calibration.json")
            if not os.path.exists(path):
                return {"semantic": {"coefficients": [1.0, 0.0]}, "ner": {"coefficients": [1.0, 0.0]}}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.log("CalibrationLoadFailed", {"error": str(e), "path": path})
            return {"semantic": {"coefficients": [1.0, 0.0]}, "ner": {"coefficients": [1.0, 0.0]}}

    # --------------- combiner ---------------

    def _combine_semantic_and_ner(self, semantic: List[Dict], ner: List[Dict], *, query: str, domain: Optional[str]):
        # Normalize
        self._normalize_scores(semantic, "score")
        self._normalize_scores(ner, "similarity")

        # Calibrate
        self._calibrate_confidence(semantic, ner, query)

        # Standardize
        semantic = self._standardize_result_schema(semantic)
        ner = self._standardize_result_schema(ner)
        self.logger.log("CombinedSearchPostStd", {"event": "combine_sizes_post_std", "sem_std": len(semantic), "ner_std": len(ner)})

        # Early empty handling (no misleading "SemanticFailure")
        if not semantic and not ner:
            self.logger.log("RetrievalEmpty", {"query": (query or "")[:120]})
            return [{
                "retrieval_type": "empty",
                "combined_score": 0.0,
                "score": 0.0,
                "text": "",
                "warning": "no_retrieval_hits"
            }]

        # Only one branch present
        if semantic and not ner:
            for r in semantic:
                r["combined_score"] = float(r.get("calibrated_score", r.get("norm_score", r.get("score", 0.0))))
                r.setdefault("retrieval_type", "semantic_only")
            self.logger.log("NERMissing", {"query": (query or "")[:120], "semantic_results": len(semantic)})
            return sorted(semantic, key=lambda x: x["combined_score"], reverse=True)

        if ner and not semantic:
            for r in ner:
                r["combined_score"] = float(r.get("calibrated_similarity", r.get("norm_similarity", r.get("similarity", 0.0)))) * 1.2  # slight boost in ner-only
                r.setdefault("retrieval_type", "ner_fallback")
            self.logger.log("SemanticMissing", {"query": (query or "")[:120], "ner_results": len(ner)})
            return sorted(ner, key=lambda x: x["combined_score"], reverse=True)

        # Both present → deterministic weighted blend
        ner_w, sem_w = self._determine_dynamic_weights(query, domain)
        all_results: List[Dict] = []

        for r in semantic:
            base = float(r.get("calibrated_score", r.get("norm_score", r.get("score", 0.0))))
            r["combined_score"] = sem_w * base
            r.setdefault("retrieval_type", "semantic")
            r["score"] = r["combined_score"]
            all_results.append(r)

        for r in ner:
            base = float(r.get("calibrated_similarity", r.get("norm_similarity", r.get("similarity", 0.0))))
            r["combined_score"] = ner_w * base
            r.setdefault("retrieval_type", "ner_entity")
            r["score"] = r["combined_score"]
            all_results.append(r)

        _logger.info(f"Combined {len(semantic)} semantic + {len(ner)} NER results → {len(all_results)} total")
        self.logger.log("CombinedSearchResults", {"event": "combine_sizes", "sem_raw": len(semantic), "ner_raw": len(ner)})
        return sorted(all_results, key=lambda x: x["combined_score"], reverse=True)

    # --------------- neighbors / misc ---------------

    def find_neighbors(self, embedding, k: int = 5):
        """Given an embedding vector, return nearest neighbor texts from the table."""
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().tolist()

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT e.id, e.text, 1 - (e.embedding <-> %s::vector) AS score
                    FROM {self.table} e
                    WHERE e.embedding IS NOT NULL
                    ORDER BY e.embedding <-> %s::vector
                    LIMIT %s;
                    """,
                    (embedding, embedding, k),
                )
                rows = cur.fetchall()
            return [{"id": row[0], "text": row[1], "score": float(row[2])} for row in rows]
        except Exception as e:
            self.logger and self.logger.log("FindNeighborsFailed", {"error": str(e)})
            return []

    # --------------- NER indexing / stats (unchanged logic) ---------------

    def index_entities_from_scorables(self, scorables: list, batch_size: int = 10) -> int:
        if not self.ner_enabled or not self.ner_retriever:
            return 0
        total_indexed = 0
        try:
            for i in range(0, len(scorables), batch_size):
                batch = scorables[i : i + batch_size]
                count = self.ner_retriever.index_scorables(batch)
                total_indexed += count
            self.ner_index_initialized = True
            self.logger and self.logger.log(
                "NERIndexingComplete",
                {"scorables_processed": len(scorables), "entities_indexed": total_indexed, "batch_size": batch_size},
            )
            return total_indexed
        except Exception as e:
            self.logger and self.logger.log(
                "NERIndexingFailed", {"error": str(e), "scorables_count": len(scorables), "batch_size": batch_size}
            )
            return 0

    def search_ner_entities(self, query: str, k: int = 5, min_similarity: float = 0.6) -> list:
        if not self.ner_enabled or not self.ner_retriever:
            return []
        try:
            q = self._sanitize_query(query)
            results = self.ner_retriever.retrieve_entities(query=q, k=k, min_similarity=min_similarity) or []
            for result in results:
                if "scorable_id" in result:
                    try:
                        with self.conn.cursor() as cur:
                            cur.execute("SELECT text FROM scorables WHERE id = %s", (result["scorable_id"],))
                            row = cur.fetchone()
                            if row:
                                result["full_text"] = row[0]
                    except Exception as e:
                        self.logger and self.logger.log("NERMetadataFetchFailed", {"error": str(e), "scorable_id": result["scorable_id"]})
            self.logger and self.logger.log(
                "NERSearchResults",
                {"query": q, "results_count": len(results), "top_results": [r.get("entity_text") for r in results[:3]] if results else []},
            )
            return results
        except Exception as e:
            self.logger and self.logger.log("NERSearchFailed", {"error": str(e), "query": query})
            return []

    def train_ner_projection(self, scorables: list, max_triplets: int = 1000, batch_size: int = 32, epochs: int = 3, lr: float = 1e-4) -> bool:
        if not self.ner_enabled or not self.ner_retriever:
            return False
        try:
            triplets = self.ner_retriever.generate_triplets(scorables, max_triplets)
            if not triplets:
                self.logger and self.logger.log("NERTrainingSkipped", {"reason": "no_triplets_generated", "scorables_count": len(scorables)})
                return False

            self.ner_retriever.train_projection(triplets=triplets, batch_size=batch_size, epochs=epochs, lr=lr)
            self.index_entities_from_scorables(scorables)
            self.logger and self.logger.log("NERTrainingComplete", {"triplets": len(triplets), "epochs": epochs, "batch_size": batch_size, "learning_rate": lr})
            return True
        except Exception as e:
            self.logger and self.logger.log("NERTrainingFailed", {"error": str(e)})
            return False

    def get_ner_index_stats(self) -> dict:
        if not self.ner_enabled or not self.ner_retriever or not getattr(self.ner_retriever, "index", None):
            return {"enabled": False, "initialized": False}
        try:
            stats = self.ner_retriever.index.get_stats()
            stats["enabled"] = True
            stats["initialized"] = self.ner_index_initialized
            return stats
        except Exception as e:
            self.logger and self.logger.log("NERStatsFailed", {"error": str(e)})
            return {"enabled": True, "initialized": False, "error": str(e)}

    def is_ner_enabled(self) -> bool:
        return self.ner_enabled and self.ner_retriever is not None

    def get_ner_index_status(self) -> dict:
        return {
            "enabled": self.ner_enabled,
            "initialized": self.ner_index_initialized,
            "retriever_available": self.ner_retriever is not None,
            "index_stats": self.get_ner_index_stats(),
        }

    # --------------- calibration training ---------------

    def train_calibration(self, historical_data: List[Dict], domain: str = "general"):
        """
        Train polynomial calibration for semantic and NER scores to expected accuracy.
        """
        semantic_data = [d for d in historical_data if d.get("system") == "semantic"]
        ner_data = [d for d in historical_data if d.get("system") == "ner"]

        semantic_calibration = self._train_calibration_model(semantic_data)
        ner_calibration = self._train_calibration_model(ner_data)

        calib = {"semantic": semantic_calibration, "ner": ner_calibration, "timestamp": datetime.now().isoformat()}
        os.makedirs(self.calib_dir, exist_ok=True)
        path = os.path.join(self.calib_dir, f"{domain}_calibration.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(calib, f, indent=2)

        self.logger.log(
            "CalibrationTrained",
            {"domain": domain, "semantic_rmse": semantic_calibration.get("rmse", 0), "ner_rmse": ner_calibration.get("rmse", 0)},
        )

    def _train_calibration_model(self, data: List[Dict]) -> Dict:
        if not data:
            return {"coefficients": [1.0, 0.0]}
        try:
            scores = [float(d["score"]) for d in data]
            acc = [float(d["accuracy"]) for d in data]
            coeffs = np.polyfit(scores, acc, 2)
            predicted = np.polyval(coeffs, scores)
            rmse = float(np.sqrt(np.mean((predicted - np.asarray(acc)) ** 2)))
            return {"coefficients": coeffs.tolist(), "rmse": rmse, "sample_size": len(data)}
        except Exception as e:
            self.logger.log("CalibrationTrainingFailed", {"error": str(e)})
            return {"coefficients": [1.0, 0.0]}
