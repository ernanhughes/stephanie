# stephanie/memory/base_embedding_store.py
import hashlib
from typing import Dict, List, Tuple
import numpy as np
import torch
import os
import json
from datetime import datetime

from stephanie.memory import BaseStore
from stephanie.utils.lru_cache import SimpleLRUCache

class BaseEmbeddingStore(BaseStore):
    def __init__(self, cfg, conn, db, table: str, name: str, embed_fn, logger=None, cache_size=10000):
        super().__init__(db, logger)
        self.cfg = cfg
        self.conn = conn
        self.dim = cfg.get("dim", 1024)
        self.hdim = self.dim // 2
        self.table = table
        self.name = name
        self.type = cfg.get("type", name)  # e.g. "hnet", "hf"
        self.embed_fn = embed_fn

        # Cache: {hash -> (embedding_id, embedding_vector)}
        self._cache = SimpleLRUCache(max_size=cache_size)
        
        # Initialize NER Retriever if enabled in config
        self.ner_retriever = None
        self.ner_enabled = cfg.get("enable_ner", False)
        self.ner_index_initialized = False

        
        if self.ner_enabled:
            try:
                from stephanie.scoring.model.ner_retriever import NERRetrieverEmbedder
                self.ner_retriever = NERRetrieverEmbedder(
                    model_name=cfg.get("ner_model", "meta-llama/Llama-3-8b"),
                    layer=cfg.get("ner_layer", 17),
                    device=cfg.get("ner_device", "cuda" if torch.cuda.is_available() else "cpu"),
                    embedding_dim=cfg.get("ner_dim", 500),
                    index_path=cfg.get("ner_index_path", "data/ner_retriever/index")
                )
                self.logger.log("NERRetrieverInitialized", {
                    "model": cfg.get("ner_model", "meta-llama/Llama-3-8b"),
                    "layer": cfg.get("ner_layer", 17),
                    "dim": cfg.get("ner_dim", 500),
                    "index_path": cfg.get("ner_index_path", "data/ner_retriever/index")
                })
                self.ner_index_initialized = True
            except Exception as e:
                self.logger.log("NERRetrieverInitFailed", {
                    "error": str(e),
                    "message": "Failed to initialize NER Retriever, disabling"
                })
                self.ner_retriever = None
                self.ner_enabled = False

    def name(self) -> str:
        return self.name

    def __repr__(self):
        return f"<{self.__class__.__name__} table={self.table} type={self.type}>"

    def get_text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
            if self.logger:
                self.logger.log("EmbeddingFetchFailed", {"error": str(e)})

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
            if self.logger:
                self.logger.log("EmbeddingInsertFailed", {"error": str(e)})

        # Fall back: lookup id if INSERT didn't return
        if embedding_id is None:
            embedding_id = self.get_id_for_text(text)

        self._cache.set(text_hash, (embedding_id, embedding))
        return embedding

    def get_id_for_text(self, text: str) -> int | None:
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
            if self.logger:
                self.logger.log("EmbeddingIdFetchFailed", {"error": str(e)})
        return None

    def search_related_scorables(
        self,
        query: str,
        target_type: str = "document", 
        top_k: int = 10,
        with_metadata: bool = True,
        include_ner: bool = True,
        domain: str = None
    ):
        """
        Search for scorables with configurable NER + semantic integration.
        
        Args:
            query: search text
            target_type: "document", "hypothesis", etc.
            top_k: number of results to return
            with_metadata: whether to fetch metadata (titles, text, etc.)
            include_ner: whether to run NER retrieval
            domain: optional domain context for dynamic weighting
        """
        hybrid_mode = self.cfg.get("ner_hybrid_mode", "merge")

        semantic_results, ner_results = [], []

        if hybrid_mode in ["merge", "semantic_only", "separate"]:
            semantic_results = self._get_semantic_results(query, target_type, top_k, with_metadata)

        if include_ner and self.ner_enabled and hybrid_mode in ["merge", "ner_only", "separate"]:
            ner_results = self._get_ner_results(query, top_k)

        if hybrid_mode == "merge":
            # 👇 fallback-safe combination
            combined = self._combine_results(semantic_results, ner_results, query=query, domain=domain)
            return combined[:top_k]

        elif hybrid_mode == "separate":
            return {
                "semantic": semantic_results[:top_k],
                "ner": ner_results[:top_k],
                "combined": self._combine_results(semantic_results, ner_results, query=query, domain=domain)[:top_k]
            }

        elif hybrid_mode == "ner_only":
            return ner_results[:top_k]

        else:  # semantic_only
            return semantic_results[:top_k]

    def _get_semantic_results(self, query: str, target_type: str, top_k: int, with_metadata: bool) -> List[Dict]:
        """Get standard semantic search results"""
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
            
            results = []
            for row in rows:
                base = {
                    "id": row[0],
                    "scorable_id": row[0],
                    "scorable_type": row[1],
                    "embedding_id": row[2],
                    "score": float(row[3]),
                }
                if with_metadata and target_type == "document":
                    base.update({
                        "title": row[4],
                        "summary": row[5],
                        "text": row[6]
                    })
                results.append(base)
            return results
        except Exception as e:
            if self.logger:
                self.logger.log("ScorableSearchFailed", {"error": str(e), "query": query})
            return []

    def _get_ner_results(self, query: str, top_k: int) -> List[Dict]:
        """Get NER entity results with proper error handling"""
        if not self.ner_enabled or not self.ner_retriever:
            return []
            
        try:
            return self.ner_retriever.retrieve_entities(
                query=query,
                k=top_k,
                min_similarity=self.cfg.get("ner_min_similarity", 0.6)
            )
        except Exception as e:
            if self.logger:
                self.logger.log("NERSearchFailed", {
                    "error": str(e),
                    "query": query
                })
            return []

    def find_neighbors(self, embedding, k: int = 5):
        """
        Given an embedding vector, return nearest neighbor texts from the table.
        """
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
            return [
                {"id": row[0], "text": row[1], "score": float(row[2])}
                for row in rows
            ]
        except Exception as e:
            if self.logger:
                self.logger.log("FindNeighborsFailed", {"error": str(e)})
            return []
    
    # ==============================
    # NER Retriever Integration
    # ==============================
    
    def _normalize_scores(self, results: List[Dict], score_key: str = "score") -> None:
        """
        Normalize scores using distribution-aware methods.

        - Semantic scores (cosine similarity / vector distance) → assume ~normal distribution.
        Normalize with mean & std (3-sigma scaling).
        - NER similarity scores → typically skewed.
        Normalize with percentile-based method (IQR scaling).

        Adds either `norm_score` or `norm_similarity` depending on score_key.
        """

        if not results:
            return

        scores = [r.get(score_key, 0.0) for r in results if r.get(score_key) is not None]
        if not scores:
            return

        # --- Case 1: semantic similarity ---
        if score_key == "score":
            mean = np.mean(scores)
            std = np.std(scores)
            for r in results:
                raw = r.get(score_key, 0.0)
                # Map to [0,1] with 3σ window
                norm = (raw - (mean - 3 * std)) / (6 * std + 1e-8)
                r["norm_score"] = float(min(1.0, max(0.0, norm)))

        # --- Case 2: NER similarity ---
        elif score_key == "similarity":
            p25 = np.percentile(scores, 25)
            p75 = np.percentile(scores, 75)
            iqr = max(p75 - p25, 1e-8)
            upper_bound = p75 + 1.5 * iqr

            for r in results:
                raw = r.get(score_key, 0.0)
                if raw >= p75:
                    # High-confidence → emphasize upper half
                    norm = 0.5 + 0.5 * min(1.0, (raw - p75) / (upper_bound - p75 + 1e-8))
                else:
                    # Low-confidence → squeeze into lower half
                    norm = 0.5 * (raw - p25) / (p75 - p25 + 1e-8)
                r["norm_similarity"] = float(min(1.0, max(0.0, norm)))

    def _standardize_result_schema(self, results: List[Dict]) -> List[Dict]:
        """Ensure all results have consistent schema"""
        standard_keys = {
            "id", "scorable_id", "scorable_type", "embedding_id", "score", 
            "norm_score", "combined_score", "retrieval_type", "entity_text", 
            "entity_type", "start", "end", "full_text", "title", "summary"
        }
        
        for result in results:
            # Add missing keys with None values
            for key in standard_keys:
                if key not in result:
                    result[key] = None
            
            # Ensure consistent ID field
            if "id" not in result and "scorable_id" in result:
                result["id"] = result["scorable_id"]
            elif "id" not in result:
                result["id"] = None
        
        return results

    def _combine_results(self, semantic_results: List[Dict], ner_results: List[Dict], query: str = "", domain: str = None) -> List[Dict]:
        # Normalize scores
        self._normalize_scores(semantic_results, "score")
        self._normalize_scores(ner_results, "similarity")
        
        # Apply confidence calibration
        self._calibrate_confidence(semantic_results, ner_results, query)
        
        # Standardize schema
        semantic_results = self._standardize_result_schema(semantic_results)
        ner_results = self._standardize_result_schema(ner_results)
        
        # Determine weights
        ner_weight, semantic_weight = self._determine_dynamic_weights(query, domain)
        
        # Apply calibrated scores
        all_results = []
        for r in semantic_results:
            r["combined_score"] = semantic_weight * r.get("calibrated_score", r.get("norm_score", 0.0))
            all_results.append(r)
            
        for r in ner_results:
            r["combined_score"] = ner_weight * r.get("calibrated_similarity", r.get("norm_similarity", 0.0))
            all_results.append(r)

        if not semantic_results:
            self.logger.log("SemanticFailure", {
                "query": query,
                "ner_results": len(ner_results),
                "message": "Semantic search failed, using NER only"
            })
            for r in ner_results:
                r["combined_score"] = r.get("norm_similarity", 0.0) * 1.2  # boost NER slightly
                r["retrieval_type"] = "ner_fallback"
            return sorted(ner_results, key=lambda x: x["combined_score"], reverse=True)

        # --- Case 3: Only semantic succeeded ---
        if not ner_results:
            self.logger.log("NERFailure", {
                "query": query,
                "semantic_results": len(semantic_results),
                "message": "NER search failed, using semantic only"
            })
            for r in semantic_results:
                r["combined_score"] = r.get("norm_score", 0.0)
                r["retrieval_type"] = "semantic_only"
                r["warning"] = "NER system unavailable"
            return sorted(semantic_results, key=lambda x: x["combined_score"], reverse=True)

        # --- Case 4: Both succeeded ---
        ner_weight, semantic_weight = self._determine_dynamic_weights(query, domain)
        
        all_results = []
        for r in semantic_results:
            r["combined_score"] = semantic_weight * r.get("norm_score", 0.0)
            r["retrieval_type"] = "semantic"
            r["score"] = r["combined_score"]
            all_results.append(r)
            
        for r in ner_results:
            r["combined_score"] = ner_weight * r.get("norm_similarity", 0.0)
            r["retrieval_type"] = "ner_entity"
            r["score"] = r["combined_score"]
            all_results.append(r)
        
        return sorted(all_results, key=lambda x: x["combined_score"], reverse=True)
       
 
    def index_entities_from_scorables(self, scorables: list, batch_size: int = 10) -> int:
        """
        Index entities from a list of scorables using NER Retriever.
        
        Args:
            scorables: List of Scorable objects to index
            batch_size: Number of scorables to process at once
            
        Returns:
            int: Total number of entities indexed
        """
        if not self.ner_enabled or not self.ner_retriever:
            return 0
            
        total_indexed = 0
        try:
            # Process in batches to avoid memory issues
            for i in range(0, len(scorables), batch_size):
                batch = scorables[i:i+batch_size]
                count = self.ner_retriever.index_scorables(batch)
                total_indexed += count
                
            self.ner_index_initialized = True
            if self.logger:
                self.logger.log("NERIndexingComplete", {
                    "scorables_processed": len(scorables),
                    "entities_indexed": total_indexed,
                    "batch_size": batch_size
                })
            return total_indexed
        except Exception as e:
            if self.logger:
                self.logger.log("NERIndexingFailed", {
                    "error": str(e),
                    "scorables_count": len(scorables),
                    "batch_size": batch_size
                })
            return 0
    
    def search_ner_entities(self, query: str, k: int = 5, min_similarity: float = 0.6) -> list:
        """
        Search for entities matching a type description using NER Retriever.
        
        Args:
            query: Type description (e.g., "financial derivative")
            k: Number of results to return
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of entity results with metadata
        """
        if not self.ner_enabled or not self.ner_retriever:
            return []
            
        try:
            results = self.ner_retriever.retrieve_entities(
                query=query,
                k=k,
                min_similarity=min_similarity
            )
            
            # Add additional metadata from our store
            for result in results:
                # Get the full scorable text if available
                if "scorable_id" in result:
                    try:
                        with self.conn.cursor() as cur:
                            cur.execute(
                                "SELECT text FROM scorables WHERE id = %s",
                                (result["scorable_id"],)
                            )
                            row = cur.fetchone()
                            if row:
                                result["full_text"] = row[0]
                    except Exception as e:
                        if self.logger:
                            self.logger.log("NERMetadataFetchFailed", {
                                "error": str(e),
                                "scorable_id": result["scorable_id"]
                            })
            
            if self.logger:
                self.logger.log("NERSearchResults", {
                    "query": query,
                    "results_count": len(results),
                    "top_results": [r["entity_text"] for r in results[:3]] if results else []
                })
                
            return results
        except Exception as e:
            if self.logger:
                self.logger.log("NERSearchFailed", {
                    "error": str(e),
                    "query": query
                })
            return []
    
    def train_ner_projection(self, scorables: list, max_triplets: int = 1000, 
                           batch_size: int = 32, epochs: int = 3, lr: float = 1e-4) -> bool:
        """
        Train the NER Retriever's projection network using contrastive learning.
        
        Args:
            scorables: List of Scorable objects to generate triplets from
            max_triplets: Maximum number of triplets to generate
            batch_size: Training batch size
            epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            bool: True if training succeeded
        """
        if not self.ner_enabled or not self.ner_retriever:
            return False
            
        try:
            # Generate triplets from scorables
            triplets = self.ner_retriever.generate_triplets(scorables, max_triplets)
            
            if not triplets:
                if self.logger:
                    self.logger.log("NERTrainingSkipped", {
                        "reason": "no_triplets_generated",
                        "scorables_count": len(scorables)
                    })
                return False
                
            # Train the projection network
            self.ner_retriever.train_projection(
                triplets=triplets,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr
            )
            
            # Re-index entities with the improved projection
            self.index_entities_from_scorables(scorables)
            
            if self.logger:
                self.logger.log("NERTrainingComplete", {
                    "triplets": len(triplets),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr
                })
            return True
        except Exception as e:
            if self.logger:
                self.logger.log("NERTrainingFailed", {
                    "error": str(e)
                })
            return False
    
    def get_ner_index_stats(self) -> dict:
        """
        Get statistics about the NER index.
        
        Returns:
            Dict: Index statistics or empty dict if NER is not enabled
        """
        if not self.ner_enabled or not self.ner_retriever or not self.ner_retriever.index:
            return {"enabled": False, "initialized": False}
            
        try:
            stats = self.ner_retriever.index.get_stats()
            stats["enabled"] = True
            stats["initialized"] = self.ner_index_initialized
            return stats
        except Exception as e:
            if self.logger:
                self.logger.log("NERStatsFailed", {
                    "error": str(e)
                })
            return {"enabled": True, "initialized": False, "error": str(e)}
    
    def is_ner_enabled(self) -> bool:
        """Check if NER Retriever is enabled and properly initialized."""
        return self.ner_enabled and self.ner_retriever is not None
    
    def get_ner_index_status(self) -> dict:
        """Get detailed status of the NER index."""
        return {
            "enabled": self.ner_enabled,
            "initialized": self.ner_index_initialized,
            "retriever_available": self.ner_retriever is not None,
            "index_stats": self.get_ner_index_stats()
        }
    

    def _determine_dynamic_weights(self, query: str, domain: str = None) -> Tuple[float, float]:
        """
        Determine optimal NER vs Semantic weights based on query type and domain.

        Returns:
            (ner_weight, semantic_weight) in [0,1].
        """
        query_lower = query.lower()

        # --- Keyword-based heuristics ---
        entity_keywords = ["who", "what", "where", "when", "which", "name", "list", "find"]
        conceptual_keywords = ["how", "why", "explain", "describe", "understand", "relationship"]

        entity_count = sum(1 for kw in entity_keywords if kw in query_lower)
        conceptual_count = sum(1 for kw in conceptual_keywords if kw in query_lower)

        # --- Base weights ---
        if entity_count > conceptual_count:
            ner_weight, semantic_weight = 0.7, 0.3
        elif conceptual_count > entity_count:
            ner_weight, semantic_weight = 0.3, 0.7
        else:
            ner_weight, semantic_weight = 0.5, 0.5

        # --- Domain adjustments ---
        domain_weights = {
            "legal": (0.6, 0.4),       # legal docs → entity precision matters
            "scientific": (0.55, 0.45),
            "creative": (0.4, 0.6),    # conceptual similarity more important
        }

        if domain and domain in domain_weights:
            ner_weight, semantic_weight = domain_weights[domain]

        # Clamp to valid range
        ner_weight = max(0.0, min(1.0, ner_weight))
        semantic_weight = max(0.0, min(1.0, semantic_weight))

        self.logger.log("DynamicWeighting", {
            "query": query[:60],
            "ner_weight": ner_weight,
            "semantic_weight": semantic_weight,
            "domain": domain or "unknown"
        })

        return ner_weight, semantic_weight


    def _calibrate_confidence(self, semantic_results: List[Dict], ner_results: List[Dict], query: str) -> None:
        """Calibrate confidence between systems based on historical performance"""
        # Get domain from query or context
        domain = self._get_current_domain(query)
        
        # Load historical calibration data
        calibration = self._load_calibration_data(domain)
        
        # Apply domain-specific calibration
        for r in semantic_results:
            r["calibrated_score"] = self._apply_calibration(
                r["norm_score"], 
                calibration.get("semantic", {})
            )
        
        for r in ner_results:
            r["calibrated_similarity"] = self._apply_calibration(
                r["norm_similarity"], 
                calibration.get("ner", {})
            )

    def _apply_calibration(self, score: float, calibration: Dict) -> float:
        """Apply polynomial calibration based on historical accuracy"""
        if not calibration or "coefficients" not in calibration:
            return score
        
        # Apply polynomial transformation
        poly = np.poly1d(calibration["coefficients"])
        return float(poly(score))

    def _get_current_domain(self, query: str) -> str:
        """Determine domain from query using classifier"""
        if not hasattr(self, '_domain_classifier'):
            from stephanie.analysis.scorable_classifier import ScorableClassifier
            self._domain_classifier = ScorableClassifier(
                memory=None,
                embed_fn=self.get_or_create,
                logger=self.logger,
                config_path=self.cfg.get("domain_config", "config/domain/seeds.yaml")
            )
        
        return self._domain_classifier.classify(query)

    def _load_calibration_data(self, domain: str) -> Dict:
        """Load historical calibration data for the given domain"""
        calibration_path = f"data/calibration/{domain}_calibration.json"
        
        if not os.path.exists(calibration_path):
            # Fallback to general calibration
            calibration_path = "data/calibration/general_calibration.json"
            
            if not os.path.exists(calibration_path):
                # Default calibration coefficients
                return {
                    "semantic": {"coefficients": [1.0, 0.0]},
                    "ner": {"coefficients": [1.0, 0.0]}
                }
        try:
            with open(calibration_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.log("CalibrationLoadFailed", {
                "error": str(e),
                "path": calibration_path
            })
            return {
                "semantic": {"coefficients": [1.0, 0.0]},
                "ner": {"coefficients": [1.0, 0.0]}
            }
    
    def train_calibration(self, historical_data: List[Dict], domain: str = "general"):
        """
        Train calibration model using historical performance data.
        
        Args:
            historical_data: List of {query, expected_results, actual_results, accuracy}
            domain: Domain to train calibration for
        """
        # Group data by system type
        semantic_data = [d for d in historical_data if d["system"] == "semantic"]
        ner_data = [d for d in historical_data if d["system"] == "ner"]
        
        # Train calibration models
        semantic_calibration = self._train_calibration_model(semantic_data)
        ner_calibration = self._train_calibration_model(ner_data)
        
        # Save to disk
        calibration_data = {
            "semantic": semantic_calibration,
            "ner": ner_calibration,
            "timestamp": datetime.now().isoformat()
        }
        
        os.makedirs("data/calibration", exist_ok=True)
        with open(f"data/calibration/{domain}_calibration.json", "w") as f:
            json.dump(calibration_data, f, indent=2)
        
        self.logger.log("CalibrationTrained", {
            "domain": domain,
            "semantic_rmse": semantic_calibration.get("rmse", 0),
            "ner_rmse": ner_calibration.get("rmse", 0)
        })

    def _train_calibration_model(self, data: List[Dict]) -> Dict:
        """Train polynomial calibration model for a system"""
        if not data:
            return {"coefficients": [1.0, 0.0]}
        
        # Extract scores and actual accuracy
        scores = [d["score"] for d in data]
        accuracy = [d["accuracy"] for d in data]
        
        # Fit polynomial (degree 2 works well)
        try:
            coeffs = np.polyfit(scores, accuracy, 2)
            # Calculate RMSE for evaluation
            predicted = np.polyval(coeffs, scores)
            rmse = np.sqrt(np.mean((predicted - accuracy) ** 2))
            
            return {
                "coefficients": coeffs.tolist(),
                "rmse": float(rmse),
                "sample_size": len(data)
            }
        except Exception as e:
            self.logger.log("CalibrationTrainingFailed", {"error": str(e)})
            return {"coefficients": [1.0, 0.0]}