# stephanie/memory/base_embedding_store.py
import hashlib
import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

import torch

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

    def search_related_scorables(self, query: str, target_type: str = "document", 
                            top_k: int = 10, with_metadata: bool = True,
                            include_ner: bool = True):
        """
        Search for scorables with configurable NER integration.
        
        Args:
            hybrid_mode: How to integrate NER results
                - "merge": Mix semantic + NER results (default)
                - "ner_only": Return only NER results
                - "semantic_only": Return only semantic results
                - "separate": Return both as separate lists
        """
        hybrid_mode = self.cfg.get("ner_hybrid_mode", "merge")
        
        # Get semantic results
        semantic_results = []
        if hybrid_mode in ["merge", "semantic_only", "separate"]:
            semantic_results = self._get_semantic_results(query, target_type, top_k, with_metadata)
        
        # Get NER results
        ner_results = []
        if include_ner and self.ner_enabled and hybrid_mode in ["merge", "ner_only", "separate"]:
            ner_results = self._get_ner_results(query, top_k)
        
        # Handle different integration modes
        if hybrid_mode == "merge":
            combined = self._combine_results(semantic_results, ner_results)
            combined.sort(key=lambda x: x["combined_score"], reverse=True)
            return combined[:top_k]
        
        elif hybrid_mode == "separate":
            return {
                "semantic": semantic_results[:top_k],
                "ner": ner_results[:top_k],
                "combined_score": self._combine_results(semantic_results, ner_results)[:top_k]
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
        """Normalize scores to [0,1] within their distribution"""
        if not results:
            return
            
        scores = [r.get(score_key, 0.0) for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score - min_score < 1e-8:
            for r in results:
                r[f"norm_{score_key}"] = 0.5
        else:
            for r in results:
                r[f"norm_{score_key}"] = (r.get(score_key, 0.0) - min_score) / (max_score - min_score + 1e-8)

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

    def _combine_results(self, semantic_results: List[Dict], ner_results: List[Dict]) -> List[Dict]:
        """Combine semantic and NER results into a unified, weighted list."""
        
        # Normalize scores separately
        self._normalize_scores(semantic_results, "score")
        self._normalize_scores(ner_results, "similarity")
        
        # Standardize schema
        semantic_results = self._standardize_result_schema(semantic_results)
        ner_results = self._standardize_result_schema(ner_results)
        
        # Apply weights
        ner_weight = self.cfg.get("ner_weight", 0.6)
        semantic_weight = 1.0 - ner_weight
        
        all_results = []
        for r in semantic_results:
            # Use normalized score
            norm_score = r.get("norm_score", 0.0)
            r["combined_score"] = semantic_weight * norm_score
            r["retrieval_type"] = "semantic"
            r["score"] = r["combined_score"]  # unify under `score`
            all_results.append(r)
            
        for r in ner_results:
            norm_similarity = r.get("norm_similarity", 0.0)
            r["combined_score"] = ner_weight * norm_similarity
            r["retrieval_type"] = "ner_entity"
            r["score"] = r["combined_score"]  # unify under `score`
            all_results.append(r)
        
        return all_results

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