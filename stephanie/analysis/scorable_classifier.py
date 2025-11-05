# stephanie/analysis/scorable_classifier.py
"""
Domain Classifier Implementation
================================

A lightweight implementation of Domain2Vec concepts for domain classification.
This module provides seed-based domain classification using embedding similarity.

Paper Reference: Domain2Vec (Zhang et al., 2025, arXiv:2506.10952)
- Represents datasets as domain vectors (linear combinations of meta-domains)
- Uses Distribution Alignment Assumption (DA²) for optimal data mixtures
- Our implementation uses seed phrases as meta-domain proxies

Key Features:
- Multiple distance metrics (cosine, Euclidean, Huber)
- Dynamic context integration for goal-specific domains
- Centroid-based domain representation (average of seed embeddings)
- Comprehensive logging for debugging and analysis

Usage:
    classifier = ScorableClassifier(memory, logger)
    domains = classifier.classify(text, top_k=3, context=goal_context)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_similarity

from stephanie.services.bus.bus_protocol import BusProtocol
from stephanie.services.bus.idempotency import InMemoryIdempotencyStore

_logger = logging.getLogger(__name__)


class ScorableClassifier:
    """
    Seed-based domain classifier implementing Domain2Vec concepts.
    
    This class provides domain classification using embedding similarity
    to predefined domain seeds. It supports multiple distance metrics
    and can incorporate dynamic context from goals or other sources.
    
    Attributes:
        memory: Embedding storage and retrieval system
        logger: Logging interface for debug and operational messages
        metric: Distance metric used for similarity calculation
        domains: Dictionary of domain configurations from YAML
        centroids: Precomputed centroid embeddings for each domain
        bus: Optional event bus for distributed caching
        idempotency_store: For caching results across restarts
    """
    
    def __init__(self, memory, logger, config_path="config/domain/seeds.yaml", metric="cosine", bus: Optional[BusProtocol] = None):
        """
        Initialize the domain classifier with configuration and metrics.
        
        Args:
            memory: Interface for embedding storage and retrieval
            logger: Logging interface for debug messages
            config_path: Path to YAML configuration file with domain seeds
            metric: Distance metric ("cosine", "huber", or "euclidean")
            bus: Optional event bus for distributed caching
        """
        self.memory = memory
        self.logger = logger
        self.metric = metric
        self.bus = bus
        self.idempotency_store = None
        
        # Log initialization with configuration details
        _logger.debug("DomainClassifierInit"
            f"config_path: {config_path}"
            f"metric {metric}"
            f"bus_available: {bool(bus)}"
            "message: Initializing domain classifier"
        )
        
        try:
            # Load domain configuration from YAML file
            with open(config_path, "r") as f:
                self.domain_config = yaml.safe_load(f)
            
            self.domains = self.domain_config.get("domains", {})
            
            # Log domain configuration details
            self.logger.log("DomainConfigLoaded", {
                "num_domains": len(self.domains),
                "domains": list(self.domains.keys()),
                "message": "Domain configuration loaded successfully"
            })
            
            # Keys that may contain dynamic domain hints in context
            self.domain_keys = ["tags", "domains", "keywords", "attributes", "concepts"]
            
            # Precompute domain centroids from seed embeddings
            self.centroids = self._domain_centroids()
            self._classification_cache = {}
            
            # Set up idempotency store if bus is available
            if bus:
                self.idempotency_store = bus.idempotency_store
                self.logger.info("Using distributed idempotency store from bus")
            else:
                self.idempotency_store = InMemoryIdempotencyStore()
                self.logger.info("Using in-memory idempotency store")
            
            # Set up KV cache if available
            self._kv = self._attach_kv_cache()
            self._kv_ttl_sec = 3600  # 1 hour default TTL

        except Exception as e:
            self.logger.log("DomainConfigError", {
                "error": str(e),
                "config_path": config_path,
                "message": "Failed to load domain configuration"
            })
            raise

    # ----------------------- NEW: Cache helpers -----------------------

    def _cache_key(self, text: str, metric: str, top_k: int, min_value: float, context_tags: tuple) -> str:
        """Generate a stable cache key for classification results."""
        payload = {
            "t": hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16],
            "m": metric,
            "k": int(top_k),
            "min": float(min_value),
            "ctx": list(context_tags),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    async def _get_cached_results(self, key: str) -> Optional[List[Tuple[str, float]]]:
        """Retrieve cached results if available and valid."""
        if not self.idempotency_store:
            return None
            
        try:
            # Check if we've seen this key before
            if await self.idempotency_store.seen(key):
                # In a real implementation, you'd fetch the actual results here
                # For now, we'll just return None to force recomputation
                return None
        except Exception as e:
            _logger.warning(f"Cache lookup failed: {str(e)}")
            
        return None

    async def _cache_results(self, key: str, results: List[Tuple[str, float]]) -> None:
        """Cache classification results."""
        if not self.idempotency_store:
            return
            
        try:
            # Mark this key as processed
            await self.idempotency_store.mark(key)
            # In a real implementation, you'd store the results here
        except Exception as e:
            _logger.warning(f"Failed to cache results: {str(e)}")


    def classify(
        self, text: str, top_k: int = 3, min_value: float = 0.7, context: dict = None
    ) -> List[Tuple[str, float]]:
        """
        Domain classification with caching (LRU + NATS KV).
        """
        # Extract context tags
        context_tags = tuple(self._extract_domains(context)) if context else tuple()
        key = self._cache_key(text, self.metric, top_k, min_value, context_tags)

        # 1) Check NATS KV first
        kv_hit = self._kv_get(key)
        if kv_hit:
            self.logger.log("DomainCacheHit", {
                "backend": "nats_kv",
                "items": len(kv_hit),
                "message": "Using cached domain classification results"
            })
            return kv_hit

        # 2) Check in-process LRU cache
        if key in self._classification_cache:
            self.logger.log("DomainCacheHit", {
                "backend": "lru",
                "items": len(self._classification_cache[key]),
                "message": "Using LRU cached results"
            })
            return self._classification_cache[key]

        # 3) Compute fresh classification
        results = self._classify_cached(text, self.metric, top_k, min_value, context_tags)

        # 4) Write-through to both caches
        if results:
            self._classification_cache[key] = results
            self._kv_put(key, results)

        # 5) Log as usual
        if not results:
            return []
        if all(score < min_value for _, score in results):
            self.logger.log("LowDomainScore", {
                "text_snippet": text[:100],
                "top_scores": results,
                "min_value": min_value,
                "message": "All domain scores below minimum threshold"
            })
        else:
            self.logger.log("ClassificationComplete", {
                "top_matches": results,
                "message": "Domain classification completed successfully"
            })

        return results
    # ----------------------- NEW: KV wiring -----------------------

    def _attach_kv_cache(self):
        """
        Best-effort: ask the bus for a KV bucket named 'domain.classify.cache'.
        Must be a synchronous facade (background loop handled by the bus).
        """
        try:
            bus = getattr(self.memory, "bus", None) or getattr(self, "kb", None)
            if bus and hasattr(bus, "get_kv"):
                kv = bus.get_kv(
                    bucket="domain.classify.cache",
                    description="Domain classifier results (1h TTL)",
                    max_age_seconds=3600
                )
                return kv
        except Exception as e:
            self.logger.log("DomainKVUnavailable", {"error": str(e)})
        return None

    def _kv_get(self, key: str) -> Optional[List[Tuple[str, float]]]:
        """Synchronous KV get; returns parsed results or None."""
        if not self._kv:
            return None
        try:
            raw = self._kv.get(key)          # bytes | None
            if not raw:
                return None
            doc = json.loads(raw.decode("utf-8"))
            # Optional client-side staleness guard in case bucket has no TTL
            if (time.time() - float(doc.get("ts", 0))) > self._kv_ttl_sec:
                return None
            results = doc.get("results")
            if isinstance(results, list):
                return [(str(d), float(s)) for d, s in results]
        except Exception as e:
            self.logger.log("DomainKVGetError", {"error": str(e)})
        return None

    def _kv_put(self, key: str, results: List[Tuple[str, float]]) -> None:
        """Best-effort KV put; non-fatal on error."""
        if not self._kv:
            return
        try:
            payload = {
                "ver": 1,
                "ts": time.time(),
                "results": results,
            }
            self._kv.put(key, json.dumps(payload).encode("utf-8"))
        except Exception as e:
            self.logger.log("DomainKVPu tError", {"error": str(e)})

    # -------------------- existing code with hooks --------------------

    @lru_cache(maxsize=2048)
    def _classify_cached(self, text: str, metric: str, top_k: int, min_value: float, context_tags: Tuple[str, ...]) -> List[Tuple[str, float]]:
        # Guard against empty text
        if not text or not text.strip():
            self.logger.log("ClassificationSkippedEmpty", {"message": "Skipped classification because input text is empty"})
            return []

        # 1) Persistent KV check (fast-return on hit)
        key = self._cache_key(text, metric, top_k, min_value, context_tags)
        kv_hit = self._kv_get(key)
        if kv_hit:
            self.logger.log("DomainCacheHit", {"backend": "nats_kv", "items": len(kv_hit)})
            return kv_hit

        # 2) Compute as before
        emb = self.memory.embedding.get_or_create(text)
        # self.logger.log("TextEmbeddingCreated", {"text_length": len(text), "embedding_shape": len(emb), "message": "Created embedding for input text"})
        scores = {}
        for domain, centroid in self.centroids.items():
            if metric == "cosine":
                score = self._cosine_distance(emb, centroid)
            elif metric == "euclidean":
                score = self._euclidean_distance(emb, centroid)
            elif metric == "huber":
                score = self._huber_distance(emb, centroid)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            scores[domain] = score

        for tag in context_tags:
            tag_emb = self.memory.embedding.get_or_create(tag)
            score = self._cosine_distance(emb, tag_emb) * 1.5
            scores[tag] = max(scores.get(tag, 0.0), score)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = sorted_scores[:top_k]

        # 3) Best-effort persistent write-through
        self._kv_put(key, results)

        return results

    def _cosine_distance(self, emb1: np.array, emb2: np.array) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        similarity = float(cosine_similarity([emb1], [emb2])[0][0])
        _logger.debug("DistanceCalculation"
            "metric: cosine"
            f"similarity: {similarity}"
            "message Calculated cosine similarity"
        )
        return similarity

    def _euclidean_distance(self, emb1: np.array, emb2: np.array) -> float:
        """
        Calculate negative Euclidean distance between two embeddings.
        
        Note: Returns negative value so higher scores indicate better matches,
        consistent with other distance metrics in this class.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Negative Euclidean distance (higher values indicate closer matches)
        """
        distance = np.linalg.norm(emb1 - emb2)
        negative_distance = -distance  # Convert to negative for consistency
        _logger.debug("DistanceCalculation"
            "metric: euclidean"
            f"distance: {distance}"
            f"negative_distance: {negative_distance}"
            "message: Calculated Euclidean distance"
        )
        return negative_distance

    def _huber_distance(self, emb1: np.array, emb2: np.array, delta: float = 1.0) -> float:
        """
        Calculate Huber loss-based similarity between two embeddings.
        
        Huber loss is less sensitive to outliers than Euclidean distance.
        This implementation follows Domain2Vec's approach.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            delta: Threshold parameter for Huber loss
            
        Returns:
            Negative Huber loss (higher values indicate closer matches)
        """
        diff = emb1 - emb2
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        huber_loss = np.mean(0.5 * quadratic**2 + delta * linear)
        negative_huber = -huber_loss  # Convert to negative for consistency
        
        _logger.debug("DistanceCalculation"
            "metric: huber"
            f"huber_loss: {huber_loss}"
            f"negative_huber: {negative_huber}"
            f"delta: {delta}"
            "message: Calculated Huber distance"
        )
        
        return negative_huber

    def _domain_centroids(self) -> Dict[str, np.array]:
        """
        Compute centroid embeddings for each domain from seed phrases.
        
        For each domain, calculates the average embedding of all its seed phrases.
        This creates a more robust representation than individual seeds.
        
        Returns:
            Dictionary mapping domain names to centroid embeddings
        """
        centroids = {}
        total_seeds = 0
        
        start = time.time()
        self.logger.log("CentroidCalculationStart", {
            "message": "Starting centroid calculation for all domains"
        })
        
        for domain, details in self.domains.items():
            seeds = details.get("seeds", [])
            seed_embs = []
            
            # Get embeddings for all seed phrases
            for seed in seeds:
                embedding = self.memory.embedding.get_or_create(seed)
                seed_embs.append(embedding)
                total_seeds += 1
            
            # Calculate centroid as mean of all seed embeddings
            if seed_embs:
                centroids[domain] = np.mean(seed_embs, axis=0)
                _logger.debug("DomainCentroidCalculated"
                    f"domain : {domain}"
                    f"num_seeds : {len(seeds)}"
                    f"centroid_shape : {centroids[domain].shape}"
                    f"message : Calculated centroid for domain {domain}"
                )
            else:
                self.logger.log("DomainWithoutSeeds", {
                    "domain": domain,
                    "message": f"Domain {domain} has no seeds, skipping"
                })
        
        self.logger.log("CentroidCalculationComplete", {
            
            "total_domains": len(centroids),
            "total_seeds": total_seeds,
            "duration_sec": time.time() - start, 
            "message": "Completed centroid calculation for all domains"
        })
        
        return centroids

    def _extract_domains(self, context: dict) -> List[str]:
        """
        Extract domain-related tags from a context dictionary.
        
        Args:
            context: Dictionary potentially containing domain hints
            
        Returns:
            List of domain tags extracted from the context
        """
        domains = []
        
        if isinstance(context, dict):
            for key in self.domain_keys:
                if key in context and isinstance(context[key], (list, tuple)):
                    domains.extend([str(t).strip() for t in context[key] if t])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_domains = []
        for domain in domains:
            if domain not in seen:
                seen.add(domain)
                unique_domains.append(domain)
        
        self.logger.log("DomainsExtractedFromContext", {
            "context_keys": list(context.keys()) if context else [],
            "extracted_domains": unique_domains,
            "message": "Extracted domains from context"
        })
        
        return unique_domains