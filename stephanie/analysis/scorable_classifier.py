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

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

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
    """
    
    def __init__(self, memory, logger, config_path="config/domain/seeds.yaml", metric="cosine", embed_fn=None):
        """
        Initialize the domain classifier with configuration and metrics.
        
        Args:
            memory: Interface for embedding storage and retrieval
            logger: Logging interface for debug messages
            config_path: Path to YAML configuration file with domain seeds
            metric: Distance metric ("cosine", "huber", or "euclidean")
        """
        self.memory = memory
        self.logger = logger
        self.metric = metric
        self.embed_fn = embed_fn

        # Log initialization with configuration details
        self.logger.log("DomainClassifierInit", {
            "config_path": config_path, 
            "metric": metric,
            "message": "Initializing domain classifier"
        })
        
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
            
        except Exception as e:
            self.logger.log("DomainConfigError", {
                "error": str(e),
                "config_path": config_path,
                "message": "Failed to load domain configuration"
            })
            raise

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
        self.logger.log("DistanceCalculation", {
            "metric": "cosine",
            "similarity": similarity,
            "message": "Calculated cosine similarity"
        })
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
        self.logger.log("DistanceCalculation", {
            "metric": "euclidean",
            "distance": distance,
            "negative_distance": negative_distance,
            "message": "Calculated Euclidean distance"
        })
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
        
        self.logger.log("DistanceCalculation", {
            "metric": "huber",
            "huber_loss": huber_loss,
            "negative_huber": negative_huber,
            "delta": delta,
            "message": "Calculated Huber distance"
        })
        
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
        
        self.logger.log("CentroidCalculationStart", {
            "message": "Starting centroid calculation for all domains"
        })
        
        for domain, details in self.domains.items():
            seeds = details.get("seeds", [])
            seed_embs = []
            
            # Get embeddings for all seed phrases
            for seed in seeds:
                if self.embed_fn:
                    embedding = self.embed_fn(seed)
                else:
                    embedding = self.memory.embedding.get_or_create(seed)
                seed_embs.append(embedding)
                total_seeds += 1
            
            # Calculate centroid as mean of all seed embeddings
            if seed_embs:
                centroids[domain] = np.mean(seed_embs, axis=0)
                logger.debug("DomainCentroidCalculated"
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
            "message": "Completed centroid calculation for all domains"
        })
        
        return centroids

    def classify(self, text: str, top_k: int = 3, min_value: float = 0.7, context: dict = None) -> List[Tuple[str, float]]:
        """
        Classify text into domains based on embedding similarity.
        
        Args:
            text: Input text to classify
            top_k: Number of top domains to return
            min_value: Minimum similarity score to consider a valid match
            context: Optional context dictionary with domain hints
            
        Returns:
            List of (domain, score) tuples sorted by score descending
        """
        self.logger.log("ClassificationStart", {
            "text_snippet": text[:100] + "..." if len(text) > 100 else text,
            "top_k": top_k,
            "min_value": min_value,
            "has_context": context is not None,
            "message": "Starting domain classification"
        })
        cache_key = (
            text,
            self.metric,
            tuple(sorted(self.centroids.keys()))  # domains being compared against
        )
        if cache_key in self._classification_cache:
            return self._classification_cache[cache_key]

        # Get embedding for input text
        if self.embed_fn:
            emb = self.embed_fn(text)
        else:
            emb = self.memory.embedding.get_or_create(text)
        self.logger.log("TextEmbeddingCreated", {
            "text_length": len(text),
            "embedding_shape": emb.shape,
            "message": "Created embedding for input text"
        })
        
        scores = {}
        
        # Calculate similarity to each domain centroid
        for domain, centroid in self.centroids.items():
            if self.metric == "cosine":
                score = self._cosine_distance(emb, centroid)
            elif self.metric == "euclidean":
                score = self._euclidean_distance(emb, centroid)
            elif self.metric == "huber":
                score = self._huber_distance(emb, centroid)
            else:
                error_msg = f"Unknown metric: {self.metric}"
                self.logger.log("ClassificationError", {
                    "error": error_msg,
                    "message": "Invalid distance metric specified"
                })
                raise ValueError(error_msg)
            
            scores[domain] = score
            
            self.logger.log("DomainSimilarityCalculated", {
                "domain": domain,
                "score": score,
                "metric": self.metric,
                "message": f"Calculated similarity for domain {domain}"
            })
        
        # Add context-specific tags with weight boost
        if context:
            goal_tags = self._extract_domains(context)
            self.logger.log("ContextProcessing", {
                "goal_tags": goal_tags,
                "message": "Processing context for additional domain hints"
            })
            
            for tag in goal_tags:
                if self.embed_fn:
                    tag_emb = self.embed_fn(tag)
                else:
                    tag_emb = self.memory.embedding.get_or_create(tag)

                # Boost context tags by 50% to prioritize them
                score = self._cosine_distance(emb, tag_emb) * 1.5
                # Keep the highest score if tag appears multiple times
                scores[tag] = max(scores.get(tag, 0.0), score)
                
                self.logger.log("ContextTagProcessed", {
                    "tag": tag,
                    "score": score,
                    "message": f"Processed context tag {tag}"
                })
        
        # Sort scores in descending order
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # compute as usual...
        top_matches = sorted_scores[:top_k]

        # cache result
        self._classification_cache[cache_key] = top_matches
        
        # Log if all scores are below minimum threshold
        if all(score < min_value for _, score in top_matches):
            self.logger.log("LowDomainScore", {
                "text_snippet": text[:100],
                "top_scores": top_matches,
                "min_value": min_value,
                "message": "All domain scores below minimum threshold"
            })
        else:
            self.logger.log("ClassificationComplete", {
                "top_matches": top_matches,
                "message": "Domain classification completed successfully"
            })
        
        return top_matches

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