"""
NER Retriever Module

This module implements a Named Entity Recognition (NER) based retrieval system for legal case documents.
It combines transformer-based entity detection, embedding generation, and approximate nearest neighbor
search to enable efficient entity retrieval from casebooks.

Key Components:
1. NERRetrieverProjection: Neural network for projecting entity embeddings to lower dimension
2. EntityDetector: BERT-based named entity recognition with fallback heuristic
3. AnnoyIndex: Wrapper for Annoy approximate nearest neighbor index
4. NERRetrieverEmbedder: Main class handling entity detection, embedding, and retrieval

Primary Features:
- Entity extraction from legal documents using BERT-NER
- Contrastive learning for entity type representations
- Efficient similarity search using Annoy index
- Batch processing for large document collections

Typical Usage:
1. Initialize NERRetrieverEmbedder with pre-trained model
2. Index casebooks using index_casebook_entities()
3. Query entities using retrieve_entities()

Dependencies:
- PyTorch, Transformers, Annoy, Numpy
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta
import random
import re

import annoy
import numpy as np
# stephanie/scoring/model/ner_retriever.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline

from stephanie.scoring.scorable import Scorable

logger = logging.getLogger(__name__)

# -------------------------------
# Projection Network
# -------------------------------
class NERRetrieverProjection(nn.Module):
    """Neural projection network to reduce embedding dimensionality while preserving semantic information."""
    def __init__(self, input_dim: int = 4096, output_dim: int = 500, dropout: float = 0.1):
        super().__init__()
        # Two-layer MLP with SiLU activation and dropout
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(output_dim, output_dim)

        # Initialize weights using Xavier uniform initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # Forward pass through projection layers
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=-1)  # L2 normalize output for cosine similarity


# -------------------------------
# Entity Detector
# -------------------------------
class EntityDetector:
    """Detects named entities in text using BERT-NER with fallback to heuristic rules."""
    def __init__(self, device: str = "cuda"):
        try:
            # Initialize HuggingFace NER pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple",
                device=0 if device == "cuda" else -1
            )
            logger.info("Initialized NER pipeline with dslim/bert-base-NER")
        except Exception as e:
            logger.error(f"Failed to init NER pipeline: {e}")
            self.ner_pipeline = None

    def detect_entities(self, text: str) -> List[Tuple[int, int, str]]:
        """Detect entities in text using either BERT-NER or fallback heuristic."""
        if not text or len(text.strip()) < 2:
            return []
        if self.ner_pipeline:
            try:
                # Use BERT-NER if available
                results = self.ner_pipeline(text)
                return [(r["start"], r["end"], self._map_entity_type(r["entity_group"])) for r in results]
            except Exception as e:
                logger.warning(f"NER pipeline failed: {e}")
        # Fallback to heuristic-based detection
        return self._heuristic_entity_detection(text)

    def _map_entity_type(self, group: str) -> str:
        """Map BERT-NER entity types to simplified categories."""
        return {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC", "MISC": "MISC"}.get(group, "UNKNOWN")

    def _heuristic_entity_detection(self, text: str) -> List[Tuple[int, int, str]]:
        """Fallback entity detection using capitalization rules."""
        entities = []
        for word in text.split():
            # Simple heuristic: capitalized words longer than 2 characters
            if word and word[0].isupper() and len(word) > 2:
                start = text.find(word)
                if start != -1:
                    entities.append((start, start + len(word), "UNKNOWN"))
        return entities


# -------------------------------
# Annoy Index Wrapper
# -------------------------------
class AnnoyIndex:
    """Wrapper for Annoy approximate nearest neighbor index with metadata management."""
    def __init__(self, dim: int = 500, index_path: str = "data/ner_retriever/index"):
        self.dim = dim
        self.index_path = index_path
        self.index = annoy.AnnoyIndex(dim, "angular")  # Angular distance for cosine similarity
        self.metadata = []
        self._load_index()

    def _load_index(self):
        """Load existing index and metadata from disk if available."""
        index_file = f"{self.index_path}.ann"
        meta_file = f"{self.index_path}_metadata.json"
        if os.path.exists(index_file) and os.path.exists(meta_file):
            try:
                self.index.load(index_file)
                with open(meta_file, "r") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded Annoy index with {self.index.get_n_items()} items")
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self.index = annoy.AnnoyIndex(self.dim, "angular")
                self.metadata = []
        else:
            logger.info("Starting with empty Annoy index")

    def add(self, embeddings: np.ndarray, metadata_list: List[Dict], n_trees: int = 10):
        """Add new embeddings with metadata and rebuild the Annoy index."""
        if not len(embeddings):
            return
        
        n_existing = self.index.get_n_items()
        added_count = 0
        
        for i, (vec, meta) in enumerate(zip(embeddings, metadata_list)):
            # Deduplicate by (scorable_id, entity_text)
            if any(
                m.get("scorable_id") == meta.get("scorable_id") and
                m.get("entity_text") == meta.get("entity_text")
                for m in self.metadata
            ):
                continue
            
            self.index.add_item(n_existing + added_count, vec)
            self.metadata.append(meta)
            added_count += 1
        
        if added_count > 0:
            # Rebuild with fresh trees
            self.index.build(n_trees)
            self._save_index()
            logger.info(f"Annoy index rebuilt with {self.index.get_n_items()} items "
                        f"(added {added_count}, skipped {len(metadata_list) - added_count})")

    def search(self, query: np.ndarray, k: int = 10):
        """Search index for nearest neighbors to query vector."""
        if self.index.get_n_items() == 0:
            return []
            
        query = query.astype(np.float32)
        indices, distances = self.index.get_nns_by_vector(query, k, include_distances=True)
        
        results = []
        for idx, dist in zip(indices, distances):
            # Convert angular distance to cosine similarity
            sim = 1 - (dist ** 2) / 2
            if idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                meta["similarity"] = float(sim)
                meta["distance"] = float(dist)
                results.append(meta)
                
        return results

    def _save_index(self):
        """Save index and metadata with atomic file replacement."""
        temp_index = f"{self.index_path}.ann.tmp"
        temp_meta = f"{self.index_path}_metadata.json.tmp"
        
        try:
            # Save to temporary files first
            self.index.save(temp_index)
            with open(temp_meta, "w") as f:
                json.dump(self.metadata, f, indent=2)
                
            # Atomically replace old files
            os.replace(temp_index, f"{self.index_path}.ann")
            os.replace(temp_meta, f"{self.index_path}_metadata.json")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            # Clean up temp files on failure
            if os.path.exists(temp_index):
                os.remove(temp_index)
            if os.path.exists(temp_meta):
                os.remove(temp_meta)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index for monitoring."""
        entity_types = {}
        for meta in self.metadata:
            etype = meta.get("entity_type", "UNKNOWN")
            entity_types[etype] = entity_types.get(etype, 0) + 1
            
        return {
            "total_entities": len(self.metadata),
            "unique_scorables": len(set(m["scorable_id"] for m in self.metadata)),
            "entity_types": entity_types,
            "index_size": os.path.getsize(f"{self.index_path}.ann") if os.path.exists(f"{self.index_path}.ann") else 0,
            "avg_similarity": np.mean([m["similarity"] for m in self.metadata if "similarity" in m]) 
                            if self.metadata else 0
        }
    
    def validate(self) -> bool:
        """Validate that the index and metadata are consistent."""
        if self.index.get_n_items() != len(self.metadata):
            logger.error(f"Index validation failed: {self.index.get_n_items()} items in index, {len(self.metadata)} in metadata")
            return False
        return True


class NERRetrieverEmbedder:
    """Entity retriever using embedding store instead of custom projections."""

    def __init__(self, 
                 model_name="meta-llama/Llama-3-8b", 
                 layer=17,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 embedding_dim=500, 
                 index_path="data/ner_retriever/index",
                 projection_enabled=False, 
                 projection_dim=500, 
                 projection_dropout=0.1,
                 logger=None,
                 memory=None,   # <-- Fix 1: accept memory + logger
                 cfg=None):

        self.device = device
        self.logger = logger or logging.getLogger(__name__)  # <-- Fix 2: self.logger always available
        self.memory = memory
        self.cfg = cfg or {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, trust_remote_code=True
        ).to(device).eval()

        self.layer = layer
        self.embedding_dim = embedding_dim
        self.index = AnnoyIndex(dim=embedding_dim, index_path=index_path)
        self.entity_detector = EntityDetector(device)

        # optional projection
        self.projection_enabled = projection_enabled
        self.projection = None
        if projection_enabled:
            self.projection = NERRetrieverProjection(
                input_dim=self.model.config.hidden_size,
                output_dim=projection_dim,
                dropout=projection_dropout,
            ).to(device).eval()

        self.logger.info(
            f"NER Retriever initialized with {model_name} "
            f"layer {layer}, projection_enabled={projection_enabled}"
        )

    # ----------------------------
    # Embedding methods
    # ----------------------------
    def embed_entity(self, text: str, span: Tuple[int, int]) -> torch.Tensor:
        """Embed an entity span with robust character-to-token alignment."""
        if span[0] >= span[1] or span[0] < 0 or span[1] > len(text):
            return torch.zeros(self.projection.fc2.out_features if self.projection_enabled else self.model.config.hidden_size, 
                            device=self.device)
            
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.layer]
        
        # Get token indices for the span with robust fallbacks
        start_token = inputs.char_to_token(0, span[0])
        end_token = inputs.char_to_token(0, span[1] - 1)
        
        # Handle cases where char_to_token returns None
        if start_token is None:
            # Search backward from span start
            for offset in range(1, min(10, span[0] + 1)):
                start_token = inputs.char_to_token(0, span[0] - offset)
                if start_token is not None:
                    break
            # If still None, use the first meaningful token
            if start_token is None:
                start_token = 1  # Skip [CLS]
                
        if end_token is None:
            # Search forward from span end
            for offset in range(1, min(10, len(text) - span[1] + 1)):
                end_token = inputs.char_to_token(0, span[1] - 1 + offset)
                if end_token is not None:
                    break
            # If still None, set to start_token or next token
            if end_token is None or end_token < start_token:
                end_token = min(inputs.input_ids.shape[1] - 1, start_token + 2)
        
        # Ensure valid span
        start_token = max(1, start_token)  # Skip [CLS]
        end_token = min(inputs.input_ids.shape[1] - 1, max(start_token, end_token))
        
        # Pool across the entity span (mean pooling)
        entity_vec = hidden_states[0, start_token:end_token+1, :].mean(dim=0)
        
        # Project if enabled
        if self.projection_enabled and self.projection is not None:
            entity_vec = self.projection(entity_vec)

        return entity_vec

    # ----------------------------
    # Indexing
    # ----------------------------
    def index_scorables(self, scorables: List[Scorable]) -> int:
        """Index entities from scorables into Annoy with metadata."""
        new_embeddings, new_metadata = [], []
        for scorable in tqdm(scorables, desc="Indexing entities"):
            for start, end, entity_type in self.entity_detector.detect_entities(scorable.text):
                entity_text = scorable.text[start:end].strip()
                if len(entity_text) < 2:
                    continue
                try:
                    emb = self.embed_entity(scorable.text, (start, end))
                    new_embeddings.append(emb.detach().cpu().numpy())  # <-- Fix 3: always NumPy
                    new_metadata.append({
                        "scorable_id": str(scorable.id),
                        "scorable_type": scorable.target_type,
                        "entity_text": entity_text,
                        "start": start,
                        "end": end,
                        "entity_type": entity_type,
                        "source_text": scorable.text[:100] + "..."
                    })
                except Exception as e:
                    self.logger.error(f"Entity embedding failed: {e}")
        if new_embeddings:
            self.index.add(np.array(new_embeddings), new_metadata)
        return len(new_embeddings)

    def retrieve_entities(self, query: str, k: int = 5, min_similarity: float = 0.6, domain: str = None) -> List[Dict]:
        """Search for entities similar to the query with domain-aware calibration."""
        # Preprocess and embed the query
        query_emb = self.embed_type_query(query)
        
        # Search index
        results = self.index.search(query_emb, k*2)
        
        # Apply domain-specific calibration
        if domain is None:
            domain = self._get_current_domain(query)
        
        # Load calibration data
        calibration = self._load_calibration_data(domain)
        
        # Apply calibration to results
        for result in results:
            if "similarity" in result:
                # Use polynomial calibration
                if "ner" in calibration:
                    poly = np.poly1d(calibration["ner"]["coefficients"])
                    calibrated = float(poly(result["similarity"]))
                    # Apply system-specific constraints
                    if result["similarity"] > 0.8:
                        calibrated = min(1.0, calibrated * 1.05)
                    result["calibrated_similarity"] = max(0.0, min(1.0, calibrated))
        
        # Filter by similarity threshold (using calibrated if available)
        filtered_results = [
            r for r in results 
            if r.get("calibrated_similarity", r.get("similarity", 0.0)) >= min_similarity
        ][:k]
        
        # Log for monitoring
        if filtered_results:
            logger.info(f"Found {len(filtered_results)} entities matching '{query}'")
            for i, result in enumerate(filtered_results[:3]):  # Log top 3
                cal_sim = result.get("calibrated_similarity", result.get("similarity", 0.0))
                logger.debug(f"Top {i+1} match: '{result['entity_text']}' (sim={cal_sim:.4f})")
        else:
            logger.info(f"No entities found matching '{query}'")
            
        return filtered_results

    def collect_calibration_data(self, query: str, results: List[Dict], ground_truth: List[str], 
                            domain: str = None) -> None:
        """
        Collect calibration data by comparing results against ground truth.
        
        Args:
            query: User query
            results: Search results with scores
            ground_truth: List of relevant item IDs
            domain: Optional domain context
        """
        if not domain:
            domain = self._get_current_domain(query)
        
        # Calculate relevance for each result
        for result in results:
            is_relevant = str(result["id"]) in ground_truth
            system_type = "ner" if result.get("retrieval_type") == "ner_entity" else "semantic"
            
            # Log for calibration
            calibration_entry = {
                "query": query,
                "score": result.get("score", 0.0),
                "system": system_type,
                "is_relevant": is_relevant,
                "domain": domain,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to persistent storage
            self._save_calibration_entry(calibration_entry)
        
        # Log for monitoring
        self.logger.log("CalibrationDataCollected", {
            "query": query[:50] + "..." if len(query) > 50 else query,
            "domain": domain,
            "total_results": len(results),
            "relevant_count": sum(1 for r in results if str(r["id"]) in ground_truth)
        })

    def _save_calibration_entry(self, entry: Dict):
        """Save calibration entry to persistent storage"""
        # Create directory if needed
        os.makedirs("data/calibration/history", exist_ok=True)
        
        # Save as timestamped JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        with open(f"data/calibration/history/{timestamp}.json", "w") as f:
            json.dump(entry, f, indent=2)

    def _load_historical_data(self, domain: str = "general", days: int = 30) -> List[Dict]:
        """Load historical calibration data for the given domain"""
        history_dir = "data/calibration/history"
        if not os.path.exists(history_dir):
            return []
        
        # Filter by domain and time window
        historical_data = []
        cutoff_time = datetime.now() - timedelta(days=days)
        
        for filename in os.listdir(history_dir):
            if not filename.endswith(".json"):
                continue
                
            filepath = os.path.join(history_dir, filename)
            try:
                with open(filepath, "r") as f:
                    entry = json.load(f)
                    
                # Skip entries outside our time window
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time < cutoff_time:
                    continue
                    
                # Filter by domain
                if domain == "all" or entry.get("domain") == domain:
                    historical_data.append({
                        "score": entry["score"],
                        "accuracy": 1.0 if entry["is_relevant"] else 0.0,
                        "system": entry["system"]
                    })
            except Exception as e:
                self.logger.log("CalibrationDataLoadFailed", {
                    "error": str(e),
                    "file": filename
                })
        
        return historical_data
    
    def _load_calibration_data(self, domain: str) -> Dict:
        """Load historical calibration data for the given domain with fallbacks"""
        # Try domain-specific calibration
        domain_path = f"data/calibration/{domain}_calibration.json"
        if os.path.exists(domain_path):
            try:
                with open(domain_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.log("DomainCalibrationLoadFailed", {
                    "error": str(e),
                    "domain": domain
                })
        
        # Try general calibration
        general_path = "data/calibration/general_calibration.json"
        if os.path.exists(general_path):
            try:
                with open(general_path, "r") as f:
                    calibration = json.load(f)
                    self.logger.log("UsingFallbackCalibration", {
                        "domain": domain,
                        "fallback": "general"
                    })
                    return calibration
            except Exception as e:
                self.logger.log("GeneralCalibrationLoadFailed", {"error": str(e)})
        
        # Default to identity function
        self.logger.log("UsingDefaultCalibration", {"domain": domain})
        return {
            "semantic": {"coefficients": [1.0, 0.0], "rmse": 0.0, "sample_size": 0},
            "ner": {"coefficients": [1.0, 0.0], "rmse": 0.0, "sample_size": 0},
            "timestamp": datetime.now().isoformat()
        }

    def _calibrate_confidence(self, semantic_results: List[Dict], ner_results: List[Dict], 
                            query: str, domain: str = None) -> None:
        """Calibrate confidence between systems based on historical performance"""
        if not domain:
            domain = self._get_current_domain(query)
        
        # Load calibration data
        calibration = self._load_calibration_data(domain)
        
        # Apply domain-specific calibration
        for r in semantic_results:
            if "norm_score" in r:
                r["calibrated_score"] = self._apply_calibration(
                    r["norm_score"], 
                    calibration.get("semantic", {}),
                    system_type="semantic"
                )
        
        for r in ner_results:
            if "norm_similarity" in r:
                r["calibrated_similarity"] = self._apply_calibration(
                    r["norm_similarity"], 
                    calibration.get("ner", {}),
                    system_type="ner"
                )

    def _apply_calibration(self, score: float, calibration: Dict, system_type: str) -> float:
        """Apply advanced calibration based on historical accuracy"""
        if not calibration or "coefficients" not in calibration:
            return score
        
        # Use polynomial calibration
        poly = np.poly1d(calibration["coefficients"])
        calibrated = float(poly(score))
        
        # Apply system-specific constraints
        if system_type == "semantic":
            # Semantic scores should be more conservative
            calibrated = min(0.95, calibrated)
        else:  # ner
            # NER scores benefit from slight boosting at high confidence
            if score > 0.8:
                calibrated = min(1.0, calibrated * 1.05)
        
        # Ensure valid range
        return max(0.0, min(1.0, calibrated))

    def _get_current_domain(self, query: str) -> str:
        """Determine domain from query using classifier with fallbacks"""
        if not hasattr(self, '_domain_classifier'):
            try:
                from stephanie.analysis.scorable_classifier import ScorableClassifier
                self._domain_classifier = ScorableClassifier(
                    memory=self.memory,
                    logger=self.logger,
                    config_path=self.cfg.get("domain_config", "config/domain/seeds.yaml")
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize domain classifier: {e}")
                # Fallback to simple keyword-based domain detection
                return self._keyword_based_domain_detection(query)
        
        try:
            return self._domain_classifier.classify(query)
        except Exception as e:
            self.logger.warning(f"Domain classification failed: {e}")
            return self._keyword_based_domain_detection(query)

    def _keyword_based_domain_detection(self, query: str) -> str:
        """Simple keyword-based domain detection as fallback"""
        query_lower = query.lower()
        
        domain_keywords = {
            "legal": ["law", "court", "judge", "case", "legal", "act", "statute"],
            "scientific": ["science", "research", "study", "experiment", "data", "hypothesis"],
            "creative": ["story", "poem", "novel", "creative", "art", "music"],
            "technical": ["code", "algorithm", "software", "programming", "technical"]
        }
        
        # Count keyword matches
        scores = {domain: 0 for domain in domain_keywords}
        for domain, keywords in domain_keywords.items():
            for kw in keywords:
                if kw in query_lower:
                    scores[domain] += 1
        
        # Return highest scoring domain or 'general'
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "general" 

    def _should_retrain_calibration(self, domain: str) -> bool:
        """Determine if calibration should be retrained for this domain"""
        # Check if calibration file exists
        cal_path = f"data/calibration/{domain}_calibration.json"
        if not os.path.exists(cal_path):
            return True
        
        # Get last training time
        try:
            with open(cal_path, "r") as f:
                cal_data = json.load(f)
                last_train = datetime.fromisoformat(cal_data["timestamp"])
                
                # Retrain if older than 7 days
                if datetime.now() - last_train > timedelta(days=7):
                    return True
        except Exception as e:
            self.logger.log("CalibrationTimestampCheckFailed", {"error": str(e)})
            return True
        
        # Check if enough new data has accumulated
        historical_data = self._load_historical_data(domain, days=1)
        return len(historical_data) >= self.cfg.get("min_calibration_samples", 100)

    def auto_train_calibration(self, domain: str = "general"):
        """Automatically train calibration if needed"""
        if self._should_retrain_calibration(domain):
            historical_data = self._load_historical_data(domain)
            if historical_data:
                self.train_calibration(historical_data, domain)
                return True
        return False
    
    def evaluate_calibration(self, domain: str = "general", test_size: int = 100) -> Dict[str, float]:
        """
        Evaluate current calibration quality using held-out test data.
        
        Args:
            domain: Domain to evaluate
            test_size: Number of samples to use for testing
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Load test data
        historical_data = self._load_historical_data(domain)
        if len(historical_data) < test_size * 2:  # Need enough data for train/test split
            return {"status": "insufficient_data", "sample_size": len(historical_data)}
        
        # Split into train and test
        random.shuffle(historical_data)
        train_data = historical_data[:-test_size]
        test_data = historical_data[-test_size:]
        
        # Train temporary model
        semantic_data = [d for d in train_data if d["system"] == "semantic"]
        ner_data = [d for d in train_data if d["system"] == "ner"]
        
        semantic_model = self._train_calibration_model(semantic_data)
        ner_model = self._train_calibration_model(ner_data)
        
        # Evaluate on test data
        semantic_rmse = self._evaluate_calibration_model(semantic_model, 
                                                    [d for d in test_data if d["system"] == "semantic"])
        ner_rmse = self._evaluate_calibration_model(ner_model, 
                                                [d for d in test_data if d["system"] == "ner"])
        
        # Compare to uncalibrated baseline
        baseline_rmse = self._evaluate_calibration_model(
            {"coefficients": [1.0, 0.0]}, 
            test_data
        )
        
        improvement = (baseline_rmse - (semantic_rmse + ner_rmse)/2) / baseline_rmse * 100
        
        self.logger.log("CalibrationEvaluation", {
            "domain": domain,
            "semantic_rmse": semantic_rmse,
            "ner_rmse": ner_rmse,
            "baseline_rmse": baseline_rmse,
            "improvement_pct": improvement,
            "test_size": test_size
        })
        
        return {
            "semantic_rmse": semantic_rmse,
            "ner_rmse": ner_rmse,
            "baseline_rmse": baseline_rmse,
            "improvement_pct": improvement,
            "sample_size": len(test_data)
        }

    def _evaluate_calibration_model(self, model: Dict, data: List[Dict]) -> float:
        """Evaluate calibration model RMSE on test data"""
        if not data or "coefficients" not in model:
            return 1.0  # Worst possible RMSE
        
        # Apply calibration
        calibrated_scores = [
            float(np.polyval(model["coefficients"], d["score"])) 
            for d in data
        ]
        
        # Calculate RMSE
        errors = [
            (calibrated - d["accuracy"]) ** 2
            for calibrated, d in zip(calibrated_scores, data)
        ]
        
        return float(np.sqrt(np.mean(errors))) if errors else 1.0

    def generate_triplets(self, scorables: List[Scorable], max_triplets: int = 1000) -> List[Tuple[str, str, str]]:
        """Generate contrastive learning triplets from CaseBooks"""
        # Group entities by type
        entities_by_type = {}
        for scorable in scorables:
            for start, end, etype in self.entity_detector.detect_entities(scorable.text):
                entity_text = scorable.text[start:end].strip() 
                if len(entity_text) >= 2:
                    if etype not in entities_by_type:
                        entities_by_type[etype] = []
                    entities_by_type[etype].append(entity_text)
        
        # Generate triplets (anchor, positive, negative)
        triplets = []
        valid_types = [t for t in entities_by_type.keys() if len(entities_by_type[t]) >= 2]
        
        for _ in range(min(max_triplets, 10 * len(valid_types))):
            etype = random.choice(valid_types)
            entities = entities_by_type[etype]
            
            if len(entities) < 2:
                continue
                
            anchor, positive = random.sample(entities, 2)
            
            # Find negative example from different type
            other_types = [t for t in valid_types if t != etype]
            if not other_types:
                continue
                
            neg_type = random.choice(other_types)
            negative = random.choice(entities_by_type[neg_type])
            
            triplets.append((anchor, positive, negative))
            
            if len(triplets) >= max_triplets:
                break
                
        return triplets

    def train_projection(self, triplets: List[Tuple[str, str, str]], 
                        batch_size: int = 32, epochs: int = 3, lr: float = 1e-4):
        """Train projection network with contrastive learning"""
        if not triplets or not self.projection_enabled or self.projection is None:
            return

        self.projection.train()
        optimizer = torch.optim.Adam(self.projection.parameters(), lr=lr)
        loss_fn = nn.TripletMarginLoss(margin=0.2)

        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(triplets)

            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i+batch_size]
                anchor_batch, pos_batch, neg_batch = zip(*batch)

                anchor_embs = [self._embed_text_for_training(a) for a in anchor_batch]
                pos_embs = [self._embed_text_for_training(p) for p in pos_batch]
                neg_embs = [self._embed_text_for_training(n) for n in neg_batch]

                anchor_tensor = self.projection(torch.stack(anchor_embs))   # <-- Fix 4: project during training
                pos_tensor = self.projection(torch.stack(pos_embs))
                neg_tensor = self.projection(torch.stack(neg_embs))

                loss = loss_fn(anchor_tensor, pos_tensor, neg_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(triplets)/batch_size)
            self.logger.info(f"Epoch {epoch+1}: avg loss {avg_loss:.4f}")

        self.projection.eval()
        logger.info("Projection network training completed")

    def _embed_text_for_training(self, text: str) -> torch.Tensor:
        """Embed text using mid-layer representation for training"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=32
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.layer]
        
        # Use last token representation
        seq_len = (inputs["attention_mask"][0] == 1).sum().item()
        query_vec = hidden_states[0, seq_len-1, :]
        
        return query_vec

    def preprocess_query(self, query: str) -> str:
        """Preprocess query to improve retrieval performance."""
        # Remove common prefixes
        query = re.sub(r"^(find all|show me|retrieve|get|list|identify)\s+", "", query, flags=re.IGNORECASE)
        # Convert to lowercase for consistency
        query = query.lower()
        # Remove trailing punctuation
        query = query.rstrip(" .,;:")
        return query.strip()

    def embed_type_query(self, query: str) -> np.ndarray:
        """Embed a user-provided type description using last token representation."""
        # Preprocess query
        query = self.preprocess_query(query)
        
        inputs = self.tokenizer(
            query, 
            return_tensors="pt", 
            truncation=True, 
            max_length=32
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.layer]
        
        # Get sequence length from attention mask
        seq_len = (inputs["attention_mask"][0] == 1).sum().item()
        
        # Use last token representation (paper-compliant)
        query_vec = hidden_states[0, seq_len-1, :]
        
        # Project if enabled
        if self.projection_enabled and self.projection is not None:
            query_vec = self.projection(query_vec)
        
        return query_vec.cpu().detach().numpy()

    def _log_retrieval_metrics(self, query: str, results: List[Dict], domain: str):
        """Log metrics for monitoring retrieval performance"""
        # Track entity types retrieved
        entity_types = {}
        for r in results:
            etype = r.get("entity_type", "UNKNOWN")
            entity_types[etype] = entity_types.get(etype, 0) + 1
        
        # Track similarity distribution
        similarities = [r["similarity"] for r in results if "similarity" in r]
        calibrated_similarities = [r["calibrated_similarity"] for r in results if "calibrated_similarity" in r]
        
        self.logger.log("NERRetrievalMetrics", {
            "query": query[:50] + "..." if len(query) > 50 else query,
            "domain": domain,
            "total_results": len(results),
            "entity_types": entity_types,
            "similarity_mean": np.mean(similarities) if similarities else 0,
            "similarity_std": np.std(similarities) if similarities else 0,
            "calibrated_mean": np.mean(calibrated_similarities) if calibrated_similarities else 0
        })

    def train_projection(self, triplets: List[Tuple[str, str, str]], 
                        batch_size: int = 32, epochs: int = 3, lr: float = 1e-4):
        """Train projection network with contrastive learning and progress monitoring"""
        if not triplets or not self.projection_enabled or self.projection is None:
            return

        self.projection.train()
        optimizer = torch.optim.Adam(self.projection.parameters(), lr=lr)
        loss_fn = nn.TripletMarginLoss(margin=0.2)
        
        # Initialize progress tracking
        total_batches = (len(triplets) + batch_size - 1) // batch_size
        progress_bar = tqdm(total=total_batches * epochs, desc="Training Projection")
        
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(triplets)

            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i+batch_size]
                anchor_batch, pos_batch, neg_batch = zip(*batch)

                anchor_embs = [self._embed_text_for_training(a) for a in anchor_batch]
                pos_embs = [self._embed_text_for_training(p) for p in pos_batch]
                neg_embs = [self._embed_text_for_training(n) for n in neg_batch]

                anchor_tensor = self.projection(torch.stack(anchor_embs))
                pos_tensor = self.projection(torch.stack(pos_embs))
                neg_tensor = self.projection(torch.stack(neg_embs))

                loss = loss_fn(anchor_tensor, pos_tensor, neg_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress
                batch_loss = loss.item()
                total_loss += batch_loss
                progress_bar.update(1)
                progress_bar.set_postfix({"epoch": epoch+1, "loss": batch_loss:.4f})

            avg_loss = total_loss / max(1, len(triplets)/batch_size)
            self.logger.info(f"Epoch {epoch+1}: avg loss {avg_loss:.4f}")

        progress_bar.close()
        self.projection.eval()
        self.logger.info("Projection network training completed")
