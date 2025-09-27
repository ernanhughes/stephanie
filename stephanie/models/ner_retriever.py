"""
NER Retriever Module

This module implements a Named Entity Recognition (NER) based retrieval system for legal case documents.
It combines transformer-based entity detection, embedding generation, and approximate nearest neighbor
search to enable efficient entity retrieval from casebooks.

Key Components:
1. NERRetrieverProjection: Neural network for projecting entity embeddings to lower dimension
2. EntityDetector: BERT-based named entity recognition with fallback heuristic
3. HNSWIndex: Wrapper for HNSW approximate nearest neighbor index
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
import random
import re
from datetime import datetime, timedelta
from time import time
from typing import Any, Dict, List, Tuple

import numpy as np
# stephanie/scoring/model/ner_retriever.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, pipeline

from stephanie.models.hnsw_index import HNSWIndex
from stephanie.scoring.scorable import Scorable

_logger = logging.getLogger(__name__)


# -------------------------------
# Projection Network
# -------------------------------
class NERRetrieverProjection(nn.Module):
    """Neural projection network to reduce embedding dimensionality while preserving semantic information."""

    def __init__(
        self,
        input_dim: int = 4096,
        output_dim: int = 500,
        dropout: float = 0.1,
    ):
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
        return F.normalize(
            x, p=2, dim=-1
        )  # L2 normalize output for cosine similarity


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
                device=0 if device == "cuda" else -1,
            )
            _logger.info("Initialized NER pipeline with dslim/bert-base-NER")
        except Exception as e:
            _logger.error(f"Failed to init NER pipeline: {e}")
            self.ner_pipeline = None

    def detect_entities(self, text: str) -> List[Dict[str, Any]]:
        """Detect entities in text using either BERT-NER or fallback heuristic.
        Always returns list of dicts with keys: text, type, start, end, score.
        """
        if not text or len(text.strip()) < 2:
            return []

        entities: List[Dict[str, Any]] = []
        try:
            if self.ner_pipeline:
                results = self.ner_pipeline(text)
                for r in results:
                    entities.append(
                        {
                            "text": text[r["start"] : r["end"]],
                            "type": self._map_entity_type(r["entity_group"]),
                            "start": r["start"],
                            "end": r["end"],
                            "score": float(r.get("score", 1.0)),
                        }
                    )
            else:
                # fallback
                entities = self._heuristic_entity_detection(text)
        except Exception as e:
            _logger.warning(f"NER pipeline failed: {e}")
            entities = self._heuristic_entity_detection(text)

        return entities

    def _map_entity_type(self, group: str) -> str:
        """Map BERT-NER entity types to simplified categories."""
        return {
            "PER": "PERSON",
            "ORG": "ORG",
            "LOC": "LOC",
            "MISC": "MISC",
        }.get(group, "UNKNOWN")

    def _heuristic_entity_detection(self, text: str) -> List[Dict[str, Any]]:
        """Fallback entity detection using capitalization rules.
        Returns dicts consistent with detect_entities.
        """
        entities: List[Dict[str, Any]] = []
        for word in text.split():
            if word and word[0].isupper() and len(word) > 2:
                start = text.find(word)
                if start != -1:
                    entities.append(
                        {
                            "text": word,
                            "type": "UNKNOWN",
                            "start": start,
                            "end": start + len(word),
                            "score": 0.5,  # heuristic confidence
                        }
                    )
        return entities


class NERRetrieverEmbedder:
    """Entity retriever using embedding store instead of custom projections."""

    def __init__(
        self,
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        layer=17,
        device="cuda" if torch.cuda.is_available() else "cpu",
        embedding_dim=2048,
        index_path="data/ner_retriever/index",
        projection_enabled=False,
        projection_dim=2048,
        projection_dropout=0.1,
        logger=None,
        memory=None,
        cfg=None,
    ):
        self.device = device
        self.logger = logger
        self.memory = memory
        self.cfg = cfg or {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = (
            AutoModel.from_pretrained(
                model_name, output_hidden_states=True, 
                trust_remote_code=True,
                ignore_mismatched_sizes=True  # Explicitly ignore size mismatches
            )
            .to(device)
            .eval()
        )

        self.layer = layer
        self.embedding_dim = embedding_dim
        from stephanie.scoring.calibration_manager import CalibrationManager

        self.index = HNSWIndex(dim=embedding_dim, index_path=index_path)
        self.entity_detector = EntityDetector(device)

        # optional projection
        self.projection_enabled = projection_enabled
        self.projection = None
        if projection_enabled:
            self.projection = (
                NERRetrieverProjection(
                    input_dim=self.model.config.hidden_size,
                    output_dim=projection_dim,
                    dropout=projection_dropout,
                )
                .to(device)
                .eval()
            )

        self.calibration = CalibrationManager(
            cfg=self.cfg, memory=self.memory, logger=self.logger
        )

        # Attach KV cache (best-effort)
        self._kv = self._attach_kv_cache()
        self._kv_ttl_sec = self.cfg.get("ner_kv_ttl_sec", 3600)  # default 1h

        _logger.info(
            f"NER Retriever initialized with {model_name} "
            f"layer {layer}, projection_enabled={projection_enabled}"
        )

    # ----------------------------
    # Embedding methods
    # ----------------------------
    def embed_entity(self, text: str, span: Tuple[int, int]) -> torch.Tensor:
        """Embed an entity span with robust character-to-token alignment + layer clamping."""
        hidden_size = (
            self.projection.fc2.out_features
            if self.projection_enabled
            else self.model.config.hidden_size
        )

        if span[0] >= span[1] or span[0] < 0 or span[1] > len(text):
            return torch.zeros(hidden_size, device=self.device)

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            available_layers = len(outputs.hidden_states)

            # ✅ Clamp layer index if too high
            if self.layer >= available_layers:
                self.logger.log(
                    "NERLayerAdjusted",
                    {
                        "requested": self.layer,
                        "available": available_layers - 1,
                        "using": available_layers - 1,
                    },
                )
                self.layer = available_layers - 1

            hidden_states = outputs.hidden_states[self.layer]

        # --- Robust char-to-token alignment ---
        start_token = inputs.char_to_token(0, span[0])
        end_token = inputs.char_to_token(0, span[1] - 1)

        if start_token is None:
            for offset in range(1, min(10, span[0] + 1)):
                start_token = inputs.char_to_token(0, span[0] - offset)
                if start_token is not None:
                    break
            if start_token is None:
                start_token = 1  # skip [CLS]

        if end_token is None:
            for offset in range(1, min(10, len(text) - span[1] + 1)):
                end_token = inputs.char_to_token(0, span[1] - 1 + offset)
                if end_token is not None:
                    break
            if end_token is None or end_token < start_token:
                end_token = min(inputs.input_ids.shape[1] - 1, start_token + 2)

        start_token = max(1, start_token)  # skip [CLS]
        end_token = min(
            inputs.input_ids.shape[1] - 1, max(start_token, end_token)
        )

        # --- Mean pooling across span ---
        entity_vec = hidden_states[0, start_token : end_token + 1, :].mean(
            dim=0
        )

        if self.projection_enabled and self.projection is not None:
            entity_vec = self.projection(entity_vec)

        return entity_vec

    def embed_entities_for_batch(
        self, texts: List[str], spans_list: List[List[Tuple[int, int]]]
    ):
        """
        Robust entity embedding:
        - Handles bad spans (cs >= ce, out of range)
        - Handles tokenizer char_to_token returning None
        - Falls back to CLS or full-sequence average if span mapping fails
        - Never throws; always returns a list of vectors
        """
        # Handle empty inputs early
        if not texts or not spans_list:
            return [[] for _ in range(len(texts))]

        # Tokenize as a batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            available_layers = len(outputs.hidden_states)

            # Validate/adjust layer index
            if self.layer >= available_layers:
                _logger.info(
                    f"NERLayerAdjusted requested: {self.layer}"
                    f" available: {available_layers - 1}, "
                    f" using: {available_layers - 1}"
                )
                layer_idx = available_layers - 1
            else:
                layer_idx = self.layer

            H = outputs.hidden_states[layer_idx]  # [B, T, D]

        batch_vecs = []
        for b, spans in enumerate(spans_list):
            text = texts[b]
            if not text or len(text.strip()) == 0:
                # Handle empty text
                batch_vecs.append(
                    [
                        torch.zeros(
                            self.projection.fc2.out_features
                            if self.projection_enabled
                            else self.model.config.hidden_size,
                            device=self.device,
                        )
                    ]
                )
                continue

            vecs = []
            for cs, ce in spans:
                # Skip invalid spans
                if ce <= cs or cs < 0 or ce > len(text):
                    continue

                st = inputs.char_to_token(b, cs)
                et = inputs.char_to_token(b, ce - 1)

                # Defensive fallbacks
                if st is None:
                    st = 1  # CLS fallback
                if et is None or et < st:
                    et = st

                try:
                    vec = H[b, st : et + 1, :].mean(dim=0)
                except Exception as e:
                    # Log the issue
                    self.logger.log(
                        "NEREmbeddingFallback",
                        {
                            "reason": str(e),
                            "text_idx": b,
                            "span": (cs, ce),
                            "token_span": (st, et),
                            "seq_length": H.shape[1],
                        },
                    )

                    # Tiered fallback strategy
                    if H.shape[1] > 1:
                        # Try [CLS] token first
                        vec = H[b, 0, :]
                        # If that fails, use mean of entire sequence
                        if torch.isnan(vec).any():
                            vec = H[b].mean(dim=0)
                    else:
                        vec = H[b, 0, :]  # Last resort

                vecs.append(vec)

            # If no valid entities found, fall back to CLS representation
            if not vecs:
                vecs = [H[b, 0, :]]

            batch_vecs.append(vecs)

        # Apply projection if enabled
        if self.projection_enabled and self.projection is not None:
            try:
                # Stack all vectors for efficient projection
                all_vecs = [v for sublist in batch_vecs for v in sublist]
                stacked = torch.stack(all_vecs)
                projected = self.projection(stacked)

                # Reconstruct nested structure
                projected_vecs = []
                idx = 0
                for sublist in batch_vecs:
                    projected_vecs.append(
                        projected[idx : idx + len(sublist)].tolist()
                    )
                    idx += len(sublist)
                batch_vecs = projected_vecs
            except Exception as e:
                self.logger.log("ProjectionError", {"error": str(e)})
                # Continue with non-projected vectors

        out = []
        for sublist in batch_vecs:
            out.append(
                [
                    v.detach().cpu().numpy()
                    if isinstance(v, torch.Tensor)
                    else np.array(v)
                    for v in sublist
                ]
            )
        return out

    def embed_entities_for_text(
        self, text: str, spans: List[Tuple[int, int]]
    ) -> List[torch.Tensor]:
        if not spans:
            return []
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            layers = len(outputs.hidden_states)
            if self.layer is None:
                self.layer = layers // 2
            elif self.layer >= layers:
                self.layer = layers - 1
            H = outputs.hidden_states[self.layer]  # [1, T, D]

        vecs = []
        # robust char->token mapping
        for cs, ce in spans:
            st = inputs.char_to_token(0, cs)
            et = inputs.char_to_token(0, ce - 1)
            if st is None:
                for off in range(1, min(10, cs + 1)):
                    st = inputs.char_to_token(0, cs - off)
                    if st is not None:
                        break
            if st is None:
                st = 1
            if et is None:
                for off in range(1, min(10, len(text) - ce + 1)):
                    et = inputs.char_to_token(0, ce - 1 + off)
                    if et is not None:
                        break
            if et is None or et < st:
                et = min(inputs.input_ids.shape[1] - 1, st + 2)

            st = max(1, st)
            et = min(inputs.input_ids.shape[1] - 1, max(st, et))
            vecs.append(H[0, st : et + 1, :].mean(dim=0))  # mean over span
        if self.projection_enabled and self.projection is not None:
            vecs = [self.projection(v) for v in vecs]
        return vecs

    # ----------------------------
    # Indexing
    # ----------------------------
    def index_scorables(self, scorables: List[Scorable]) -> int:
        """
        Index entities from scorables into HNSW with proper metadata.
        """
        new_embeddings, new_metadata = [], []
        total_entities = 0

        for scorable in tqdm(scorables, desc="Indexing entities"):
            try:
                entities = self.entity_detector.detect_entities(scorable.text)
                if not entities:
                    continue

                entities = entities[
                    : self.cfg.get("max_entities_per_scorable", 300)
                ]

                for ent in entities:
                    entity_text = ent["text"].strip()
                    if len(entity_text) < 2:
                        continue

                    emb = self.embed_entity(
                        scorable.text, (ent["start"], ent["end"])
                    )

                    new_embeddings.append(emb.detach().cpu().numpy())
                    new_metadata.append(
                        {
                            "scorable_id": scorable.id,
                            "scorable_type": scorable.target_type,
                            "entity_text": entity_text,
                            "start": ent["start"],
                            "end": ent["end"],
                            "entity_type": ent["type"],
                            "source_text": scorable.text[:100] + "...",
                        }
                    )

                    total_entities += 1

            except Exception as e:
                self.logger.log(
                    "NERIndexingError",
                    {"scorable_id": scorable.id, "error": str(e)},
                )

        if new_embeddings:
            self.index.add(np.array(new_embeddings), new_metadata, save=True)

        self.logger.log(
            "NERIndexingComplete",
            {
                "scorables_processed": len(scorables),
                "entities_indexed": total_entities,
            },
        )

        return total_entities

    def _attach_kv_cache(self):
        """Attach NATS KV cache for entity retrieval results."""
        try:
            bus = getattr(self.memory, "bus", None)
            if bus and hasattr(bus, "get_kv"):
                kv = bus.get_kv(
                    bucket="ner.retrieve.cache",
                    description="NER retrieval results (TTL 1h)",
                    max_age_seconds=3600,
                )
                return kv
        except Exception as e:
            if self.logger:
                self.logger.log("NERKVUnavailable", {"error": str(e)})
        return None

    def _cache_key(self, query: str, k: int, domain: str) -> str:
        """Stable key for caching retrieval results."""
        import hashlib
        import json

        payload = {"q": query, "k": k, "d": domain or "general"}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _kv_get(self, key: str):
        if not self._kv:
            return None
        try:
            raw = self._kv.get(key)
            if not raw:
                return None
            doc = json.loads(raw.decode("utf-8"))
            if (time.time() - float(doc.get("ts", 0))) > self._kv_ttl_sec:
                return None
            return doc.get("results")
        except Exception as e:
            if self.logger:
                self.logger.log("NERKVGetError", {"error": str(e)})
        return None

    def _kv_put(self, key: str, results: List[Dict]):
        if not self._kv:
            return
        try:
            payload = {"ts": time.time(), "results": results}
            self._kv.put(key, json.dumps(payload).encode("utf-8"))
            if self.logger:
                self.logger.log(
                    "NERKVStored", {"key": key, "items": len(results)}
                )
        except Exception as e:
            if self.logger:
                self.logger.log("NERKVPu tError", {"error": str(e)})

    def _process_batch(self, texts, spans_list, scorables, new_embs, new_meta):
        all_spans = [[(a, b) for a, b, _ in spans] for spans in spans_list]
        batched_vecs = self.embed_entities_for_batch(
            texts, all_spans
        )  # new batched method

        for s, spans, vecs in zip(scorables, spans_list, batched_vecs):
            for (start, end, etype), emb in zip(spans, vecs):
                ent = s.text[start:end].strip()
                if len(ent) < 2:
                    continue
                new_embs.append(emb.detach().cpu().numpy())
                new_meta.append(
                    {
                        "scorable_id": str(s.id),
                        "scorable_type": s.target_type,
                        "entity_text": ent,
                        "start": start,
                        "end": end,
                        "entity_type": etype,
                        "source_text": s.text[:100] + "...",
                    }
                )
        return new_embs, new_meta

    def retrieve_entities(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.6,
        min_calibrated_similarity: float = 0.45,  # Critical: separate calibrated threshold
        domain: str = None,
    ) -> List[Dict]:
        """Search for entities similar to the query with robust domain-aware calibration.

        Key improvements:
        - Proper fallbacks when calibration data is missing
        - Confidence calculation for downstream use
        - Enhanced monitoring of calibration effects
        - Metadata completeness for knowledge graph integration
        - Robust error handling for polynomial calibration
        - Domain-aware thresholding
        """

        key = self._cache_key(query, top_k, domain)

        # 1) Check KV cache first
        kv_hit = self._kv_get(key)
        if kv_hit:
            if self.logger:
                self.logger.log(
                    "NERCacheHit",
                    {
                        "backend": "nats_kv",
                        "items": len(kv_hit),
                        "query": query[:50],
                    },
                )
            return kv_hit

        # Preprocess and embed the query
        query_emb = self.embed_type_query(query)

        # Search index
        results = self.index.search(query_emb, top_k * 2)

        # Apply domain-specific calibration
        if domain is None:
            domain = self._get_current_domain(query)

        # Load calibration data with fallbacks
        calibration = self._load_calibration_data(domain)
        if not calibration:
            _logger.warning(
                f"No calibration data found for domain: {domain}. Using default."
            )
            calibration = {
                "ner": {
                    "coefficients": [
                        1.0,
                        0.0,
                    ],  # Identity function as fallback
                    "description": "default_calibration",
                }
            }

        # Track calibration effects for monitoring
        calibration_effects = []
        calibrated_count = 0

        # Apply calibration to results
        for result in results:
            if "similarity" not in result:
                continue

            raw_sim = result["similarity"]
            calibrated_sim = raw_sim  # Default to raw if calibration fails

            # Apply polynomial calibration if available
            if "ner" in calibration and "coefficients" in calibration["ner"]:
                try:
                    poly = np.poly1d(calibration["ner"]["coefficients"])
                    calibrated_sim = float(poly(raw_sim))

                    # Apply system-specific constraints
                    if raw_sim > 0.8:
                        calibrated_sim = min(1.0, calibrated_sim * 1.05)

                    # Ensure valid range
                    calibrated_sim = max(0.0, min(1.0, calibrated_sim))
                    calibrated_count += 1

                    # Track effect for monitoring
                    calibration_effects.append(
                        {
                            "raw": raw_sim,
                            "calibrated": calibrated_sim,
                            "delta": calibrated_sim - raw_sim,
                        }
                    )

                except Exception as e:
                    _logger.error(
                        f"Calibration failed for entity '{result.get('entity_text', 'unknown')}': {e}",
                        extra={
                            "coefficients": calibration["ner"]["coefficients"]
                        },
                    )

            # Store both scores for transparency
            result["calibrated_similarity"] = calibrated_sim
            result["raw_similarity"] = raw_sim

            # Calculate confidence (more nuanced than binary threshold)
            confidence = min(1.0, max(0.0, (calibrated_sim - 0.3) / 0.7))
            result["confidence"] = confidence

        # Filter by similarity threshold (using calibrated if available)
        filtered_results = []
        for r in results:
            # Critical: Use calibrated_similarity for filtering when available
            sim = r.get("calibrated_similarity", r.get("similarity", 0.0))

            # Apply domain-specific thresholding
            if sim >= (
                min_calibrated_similarity
                if "calibrated_similarity" in r
                else min_similarity
            ):
                filtered_results.append(r)

        # Sort by calibrated similarity (or raw if calibrated not available)
        filtered_results.sort(
            key=lambda x: x.get(
                "calibrated_similarity", x.get("similarity", 0.0)
            ),
            reverse=True,
        )

        # Limit to requested number
        filtered_results = filtered_results[:top_k]

        self._kv_put(key, filtered_results)

        # Log for monitoring
        self._log_retrieval_metrics(
            query,
            domain,
            results,
            filtered_results,
            calibration_effects,
            calibrated_count,
        )

        return filtered_results

    def _log_retrieval_metrics(
        self,
        query: str,
        domain: str,
        all_results: List[Dict],
        filtered_results: List[Dict],
        calibration_effects: List[Dict],
        calibrated_count: int,
    ) -> None:
        """Log detailed metrics for monitoring and debugging - PACS-compliant."""
        # Calculate statistics
        raw_sims = [r["similarity"] for r in all_results if "similarity" in r]
        calibrated_sims = [
            r["calibrated_similarity"]
            for r in all_results
            if "calibrated_similarity" in r
        ]

        # Log summary with PACS alignment
        _logger.debug(
            f"Entity retrieval: '{query[:50]}{'...' if len(query) > 50 else ''}' "
            f"| Domain: {domain} "
            f"| Found: {len(all_results)} "
            f"| Returned: {len(filtered_results)}"
        )

        # Log detailed metrics for PACS monitoring
        metrics = {
            "query": query[:100] + "..." if len(query) > 100 else query,
            "domain": domain,
            "total_candidates": len(all_results),
            "returned_results": len(filtered_results),
            "calibrated_count": calibrated_count,
            "raw_similarity_mean": float(np.mean(raw_sims))
            if raw_sims
            else 0.0,
            "raw_similarity_std": float(np.std(raw_sims)) if raw_sims else 0.0,
            "calibrated_similarity_mean": float(np.mean(calibrated_sims))
            if calibrated_sims
            else 0.0,
            "calibrated_similarity_std": float(np.std(calibrated_sims))
            if calibrated_sims
            else 0.0,
        }

        # Add calibration effect metrics if available (PACS: "learning from history")
        if calibration_effects:
            deltas = [e["delta"] for e in calibration_effects]
            metrics.update(
                {
                    "calibration_mean_delta": float(np.mean(deltas)),
                    "calibration_max_delta": float(max(deltas, default=0.0)),
                    "calibration_min_delta": float(min(deltas, default=0.0)),
                    "calibration_effectiveness": self._calculate_calibration_effectiveness(
                        calibration_effects
                    ),
                }
            )

        # Log top results for debugging (PACS: "contextual glue")
        if filtered_results:
            top_entities = [
                f"{r['entity_text']} ({r.get('calibrated_similarity', r.get('similarity', 0)):.3f})"
                for r in filtered_results[:3]
            ]
            _logger.debug(
                f"Top matches for '{query[:30]}...': "
                + ", ".join(top_entities)
            )

        # Send to monitoring system (PACS: "self-correcting" capability)
        _logger.debug(f"EntityRetrievalMetrics: {metrics}")

        # PACS-specific alerting for calibration issues
        if metrics.get("calibration_mean_delta", 0) > 0.25:
            _logger.warning(
                f"Large calibration shift detected: {metrics['calibration_mean_delta']:.3f} "
                f"(domain: {domain}, query: {query[:20]}...)"
            )

    def _calculate_calibration_effectiveness(
        self, calibration_effects: List[Dict]
    ) -> float:
        """Calculate how well calibration aligns with relevance (PACS metric)."""
        if not calibration_effects:
            return 0.0

        # Calculate mean delta for relevant vs non-relevant
        relevant_deltas = [
            e["delta"] for e in calibration_effects if e.get("is_relevant")
        ]
        non_relevant_deltas = [
            e["delta"]
            for e in calibration_effects
            if not e.get("is_relevant", False)
        ]

        if not relevant_deltas or not non_relevant_deltas:
            return 0.0

        relevant_mean = np.mean(relevant_deltas)
        non_relevant_mean = np.mean(non_relevant_deltas)

        # Effectiveness = how much more we boosted relevant items
        return max(0.0, min(1.0, relevant_mean - non_relevant_mean))

    def collect_calibration_data(
        self,
        query: str,
        results: List[Dict],
        ground_truth: List[str],
        domain: str = None,
    ) -> None:
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
            system_type = (
                "ner"
                if result.get("retrieval_type") == "ner_entity"
                else "semantic"
            )

            # Log for calibration
            calibration_entry = {
                "query": query,
                "score": result.get("score", 0.0),
                "system": system_type,
                "is_relevant": is_relevant,
                "domain": domain,
                "timestamp": datetime.now().isoformat(),
            }

            # Save to persistent storage
            self._save_calibration_entry(calibration_entry)

        # Log for monitoring
        self.logger.log(
            "CalibrationDataCollected",
            {
                "query": query[:50] + "..." if len(query) > 50 else query,
                "domain": domain,
                "total_results": len(results),
                "relevant_count": sum(
                    1 for r in results if str(r["id"]) in ground_truth
                ),
            },
        )

    def _save_calibration_entry(self, entry: Dict):
        """Save calibration entry to persistent storage"""
        # Create directory if needed
        os.makedirs("data/calibration/history", exist_ok=True)

        # Save as timestamped JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        with open(f"data/calibration/history/{timestamp}.json", "w") as f:
            json.dump(entry, f, indent=2)

    def _load_historical_data(
        self, domain: str = "general", days: int = 30
    ) -> List[Dict]:
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
                    historical_data.append(
                        {
                            "score": entry["score"],
                            "accuracy": 1.0 if entry["is_relevant"] else 0.0,
                            "system": entry["system"],
                        }
                    )
            except Exception as e:
                self.logger.log(
                    "CalibrationDataLoadFailed",
                    {"error": str(e), "file": filename},
                )

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
                self.logger.log(
                    "DomainCalibrationLoadFailed",
                    {"error": str(e), "domain": domain},
                )

        # Try general calibration
        general_path = "data/calibration/general_calibration.json"
        if os.path.exists(general_path):
            try:
                with open(general_path, "r") as f:
                    calibration = json.load(f)
                    self.logger.log(
                        "UsingFallbackCalibration",
                        {"domain": domain, "fallback": "general"},
                    )
                    return calibration
            except Exception as e:
                self.logger.log(
                    "GeneralCalibrationLoadFailed", {"error": str(e)}
                )

        # Default to identity function
        _logger.debug(f"UsingDefaultCalibration for domain: {domain}")
        return {
            "semantic": {
                "coefficients": [1.0, 0.0],
                "rmse": 0.0,
                "sample_size": 0,
            },
            "ner": {"coefficients": [1.0, 0.0], "rmse": 0.0, "sample_size": 0},
            "timestamp": datetime.now().isoformat(),
        }

    def _calibrate_confidence(
        self,
        semantic_results: List[Dict],
        ner_results: List[Dict],
        query: str,
        domain: str = None,
    ) -> None:
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
                    system_type="semantic",
                )

        for r in ner_results:
            if "norm_similarity" in r:
                r["calibrated_similarity"] = self._apply_calibration(
                    r["norm_similarity"],
                    calibration.get("ner", {}),
                    system_type="ner",
                )

    def _apply_calibration(
        self, score: float, calibration: Dict, system_type: str
    ) -> float:
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
        if not hasattr(self, "_domain_classifier"):
            try:
                from stephanie.analysis.scorable_classifier import \
                    ScorableClassifier

                self._domain_classifier = ScorableClassifier(
                    memory=self.memory,
                    logger=self.logger,
                    config_path=self.cfg.get(
                        "domain_config", "config/domain/seeds.yaml"
                    ),
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize domain classifier: {e}"
                )
                # Fallback to simple keyword-based domain detection
                return self._keyword_based_domain_detection(query)

        try:
            return self._domain_classifier.classify(query)
        except Exception as e:
            _logger.warning(f"Domain classification failed: {e}")
            return self._keyword_based_domain_detection(query)

    def _keyword_based_domain_detection(self, query: str) -> str:
        """Simple keyword-based domain detection as fallback"""
        query_lower = query.lower()

        domain_keywords = {
            "legal": [
                "law",
                "court",
                "judge",
                "case",
                "legal",
                "act",
                "statute",
            ],
            "scientific": [
                "science",
                "research",
                "study",
                "experiment",
                "data",
                "hypothesis",
            ],
            "creative": ["story", "poem", "novel", "creative", "art", "music"],
            "technical": [
                "code",
                "algorithm",
                "software",
                "programming",
                "technical",
            ],
        }

        # Count keyword matches
        scores = {domain: 0 for domain in domain_keywords}
        for domain, keywords in domain_keywords.items():
            for kw in keywords:
                if kw in query_lower:
                    scores[domain] += 1

        # Return highest scoring domain or 'general'
        return (
            max(scores, key=scores.get)
            if max(scores.values()) > 0
            else "general"
        )

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
            self.logger.log(
                "CalibrationTimestampCheckFailed", {"error": str(e)}
            )
            return True

        # Check if enough new data has accumulated
        historical_data = self._load_historical_data(domain, days=1)
        return len(historical_data) >= self.cfg.get(
            "min_calibration_samples", 100
        )

    def auto_train_calibration(self, domain: str = "general"):
        """Automatically train calibration if needed"""
        if self._should_retrain_calibration(domain):
            historical_data = self._load_historical_data(domain)
            if historical_data:
                self.train_calibration(historical_data, domain)
                return True
        return False

    def evaluate_calibration(
        self, domain: str = "general", test_size: int = 100
    ) -> Dict[str, float]:
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
        if (
            len(historical_data) < test_size * 2
        ):  # Need enough data for train/test split
            return {
                "status": "insufficient_data",
                "sample_size": len(historical_data),
            }

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
        semantic_rmse = self._evaluate_calibration_model(
            semantic_model, [d for d in test_data if d["system"] == "semantic"]
        )
        ner_rmse = self._evaluate_calibration_model(
            ner_model, [d for d in test_data if d["system"] == "ner"]
        )

        # Compare to uncalibrated baseline
        baseline_rmse = self._evaluate_calibration_model(
            {"coefficients": [1.0, 0.0]}, test_data
        )

        improvement = (
            (baseline_rmse - (semantic_rmse + ner_rmse) / 2)
            / baseline_rmse
            * 100
        )

        self.logger.log(
            "CalibrationEvaluation",
            {
                "domain": domain,
                "semantic_rmse": semantic_rmse,
                "ner_rmse": ner_rmse,
                "baseline_rmse": baseline_rmse,
                "improvement_pct": improvement,
                "test_size": test_size,
            },
        )

        return {
            "semantic_rmse": semantic_rmse,
            "ner_rmse": ner_rmse,
            "baseline_rmse": baseline_rmse,
            "improvement_pct": improvement,
            "sample_size": len(test_data),
        }

    def _evaluate_calibration_model(
        self, model: Dict, data: List[Dict]
    ) -> float:
        """Evaluate calibration model RMSE on test data"""
        if not data or "coefficients" not in model:
            return 1.0  # Worst possible RMSE

        # Apply calibration
        calibrated_scores = [
            float(np.polyval(model["coefficients"], d["score"])) for d in data
        ]

        # Calculate RMSE
        errors = [
            (calibrated - d["accuracy"]) ** 2
            for calibrated, d in zip(calibrated_scores, data)
        ]

        return float(np.sqrt(np.mean(errors))) if errors else 1.0

    def generate_triplets(
        self, scorables: List[Scorable], max_triplets: int = 1000
    ) -> List[Tuple[str, str, str]]:
        """Generate contrastive learning triplets from CaseBooks"""
        entities_by_type = {}
        for scorable in scorables:
            for ent in self.entity_detector.detect_entities(scorable.text):
                entity_text = ent["text"].strip()
                if len(entity_text) >= 2:
                    entities_by_type.setdefault(ent["type"], []).append(
                        entity_text
                    )

        triplets = []
        valid_types = [
            t for t in entities_by_type if len(entities_by_type[t]) >= 2
        ]

        for _ in range(min(max_triplets, 10 * len(valid_types))):
            etype = random.choice(valid_types)
            entities = entities_by_type[etype]
            if len(entities) < 2:
                continue

            anchor, positive = random.sample(entities, 2)
            other_types = [t for t in valid_types if t != etype]
            if not other_types:
                continue

            neg_type = random.choice(other_types)
            negative = random.choice(entities_by_type[neg_type])

            triplets.append((anchor, positive, negative))
            if len(triplets) >= max_triplets:
                break

        return triplets

    def _get_domain_calibration(self, domain: str) -> Dict:
        """Get calibration with domain hierarchy fallbacks."""
        # Try specific domain first
        cal = self._load_calibration_data(domain)
        if cal:
            return cal

        # Try parent domain (e.g., "computer_vision" → "ai")
        parent_domain = self._get_parent_domain(domain)
        if parent_domain:
            cal = self._load_calibration_data(parent_domain)
            if cal:
                return cal

        # Try general domain
        cal = self._load_calibration_data("general")
        if cal:
            return cal

        # Final fallback: identity function
        return {
            "ner": {
                "coefficients": [1.0, 0.0],
                "description": "default_identity",
            }
        }

    def _calculate_calibration_confidence(
        self, calibration_data: List[Dict]
    ) -> float:
        """Calculate confidence in calibration based on data quality."""
        if not calibration_data:
            return 0.0

        # Confidence factors
        sample_size = min(
            1.0, len(calibration_data) / 200
        )  # Max at 200 samples

        # Temporal recency (1.0 = today, 0.5 = 7 days ago)
        if "timestamp" in calibration_data[0]:
            latest = max(d["timestamp"] for d in calibration_data)
            days_old = (datetime.now() - latest).days
            recency = max(
                0.0, 1.0 - (days_old / 14)
            )  # Full confidence within 14 days
        else:
            recency = 0.5

        # Data diversity (how many different queries)
        query_variety = min(
            1.0, len(set(d["query"] for d in calibration_data)) / 50
        )

        # Weighted combination
        return (0.5 * sample_size) + (0.3 * recency) + (0.2 * query_variety)

    def train_projection(
        self,
        triplets: List[Tuple[str, str, str]],
        batch_size: int = 32,
        epochs: int = 3,
        lr: float = 1e-4,
    ):
        """Train projection network with contrastive learning"""
        if (
            not triplets
            or not self.projection_enabled
            or self.projection is None
        ):
            return

        self.projection.train()
        optimizer = torch.optim.Adam(self.projection.parameters(), lr=lr)
        loss_fn = nn.TripletMarginLoss(margin=0.2)

        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(triplets)

            for i in range(0, len(triplets), batch_size):
                batch = triplets[i : i + batch_size]
                anchor_batch, pos_batch, neg_batch = zip(*batch)

                anchor_embs = [
                    self._embed_text_for_training(a) for a in anchor_batch
                ]
                pos_embs = [
                    self._embed_text_for_training(p) for p in pos_batch
                ]
                neg_embs = [
                    self._embed_text_for_training(n) for n in neg_batch
                ]

                anchor_tensor = self.projection(
                    torch.stack(anchor_embs)
                )  # <-- Fix 4: project during training
                pos_tensor = self.projection(torch.stack(pos_embs))
                neg_tensor = self.projection(torch.stack(neg_embs))

                loss = loss_fn(anchor_tensor, pos_tensor, neg_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(triplets) / batch_size)
            self.logger.info(f"Epoch {epoch + 1}: avg loss {avg_loss:.4f}")

        self.projection.eval()
        _logger.info("Projection network training completed")

    def _embed_text_for_training(self, text: str) -> torch.Tensor:
        """Embed text using mid-layer representation for training"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=32,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.layer]

        # Use last token representation
        seq_len = (inputs["attention_mask"][0] == 1).sum().item()
        query_vec = hidden_states[0, seq_len - 1, :]

        return query_vec

    def preprocess_query(self, query: str) -> str:
        """Preprocess query to improve retrieval performance."""
        # Remove common prefixes
        query = re.sub(
            r"^(find all|show me|retrieve|get|list|identify)\s+",
            "",
            query,
            flags=re.IGNORECASE,
        )
        # Convert to lowercase for consistency
        query = query.lower()
        # Remove trailing punctuation
        query = query.rstrip(" .,;:")
        return query.strip()

    def embed_type_query(self, query: str) -> np.ndarray:
        """Embed a user-provided type description using last token representation,
        robust to layer mismatches."""
        # Preprocess query
        query = self.preprocess_query(query)

        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=32,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states_all = outputs.hidden_states
            available_layers = len(hidden_states_all)

            # Validate / adjust layer index
            if self.layer is None:
                self.layer = available_layers // 2  # middle layer default
            elif self.layer >= available_layers:
                _logger.info(
                    "NERLayerAdjusted"
                    f"requested: {self.layer}"
                    f"available: {available_layers - 1}"
                    f"using: {available_layers - 1}"
                )
                self.layer = available_layers - 1  # last available layer

            hidden_states = hidden_states_all[self.layer]

        # Get sequence length from attention mask
        seq_len = (inputs["attention_mask"][0] == 1).sum().item()

        # Use last token representation (paper-compliant)
        query_vec = hidden_states[0, seq_len - 1, :]

        # Project if enabled
        if self.projection_enabled and self.projection is not None:
            query_vec = self.projection(query_vec)

        return query_vec.cpu().detach().numpy()

    def _log_calibration_effectiveness(
        self, domain: str, calibration_effects: List[Dict]
    ):
        """Log metrics about how calibration affects retrieval quality."""
        if not calibration_effects:
            return

        # Calculate how calibration changes result distribution
        deltas = [e["delta"] for e in calibration_effects]
        positive_shifts = [d for d in deltas if d > 0.05]
        negative_shifts = [d for d in deltas if d < -0.05]

        metrics = {
            "domain": domain,
            "total_effects": len(calibration_effects),
            "positive_shifts": len(positive_shifts),
            "negative_shifts": len(negative_shifts),
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas)),
            "max_positive": float(max(deltas, default=0.0)),
            "max_negative": float(min(deltas, default=0.0)),
            "shift_ratio": len(positive_shifts)
            / max(1, len(calibration_effects)),
        }

        # Log to monitoring system
        self.logger.log("CalibrationEffectiveness", metrics)

        # Alert if calibration is causing excessive shifts
        if abs(metrics["mean_delta"]) > 0.2:
            self.logger.log(
                "CalibrationWarning",
                {
                    "message": "Calibration causing large mean shift",
                    "domain": domain,
                    "mean_delta": metrics["mean_delta"],
                },
            )

    def train_projection(
        self,
        triplets: List[Tuple[str, str, str]],
        batch_size: int = 32,
        epochs: int = 3,
        lr: float = 1e-4,
    ):
        """Train projection network with contrastive learning and progress monitoring"""
        if (
            not triplets
            or not self.projection_enabled
            or self.projection is None
        ):
            return

        self.projection.train()
        optimizer = torch.optim.Adam(self.projection.parameters(), lr=lr)
        loss_fn = nn.TripletMarginLoss(margin=0.2)

        # Initialize progress tracking
        total_batches = (len(triplets) + batch_size - 1) // batch_size
        progress_bar = tqdm(
            total=total_batches * epochs, desc="Training Projection"
        )

        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(triplets)

            for i in range(0, len(triplets), batch_size):
                batch = triplets[i : i + batch_size]
                anchor_batch, pos_batch, neg_batch = zip(*batch)

                anchor_embs = [
                    self._embed_text_for_training(a) for a in anchor_batch
                ]
                pos_embs = [
                    self._embed_text_for_training(p) for p in pos_batch
                ]
                neg_embs = [
                    self._embed_text_for_training(n) for n in neg_batch
                ]

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
                progress_bar.set_postfix(
                    {"epoch": epoch + 1, "loss": batch_loss}
                )

            avg_loss = total_loss / max(1, len(triplets) / batch_size)
            self.logger.info(f"Epoch {epoch + 1}: avg loss {avg_loss:.4f}")

        progress_bar.close()
        self.projection.eval()
        self.logger.info("Projection network training completed")
