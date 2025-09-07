# stephanie/scoring/model/ner_retriever.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import logging
import re
import random
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from transformers import AutoModel, AutoTokenizer, pipeline
import annoy

from stephanie.scoring.scorable import Scorable

logger = logging.getLogger(__name__)

# -------------------------------
# Projection Network
# -------------------------------
class NERRetrieverProjection(nn.Module):
    def __init__(self, input_dim: int = 4096, output_dim: int = 500, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(output_dim, output_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=-1)


# -------------------------------
# Entity Detector
# -------------------------------
class EntityDetector:
    def __init__(self, device: str = "cuda"):
        try:
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
        if not text or len(text.strip()) < 2:
            return []
        if self.ner_pipeline:
            try:
                results = self.ner_pipeline(text)
                return [(r["start"], r["end"], self._map_entity_type(r["entity_group"])) for r in results]
            except Exception as e:
                logger.warning(f"NER pipeline failed: {e}")
        # Fallback: naive heuristic
        return self._heuristic_entity_detection(text)

    def _map_entity_type(self, group: str) -> str:
        return {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC", "MISC": "MISC"}.get(group, "UNKNOWN")

    def _heuristic_entity_detection(self, text: str) -> List[Tuple[int, int, str]]:
        entities = []
        for word in text.split():
            if word and word[0].isupper() and len(word) > 2:
                start = text.find(word)
                if start != -1:
                    entities.append((start, start + len(word), "UNKNOWN"))
        return entities


# -------------------------------
# Annoy Index Wrapper
# -------------------------------
class AnnoyIndex:
    def __init__(self, dim: int = 500, index_path: str = "data/ner_retriever/index"):
        self.dim = dim
        self.index_path = index_path
        self.index = annoy.AnnoyIndex(dim, "angular")
        self.metadata = []
        self._load_index()

    def _load_index(self):
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

    def add(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        if not len(embeddings):
            return
            
        n_existing = self.index.get_n_items()
        added_count = 0
        
        for i, (vec, meta) in enumerate(zip(embeddings, metadata_list)):
            # Deduplicate by scorable_id + entity_text
            if any(m.get("scorable_id") == meta.get("scorable_id") and 
                   m.get("entity_text") == meta.get("entity_text")
                   for m in self.metadata):
                continue
                
            # Add to index
            self.index.add_item(n_existing + added_count, vec)
            self.metadata.append(meta)
            added_count += 1
            
        if added_count > 0:
            self._save_index()
            logger.info(f"Added {added_count} new entities to index (skipped {len(metadata_list) - added_count} duplicates)")

    def search(self, query: np.ndarray, k: int = 10):
        if self.index.get_n_items() == 0:
            return []
            
        query = query.astype(np.float32)
        indices, distances = self.index.get_nns_by_vector(query, k, include_distances=True)
        
        results = []
        for idx, dist in zip(indices, distances):
            # Convert angular distance to cosine similarity
            # Formula: cos_sim = 1 - (angular_dist^2)/2
            sim = 1 - (dist ** 2) / 2
            if idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                meta["similarity"] = float(sim)
                meta["distance"] = float(dist)
                results.append(meta)
                
        return results

    def _save_index(self):
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
            # Clean up temp files if they exist
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


# -------------------------------
# Retriever Embedder
# -------------------------------
class NERRetrieverEmbedder:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-8b",
        layer: int = 17,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        embedding_dim: int = 500,
        index_path: str = "data/ner_retriever/index",
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, 
            output_hidden_states=True,
            trust_remote_code=True
        ).to(device).eval()
        self.layer = layer
        self.embedding_dim = embedding_dim
        self.projection = NERRetrieverProjection(
            self.model.config.hidden_size, 
            embedding_dim
        ).to(device)
        self.index = AnnoyIndex(dim=embedding_dim, index_path=index_path)
        self.entity_detector = EntityDetector(device)
        
        logger.info(f"NER Retriever initialized with {model_name} layer {layer}")

    def preprocess_query(self, query: str) -> str:
        """Preprocess query to improve retrieval performance."""
        # Remove common prefixes
        query = re.sub(r"^(find all|show me|retrieve|get|list|identify)\s+", "", query, flags=re.IGNORECASE)
        # Convert to lowercase for consistency
        query = query.lower()
        # Remove trailing punctuation
        query = query.rstrip(" .,;:")
        return query.strip()

    def embed_entity(self, text: str, span: Tuple[int, int]) -> torch.Tensor:
        """Embed an entity span with robust character-to-token alignment."""
        if span[0] >= span[1] or span[0] < 0 or span[1] > len(text):
            return torch.zeros(self.projection.fc2.out_features, device=self.device)
            
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
        return self.projection(entity_vec)

    def embed_type_query(self, query: str) -> torch.Tensor:
        """Embed a user-provided type description using last token representation (paper-compliant)."""
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
        return self.projection(query_vec)

    def batch_embed_entities(self, texts: List[str], spans: List[Tuple[int, int]]) -> np.ndarray:
        """Embed multiple entities in a single batch for efficiency."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
            
        # Tokenize all texts at once
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.layer]
        
        embeddings = []
        for i, (start, end) in enumerate(spans):
            # Find token indices for the span
            start_token = inputs.char_to_token(i, start)
            end_token = inputs.char_to_token(i, end-1)
            
            # Fallback logic (similar to embed_entity)
            if start_token is None:
                for offset in range(1, min(10, start + 1)):
                    start_token = inputs.char_to_token(i, start - offset)
                    if start_token is not None:
                        break
                if start_token is None:
                    start_token = 1  # Skip [CLS]
                    
            if end_token is None:
                for offset in range(1, min(10, len(texts[i]) - end + 1)):
                    end_token = inputs.char_to_token(i, end - 1 + offset)
                    if end_token is not None:
                        break
                if end_token is None or end_token < start_token:
                    end_token = min(hidden_states.shape[1] - 1, start_token + 2)
            
            # Ensure valid span
            start_token = max(1, start_token)
            end_token = min(hidden_states.shape[1] - 1, max(start_token, end_token))
            
            # Pool across the entity span
            entity_vec = hidden_states[i, start_token:end_token+1, :].mean(dim=0)
            embeddings.append(self.projection(entity_vec).cpu().numpy())
        
        return np.array(embeddings)

    def index_scorables(self, scorables: List[Scorable], memory) -> int:
        """Index all entities in a list of scorables for retrieval."""
        logger.info("Indexing entities from scorables")
        
        
        if not scorables:
            logger.info("No scorables found for indexing")
            return 0
        
        new_embeddings = []
        new_metadata = []
        
        # Process each scorable
        for scorable in tqdm(scorables, desc="Processing scorables"):
            # Detect entities in the scorable text
            entities = self.entity_detector.detect_entities(scorable.text)
            
            for start, end, entity_type in entities:
                # Extract entity text
                entity_text = scorable.text[start:end].strip()
                
                # Skip very short entities
                if len(entity_text) < 2:
                    continue
                
                # Create embedding
                try:
                    embedding = self.embed_entity(scorable.text, (start, end))
                    embedding_np = embedding.cpu().numpy()
                    
                    # Store for indexing
                    new_embeddings.append(embedding_np)
                    new_metadata.append({
                        "scorable_id": str(scorable.id),
                        "scorable_type": scorable.target_type,
                        "entity_text": entity_text,
                        "start": start,
                        "end": end,
                        "entity_type": entity_type,
                        "source_text": scorable.text[:100] + "..." if len(scorable.text) > 100 else scorable.text
                    })
                except Exception as e:
                    logger.error(f"Failed to embed entity '{entity_text}': {e}")
        
        # Add to index
        if new_embeddings:
            self.index.add(np.array(new_embeddings), new_metadata)
            logger.info(f"Indexed {len(new_embeddings)} entities")
            return len(new_embeddings)

        logger.info("No entities found for indexing")
        return 0

    def retrieve_entities(self, query: str, k: int = 5, min_similarity: float = 0.6) -> List[Dict]:
        """Retrieve entities matching the query description."""
        # Preprocess and embed the query
        query_emb = self.embed_type_query(query).cpu().numpy()
        
        # Search index
        results = self.index.search(query_emb, k*2)
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in results 
            if r.get("similarity", 0.0) >= min_similarity
        ][:k]
        
        # Log for monitoring
        if filtered_results:
            logger.info(f"Found {len(filtered_results)} entities matching '{query}'")
            for i, result in enumerate(filtered_results[:3]):  # Log top 3
                logger.debug(f"Top {i+1} match: '{result['entity_text']}' (sim={result['similarity']:.4f})")
        else:
            logger.info(f"No entities found matching '{query}'")
            
        return filtered_results

    def generate_triplets(
        self, 
        scorables: List[Scorable], 
        max_triplets: int = 1000
    ) -> List[Tuple[str, str, str]]:
        """
        Generate contrastive learning triplets from CaseBooks.
        
        Triplets are in the form (anchor, positive, negative) where:
        - anchor: entity text
        - positive: similar entity text (same type)
        - negative: dissimilar entity text (different type)
        """
        logger.info("Generating contrastive learning triplets...")
        
        # Group entities by type
        entities_by_type = {}
        for scorable in scorables:
            for start, end, etype in self.entity_detector.detect_entities(scorable.text):
                entity_text = scorable.text[start:end].strip()
                if len(entity_text) >= 2:  # Skip very short entities
                    if etype not in entities_by_type:
                        entities_by_type[etype] = []
                    entities_by_type[etype].append(entity_text)
        
        # Filter out types with too few examples
        valid_types = [t for t in entities_by_type.keys() if len(entities_by_type[t]) >= 2]
        
        if not valid_types:
            logger.warning("No valid entity types found for triplet generation")
            return []
            
        logger.info(f"Found {len(valid_types)} valid entity types for triplet generation")
        
        # Generate triplets
        triplets = []
        for _ in range(min(max_triplets, 10 * len(valid_types))):
            # Randomly select a type with at least 2 entities
            etype = random.choice(valid_types)
            entities = entities_by_type[etype]
            
            if len(entities) < 2:
                continue
                
            # Randomly select two entities of the same type (anchor and positive)
            anchor, positive = random.sample(entities, 2)
            
            # Find a negative example (different type)
            other_types = [t for t in valid_types if t != etype]
            if not other_types:
                continue
                
            neg_type = random.choice(other_types)
            negative = random.choice(entities_by_type[neg_type])
            
            triplets.append((anchor, positive, negative))
            
            if len(triplets) >= max_triplets:
                break
        
        logger.info(f"Generated {len(triplets)} triplets for contrastive learning")
        return triplets

    def train_projection(
        self,
        triplets: List[Tuple[str, str, str]],
        batch_size=32,
        epochs=3,
        lr=1e-4
    ):
        """Train projection network with contrastive learning."""
        if not triplets:
            logger.warning("No triplets provided for training - skipping projection training")
            return
            
        logger.info(f"Training projection network with {len(triplets)} triplets for {epochs} epochs")
        
        self.projection.train()
        optimizer = torch.optim.Adam(self.projection.parameters(), lr=lr)
        loss_fn = nn.TripletMarginLoss(margin=0.2)
        
        # Pre-tokenize all texts for efficiency
        all_texts = []
        for anchor, positive, negative in triplets:
            all_texts.extend([anchor, positive, negative])
        
        # Tokenize in batches
        anchor_embeddings = []
        positive_embeddings = []
        negative_embeddings = []
        
        for i in tqdm(range(0, len(triplets), batch_size), desc="Preprocessing"):
            batch = triplets[i:i+batch_size]
            
            # Process anchor texts
            anchor_batch = [t[0] for t in batch]
            anchor_inputs = self.tokenizer(
                anchor_batch, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                anchor_outputs = self.model(**anchor_inputs, output_hidden_states=True)
                anchor_hidden = anchor_outputs.hidden_states[self.layer]
                # Use last token representation
                anchor_vec = anchor_hidden[:, -1, :]
            
            anchor_embeddings.append(self.projection(anchor_vec))
            
            # Process positive entities
            positive_batch = [t[1] for t in batch]
            positive_inputs = self.tokenizer(
                positive_batch, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=32
            ).to(self.device)
            
            with torch.no_grad():
                positive_outputs = self.model(**positive_inputs, output_hidden_states=True)
                positive_hidden = positive_outputs.hidden_states[self.layer]
                # Use last token representation
                positive_vec = positive_hidden[:, -1, :]
            
            positive_embeddings.append(self.projection(positive_vec))
            
            # Process negative entities
            negative_batch = [t[2] for t in batch]
            negative_inputs = self.tokenizer(
                negative_batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=32
            ).to(self.device)
            
            with torch.no_grad():
                negative_outputs = self.model(**negative_inputs, output_hidden_states=True)
                negative_hidden = negative_outputs.hidden_states[self.layer]
                # Use last token representation
                negative_vec = negative_hidden[:, -1, :]
            
            negative_embeddings.append(self.projection(negative_vec))
        
        # Combine all batches
        anchor_emb = torch.cat(anchor_embeddings, dim=0)
        positive_emb = torch.cat(positive_embeddings, dim=0)
        negative_emb = torch.cat(negative_embeddings, dim=0)
        
        # Train in epochs
        for epoch in range(epochs):
            epoch_loss = 0
            indices = torch.randperm(len(triplets))
            
            for i in tqdm(range(0, len(triplets), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
                batch_indices = indices[i:i+batch_size]
                
                # Get batch embeddings
                a = anchor_emb[batch_indices]
                p = positive_emb[batch_indices]
                n = negative_emb[batch_indices]
                
                # Compute loss
                loss = loss_fn(a, p, n)
                epoch_loss += loss.item()
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_loss = epoch_loss / (len(triplets) / batch_size)
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        self.projection.eval()
        logger.info("Projection network training completed")
        
        # Validate index after training
        if self.index.validate():
            logger.info("Index validation passed after training")
        else:
            logger.warning("Index validation failed after training - consider reindexing")