# stephanie/scoring/model/ner_retriever.py
import json
import logging
import os
from typing import Dict, List, Tuple

import annoy
import numpy as np
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
            self.index.load(index_file)
            with open(meta_file, "r") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded Annoy index with {self.index.get_n_items()} items")
        else:
            logger.info("Starting with empty Annoy index")

    def add(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        if not len(embeddings):
            return
        n_existing = self.index.get_n_items()
        for i, (vec, meta) in enumerate(zip(embeddings, metadata_list)):
            # Deduplicate by scorable_id + entity_text
            if any(m["scorable_id"] == meta["scorable_id"] and m["entity_text"] == meta["entity_text"]
                   for m in self.metadata):
                continue
            self.index.add_item(n_existing + i, vec)
            self.metadata.append(meta)
        self._save_index()

    def search(self, query: np.ndarray, k: int = 10):
        indices, distances = self.index.get_nns_by_vector(query, k, include_distances=True)
        results = []
        for idx, dist in zip(indices, distances):
            # Convert angular distance to cosine similarity
            sim = 1 - (dist ** 2) / 2
            if idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                meta["similarity"] = float(sim)
                results.append(meta)
        return results

    def _save_index(self):
        self.index.save(f"{self.index_path}.ann")
        with open(f"{self.index_path}_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)


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
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()
        self.layer = layer
        self.projection = NERRetrieverProjection(self.model.config.hidden_size, embedding_dim).to(device)
        self.index = AnnoyIndex(dim=embedding_dim, index_path=index_path)
        self.entity_detector = EntityDetector(device)

    def embed_entity(self, text: str, span: Tuple[int, int]) -> torch.Tensor:
        if span[0] >= span[1]:
            return torch.zeros(self.projection.fc2.out_features, device=self.device)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs, output_hidden_states=True).hidden_states[self.layer]
        start_token = inputs.char_to_token(0, span[0]) or 1
        end_token = inputs.char_to_token(0, span[1]-1) or start_token
        entity_vec = hidden[0, start_token:end_token+1, :].mean(dim=0)
        return self.projection(entity_vec)

    def embed_type_query(self, query: str) -> torch.Tensor:
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=32).to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs, output_hidden_states=True).hidden_states[self.layer]
        seq_len = (inputs["attention_mask"][0] == 1).sum().item()
        query_vec = hidden[0, :seq_len, :].mean(dim=0)  # mean pool all tokens
        return self.projection(query_vec)

    def index_casebook_entities(self, scorables: List[Scorable]) -> int:
        new_embeddings, new_metadata = [], []
        for scorable in tqdm(scorables, desc="Indexing entities"):
            for start, end, etype in self.entity_detector.detect_entities(scorable.text):
                entity_text = scorable.text[start:end].strip()
                if len(entity_text) < 2:
                    continue
                try:
                    emb = self.embed_entity(scorable.text, (start, end)).cpu().numpy()
                    new_embeddings.append(emb)
                    new_metadata.append({
                        "scorable_id": str(scorable.id),
                        "scorable_type": scorable.target_type,
                        "entity_text": entity_text,
                        "start": start,
                        "end": end,
                        "entity_type": etype,
                        "source_text": scorable.text[:100] + "..."
                    })
                except Exception as e:
                    logger.error(f"Entity embedding failed: {e}")
        if new_embeddings:
            self.index.add(np.array(new_embeddings), new_metadata)
        return len(new_embeddings)

    def retrieve_entities(self, query: str, k: int = 5, min_similarity: float = 0.6) -> List[Dict]:
        query_emb = self.embed_type_query(query).cpu().numpy()
        results = self.index.search(query_emb, k*2)
        return [r for r in results if r["similarity"] >= min_similarity][:k]

    def train_projection(self, triplets: List[Tuple[str, str, str]], batch_size=32, epochs=3, lr=1e-4):
        self.projection.train()
        optimizer = torch.optim.Adam(self.projection.parameters(), lr=lr)
        loss_fn = nn.TripletMarginLoss(margin=0.2)

        for epoch in range(epochs):
            np.random.shuffle(triplets)
            total_loss = 0
            for i in tqdm(range(0, len(triplets), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
                batch = triplets[i:i+batch_size]
                anchor_texts, pos_entities, neg_entities = zip(*batch)

                def encode_batch(texts, max_len=64):
                    inputs = self.tokenizer(list(texts), return_tensors="pt", padding=True,
                                            truncation=True, max_length=max_len).to(self.device)
                    with torch.no_grad():
                        hidden = self.model(**inputs, output_hidden_states=True).hidden_states[self.layer]
                    return hidden.mean(dim=1)  # mean pool sequence

                anchor_emb = self.projection(encode_batch(anchor_texts, max_len=128))
                pos_emb = self.projection(encode_batch(pos_entities, max_len=32))
                neg_emb = self.projection(encode_batch(neg_entities, max_len=32))

                loss = loss_fn(anchor_emb, pos_emb, neg_emb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(triplets)/batch_size)
            logger.info(f"Epoch {epoch+1}: avg loss {avg_loss:.4f}")
        self.projection.eval()
