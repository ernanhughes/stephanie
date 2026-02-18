import hashlib
import sqlite3
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EntailmentModel:

    def __init__(
        self,
        model_name: str,
        db_path: str = "nli_cache.db",
        device: str = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.entailment_index = 2

        # Setup SQLite
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS nli_cache (
            key TEXT PRIMARY KEY,
            entailment_prob REAL
        )
        """)
        self.conn.commit()

    def _make_key(self, premise: str, hypothesis: str) -> str:
        combined = premise.strip() + "|||" + hypothesis.strip()
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _lookup_cache(self, keys: List[str]):
        placeholders = ",".join("?" for _ in keys)
        query = f"SELECT key, entailment_prob FROM nli_cache WHERE key IN ({placeholders})"
        rows = self.conn.execute(query, keys).fetchall()
        return {k: v for k, v in rows}

    def _insert_cache(self, key_score_pairs):
        self.conn.executemany(
            "INSERT OR REPLACE INTO nli_cache (key, entailment_prob) VALUES (?, ?)",
            key_score_pairs
        )
        self.conn.commit()

    @torch.no_grad()
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:

        keys = [self._make_key(p, h) for p, h in pairs]
        cached = self._lookup_cache(keys)

        results = []
        to_compute = []
        compute_indices = []

        # Separate cached vs new
        for i, key in enumerate(keys):
            if key in cached:
                results.append(cached[key])
            else:
                results.append(None)
                to_compute.append(pairs[i])
                compute_indices.append(i)

        # If nothing new → return immediately
        if not to_compute:
            return results

        # Compute missing in batches
        new_scores = []

        for i in range(0, len(to_compute), self.batch_size):
            batch = to_compute[i:i+self.batch_size]
            premises = [p for p, h in batch]
            hypotheses = [h for p, h in batch]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
            entailment_probs = probs[:, self.entailment_index]

            new_scores.extend(entailment_probs.detach().cpu().numpy().tolist())

        # Insert into cache
        new_key_score_pairs = [
            (keys[idx], score)
            for idx, score in zip(compute_indices, new_scores)
        ]
        self._insert_cache(new_key_score_pairs)

        # Fill in results
        for idx, score in zip(compute_indices, new_scores):
            results[idx] = score

        return results
