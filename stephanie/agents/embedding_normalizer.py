# stephanie/agents/embedding_normalizer.py
"""
This agent retroactively **normalizes all stored embeddings** across Stephanie’s
embedding tables (HNet, HuggingFace, Ollama, etc.) so that every vector
has approximately unit length (‖v‖ ≈ 1.0).  

In 2025, during visual differential analysis and Q-field studies,
we discovered that older embeddings—particularly those produced before
H-Net integration—had widely varying magnitudes (norms ranging from ~0.5 to 30+).
This caused cosine and Euclidean similarity metrics in the database
(e.g., `1 - (embedding <-> query_vector)`) to return negative or
nonsensical values, breaking band-searches, clustering, and visualization.

This agent was introduced to:
    1. **Repair all legacy embeddings** by re-normalizing them to unit norm.
    2. **Standardize vector geometry** across all embedding types for
       consistent similarity math.
    3. **Future-proof the system** so that visualization agents (PhōsAgent,
       ZeroModelService, VPMWorker, etc.) operate on reliable, bounded data.

Design
------
- Runs in **batches** to avoid memory pressure and long transactions.
- Operates **in-place**: updates only when the vector norm is clearly off
  (‖v‖ < 0.5 or ‖v‖ > 2.0), preserving already-normalized embeddings.
- Supports **multiple tables** (configurable via Hy That's right dra).
- **Transactional & resumable** — commits after each batch and logs progress.
- Compatible with **Stephanie’s Supervisor pipeline system**, so it can
  be scheduled, monitored, or replayed as part of a maintenance run.

When to Use
-----------
- After changing or upgrading an embedding backend.
- After importing historical embeddings (e.g., from MXBAI, HuggingFace, or
  LLaMA sources).
- Before running large-scale similarity, clustering, or contrastive-visual
  experiments (Phōs, HRM, SICQL, or ZeroModel).

Related Components
------------------
- :mod:`stephanie.memory.base_embedding_store` – now performs normalization
  for *new* embeddings on creation.
- :mod:`stephanie.services.scoring_service` – relies on consistent vector
  distances for document scoring.
- :mod:`stephanie.analysis.vpm_differential_analyzer` – depends on normalized
  fields for accurate heatmap generation.

Version History
---------------
- **v1.0 (2025-10)** — Initial version introduced after discovery of
  negative similarity bands in H-Net vectors.

"""

from __future__ import annotations

import logging

import numpy as np

from stephanie.agents.base_agent import BaseAgent

log = logging.getLogger(__name__)

class EmbeddingNormalizerAgent(BaseAgent):
    """
    Normalize all embeddings in all embedding tables to unit length.
    Works safely and transactionally in batches.
    """

    def __init__(self, cfg, memory, container, logger=None):
        super().__init__(cfg, memory, container, logger or _logger)
        self.batch_size = cfg.get("batch_size", 500)
        # By default normalize all known tables
        self.tables = cfg.get("tables", [
            "hf_embeddings",
            "hnet_embeddings",
            "embeddings"
        ])

    async def run(self, context=None):
        log.debug(f"🔄 Starting embedding normalization for tables: {self.tables}")
        summary = {}

        for table in self.tables:
            total, normalized = self._normalize_table(table)
            summary[table] = {"total": total, "normalized": normalized}

        log.debug(f"✅ Normalization complete → {summary}")
        context[self.output_key] = summary
        return context

    # -------------------------
    # Core logic
    # -------------------------
    def _normalize_table(self, table: str):
        """Normalize all vectors in a single table."""
        try:
            with self.memory.conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                total = cur.fetchone()[0]
        except Exception as e:
            log.error(f"❌ Failed to count rows in {table}: {e}")
            return 0, 0

        offset = 0
        normalized = 0
        log.debug(f"⚙️  Normalizing {total} embeddings in '{table}'...")

        while True:
            with self.memory.conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, embedding FROM {table} WHERE embedding IS NOT NULL "
                    f"ORDER BY id ASC LIMIT %s OFFSET %s",
                    (self.batch_size, offset),
                )
                rows = cur.fetchall()

            if not rows:
                break

            updates = []
            for rid, vec in rows:
                try:
                    arr = np.asarray(vec, dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    if norm > 2.0 or norm < 0.5:  # normalize only if out of expected range
                        arr = arr / (norm + 1e-8)
                        updates.append((arr.tolist(), rid))
                        normalized += 1
                except Exception as e:
                    logning(f"⚠️  Skip id={rid} due to {e}")

            if updates:
                with self.memory.conn.cursor() as cur:
                    for emb, rid in updates:
                        cur.execute(
                            f"UPDATE {table} SET embedding = %s WHERE id = %s",
                            (emb, rid),
                        )
                self.memory.conn.commit()

            offset += self.batch_size
            log.debug(
                "🧮 Table=%s → %s/%s processed, %s normalized so far...",
                table, offset, total, normalized
            )

        log.debug("✅ Table '%s' normalized: %s/%s", table, normalized, total)
        return total, normalized
