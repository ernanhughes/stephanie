from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

# Your real factory + embeddings
from stephanie.scoring.scorable import ScorableFactory
from stephanie.tools.hnet_embedder import get_embedding

from ..app.types import NexusEdge, NexusNode
from ..stores.dict_store import NexusGraphStore


class NexusIndexer:
    def __init__(self, store: NexusGraphStore, cfg: dict, memory) -> None:
        self.store = store
        self.cfg = cfg
        self.memory = memory

    # (A) Scorable-based nodes (conversation_turn / document / goal ...)
    def add_nodes_from_scorables(
        self,
        items: Iterable[Tuple[str, str, str, dict]],
    ) -> List[NexusNode]:
        """
        items: iterable of (node_id, scorable_type, scorable_id, meta)
          meta may include: patchgrid_path, metrics, policy, outcome, memcube_key
        """
        out: List[NexusNode] = []
        for node_id, scorable_type, scorable_id, meta in items:
            try:
                scorable = ScorableFactory.from_id(self.memory, scorable_type, scorable_id)
            except Exception as e:
                print(f"Failed to resolve scorable {scorable_type}/{scorable_id}: {e}")
                continue

            base_text = scorable.text or ""

            # Optional enrichment: only if risky (keeps embed space stable)
            m = meta.get("metrics") or {}
            hall = float(m.get("hall", 0.0))
            ent = float(m.get("uncertainty", 0.0))
            if hall > 0.3 or ent > 0.4:
                base_text += (
                    f"\n\n[METRICS]\nHallucination Risk: {hall:.3f}\n"
                    f"Uncertainty: {ent:.3f}\n"
                    f"HRM↔Tiny Disagreement: {float(m.get('disagree', 0.0)):.3f}"
                )

            embed_global = np.asarray(get_embedding(base_text, self.cfg), dtype=np.float32)

            n = NexusNode(
                node_id=node_id,
                scorable_id=scorable_id,
                scorable_type=scorable_type,
                memcube_key=meta.get("memcube_key"),
                embed_global=embed_global,
                patchgrid_path=meta.get("patchgrid_path"),
                metrics=m,
                policy=meta.get("policy"),
                outcome=meta.get("outcome"),
            )
            self.store.upsert_node(n)
            out.append(n)
        return out

    # (B) VPM-based nodes (filmstrips / tiles) — lean scorable: "vpm"
    def add_nodes_from_vpms(self, items: Iterable[Tuple[str, dict]]) -> List[NexusNode]:
        """
        items: iterable of (node_id, meta)
          meta must include: memcube_key, patchgrid_path, metrics
          optional: scorable_id (else we use memcube_key)
        """
        out: List[NexusNode] = []
        for node_id, meta in items:
            embed_global = self._encode_global(meta["patchgrid_path"])

            n = NexusNode(
                node_id=node_id,
                scorable_id=meta.get("scorable_id") or meta["memcube_key"],
                scorable_type="vpm",
                memcube_key=meta["memcube_key"],
                embed_global=embed_global,
                patchgrid_path=meta["patchgrid_path"],
                metrics=meta.get("metrics", {}),
                policy=meta.get("policy"),
                outcome=meta.get("outcome"),
            )
            self.store.upsert_node(n)
            out.append(n)
        return out

    def build_knn_edges(self, k: int) -> List[NexusEdge]:
        # TODO: replace with FAISS index (IVF-PQ) – this is a simple O(N^2) fallback
        nodes = list(self.store.nodes.values())
        edges: List[NexusEdge] = []
        if len(nodes) <= 1:
            return edges

        X = np.stack([n.embed_global for n in nodes])  # assumes not None
        sims = X @ X.T  # dot-product

        for i, src in enumerate(nodes):
            order = np.argsort(-sims[i])
            kept = 0
            for j in order:
                if i == j:
                    continue
                edges.append(NexusEdge(
                    src=src.node_id,
                    dst=nodes[j].node_id,
                    type="knn_global",
                    weight=float(sims[i, j]),
                ))
                kept += 1
                if kept >= k:
                    break
        self.store.add_edges(edges)
        return edges

    def _encode_global(self, patchgrid_path: str) -> np.ndarray:
        # TODO: load the VPM (patchgrid) tensor and pool through your vision encoder
        return np.ones((128,), dtype=np.float32) / 128.0