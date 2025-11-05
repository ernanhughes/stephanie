from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from stephanie.components.nexus.types import NexusEdge, NexusNode

def build_edges(
    nodes: Dict[str, NexusNode],
    items: List[Dict],
    knn_k: int = 12,
    add_temporal: bool = True,
    sim_threshold: float = 0.35,
    bidirectional_knn: bool = False,
    run_id: Optional[str] = None,
) -> List[NexusEdge]:
    """
    Build edges for the Nexus graph.

    - KNN edges over node.embed_global (cosine similarity).
    - Optional temporal edges using (chat_id, turn_index).

    Args:
        nodes: node_id -> NexusNode
        items: manifest['items'] list (must contain 'item_id', and for temporal: 'chat_id','turn_index')
        knn_k: neighbors per node (max)
        add_temporal: add edges linking consecutive turns within a chat
        sim_threshold: minimum cosine similarity for KNN edges
        bidirectional_knn: add reverse edge for each KNN pair
        run_id: if provided, temporal src/dst = f"vpm://{run_id}/{item_id}";
                else, we infer by matching the trailing '/{item_id}' in nodes.

    Returns:
        List[NexusEdge]
    """
    edges: List[NexusEdge] = []
    seen: Set[Tuple[str, str, str]] = set()  # (src,dst,type)

    # -----------------------------
    # KNN over global embeddings
    # -----------------------------
    keyed = [(nid, n.embed_global) for nid, n in nodes.items() if n.embed_global is not None]
    if keyed:
        ids, mats = zip(*keyed)                     # ids: tuple[str], mats: tuple[np.ndarray]
        M = np.stack(mats, axis=0).astype(np.float32)  # [N, d]
        # normalize rows to unit length to make cosine = dot
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1e-9
        U = M / norms

        # cosine similarities to each row vector
        # For large N you’ll want FAISS; this is fine for small/medium
        for i in range(U.shape[0]):
            sims = U @ U[i]                      # [N]
            order = np.argsort(-sims)            # descending
            picked = 0
            for j in order:
                if i == j:
                    continue
                sim = float(sims[j])
                if sim < sim_threshold:
                    break
                src, dst = ids[i], ids[j]
                key = (src, dst, "knn_global")
                if key not in seen:
                    edges.append(NexusEdge(src=src, dst=dst, type="knn_global", weight=sim))
                    seen.add(key)
                if bidirectional_knn:
                    key2 = (dst, src, "knn_global")
                    if key2 not in seen:
                        edges.append(NexusEdge(src=dst, dst=src, type="knn_global", weight=sim))
                        seen.add(key2)
                picked += 1
                if picked >= knn_k:
                    break

    # -----------------------------
    # Temporal chain per chat
    # -----------------------------
    if add_temporal and items:
        # Map item_id -> node_id
        item_to_node: Dict[str, str] = {}

        if run_id is not None:
            # Fast path: we know the prefix format
            for it in items:
                iid = it.get("item_id")
                if iid:
                    nid = f"vpm://{run_id}/{iid}"
                    if nid in nodes:
                        item_to_node[iid] = nid

        # Fallback: infer by suffix match (last path component == item_id)
        if not item_to_node:
            # Precompute tail -> node_id mapping
            tail_map: Dict[str, str] = {}
            for nid in nodes.keys():
                tail = nid.rsplit("/", 1)[-1]
                tail_map[tail] = nid
            for it in items:
                iid = it.get("item_id")
                if iid and iid in tail_map:
                    item_to_node[iid] = tail_map[iid]

        # Group by chat_id
        from collections import defaultdict
        chats = defaultdict(list)
        for it in items:
            chat_id = it.get("chat_id")
            turn_idx = it.get("turn_index")
            iid = it.get("item_id")
            if chat_id and isinstance(turn_idx, int) and iid in item_to_node:
                chats[chat_id].append((turn_idx, iid))

        # Link consecutive turns
        for chat_id, arr in chats.items():
            arr.sort(key=lambda x: x[0])  # by turn_index
            for (_, iid_cur), (_, iid_next) in zip(arr, arr[1:]):
                src = item_to_node.get(iid_cur)
                dst = item_to_node.get(iid_next)
                if src and dst:
                    key = (src, dst, "temporal_next")
                    if key not in seen:
                        edges.append(NexusEdge(src=src, dst=dst, type="temporal_next", weight=1.0))
                        seen.add(key)

    return edges
