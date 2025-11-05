from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set, Dict, Any, Iterable, Optional, Union
import numpy as np
from stephanie.components.nexus.types import NexusEdge, NexusNode


def _as_manifest_dict(m: Any) -> Dict[str, Any]:
    """
    Accepts NexusRunManifest or dict and returns a plain dict:
      { "run_id": str, "items": [ { ...item fields... } ], "extras": {...} }
    """
    if isinstance(m, dict):
        return m

    # Likely NexusRunManifest
    out = {
        "run_id": getattr(m, "run_id", None),
        "items": [],
        "extras": getattr(m, "extras", {}) or {},
    }
    items: Iterable[Any] = getattr(m, "items", []) or []
    for it in items:
        if hasattr(it, "to_dict"):
            out["items"].append(it.to_dict())
        else:
            # Try dataclass-like attributes
            out["items"].append({
                "item_id": getattr(it, "item_id", None),
                "scorable_id": getattr(it, "scorable_id", None),
                "scorable_type": getattr(it, "scorable_type", None),
                "turn_index": getattr(it, "turn_index", None),
                "chat_id": getattr(it, "chat_id", None),
                "domains": getattr(it, "domains", None),
                "entities": getattr(it, "entities", None),
                "near_identity": getattr(it, "near_identity", None),
                "metrics_columns": getattr(it, "metrics_columns", None),
                "metrics_values": getattr(it, "metrics_values", None),
                "metrics_vector": getattr(it, "metrics_vector", None),
                "embeddings": getattr(it, "embeddings", None),
                "vpm_png": getattr(it, "vpm_png", None),
                "rollout": getattr(it, "rollout", None),
            })
    return out


def _make_node(node_id: str, scorable_id: str, scorable_type: str) -> NexusNode:
    """
    Construct a NexusNode regardless of the constructor signature.
    Prefers NexusNode(id=...), falls back to NexusNode(node_id).
    """
    return NexusNode(node_id=node_id, scorable_id=scorable_id, scorable_type=scorable_type)  # type: ignore


def _pick_title_text(item: Dict[str, Any]) -> tuple[str, str]:
    """
    Choose a human label and a longer text for the node from near_identity /
    scorable hints. Falls back to item_id.
    """
    near = item.get("near_identity") or {}
    title = (near.get("title") or near.get("summary") or item.get("scorable_id")
             or item.get("item_id") or "item")
    text = (near.get("text") or near.get("body") or near.get("snippet")
            or title)
    return str(title), str(text)


def _pick_embed_global(emb: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    """
    Return a float32 numpy vector if available. Prefers 'global', otherwise the
    first numeric vector found.
    """
    if not emb:
        return None
    if "global" in emb and isinstance(emb["global"], (list, tuple)):
        try:
            v = np.asarray(emb["global"], dtype=np.float32)
            return v if v.size else None
        except Exception:
            pass
    # fallback: first numeric
    for k, v in emb.items():
        if isinstance(v, (list, tuple)):
            try:
                vec = np.asarray(v, dtype=np.float32)
                if vec.size:
                    return vec
            except Exception:
                continue
    return None


def build_nodes_from_manifest(
    manifest: Union[Dict[str, Any], Any],
    *,
    namespace: str = "vpm",
) -> Dict[str, NexusNode]:
    """
    Build {node_id -> NexusNode} from a Nexus manifest.

    Node id scheme:  f"{namespace}://{run_id}/{item_id}"
      - Ensures uniqueness across runs
      - Matches what your edge builder expects

    Populated attributes (if NexusNode supports dynamic attrs):
      - id, run_id, item_id, scorable_id, target_type
      - chat_id, turn_index
      - domains, entities, near_identity
      - metrics_columns, metrics_values, metrics_vector
      - embeddings, embed_global (np.ndarray or None)
      - title, text, degree (init 0)
      - vpm_png, rollout
    """
    m = _as_manifest_dict(manifest)
    run_id = str(m.get("run_id") or "")
    items = list(m.get("items") or [])
    nodes: Dict[str, NexusNode] = {}

    for it in items:
        item_id = it.get("item_id") or it.get("scorable_id") or ""
        node_id = f"{namespace}://{run_id}/{item_id}"
        scorable_type = it.get("scorable_type") or "unknown"
        scorable_id = it.get("scorable_id") or item_id
        node = _make_node(node_id, scorable_id=scorable_id, scorable_type=scorable_type)

        # Core ids
        setattr(node, "id", node_id)
        setattr(node, "run_id", run_id)
        setattr(node, "item_id", item_id)
        setattr(node, "scorable_id", it.get("scorable_id"))

        # Label + body text
        title, text = _pick_title_text(it)
        setattr(node, "title", title)
        setattr(node, "text", text)

        # Type / indices
        setattr(node, "target_type", it.get("scorable_type") or "unknown")
        setattr(node, "chat_id", it.get("chat_id"))
        setattr(node, "turn_index", it.get("turn_index"))

        # Semantic/contextual tags
        setattr(node, "domains", it.get("domains") or [])
        # entities can be list or dict keys
        ents = it.get("entities")
        if isinstance(ents, dict):
            ents = list(ents.keys())
        setattr(node, "entities", ents or [])
        setattr(node, "near_identity", it.get("near_identity") or {})

        # Metrics
        setattr(node, "metrics_columns", it.get("metrics_columns") or [])
        setattr(node, "metrics_values", it.get("metrics_values") or [])
        setattr(node, "metrics_vector", it.get("metrics_vector") or {})

        # Embeddings
        embs = it.get("embeddings") or {}
        setattr(node, "embeddings", embs)
        setattr(node, "embed_global", _pick_embed_global(embs))

        # VPM / rollout artifacts
        setattr(node, "vpm_png", it.get("vpm_png"))
        setattr(node, "rollout", it.get("rollout") or {})

        # Degree starts at 0; router JS recomputes from edges
        setattr(node, "degree", 0)

        nodes[node_id] = node

    return nodes

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
