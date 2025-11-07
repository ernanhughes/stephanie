# stephanie/components/nexus/graph/builder.py
from __future__ import annotations

import math
from collections import defaultdict
from typing import (Any, DefaultDict, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)

import numpy as np

from stephanie.components.nexus.app.types import NexusEdge, NexusNode
from stephanie.utils.json_sanitize import dumps_safe


def _topk_indices_desc(row: np.ndarray, k: int, exclude_self: int) -> np.ndarray:
    # returns indices of the largest k entries (descending), excluding self
    # uses argpartition for O(N) selection then sorts that small set
    N = row.shape[0]
    k_eff = min(k + 1, N)  # (+1) because we may exclude self
    idx = np.argpartition(-row, kth=k_eff-1)[:k_eff]
    # full sort of the small slice
    idx = idx[np.argsort(-row[idx])]
    # drop self if present
    if exclude_self in idx:
        idx = idx[idx != exclude_self]
    return idx[:k]

def build_edges_enhanced(
    nodes: Dict[str, Any],
    items: List[Dict[str, Any]],
    *,
    run_id: Optional[str],
    # KNN blend
    knn_k: int = 8,
    sim_threshold: float = 0.35,
    max_edges_per_node: int = 24,
    # channel weights
    weights: Optional[Dict[str, float]] = None,
    # per-type caps (extra safety)
    caps: Optional[Dict[str, int]] = None,
    # structure helpers
    add_temporal: bool = True,
    add_mst_backbone: bool = True,
    # optional extra edge families
    add_domain_edges: bool = True,
    add_entity_edges: bool = True,
    max_domain_edges_per_node: int = 6,
    max_entity_edges_per_node: int = 6,
) -> List[NexusEdge]:
    ids = list(nodes.keys())
    if not ids:
        return []

    # weights (don’t need to sum to 1; final S renormalized)
    w = {
        "embed": 0.50,
        "metrics": 0.25,
        "lexical": 0.10,
        "domains": 0.08,
        "entities": 0.05,
        "agreement": 0.02,
        "stability": 0.00,
        **(weights or {}),
    }

    S, C = _pairwise_blend(ids, nodes, w)

    edges: List[NexusEdge] = []
    seen: Set[Tuple[str, str, str]] = set()       # (src,dst,type)
    degree = defaultdict(int)                     # node -> deg cap
    per_type_count = defaultdict(int)
    cap = {
        "knn_blend": 10**9,
        "temporal_next": 10**9,
        "backbone_mst": 10**9,
        "shared_domain": 100000,
        "shared_entity": 100000,
        **(caps or {}),
    }

    # ---------- blended KNN (fast top-k per row) ----------
    N = len(ids)
    for i, src in enumerate(ids):
        row = S[i]
        if row.max(initial=0.0) < sim_threshold:
            continue
        nbr_idx = _topk_indices_desc(row, k=knn_k, exclude_self=i)
        for j in nbr_idx:
            sim = float(row[j])
            if sim < sim_threshold:
                continue
            dst = ids[j]
            if degree[src] >= max_edges_per_node: break
            if degree[dst] >= max_edges_per_node: continue
            if per_type_count["knn_blend"] >= cap["knn_blend"]:
                break
            key = (src, dst, "knn_blend")
            if key in seen or src == dst:
                continue

            ch = {k: float(C[k][i, j]) for k in C.keys()}
            edges.append(NexusEdge(src, dst, "knn_blend", sim, channels=ch))
            seen.add(key)
            degree[src] += 1
            degree[dst] += 1
            per_type_count["knn_blend"] += 1

    # ---------- temporal chain ----------
    if add_temporal:
        t_edges = _temporal_edges(items, nodes, run_id)
        for e in t_edges:
            if e.src == e.dst:
                continue
            key = (e.src, e.dst, e.type)
            if key in seen:
                continue
            if per_type_count["temporal_next"] >= cap["temporal_next"]:
                break
            if degree[e.src] >= max_edges_per_node or degree[e.dst] >= max_edges_per_node:
                continue
            edges.append(e)
            seen.add(key)
            degree[e.src] += 1
            degree[e.dst] += 1
            per_type_count["temporal_next"] += 1

    # ---------- optional domain/entity cliques (bounded) ----------
    if (add_domain_edges or add_entity_edges) and items:
        # Build quick maps: item_id -> node_id
        item_to_node = {}
        for nid in ids:
            tail = nid.rsplit("/", 1)[-1]
            item_to_node[tail] = nid

        # Domains
        if add_domain_edges:
            dmap: DefaultDict[str, List[str]] = defaultdict(list)
            for it in items:
                iid = it.get("item_id")
                nid = item_to_node.get(iid)
                if not nid: continue
                for d in it.get("domains") or []:
                    dmap[str(d)].append(nid)

            for domain, nids in dmap.items():
                if len(nids) <= 1:
                    continue
                # create a light clique with per-node edge cap
                for a_i, src in enumerate(nids):
                    added = 0
                    if degree[src] >= max_edges_per_node:
                        continue
                    # connect to next few neighbors only (to avoid O(m^2))
                    for dst in nids[a_i+1:a_i+1+max_domain_edges_per_node]:
                        if src == dst:
                            continue
                        if degree[src] >= max_edges_per_node or degree[dst] >= max_edges_per_node:
                            continue
                        if per_type_count["shared_domain"] >= cap["shared_domain"]:
                            break
                        key = (src, dst, "shared_domain")
                        if key in seen:
                            continue
                        edges.append(NexusEdge(src, dst, "shared_domain", 0.5, channels={"domain": domain}))
                        seen.add(key)
                        degree[src] += 1
                        degree[dst] += 1
                        per_type_count["shared_domain"] += 1
                        added += 1
                        if added >= max_domain_edges_per_node:
                            break

        # Entities
        if add_entity_edges:
            emap: DefaultDict[str, List[str]] = defaultdict(list)
            for it in items:
                iid = it.get("item_id")
                nid = item_to_node.get(iid)
                if not nid: continue
                ents = it.get("entities") or []
                if isinstance(ents, dict):
                    ents = list(ents.keys())
                for ent in ents:
                    emap[str(ent)].append(nid)

            for ent, nids in emap.items():
                if len(nids) <= 1:
                    continue
                for a_i, src in enumerate(nids):
                    added = 0
                    if degree[src] >= max_edges_per_node:
                        continue
                    for dst in nids[a_i+1:a_i+1+max_entity_edges_per_node]:
                        if src == dst:
                            continue
                        if degree[src] >= max_edges_per_node or degree[dst] >= max_edges_per_node:
                            continue
                        if per_type_count["shared_entity"] >= cap["shared_entity"]:
                            break
                        key = (src, dst, "shared_entity")
                        if key in seen:
                            continue
                        edges.append(NexusEdge(src, dst, "shared_entity", 0.45, channels={"entity": ent}))
                        seen.add(key)
                        degree[src] += 1
                        degree[dst] += 1
                        per_type_count["shared_entity"] += 1
                        added += 1
                        if added >= max_entity_edges_per_node:
                            break

    # ---------- MST backbone (guarantee connectivity) ----------
    if add_mst_backbone:
        mst = _mst_edges(ids, S)
        for e in mst:
            if e.src == e.dst:
                continue
            key = (e.src, e.dst, e.type)
            if key in seen:
                continue
            if per_type_count["backbone_mst"] >= cap["backbone_mst"]:
                break
            if degree[e.src] >= max_edges_per_node or degree[e.dst] >= max_edges_per_node:
                continue
            edges.append(e)
            seen.add(key)
            degree[e.src] += 1
            degree[e.dst] += 1
            per_type_count["backbone_mst"] += 1

    return edges


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

def _cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None: return 0.0
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _z(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if x is None: return None
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0: return x
    mu = x.mean(); sd = x.std() + 1e-8
    return (x - mu) / sd

def _normalize_token_list(v, *, key: str | None = None) -> list[str]:
    """
    Turn a heterogeneous list (strings / dicts / scalars) into a list of
    lowercase string tokens that are safe to put in a set.
    If key is provided and an element is a dict, use dict[key] when present,
    else fall back to a canonical JSON form of the dict.
    """
    if v is None:
        return []
    if isinstance(v, (str, int, float, bool, dict)):
        v = [v]  # make it iterable

    out: list[str] = []
    for item in v:
        if isinstance(item, str):
            out.append(item.strip().lower())
        elif isinstance(item, dict):
            if key and (key in item) and item[key] is not None:
                out.append(str(item[key]).strip().lower())
            else:
                # canonicalize whole dict to a stable string
                out.append(dumps_safe(item, sort_keys=True, ensure_ascii=False).lower())
        elif item is None:
            continue
        else:
            out.append(str(item).strip().lower())
    return out

def _jaccard_list(a, b, *, key: str | None = None) -> float:
    A = set(_normalize_token_list(a, key=key))
    B = set(_normalize_token_list(b, key=key))
    if not A and not B:
        return 0.0
    return len(A & B) / float(len(A | B))

def _char_shingles_jaccard(a: Optional[str], b: Optional[str], k: int = 5, max_len: int = 4000) -> float:
    if not a or not b: return 0.0
    # limit to keep it fast
    a = a[:max_len]; b = b[:max_len]
    Sa = {a[i:i+k] for i in range(0, max(0, len(a)-k+1))}
    Sb = {b[i:i+k] for i in range(0, max(0, len(b)-k+1))}
    if not Sa and not Sb: return 0.0
    return len(Sa & Sb) / max(1, len(Sa | Sb))

def _norm01_rowwise(S: np.ndarray) -> np.ndarray:
    # normalize each row to 0..1 to make thresholds robust across corpora
    out = S.copy()
    for i in range(out.shape[0]):
        r = out[i]
        mn, mx = r.min(initial=0.0), r.max(initial=0.0)
        denom = (mx - mn) if (mx - mn) > 1e-8 else 1.0
        out[i] = (r - mn) / denom
    # keep symmetric
    return (out + out.T) / 2.0

# ---------- composite scoring ----------

def _pairwise_blend(ids: List[str], nodes: Dict[str, Any], w: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      S : [N,N] blended similarity in [0,1]
      C : per-channel score matrices keyed by channel name
    Channels: embed, metrics, lexical, domains, entities, agreement, stability
    """
    N = len(ids)
    S = np.zeros((N, N), dtype=np.float32)
    C: Dict[str, np.ndarray] = {
        "embed": np.zeros((N, N), dtype=np.float32),
        "metrics": np.zeros((N, N), dtype=np.float32),
        "lexical": np.zeros((N, N), dtype=np.float32),
        "domains": np.zeros((N, N), dtype=np.float32),
        "entities": np.zeros((N, N), dtype=np.float32),
        "agreement": np.zeros((N, N), dtype=np.float32),
        "stability": np.zeros((N, N), dtype=np.float32),
    }

    E = [nodes[i].embed_global if hasattr(nodes[i], "embed_global") else None for i in ids]
    M = [np.asarray(getattr(nodes[i], "metrics_values", []), dtype=np.float32) for i in ids]
    Mz = [(_z(m) if m.size else None) for m in M]
    T = [getattr(nodes[i], "text", None) for i in ids]
    D = [getattr(nodes[i], "domains", []) for i in ids]
    G = [getattr(nodes[i], "entities", []) for i in ids]
    A = [getattr(nodes[i], "agreement", None) for i in ids]
    U = [getattr(nodes[i], "stability", None) for i in ids]

    for i in range(N):
        for j in range(i+1, N):
            s_embed   = _cos(E[i], E[j])
            s_metrics = _cos(Mz[i], Mz[j]) if (Mz[i] is not None and Mz[j] is not None) else 0.0
            s_lex     = _char_shingles_jaccard(T[i], T[j], k=5)
            s_dom = _jaccard_list(D[i], D[j], key="domain")      # domains: [{"domain": "...", "score": ...}, ...]
            s_ent = _jaccard_list(E[i], E[j], key="text")        # entities: [{"text": "...", "label": ...}, ...]


            # agreement/stability: use *min* to propagate weakest link
            s_agree   = min(A[i], A[j]) if (A[i] is not None and A[j] is not None) else 0.0
            s_stab    = min(U[i], U[j]) if (U[i] is not None and U[j] is not None) else 0.0

            C["embed"][i, j]    = C["embed"][j, i]    = s_embed
            C["metrics"][i, j]  = C["metrics"][j, i]  = s_metrics
            C["lexical"][i, j]  = C["lexical"][j, i]  = s_lex
            C["domains"][i, j]  = C["domains"][j, i]  = s_dom
            C["entities"][i, j] = C["entities"][j, i] = s_ent
            C["agreement"][i, j]= C["agreement"][j, i]= s_agree
            C["stability"][i, j]= C["stability"][j, i]= s_stab

    # blend (pre-normalize rows for stability, then weight)
    for k in C.keys():
        C[k] = _norm01_rowwise(C[k])

    S = (
        w.get("embed",0)*C["embed"] +
        w.get("metrics",0)*C["metrics"] +
        w.get("lexical",0)*C["lexical"] +
        w.get("domains",0)*C["domains"] +
        w.get("entities",0)*C["entities"] +
        w.get("agreement",0)*C["agreement"] +
        w.get("stability",0)*C["stability"]
    )
    # re-normalize after blend for robust thresholding
    S = _norm01_rowwise(S)
    return S, C

def _temporal_edges(items: List[Dict[str, Any]], nodes: Dict[str, Any], run_id: Optional[str]) -> List[NexusEdge]:
    edges: List[NexusEdge] = []
    chats: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        if it.get("chat_id") and it.get("turn_index") is not None:
            chats[it["chat_id"]].append(it)
    for _, arr in chats.items():
        arr.sort(key=lambda x: x["turn_index"])
        for a, b in zip(arr, arr[1:]):
            src = f"vpm://{run_id}/{a['item_id']}" if run_id else f"vpm://{a.get('run_id','')}/{a['item_id']}"
            dst = f"vpm://{run_id}/{b['item_id']}" if run_id else f"vpm://{b.get('run_id','')}/{b['item_id']}"
            if src in nodes and dst in nodes:
                edges.append(NexusEdge(src, dst, "temporal_next", 1.0))
    return edges

def _mst_edges(ids: List[str], S: np.ndarray) -> List[NexusEdge]:
    """
    Minimum Spanning Tree over distance = 1 - S (Prim's algorithm).
    Guarantees global connectivity across disparate chats.
    """
    N = len(ids)
    if N <= 1: return []
    dist = 1.0 - S
    in_tree = np.zeros(N, dtype=bool)
    in_tree[0] = True
    best = dist[0].copy()
    parent = np.full(N, -1, dtype=int)
    parent[0] = 0
    edges: List[NexusEdge] = []
    for _ in range(N-1):
        j = np.argmin(np.where(in_tree, np.inf, best))
        if math.isinf(best[j]): break
        i = parent[j]
        if i >= 0:
            w = float(S[i, j])
            edges.append(NexusEdge(ids[i], ids[j], "backbone_mst", w))
        in_tree[j] = True
        for k in range(N):
            if not in_tree[k] and dist[j, k] < best[k]:
                best[k] = dist[j, k]
                parent[k] = j
    return edges
