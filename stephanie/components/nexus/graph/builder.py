# stephanie/components/nexus/graph/builder.py
from __future__ import annotations

import math
import logging
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from stephanie.components.nexus.app.types import NexusEdge, NexusNode
from stephanie.utils.json_sanitize import dumps_safe
from stephanie.memory.nexus_store import NexusStore

log = logging.getLogger(__name__)

ManifestLike = Union[Dict[str, Any], List[Dict[str, Any]]]


class GraphBuilder:
    """
    Build a Nexus graph (nodes + blended edges) from a scorable manifest.

    - Accepts either:
      * plain list:     [ ...scorable dicts... ]
    - Persists nodes before edges.
    - Supports multiple edge families (KNN, temporal, domain/entity cliques, MST).
    - Versioning: keep `namespace` for modality (e.g., "vpm"); encode variants
      into `run_id` e.g., "run100__exp03". Node id -> f"{namespace}://{run_id}/{item_id}"
    """

    def __init__(self, cfg, memory, logger):
        self.cfg = cfg
        self.memory = memory
        self.store: NexusStore = memory.nexus
        self.logger = logger

    def build(
        self,
        run_id: str,
        scorables_or_manifest: Any,                # NEW name
        *,
        namespace: str = "vpm",                    # NEW
        knn_k: int = 8,
        sim_threshold: float = 0.35,
        max_edges_per_node: int = 24,
        add_temporal: bool = True,
        add_mst_backbone: bool = True,
        add_domain_edges: bool = True,
        add_entity_edges: bool = True,
        knn_edge_type: str = "knn_global",
        weights: Optional[Dict[str, float]] = None,
        caps: Optional[Dict[str, int]] = None,
        export_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Normalize input to a manifest dict with an 'items' list
        if isinstance(scorables_or_manifest, dict) and "items" in scorables_or_manifest:
            manifest = scorables_or_manifest
            items = list(manifest.get("items") or [])
        elif isinstance(scorables_or_manifest, (list, tuple)):
            items = list(scorables_or_manifest)
            manifest = {"run_id": run_id, "items": items}
        else:
            items = []
            manifest = {"run_id": run_id, "items": items}

        # Build nodes with run_id + namespace (fix)
        nodes = build_nodes(run_id=run_id, manifest=manifest, namespace=namespace)

        # Items for temporal/domain/entity edges must be dicts; tolerate Scorables
        norm_items = []
        for it in items:
            if isinstance(it, dict):
                norm_items.append(it)
            else:
                # Best-effort mapping from Scorable
                norm_items.append({
                    "item_id": getattr(it, "id", None) or getattr(it, "scorable_id", None),
                    "scorable_id": getattr(it, "id", None),
                    "scorable_type": getattr(it, "target_type", "document"),
                    "near_identity": {"title": None, "text": getattr(it, "text", None)},
                    "domains": getattr(it, "domains", None),
                    "entities": getattr(it, "entities", None),
                    "chat_id": getattr(it, "chat_id", None),
                    "turn_index": getattr(it, "turn_index", None),
                    "metrics_values": getattr(it, "metrics_values", None),
                    "embeddings": {"global": getattr(it, "embed_global", None)},
                })

        edges = build_edges_enhanced(
            nodes,
            norm_items,
            run_id=run_id,
            knn_k=knn_k,
            sim_threshold=sim_threshold,
            max_edges_per_node=max_edges_per_node,
            weights=weights,
            caps=caps,
            add_temporal=add_temporal,
            add_mst_backbone=add_mst_backbone,
            add_domain_edges=add_domain_edges,
            add_entity_edges=add_entity_edges,
            knn_edge_type=knn_edge_type,
        )

        payload = [{
            "src": e.src, "dst": e.dst, "type": e.type,
            "weight": float(e.weight or 0.0),
            "channels": getattr(e, "channels", None)
        } for e in edges]
        n = self.store.write_edges(run_id, payload)

        from stephanie.components.nexus.graph.exporters.pyviz import export_graph_json
        graph_json = export_graph_json(path=export_path, nodes=nodes, edges=edges, include_channels=True)

        c = Counter(e["type"] for e in payload)
        return {
            "run_id": run_id,
            "edges_written": n,
            "by_type": dict(c),
            "node_count": len(nodes),
            "nodes": nodes,
            "edges": edges,
            "graph_json": graph_json,               # NEW convenience
        }

    # ---- store helpers -----------------------------------------------------

    def _upsert_nodes(self, run_id: str, nodes: Dict[str, NexusNode]) -> None:
        # Expect NexusStore to support a bulk upsert. If not present, implement one.
        # Minimal payload: id, run_id, scorable_id, type, title, text, domains, entities, embed dims/exists
        rows = []
        for nid, n in nodes.items():
            rows.append(
                {
                    "node_id": nid,
                    "run_id": getattr(n, "run_id", run_id),
                    "scorable_id": getattr(n, "scorable_id", None),
                    "scorable_type": getattr(n, "target_type", None),
                    "title": getattr(n, "title", None),
                    "text": getattr(n, "text", None),
                    "domains": getattr(n, "domains", None),
                    "entities": getattr(n, "entities", None),
                    "metrics_columns": getattr(n, "metrics_columns", None),
                    "metrics_values": getattr(n, "metrics_values", None),
                    "embeddings": getattr(n, "embeddings", None),
                }
            )
        # If your store API differs, adapt here.
        if hasattr(self.store, "upsert_nodes"):
            self.store.upsert_nodes(run_id, rows)
        else:
            # Fallback: create nodes one by one if that’s all you have.
            if hasattr(self.store, "upsert_node"):
                for r in rows:
                    self.store.upsert_node(run_id, r)
            else:
                log.warning("NexusStore lacks upsert_nodes/upsert_node; nodes not persisted.")


# ---- pure functions (no store side-effects) --------------------------------

def _as_manifest_dict_from_any(run_id: str, m: ManifestLike) -> Dict[str, Any]:
    """Accept a dict manifest or a plain list of items and return a manifest dict."""
    if isinstance(m, dict):
        out = _as_manifest_dict(m)
        # Guarantee run_id presence/override
        out["run_id"] = run_id
        return out
    elif isinstance(m, list):
        return {"run_id": run_id, "items": m, "extras": {}}
    else:
        raise TypeError("scorables must be a dict manifest or a list of items")


def _topk_indices_desc(row: np.ndarray, k: int, exclude_self: int) -> np.ndarray:
    N = row.shape[0]
    k_eff = min(k + 1, N)
    idx = np.argpartition(-row, kth=k_eff - 1)[:k_eff]
    idx = idx[np.argsort(-row[idx])]
    if exclude_self in idx:
        idx = idx[idx != exclude_self]
    return idx[:k]


def build_edges_enhanced(
    nodes: Dict[str, Any],
    items: List[Dict[str, Any]],
    *,
    run_id: Optional[str],
    namespace: str = "vpm",
    knn_k: int = 8,
    knn_edge_type: str = "knn_global",
    sim_threshold: float = 0.35,
    max_edges_per_node: int = 24,
    weights: Optional[Dict[str, float]] = None,
    caps: Optional[Dict[str, int]] = None,
    add_temporal: bool = True,
    add_mst_backbone: bool = True,
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
    seen: Set[Tuple[str, str, str]] = set()  # (src,dst,type)
    degree = defaultdict(int)
    per_type_count = defaultdict(int)
    # ensure a cap entry that matches the actual edge type string used
    knn_cap_key = knn_edge_type
    cap = {
        knn_cap_key: 10**9,
        "temporal_next": 10**9,
        "backbone_mst": 10**9,
        "shared_domain": 100000,
        "shared_entity": 100000,
        **(caps or {}),
    }

    # ---------- blended KNN ----------
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
            if degree[src] >= max_edges_per_node:
                break
            if degree[dst] >= max_edges_per_node:
                continue
            if per_type_count[knn_cap_key] >= cap[knn_cap_key]:
                break

            etype = knn_edge_type
            key = (src, dst, etype)
            if key in seen or src == dst:
                continue

            ch = {k: float(C[k][i, j]) for k in C.keys()}
            edges.append(NexusEdge(src, dst, etype, sim, channels=ch))
            seen.add(key)
            degree[src] += 1
            degree[dst] += 1
            per_type_count[etype] += 1

    # ---------- temporal chain ----------
    if add_temporal:
        t_edges = _temporal_edges(items, nodes, run_id, namespace)
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

    # ---------- optional domain/entity cliques ----------
    if (add_domain_edges or add_entity_edges) and items:
        item_to_node = {}
        for nid in ids:
            tail = nid.rsplit("/", 1)[-1]
            item_to_node[tail] = nid

        if add_domain_edges:
            dmap: DefaultDict[str, List[str]] = defaultdict(list)
            for it in items:
                iid = it.get("item_id")
                nid = item_to_node.get(iid)
                if not nid:
                    continue
                for d in it.get("domains") or []:
                    dmap[str(d)].append(nid)

            for domain, nids in dmap.items():
                if len(nids) <= 1:
                    continue
                for a_i, src in enumerate(nids):
                    added = 0
                    if degree[src] >= max_edges_per_node:
                        continue
                    for dst in nids[a_i + 1 : a_i + 1 + max_domain_edges_per_node]:
                        if src == dst:
                            continue
                        if degree[src] >= max_edges_per_node or degree[dst] >= max_edges_per_node:
                            continue
                        if per_type_count["shared_domain"] >= cap["shared_domain"]:
                            break
                        key = (src, dst, "shared_domain")
                        if key in seen:
                            continue
                        edges.append(
                            NexusEdge(src, dst, "shared_domain", 0.5, channels={"domain": domain})
                        )
                        seen.add(key)
                        degree[src] += 1
                        degree[dst] += 1
                        per_type_count["shared_domain"] += 1
                        added += 1
                        if added >= max_domain_edges_per_node:
                            break

        if add_entity_edges:
            emap: DefaultDict[str, List[str]] = defaultdict(list)
            for it in items:
                iid = it.get("item_id")
                nid = item_to_node.get(iid)
                if not nid:
                    continue
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
                    for dst in nids[a_i + 1 : a_i + 1 + max_entity_edges_per_node]:
                        if src == dst:
                            continue
                        if degree[src] >= max_edges_per_node or degree[dst] >= max_edges_per_node:
                            continue
                        if per_type_count["shared_entity"] >= cap["shared_entity"]:
                            break
                        key = (src, dst, "shared_entity")
                        if key in seen:
                            continue
                        edges.append(
                            NexusEdge(src, dst, "shared_entity", 0.45, channels={"entity": ent})
                        )
                        seen.add(key)
                        degree[src] += 1
                        degree[dst] += 1
                        per_type_count["shared_entity"] += 1
                        added += 1
                        if added >= max_entity_edges_per_node:
                            break

    # ---------- MST backbone ----------
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
    if isinstance(m, dict):
        out = {
            "run_id": m.get("run_id"),
            "items": list(m.get("items") or []),
            "extras": m.get("extras") or {},
        }
        return out

    # Likely NexusRunManifest-like object
    out = {"run_id": getattr(m, "run_id", None), "items": [], "extras": getattr(m, "extras", {}) or {}}
    items: Iterable[Any] = getattr(m, "items", []) or []
    for it in items:
        if hasattr(it, "to_dict"):
            out["items"].append(it.to_dict())
        else:
            out["items"].append(
                {
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
                }
            )
    return out


def _make_node(node_id: str, scorable_id: str, scorable_type: str) -> NexusNode:
    return NexusNode(node_id=node_id, scorable_id=scorable_id, scorable_type=scorable_type)  # type: ignore


def _pick_title_text(item: Dict[str, Any]) -> tuple[str, str]:
    near = item.get("near_identity") or {}
    title = (
        near.get("title")
        or near.get("summary")
        or item.get("scorable_id")
        or item.get("item_id")
        or "item"
    )
    text = (near.get("text") or near.get("body") or near.get("snippet") or title)
    return str(title), str(text)


def _pick_embed_global(emb: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not emb:
        return None
    if "global" in emb and isinstance(emb["global"], (list, tuple)):
        try:
            v = np.asarray(emb["global"], dtype=np.float32)
            return v if v.size else None
        except Exception:
            pass
    for k, v in (emb or {}).items():
        if isinstance(v, (list, tuple)):
            try:
                vec = np.asarray(v, dtype=np.float32)
                if vec.size:
                    return vec
            except Exception:
                continue
    return None


def build_nodes(
    run_id: str,
    manifest: Dict[str, Any],
    *,
    namespace: str = "vpm",
) -> Dict[str, NexusNode]:
    items = list(manifest.get("items") or [])
    nodes: Dict[str, NexusNode] = {}

    for it in items:
        item_id = it.get("item_id") or it.get("scorable_id") or ""
        node_id = f"{namespace}://{run_id}/{item_id}"
        scorable_type = it.get("scorable_type") or "unknown"
        scorable_id = it.get("scorable_id") or item_id
        node = _make_node(node_id, scorable_id=scorable_id, scorable_type=scorable_type)

        setattr(node, "id", node_id)
        setattr(node, "run_id", run_id)
        setattr(node, "item_id", item_id)
        setattr(node, "scorable_id", it.get("scorable_id"))

        title, text = _pick_title_text(it)
        setattr(node, "title", title)
        setattr(node, "text", text)

        setattr(node, "target_type", it.get("scorable_type") or "unknown")
        setattr(node, "chat_id", it.get("chat_id"))
        setattr(node, "turn_index", it.get("turn_index"))

        setattr(node, "domains", it.get("domains") or [])
        ents = it.get("entities")
        if isinstance(ents, dict):
            ents = list(ents.keys())
        setattr(node, "entities", ents or [])
        setattr(node, "near_identity", it.get("near_identity") or {})

        setattr(node, "metrics_columns", it.get("metrics_columns") or [])
        setattr(node, "metrics_values", it.get("metrics_values") or [])
        setattr(node, "metrics_vector", it.get("metrics_vector") or {})

        embs = it.get("embeddings") or {}
        setattr(node, "embeddings", embs)
        setattr(node, "embed_global", _pick_embed_global(embs))

        setattr(node, "vpm_png", it.get("vpm_png"))
        setattr(node, "rollout", it.get("rollout") or {})

        setattr(node, "degree", 0)
        nodes[node_id] = node

    return nodes


def _cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _z(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mu = x.mean()
    sd = x.std() + 1e-8
    return (x - mu) / sd


def _normalize_token_list(v, *, key: str | None = None) -> list[str]:
    if v is None:
        return []
    if isinstance(v, (str, int, float, bool, dict)):
        v = [v]
    out: list[str] = []
    for item in v:
        if isinstance(item, str):
            out.append(item.strip().lower())
        elif item is None:
            continue
        elif isinstance(item, dict):
            if key and (key in item) and item[key] is not None:
                out.append(str(item[key]).strip().lower())
            else:
                out.append(dumps_safe(item, sort_keys=True, ensure_ascii=False).lower())
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
    if not a or not b:
        return 0.0
    a = a[:max_len]
    b = b[:max_len]
    Sa = {a[i : i + k] for i in range(0, max(0, len(a) - k + 1))}
    Sb = {b[i : i + k] for i in range(0, max(0, len(b) - k + 1))}
    if not Sa and not Sb:
        return 0.0
    return len(Sa & Sb) / max(1, len(Sa | Sb))


def _norm01_rowwise(S: np.ndarray) -> np.ndarray:
    out = S.copy()
    for i in range(out.shape[0]):
        r = out[i]
        mn, mx = r.min(initial=0.0), r.max(initial=0.0)
        denom = (mx - mn) if (mx - mn) > 1e-8 else 1.0
        out[i] = (r - mn) / denom
    return (out + out.T) / 2.0


def _pairwise_blend(ids: List[str], nodes: Dict[str, Any], w: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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

    E = [getattr(nodes[i], "embed_global", None) for i in ids]
    M = [np.asarray(getattr(nodes[i], "metrics_values", []), dtype=np.float32) for i in ids]
    Mz = [(_z(m) if m.size else None) for m in M]
    T = [getattr(nodes[i], "text", None) for i in ids]
    D = [getattr(nodes[i], "domains", []) for i in ids]
    G = [getattr(nodes[i], "entities", []) for i in ids]
    A = [getattr(nodes[i], "agreement", None) for i in ids]
    U = [getattr(nodes[i], "stability", None) for i in ids]

    for i in range(N):
        for j in range(i + 1, N):
            s_embed = _cos(E[i], E[j])
            s_metrics = _cos(Mz[i], Mz[j]) if (Mz[i] is not None and Mz[j] is not None) else 0.0
            s_lex = _char_shingles_jaccard(T[i], T[j], k=5)
            s_dom = _jaccard_list(D[i], D[j], key="domain")
            s_ent = _jaccard_list(G[i], G[j], key="text")
            s_agree = min(A[i], A[j]) if (A[i] is not None and A[j] is not None) else 0.0
            s_stab = min(U[i], U[j]) if (U[i] is not None and U[j] is not None) else 0.0

            C["embed"][i, j] = C["embed"][j, i] = s_embed
            C["metrics"][i, j] = C["metrics"][j, i] = s_metrics
            C["lexical"][i, j] = C["lexical"][j, i] = s_lex
            C["domains"][i, j] = C["domains"][j, i] = s_dom
            C["entities"][i, j] = C["entities"][j, i] = s_ent
            C["agreement"][i, j] = C["agreement"][j, i] = s_agree
            C["stability"][i, j] = C["stability"][j, i] = s_stab

    for k in C.keys():
        C[k] = _norm01_rowwise(C[k])

    S = (
        w.get("embed", 0) * C["embed"]
        + w.get("metrics", 0) * C["metrics"]
        + w.get("lexical", 0) * C["lexical"]
        + w.get("domains", 0) * C["domains"]
        + w.get("entities", 0) * C["entities"]
        + w.get("agreement", 0) * C["agreement"]
        + w.get("stability", 0) * C["stability"]
    )
    S = _norm01_rowwise(S)
    return S, C


def _temporal_edges(items: List[Dict[str, Any]], nodes: Dict[str, Any], run_id: str, namespace: str) -> List[NexusEdge]:
    edges: List[NexusEdge] = []
    chats: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        if it.get("chat_id") and it.get("turn_index") is not None:
            chats[it["chat_id"]].append(it)
    for _, arr in chats.items():
        arr.sort(key=lambda x: x["turn_index"])
        for a, b in zip(arr, arr[1:]):
            src = f"{namespace}://{run_id}/{a['item_id']}" if run_id else f"{namespace}://{a.get('run_id','')}/{a['item_id']}"
            dst = f"{namespace}://{run_id}/{b['item_id']}" if run_id else f"{namespace}://{b.get('run_id','')}/{b['item_id']}"
            if src in nodes and dst in nodes:
                edges.append(NexusEdge(src, dst, "temporal_next", 1.0))
    return edges


def _mst_edges(ids: List[str], S: np.ndarray) -> List[NexusEdge]:
    N = len(ids)
    if N <= 1:
        return []
    dist = 1.0 - S
    in_tree = np.zeros(N, dtype=bool)
    in_tree[0] = True
    best = dist[0].copy()
    parent = np.full(N, -1, dtype=int)
    parent[0] = 0
    edges: List[NexusEdge] = []
    for _ in range(N - 1):
        j = np.argmin(np.where(in_tree, np.inf, best))
        if math.isinf(best[j]):
            break
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
