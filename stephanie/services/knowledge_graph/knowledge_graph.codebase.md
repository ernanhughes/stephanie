# Codebase Pack: knowledge_graph

```text
ROOT: C:\Project\stephanie\stephanie\services\knowledge_graph
GENERATED_AT_UTC: 2026-06-29T11:35:24.224223+00:00
PART: 1/1
FILES_IN_PART: 12
TOTAL_LINES_IN_PART: 1278
TOTAL_BYTES_UTF8_IN_PART: 44213
MODE: configured include extensions
LINE_NUMBERS: True
MAX_FILE_KB: 400
```

## How to cite this pack in review

Use the stable file ID plus line numbers, for example:

```text
F0007 `services/replay_service.py` L0042-L0068
```

## File Index

| ID | Path | Lang | Lines | KB | SHA256 |
|---|---|---:|---:|---:|---|
| F0001 | `__init__.py` | python | 1 | 0.0 | `e3b0c44298fc` |
| F0002 | `edge_enricher.py` | python | 24 | 0.6 | `c3f15bdbfdff` |
| F0003 | `entity_canonicalizer.py` | python | 17 | 0.5 | `37e29aa76203` |
| F0004 | `evolution.py` | python | 154 | 4.9 | `0d661f3c8ec6` |
| F0005 | `graph_indexer.py` | python | 327 | 11.9 | `efeeb3e88a78` |
| F0006 | `explorer/__init__.py` | python | 3 | 0.2 | `665510ed565f` |
| F0007 | `explorer/explorer_graph.py` | python | 123 | 3.5 | `26620c971edd` |
| F0008 | `explorer/explorer_graph_builder.py` | python | 147 | 5.7 | `96ab980e55e9` |
| F0009 | `explorer/explorer_graph_service.py` | python | 51 | 1.4 | `81e538c9b4ae` |
| F0010 | `subgraphs/edge_index.py` | python | 111 | 3.6 | `d3cd1884ff09` |
| F0011 | `subgraphs/seed_finder.py` | python | 85 | 2.6 | `bb0d0b305f49` |
| F0012 | `subgraphs/subgraph_builder.py` | python | 235 | 8.3 | `90a58127a2a6` |

## Directory Tree

```text
└─ __init__.py
└─ edge_enricher.py
└─ entity_canonicalizer.py
└─ evolution.py
📁 explorer
  └─ __init__.py
  └─ explorer_graph.py
  └─ explorer_graph_builder.py
  └─ explorer_graph_service.py
└─ graph_indexer.py
📁 subgraphs
  └─ edge_index.py
  └─ seed_finder.py
  └─ subgraph_builder.py
```

## Files


---

## F0001 — `__init__.py`

```text
FILE_ID: F0001
PATH: __init__.py
LANGUAGE: python
LINES: 1
BYTES_UTF8: 0
SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

```python
```


---

## F0002 — `edge_enricher.py`

```text
FILE_ID: F0002
PATH: edge_enricher.py
LANGUAGE: python
LINES: 24
BYTES_UTF8: 639
SHA256: c3f15bdbfdff75284750197da0db14e99718db247e50e5763b7b989c42ababb4
```

```python
0001 | # stephanie/services/knowledge_graph/edge_enricher.py
0002 | from typing import Dict, Optional
0003 | 
0004 | 
0005 | def enrich_relationship(
0006 |     rel: Dict,
0007 |     *,
0008 |     doc_hash: str,
0009 |     sentence_ix: Optional[int] = None,
0010 |     scorable_id: str,
0011 |     scorable_type: str,
0012 |     evidence_type: str,
0013 | ) -> Dict:
0014 |     """Standardize evidence metadata onto relationships."""
0015 |     base = {
0016 |         "doc_hash": doc_hash,
0017 |         "scorable_id": scorable_id,
0018 |         "scorable_type": scorable_type,
0019 |         "evidence_type": evidence_type,
0020 |     }
0021 |     if sentence_ix is not None:
0022 |         base["sentence_ix"] = sentence_ix
0023 |     rel.setdefault("properties", {}).update(base)
0024 |     return rel
```


---

## F0003 — `entity_canonicalizer.py`

```text
FILE_ID: F0003
PATH: entity_canonicalizer.py
LANGUAGE: python
LINES: 17
BYTES_UTF8: 523
SHA256: 37e29aa7620393c666bdc37d56b10e36ce2266cff39c1d60b6234276300be2ab
```

```python
0001 | # stephanie/services/knowledge_graph/entity_canonicalizer.py
0002 | import re
0003 | 
0004 | from stephanie.utils.hash_utils import hash_text
0005 | 
0006 | 
0007 | class EntityCanonicalizer:
0008 |     @staticmethod
0009 |     def normalize_surface(s: str) -> str:
0010 |         s = (s or "").strip()
0011 |         return re.sub(r"\s+", " ", s).lower()
0012 | 
0013 |     @staticmethod
0014 |     def canonical_id(entity_type: str, surface: str) -> str:
0015 |         et = (entity_type or "ENTITY").strip().upper()
0016 |         nh = hash_text(EntityCanonicalizer.normalize_surface(surface))
0017 |         return f"ent:{et}:{nh}"
```


---

## F0004 — `evolution.py`

```text
FILE_ID: F0004
PATH: evolution.py
LANGUAGE: python
LINES: 154
BYTES_UTF8: 5020
SHA256: 0d661f3c8ec6de51e7538d43c4ffff878b1f6180fecae620ba9eca4e351814ab
```

```python
0001 | # stephanie/services/knowledge_graph/evolution.py
0002 | from __future__ import annotations
0003 | 
0004 | import hashlib
0005 | import json
0006 | import os
0007 | from dataclasses import dataclass
0008 | from datetime import datetime, timezone
0009 | from typing import Any, Dict, List, Optional, Tuple
0010 | 
0011 | 
0012 | def _hash_query(q: str) -> str:
0013 |     qn = " ".join((q or "").lower().split())
0014 |     return hashlib.sha256(qn.encode("utf-8")).hexdigest()[:16]
0015 | 
0016 | 
0017 | def _edge_key(e: Dict[str, Any]) -> Tuple[str, str, str]:
0018 |     # accept both {src,dst} and {source,target}
0019 |     src = e.get("src") or e.get("source") or e.get("source_id")
0020 |     dst = e.get("dst") or e.get("target") or e.get("target_id")
0021 |     typ = e.get("type") or e.get("rel_type")
0022 |     return (str(src), str(dst), str(typ))
0023 | 
0024 | 
0025 | def _node_key(n: Dict[str, Any]) -> str:
0026 |     return str(n.get("id"))
0027 | 
0028 | 
0029 | def _quantile(xs: List[float], q: float) -> float:
0030 |     if not xs:
0031 |         return 0.0
0032 |     xs = sorted(xs)
0033 |     i = int(q * (len(xs) - 1))
0034 |     return float(xs[i])
0035 | 
0036 | 
0037 | @dataclass
0038 | class SnapshotRef:
0039 |     path: str
0040 |     offset: int  # byte offset (optional)
0041 | 
0042 | 
0043 | class KGEvolutionTracker:
0044 |     def __init__(self, *, log_path: str, logger: Any):
0045 |         self.log_path = log_path
0046 |         self.logger = logger
0047 |         os.makedirs(os.path.dirname(log_path), exist_ok=True)
0048 | 
0049 |     def snapshot_subgraph(
0050 |         self,
0051 |         *,
0052 |         query: str,
0053 |         subgraph: Dict[str, Any],
0054 |         version: str = "kg_v1",
0055 |         run_id: str = "kg_live",
0056 |         stage: str = "raw",
0057 |         extra: Optional[Dict[str, Any]] = None,
0058 |     ) -> None:
0059 |         nodes = subgraph.get("nodes", []) or []
0060 |         edges = (
0061 |             subgraph.get("edges", [])
0062 |             or subgraph.get("relationships", [])
0063 |             or []
0064 |         )
0065 | 
0066 |         confs = [
0067 |             float(e.get("confidence", 0.0))
0068 |             for e in edges
0069 |             if e.get("confidence") is not None
0070 |         ]
0071 |         has_evidence = 0
0072 |         for e in edges:
0073 |             # treat doc_hash or evidence_type as evidence-carrying
0074 |             props = e.get("properties") if isinstance(e.get("properties"), dict) else {}
0075 |             if e.get("doc_hash") or e.get("evidence_type") or props.get("doc_hash") or props.get("evidence_type"):
0076 |                 has_evidence += 1
0077 |         evidence_rate = has_evidence / max(1, len(edges))
0078 | 
0079 |         rec = {
0080 |             "ts": datetime.now(timezone.utc).isoformat(),
0081 |             "kind": "subgraph",
0082 |             "version": version,
0083 |             "run_id": run_id,
0084 |             "stage": stage,
0085 |             "scope": {
0086 |                 "query": query,
0087 |                 "query_hash": _hash_query(query),
0088 |             },
0089 |             "nodes": nodes,
0090 |             "edges": edges,
0091 |             "stats": {
0092 |                 "node_count": len(nodes),
0093 |                 "edge_count": len(edges),
0094 |                 "evidence_rate": float(evidence_rate),
0095 |                 "confidence_p50": _quantile(confs, 0.50),
0096 |                 "confidence_p10": _quantile(confs, 0.10),
0097 |                 "confidence_p90": _quantile(confs, 0.90),
0098 |             },
0099 |         }
0100 |         if extra:
0101 |             rec["extra"] = extra
0102 | 
0103 |         with open(self.log_path, "a", encoding="utf-8") as f:
0104 |             f.write(json.dumps(rec, ensure_ascii=False) + "\n")
0105 | 
0106 |     def compare_snapshots(
0107 |         self, prev: Dict[str, Any], curr: Dict[str, Any]
0108 |     ) -> Dict[str, Any]:
0109 |         prev_nodes = {_node_key(n) for n in (prev.get("nodes") or [])}
0110 |         curr_nodes = {_node_key(n) for n in (curr.get("nodes") or [])}
0111 | 
0112 |         prev_edges = {_edge_key(e) for e in (prev.get("edges") or [])}
0113 |         curr_edges = {_edge_key(e) for e in (curr.get("edges") or [])}
0114 | 
0115 |         node_j = len(prev_nodes & curr_nodes) / max(
0116 |             1, len(prev_nodes | curr_nodes)
0117 |         )
0118 |         edge_j = len(prev_edges & curr_edges) / max(
0119 |             1, len(prev_edges | curr_edges)
0120 |         )
0121 | 
0122 |         added_edges = list(curr_edges - prev_edges)
0123 |         removed_edges = list(prev_edges - curr_edges)
0124 | 
0125 |         return {
0126 |             "node_jaccard": float(node_j),
0127 |             "edge_jaccard": float(edge_j),
0128 |             "added_edge_count": len(added_edges),
0129 |             "removed_edge_count": len(removed_edges),
0130 |             "added_edges_sample": [
0131 |                 f"{a} -[{t}]-> {b}" for (a, b, t) in added_edges[:10]
0132 |             ],
0133 |             "removed_edges_sample": [
0134 |                 f"{a} -[{t}]-> {b}" for (a, b, t) in removed_edges[:10]
0135 |             ],
0136 |             "prev_stats": prev.get("stats", {}),
0137 |             "curr_stats": curr.get("stats", {}),
0138 |         }
0139 | 
0140 |     def load_snapshots_for_query(
0141 |         self, *, query_hash: str, limit: int = 50
0142 |     ) -> List[Dict[str, Any]]:
0143 |         if not os.path.exists(self.log_path):
0144 |             return []
0145 |         out = []
0146 |         with open(self.log_path, "r", encoding="utf-8") as f:
0147 |             for line in f:
0148 |                 if not line.strip():
0149 |                     continue
0150 |                 rec = json.loads(line)
0151 |                 if (rec.get("scope") or {}).get("query_hash") == query_hash:
0152 |                     out.append(rec)
0153 |         out.sort(key=lambda r: r.get("ts", ""))
0154 |         return out[-limit:]
```


---

## F0005 — `graph_indexer.py`

```text
FILE_ID: F0005
PATH: graph_indexer.py
LANGUAGE: python
LINES: 327
BYTES_UTF8: 12223
SHA256: efeeb3e88a784abe827099f168e573fc04ededdf455fa1552e90a03d17d10a8b
```

```python
0001 | # stephanie/services/knowledge_graph/graph_indexer.py
0002 | from __future__ import annotations
0003 | 
0004 | import logging
0005 | from typing import Any, Dict, List, Optional
0006 | 
0007 | from stephanie.utils.hash_utils import hash_text
0008 | from stephanie.utils.text_utils import safe_slice, sentences
0009 | 
0010 | log = logging.getLogger(__name__)
0011 | 
0012 | class GraphIndexer:
0013 |     def __init__(self, *, kg_service: Any, 
0014 |                  detect_entities_fn: Any,
0015 |                  fetch_text_fn: Any,
0016 |                  logger: Any):
0017 |         from stephanie.services.knowledge_graph_service import \
0018 |             KnowledgeGraphService
0019 |         self.kg: KnowledgeGraphService = kg_service
0020 |         self.detect_entities_fn = detect_entities_fn
0021 |         self.fetch_text_fn = fetch_text_fn
0022 |         self.logger = logger
0023 | 
0024 |     def _normalize_text(self, t: str) -> str:
0025 |         return (t or "").replace("\u200b", "").strip()
0026 | 
0027 |     def _locate_sentence_ix(self, sent_spans, start: int, end: int) -> int:
0028 |         for i, (s, e) in enumerate(sent_spans):
0029 |             if not (end <= s or start >= e):
0030 |                 return i
0031 |         return -1
0032 | 
0033 |     async def _fetch_text_for_indexing(self, scorable_id: str, payload_text: Optional[str]) -> Optional[str]:
0034 |         # 1) try service hook if present
0035 |         try:
0036 |             fn = getattr(self.kg, "fetch_text_for_scorable", None)
0037 |             if callable(fn):
0038 |                 t = fn(scorable_id)
0039 |                 if t:
0040 |                     return t
0041 |         except Exception as ex:
0042 |             log.warning(f"GraphIndexer: fetch_text_for_scorable failed: {ex}")
0043 | 
0044 |         # 2) legacy: if service has scorable_store and a getter
0045 |         try:
0046 |             store = getattr(self.kg, "scorable_store", None)
0047 |             if store and hasattr(store, "get_by_scorable_id"):
0048 |                 row = store.get_by_scorable_id(scorable_id)
0049 |                 t = getattr(row, "text", None)
0050 |                 if t:
0051 |                     return t
0052 |         except Exception as ex:
0053 |             log.warning(f"GraphIndexer: scorable_store lookup failed: {ex}")
0054 | 
0055 |         # 3) fallback
0056 |         return payload_text
0057 | 
0058 |     def _mention_id(self, scorable_id: str, ent: Dict[str, Any]) -> str:
0059 |         return f"{scorable_id}:{ent['type']}:{ent['start']}-{ent['end']}"
0060 | 
0061 |     def _repair_entities(
0062 |         self,
0063 |         *,
0064 |         entities: List[Dict[str, Any]],
0065 |         text: str,
0066 |         sent_spans,
0067 |         doc_hash: str,
0068 |     ) -> List[Dict[str, Any]]:
0069 |         fixed: List[Dict[str, Any]] = []
0070 |         for ent in (entities or []):
0071 |             etype = (ent.get("type") or "UNKNOWN").strip() or "UNKNOWN"
0072 |             surface = ent.get("text")
0073 | 
0074 |             try:
0075 |                 start = int(ent.get("start", -1))
0076 |                 end = int(ent.get("end", -1))
0077 |             except Exception:
0078 |                 start, end = -1, -1
0079 | 
0080 |             etype = (ent.get("type") or "UNKNOWN").strip() or "UNKNOWN"
0081 |             surface = ent.get("text")
0082 | 
0083 |             # repair offsets if needed using surface search
0084 |             if start < 0 or end <= start or end > len(text):
0085 |                 if surface:
0086 |                     pos = text.find(surface)
0087 |                     if pos >= 0:
0088 |                         start, end = pos, pos + len(surface)
0089 |                     else:
0090 |                         continue
0091 |                 else:
0092 |                     continue
0093 | 
0094 |             derived = safe_slice(text, start, end)
0095 |             if not surface or surface != derived:
0096 |                 surface = derived
0097 | 
0098 |             sentence_ix = self._locate_sentence_ix(sent_spans, start, end)
0099 | 
0100 |             if 0 <= sentence_ix < len(sent_spans):
0101 |                 s0, s1 = sent_spans[sentence_ix]
0102 |                 context = text[s0:s1]
0103 |             else:
0104 |                 context = safe_slice(text, max(0, start - 60), min(len(text), end + 60))
0105 | 
0106 |             fixed.append(
0107 |                 {
0108 |                     **ent,
0109 |                     "start": start,
0110 |                     "end": end,
0111 |                     "type": etype,
0112 |                     "text": surface,
0113 |                     "sentence_ix": sentence_ix,
0114 |                     "context": context,
0115 |                     "doc_hash": doc_hash,
0116 |                 }
0117 |             )
0118 |         return fixed
0119 | 
0120 |     def _cooccur_edges(
0121 |         self,
0122 |         *,
0123 |         scorable_id: str,
0124 |         scorable_type: str,
0125 |         doc_hash: str,
0126 |         fixed_entities: List[Dict[str, Any]],
0127 |     ) -> List[Dict[str, Any]]:
0128 |         by_sent: Dict[int, List[Dict[str, Any]]] = {}
0129 |         for e in fixed_entities:
0130 |             by_sent.setdefault(int(e.get("sentence_ix", -1)), []).append(e)
0131 | 
0132 |         rels: List[Dict[str, Any]] = []
0133 |         for s_ix, ents in by_sent.items():
0134 |             if s_ix < 0 or len(ents) < 2:
0135 |                 continue
0136 |             for i in range(len(ents)):
0137 |                 for j in range(i + 1, len(ents)):
0138 |                     a, b = ents[i], ents[j]
0139 |                     rels.append(
0140 |                         {
0141 |                             "source": self._mention_id(scorable_id, a),
0142 |                             "target": self._mention_id(scorable_id, b),
0143 |                             "type": "CO_OCCURS_IN_SENTENCE",
0144 |                             "confidence": 1.0,
0145 |                             "doc_hash": doc_hash,
0146 |                             "sentence_ix": s_ix,
0147 |                             "scorable_id": scorable_id,
0148 |                             "scorable_type": scorable_type,
0149 |                             "evidence_type": "sentence_cooccurrence",
0150 |                         }
0151 |                     )
0152 |         return rels
0153 | 
0154 | 
0155 | 
0156 |     def _build_relationships(
0157 |         self,
0158 |         *,
0159 |         fixed_entities: List[Dict[str, Any]],
0160 |         domains: List[Dict[str, Any]],
0161 |         scorable_id: str,
0162 |         scorable_type: str,
0163 |         doc_hash: str,
0164 |     ) -> List[Dict[str, Any]]:
0165 |         rels: List[Dict[str, Any]] = []
0166 | 
0167 |         # 1) typed rels (delegate to service if present)
0168 |         try:
0169 |             fn = getattr(self.kg, "_build_relationships", None)
0170 |             if callable(fn):
0171 |                 # expects (entities, domains, scorable_id) in your service
0172 |                 typed = fn(fixed_entities, domains, scorable_id) or []
0173 |                 # normalize keys to {source,target,type,confidence}
0174 |                 for r in typed:
0175 |                     rels.append(
0176 |                         {
0177 |                             "source": r.get("source") or r.get("source_id"),
0178 |                             "target": r.get("target") or r.get("target_id"),
0179 |                             "type": r.get("type") or r.get("rel_type") or "RELATED",
0180 |                             "confidence": float(r.get("confidence", 0.0) or 0.0),
0181 |                             "properties": r.get("properties") or {},
0182 |                         }
0183 |                     )
0184 |         except Exception as ex:
0185 |             log.warning(f"GraphIndexer: typed _build_relationships failed: {ex}")
0186 | 
0187 |         # 2) sentence co-occurrence (cheap signal)
0188 |         by_sent: Dict[int, List[Dict[str, Any]]] = {}
0189 |         for e in fixed_entities:
0190 |             by_sent.setdefault(int(e.get("sentence_ix", -1)), []).append(e)
0191 | 
0192 |         for s_ix, ents in by_sent.items():
0193 |             if s_ix < 0 or len(ents) < 2:
0194 |                 continue
0195 |             for i in range(len(ents)):
0196 |                 for j in range(i + 1, len(ents)):
0197 |                     a, b = ents[i], ents[j]
0198 |                     rels.append(
0199 |                         {
0200 |                             "source": f"{scorable_id}:{a['type']}:{a['start']}-{a['end']}",
0201 |                             "target": f"{scorable_id}:{b['type']}:{b['start']}-{b['end']}",
0202 |                             "type": "CO_OCCURS_IN_SENTENCE",
0203 |                             "confidence": 1.0,
0204 |                             "properties": {
0205 |                                 "doc_hash": doc_hash,
0206 |                                 "sentence_ix": s_ix,
0207 |                                 "scorable_id": scorable_id,
0208 |                                 "scorable_type": scorable_type,
0209 |                                 "evidence_type": "sentence_cooccurrence",
0210 |                             },
0211 |                         }
0212 |                     )
0213 | 
0214 |         return rels
0215 | 
0216 |     def _mention_id(self, scorable_id: str, ent: Dict[str, Any]) -> str:
0217 |         return f"{scorable_id}:{ent['type']}:{ent['start']}-{ent['end']}"
0218 | 
0219 |     async def handle_index_request(self, payload: Dict[str, Any]) -> None:
0220 |         scorable_id = payload["scorable_id"]
0221 |         scorable_type = payload.get("scorable_type", "unknown")
0222 |         entities = payload.get("entities") or []
0223 |         domains = payload.get("domains") or []
0224 | 
0225 |         text = await self.fetch_text_fn(scorable_id, payload.get("text"), payload)
0226 |         if not text:
0227 |             await self.kg.publish("knowledge_graph.index_failed", {"scorable_id": scorable_id, "error": "Missing text"})
0228 |             return
0229 | 
0230 |         text = self._normalize_text(text)
0231 |         doc_hash = hash_text(text)
0232 |         sent_spans = sentences(text)
0233 | 
0234 |         entities = payload.get("entities") or []
0235 |         if not entities:
0236 |             try:
0237 |                 entities = self.detect_entities_fn(text) or []
0238 |             except Exception:
0239 |                 log.warning("GraphIndexer: detect_entities_fn failed", exc_info=True)
0240 |                 entities = []
0241 | 
0242 |         fixed_entities = self._repair_entities(
0243 |             entities=entities,
0244 |             text=text,
0245 |             sent_spans=sent_spans,
0246 |             doc_hash=doc_hash,
0247 |         )
0248 | 
0249 |         if not fixed_entities:
0250 |             await self.kg.publish("knowledge_graph.index_failed", {"scorable_id": scorable_id, "error": "No entities"})
0251 |             return
0252 | 
0253 |         # upsert mention + canonical + MENTIONS edge
0254 |         for ent in fixed_entities:
0255 |             mention_id = self._mention_id(scorable_id, ent)
0256 | 
0257 |             # mention node
0258 |             self.kg.upsert_node(
0259 |                 node_id=mention_id,
0260 |                 properties={
0261 |                     "type": "entity_mention",
0262 |                     "text": ent["text"],
0263 |                     "entity_type": ent.get("type", "UNKNOWN"),
0264 |                     "scorable_id": scorable_id,
0265 |                     "scorable_type": scorable_type,
0266 |                     "doc_hash": doc_hash,
0267 |                     "sentence_ix": ent.get("sentence_ix"),
0268 |                     "context": ent.get("context"),
0269 |                     "start": ent.get("start"),
0270 |                     "end": ent.get("end"),
0271 |                     "domains": [d.get("domain") for d in domains if isinstance(d, dict)],
0272 |                 },
0273 |             )
0274 | 
0275 |             canon_id = self.kg._canonical_entity_id(ent)
0276 |             self.kg.upsert_node(
0277 |                 node_id=canon_id,
0278 |                 properties={
0279 |                     "type": "canonical_entity",
0280 |                     "text": ent["text"],
0281 |                     "entity_type": ent.get("type", "UNKNOWN"),
0282 |                     "domains": [d.get("domain") for d in domains if isinstance(d, dict)],
0283 |                 },
0284 |             )
0285 | 
0286 |             self.kg.upsert_edge(
0287 |                 source_id=mention_id,
0288 |                 target_id=canon_id,
0289 |                 rel_type="MENTIONS",
0290 |                 properties={
0291 |                     "confidence": 0.95,
0292 |                     "doc_hash": doc_hash,
0293 |                     "sentence_ix": ent.get("sentence_ix"),
0294 |                     "scorable_id": scorable_id,
0295 |                     "scorable_type": scorable_type,
0296 |                     "evidence_type": "entity_mention_link",
0297 |                 },
0298 |             )
0299 | 
0300 |         # relationships (typed + co-occurrence)
0301 |         rels = self._build_relationships(
0302 |             fixed_entities=fixed_entities,
0303 |             domains=domains,
0304 |             scorable_id=scorable_id,
0305 |             scorable_type=scorable_type,
0306 |             doc_hash=doc_hash,
0307 |         )
0308 |         for rel in rels:
0309 |             self.kg.upsert_edge(
0310 |                 source_id=rel["source"],
0311 |                 target_id=rel["target"],
0312 |                 rel_type=rel["type"],
0313 |                 properties={
0314 |                     "confidence": float(rel.get("confidence", 1.0)),
0315 |                     **(rel.get("properties") or {}),
0316 |                 },
0317 |             )
0318 | 
0319 |         await self.kg.publish(
0320 |             "knowledge_graph.index_complete",
0321 |             {
0322 |                 "scorable_id": scorable_id,
0323 |                 "node_count": len(fixed_entities),
0324 |                 "relationship_count": max(0, len(fixed_entities) - 1),
0325 |                 "doc_hash": doc_hash,
0326 |             },
0327 |         )
```


---

## F0006 — `explorer/__init__.py`

```text
FILE_ID: F0006
PATH: explorer/__init__.py
LANGUAGE: python
LINES: 3
BYTES_UTF8: 185
SHA256: 665510ed565fc0a6cbfe1a372725abe0cca10ccc5c3229ce1b81fdff187e4fe6
```

```python
0001 | # stephanie/services/knowledge_graph/explorer/__init__.py
0002 | from .explorer_graph import ExplorerGraph, ExplorerNode, ExplorerEdge
0003 | from .explorer_graph_service import ExplorerGraphService
```


---

## F0007 — `explorer/explorer_graph.py`

```text
FILE_ID: F0007
PATH: explorer/explorer_graph.py
LANGUAGE: python
LINES: 123
BYTES_UTF8: 3579
SHA256: 26620c971edd03b7088a06eb22a3bbb0cb6aef3ae852a7453e5b8418d7459821
```

```python
0001 | # stephanie/services/knowledge_graph/explorer/explorer_graph.py
0002 | from __future__ import annotations
0003 | 
0004 | from dataclasses import dataclass, field
0005 | from typing import Any, Dict, List, Optional, Iterable
0006 | from enum import Enum
0007 | 
0008 | 
0009 | class NodeType(str, Enum):
0010 |     PLAN_TRACE = "plan_trace"
0011 |     EXEC_STEP = "execution_step"
0012 |     EVALUATION = "evaluation"
0013 |     SCORE = "score"
0014 |     SCORABLE = "scorable"          # future generic
0015 |     CASE = "case"                  # future
0016 |     CANDIDATE = "candidate"        # future
0017 |     EVIDENCE_CARD = "evidence_card" # future
0018 |     BUNDLE = "bundle"              # future
0019 |     DRAFT = "draft_variant"        # future
0020 |     ARENA_MATCH = "arena_match"    # future
0021 | 
0022 | 
0023 | class EdgeType(str, Enum):
0024 |     HAS_STEP = "has_step"
0025 |     NEXT = "next"
0026 |     HAS_EVALUATION = "has_evaluation"
0027 |     HAS_SCORE = "has_score"
0028 | 
0029 | 
0030 | def make_node_id(node_type: str, node_pk: Any) -> str:
0031 |     # Stable canonical identity in the graph
0032 |     return f"{node_type}:{node_pk}"
0033 | 
0034 | 
0035 | @dataclass
0036 | class ExplorerNode:
0037 |     node_id: str
0038 |     node_type: str
0039 |     label: str = ""
0040 |     meta: Dict[str, Any] = field(default_factory=dict)
0041 | 
0042 |     def to_dict(self) -> dict:
0043 |         return {
0044 |             "node_id": self.node_id,
0045 |             "node_type": self.node_type,
0046 |             "label": self.label,
0047 |             "meta": self.meta,
0048 |         }
0049 | 
0050 |     @staticmethod
0051 |     def from_dict(d: dict) -> "ExplorerNode":
0052 |         return ExplorerNode(
0053 |             node_id=d["node_id"],
0054 |             node_type=d["node_type"],
0055 |             label=d.get("label", ""),
0056 |             meta=d.get("meta", {}) or {},
0057 |         )
0058 | 
0059 | 
0060 | @dataclass
0061 | class ExplorerEdge:
0062 |     src: str
0063 |     dst: str
0064 |     edge_type: str
0065 |     weight: Optional[float] = None
0066 |     meta: Dict[str, Any] = field(default_factory=dict)
0067 | 
0068 |     def to_dict(self) -> dict:
0069 |         return {
0070 |             "src": self.src,
0071 |             "dst": self.dst,
0072 |             "edge_type": self.edge_type,
0073 |             "weight": self.weight,
0074 |             "meta": self.meta,
0075 |         }
0076 | 
0077 |     @staticmethod
0078 |     def from_dict(d: dict) -> "ExplorerEdge":
0079 |         return ExplorerEdge(
0080 |             src=d["src"],
0081 |             dst=d["dst"],
0082 |             edge_type=d["edge_type"],
0083 |             weight=d.get("weight"),
0084 |             meta=d.get("meta", {}) or {},
0085 |         )
0086 | 
0087 | 
0088 | @dataclass
0089 | class ExplorerGraph:
0090 |     root_id: str
0091 |     nodes: Dict[str, ExplorerNode] = field(default_factory=dict)
0092 |     edges: List[ExplorerEdge] = field(default_factory=list)
0093 |     meta: Dict[str, Any] = field(default_factory=dict)
0094 | 
0095 |     def add_node(self, node: ExplorerNode) -> ExplorerNode:
0096 |         self.nodes[node.node_id] = node
0097 |         return node
0098 | 
0099 |     def add_edge(self, edge: ExplorerEdge) -> ExplorerEdge:
0100 |         self.edges.append(edge)
0101 |         return edge
0102 | 
0103 |     def get_or_add_node(self, node_id: str, node_type: str, label: str = "", meta: Optional[dict] = None) -> ExplorerNode:
0104 |         if node_id in self.nodes:
0105 |             return self.nodes[node_id]
0106 |         return self.add_node(ExplorerNode(node_id=node_id, node_type=node_type, label=label, meta=meta or {}))
0107 | 
0108 |     def to_dict(self) -> dict:
0109 |         return {
0110 |             "root_id": self.root_id,
0111 |             "nodes": [n.to_dict() for n in self.nodes.values()],
0112 |             "edges": [e.to_dict() for e in self.edges],
0113 |             "meta": self.meta,
0114 |         }
0115 | 
0116 |     @staticmethod
0117 |     def from_dict(d: dict) -> "ExplorerGraph":
0118 |         g = ExplorerGraph(root_id=d["root_id"], meta=d.get("meta", {}) or {})
0119 |         for nd in d.get("nodes", []):
0120 |             g.add_node(ExplorerNode.from_dict(nd))
0121 |         for ed in d.get("edges", []):
0122 |             g.add_edge(ExplorerEdge.from_dict(ed))
0123 |         return g
```


---

## F0008 — `explorer/explorer_graph_builder.py`

```text
FILE_ID: F0008
PATH: explorer/explorer_graph_builder.py
LANGUAGE: python
LINES: 147
BYTES_UTF8: 5824
SHA256: 96ab980e55e933a6c95a3c252716e6c17e638d7aee8b05777d3e3c540f15cf10
```

```python
0001 | # stephanie/services/knowledge_graph/explorer/explorer_graph_builder.py
0002 | from __future__ import annotations
0003 | 
0004 | from typing import Optional
0005 | 
0006 | from .explorer_graph import (
0007 |     ExplorerGraph, ExplorerEdge, ExplorerNode,
0008 |     NodeType, EdgeType, make_node_id
0009 | )
0010 | 
0011 | # NOTE: We intentionally type-hint as "Any" to avoid import cycles.
0012 | # In your repo you can replace these with real ORM imports if you want.
0013 | from typing import Any
0014 | 
0015 | 
0016 | class ExplorerGraphBuilder:
0017 |     """
0018 |     Builds an ExplorerGraph from a PlanTraceORM, optionally including:
0019 |     - ExecutionStep nodes
0020 |     - Evaluation nodes
0021 |     - Score nodes (ScoreORM rows)
0022 |     """
0023 | 
0024 |     def build_from_plan_trace(
0025 |         self,
0026 |         plan_trace: Any,
0027 |         *,
0028 |         include_evaluations: bool = True,
0029 |         include_scores: bool = True,
0030 |     ) -> ExplorerGraph:
0031 |         trace_node_id = make_node_id(NodeType.PLAN_TRACE.value, plan_trace.id)
0032 |         g = ExplorerGraph(
0033 |             root_id=trace_node_id,
0034 |             meta={
0035 |                 "plan_trace_id": plan_trace.id,
0036 |                 "pipeline_run_id": getattr(plan_trace, "pipeline_run_id", None),
0037 |                 "version": "v1",
0038 |             },
0039 |         )
0040 | 
0041 |         # Root node
0042 |         g.get_or_add_node(
0043 |             trace_node_id,
0044 |             NodeType.PLAN_TRACE.value,
0045 |             label=f"PlanTrace {plan_trace.id}",
0046 |             meta={
0047 |                 "trace_id": getattr(plan_trace, "trace_id", None),
0048 |                 "task_type": getattr(plan_trace, "task_type", None),
0049 |                 "goal_text": getattr(plan_trace, "goal_text", None),
0050 |             },
0051 |         )
0052 | 
0053 |         steps = sorted(getattr(plan_trace, "execution_steps", []) or [], key=lambda s: s.step_order)
0054 | 
0055 |         prev_step_node_id: Optional[str] = None
0056 |         for step in steps:
0057 |             step_node_id = make_node_id(NodeType.EXEC_STEP.value, step.id)
0058 | 
0059 |             g.get_or_add_node(
0060 |                 step_node_id,
0061 |                 NodeType.EXEC_STEP.value,
0062 |                 label=f"{step.step_order}: {step.step_id}",
0063 |                 meta={
0064 |                     "step_order": step.step_order,
0065 |                     "step_id": step.step_id,
0066 |                     "step_type": getattr(step, "step_type", None),
0067 |                     "agent_role": getattr(step, "agent_role", None),
0068 |                     "description": step.description,
0069 |                     "evaluation_id": getattr(step, "evaluation_id", None),
0070 |                     "output_embedding_id": getattr(step, "output_embedding_id", None),
0071 |                     # store output_text only if you want (can be big)
0072 |                     # "output_text": step.output_text,
0073 |                 },
0074 |             )
0075 | 
0076 |             # Trace -> Step edge
0077 |             g.add_edge(ExplorerEdge(
0078 |                 src=trace_node_id,
0079 |                 dst=step_node_id,
0080 |                 edge_type=EdgeType.HAS_STEP.value,
0081 |                 meta={"step_order": step.step_order},
0082 |             ))
0083 | 
0084 |             # Step ordering edges
0085 |             if prev_step_node_id is not None:
0086 |                 g.add_edge(ExplorerEdge(
0087 |                     src=prev_step_node_id,
0088 |                     dst=step_node_id,
0089 |                     edge_type=EdgeType.NEXT.value,
0090 |                     meta={},
0091 |                 ))
0092 |             prev_step_node_id = step_node_id
0093 | 
0094 |             # Evaluation + scores
0095 |             if include_evaluations and getattr(step, "evaluation", None) is not None:
0096 |                 ev = step.evaluation
0097 |                 ev_node_id = make_node_id(NodeType.EVALUATION.value, ev.id)
0098 | 
0099 |                 g.get_or_add_node(
0100 |                     ev_node_id,
0101 |                     NodeType.EVALUATION.value,
0102 |                     label=f"Evaluation {ev.id}",
0103 |                     meta={
0104 |                         "goal_id": getattr(ev, "goal_id", None),
0105 |                         "plan_trace_id": getattr(ev, "plan_trace_id", None),
0106 |                         "pipeline_run_id": getattr(ev, "pipeline_run_id", None),
0107 |                         "scorable_type": getattr(ev, "scorable_type", None),
0108 |                         "scorable_id": getattr(ev, "scorable_id", None),
0109 |                         "query_type": getattr(ev, "query_type", None),
0110 |                         "query_id": getattr(ev, "query_id", None),
0111 |                         "symbolic_rule_id": getattr(ev, "symbolic_rule_id", None),
0112 |                         "reasoning_strategy": getattr(ev, "reasoning_strategy", None),
0113 |                     },
0114 |                 )
0115 | 
0116 |                 g.add_edge(ExplorerEdge(
0117 |                     src=step_node_id,
0118 |                     dst=ev_node_id,
0119 |                     edge_type=EdgeType.HAS_EVALUATION.value,
0120 |                     meta={},
0121 |                 ))
0122 | 
0123 |                 if include_scores:
0124 |                     for s in getattr(ev, "dimension_scores", []) or []:
0125 |                         score_node_id = make_node_id(NodeType.SCORE.value, s.id)
0126 |                         g.get_or_add_node(
0127 |                             score_node_id,
0128 |                             NodeType.SCORE.value,
0129 |                             label=f"{s.dimension}={s.score}",
0130 |                             meta={
0131 |                                 "dimension": s.dimension,
0132 |                                 "score": s.score,
0133 |                                 "weight": getattr(s, "weight", None),
0134 |                                 "source": getattr(s, "source", None),
0135 |                                 "prompt_hash": getattr(s, "prompt_hash", None),
0136 |                                 # rationale can be big; include or not
0137 |                                 "rationale": getattr(s, "rationale", None),
0138 |                             },
0139 |                         )
0140 |                         g.add_edge(ExplorerEdge(
0141 |                             src=ev_node_id,
0142 |                             dst=score_node_id,
0143 |                             edge_type=EdgeType.HAS_SCORE.value,
0144 |                             meta={},
0145 |                         ))
0146 | 
0147 |         return g
```


---

## F0009 — `explorer/explorer_graph_service.py`

```text
FILE_ID: F0009
PATH: explorer/explorer_graph_service.py
LANGUAGE: python
LINES: 51
BYTES_UTF8: 1432
SHA256: 81e538c9b4ae82485fc37107d48e3def757a45507a2b2678840e50d7946e963e
```

```python
0001 | # stephanie/services/knowledge_graph/explorer/explorer_graph_service.py
0002 | from __future__ import annotations
0003 | 
0004 | from typing import Any, Optional
0005 | 
0006 | from .explorer_graph_builder import ExplorerGraphBuilder
0007 | from .explorer_graph import ExplorerGraph
0008 | 
0009 | 
0010 | class ExplorerGraphService:
0011 |     def __init__(self, db_session: Any):
0012 |         self.db = db_session
0013 |         self.builder = ExplorerGraphBuilder()
0014 | 
0015 |     def build_and_attach(
0016 |         self,
0017 |         plan_trace: Any,
0018 |         *,
0019 |         include_evaluations: bool = True,
0020 |         include_scores: bool = True,
0021 |         meta_key: str = "explorer_graph_v1",
0022 |         commit: bool = True,
0023 |     ) -> ExplorerGraph:
0024 |         g = self.builder.build_from_plan_trace(
0025 |             plan_trace,
0026 |             include_evaluations=include_evaluations,
0027 |             include_scores=include_scores,
0028 |         )
0029 | 
0030 |         # Attach to PlanTrace.meta (JSON)
0031 |         meta = getattr(plan_trace, "meta", None) or {}
0032 |         meta[meta_key] = g.to_dict()
0033 |         plan_trace.meta = meta
0034 | 
0035 |         self.db.add(plan_trace)
0036 |         if commit:
0037 |             self.db.commit()
0038 | 
0039 |         return g
0040 | 
0041 |     def load_from_plan_trace(
0042 |         self,
0043 |         plan_trace: Any,
0044 |         *,
0045 |         meta_key: str = "explorer_graph_v1",
0046 |     ) -> Optional[ExplorerGraph]:
0047 |         meta = getattr(plan_trace, "meta", None) or {}
0048 |         payload = meta.get(meta_key)
0049 |         if not payload:
0050 |             return None
0051 |         return ExplorerGraph.from_dict(payload)
```


---

## F0010 — `subgraphs/edge_index.py`

```text
FILE_ID: F0010
PATH: subgraphs/edge_index.py
LANGUAGE: python
LINES: 111
BYTES_UTF8: 3640
SHA256: d3cd1884ff09cdd32e1c38befb1966b61d5412fc6d37e68bb161e57e2df8a363
```

```python
0001 | # stephanie/services/knowledge_graph/subgraphs/edge_index.py
0002 | from __future__ import annotations
0003 | 
0004 | import json
0005 | from collections import defaultdict
0006 | from pathlib import Path
0007 | from typing import Any, DefaultDict, Dict, Iterable, List
0008 | 
0009 | 
0010 | def _rel_get(rel: Dict[str, Any], *keys: str, default=None) -> Any:
0011 |     for k in keys:
0012 |         if k in rel:
0013 |             return rel[k]
0014 |     return default
0015 | 
0016 | 
0017 | class JSONLEdgeIndex:
0018 |     """
0019 |     Loads a JSONL relationship file once, builds adjacency maps, and serves neighbors fast.
0020 | 
0021 |     Expected edge schema (flexible):
0022 |       source / source_id
0023 |       target / target_id
0024 |       type   / rel_type
0025 |       confidence (optional)
0026 |       evidence fields (optional)
0027 |     """
0028 | 
0029 |     def __init__(self, *, rel_path: str | Path, logger: Any = None) -> None:
0030 |         self.rel_path = Path(rel_path)
0031 |         self.logger = logger
0032 | 
0033 |         self._loaded = False
0034 |         self._by_src: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
0035 |         self._by_dst: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
0036 |         self._edge_count = 0
0037 | 
0038 |     def ensure_loaded(self) -> None:
0039 |         if self._loaded:
0040 |             return
0041 | 
0042 |         if not self.rel_path.exists():
0043 |             if self.logger:
0044 |                 self.logger.warning(f"JSONLEdgeIndex: rel_path missing: {self.rel_path}")
0045 |             self._loaded = True
0046 |             return
0047 | 
0048 |         try:
0049 |             with self.rel_path.open("r", encoding="utf-8") as f:
0050 |                 for line in f:
0051 |                     line = line.strip()
0052 |                     if not line:
0053 |                         continue
0054 |                     e = json.loads(line)
0055 | 
0056 |                     s = str(_rel_get(e, "source", "source_id", default="")).strip()
0057 |                     t = str(_rel_get(e, "target", "target_id", default="")).strip()
0058 |                     if not s or not t:
0059 |                         continue
0060 | 
0061 |                     self._by_src[s].append(e)
0062 |                     self._by_dst[t].append(e)
0063 |                     self._edge_count += 1
0064 | 
0065 |             self._loaded = True
0066 |             if self.logger:
0067 |                 self.logger.info(
0068 |                     f"JSONLEdgeIndex loaded {self._edge_count} edges from {self.rel_path}"
0069 |                 )
0070 |         except Exception as ex:
0071 |             self._loaded = True
0072 |             if self.logger:
0073 |                 self.logger.warning(f"JSONLEdgeIndex: failed loading {self.rel_path}: {ex}")
0074 | 
0075 |     def neighbors(self, node_id: str, *, include_reverse: bool = True) -> Iterable[Dict[str, Any]]:
0076 |         self.ensure_loaded()
0077 |         node_id = str(node_id)
0078 |         if include_reverse:
0079 |             # yield in a stable order: outgoing then incoming
0080 |             yield from self._by_src.get(node_id, [])
0081 |             yield from self._by_dst.get(node_id, [])
0082 |         else:
0083 |             yield from self._by_src.get(node_id, [])
0084 | 
0085 |     @property
0086 |     def edge_count(self) -> int:
0087 |         self.ensure_loaded()
0088 |         return self._edge_count
0089 | 
0090 | 
0091 |     def append_edge(self, e: Dict[str, Any]) -> None:
0092 |         """
0093 |         Keep the in-memory index consistent with new JSONL appends.
0094 |         Safe to call even before ensure_loaded().
0095 |         """
0096 |         self.ensure_loaded()
0097 |         s = str(_rel_get(e, "source", "source_id", default="")).strip()
0098 |         t = str(_rel_get(e, "target", "target_id", default="")).strip()
0099 |         if not s or not t:
0100 |             return
0101 |         self._by_src[s].append(e)
0102 |         self._by_dst[t].append(e)
0103 |         self._edge_count += 1
0104 | 
0105 | 
0106 |     def invalidate(self) -> None:
0107 |         """Force a reload on next query (call after appending edges)."""
0108 |         self._loaded = False
0109 |         self._by_src.clear()
0110 |         self._by_dst.clear()
0111 |         self._edge_count = 0
```


---

## F0011 — `subgraphs/seed_finder.py`

```text
FILE_ID: F0011
PATH: subgraphs/seed_finder.py
LANGUAGE: python
LINES: 85
BYTES_UTF8: 2677
SHA256: bb0d0b305f497fe9fa414da4b34391dc5cd13c760e37609da53fd436f58ee45b
```

```python
0001 | # stephanie/services/knowledge_graph/subgraphs/seed_finder.py
0002 | from __future__ import annotations
0003 | 
0004 | from typing import Any, Callable, Dict, List, Tuple
0005 | 
0006 | 
0007 | def _dedupe_keep_order(xs: List[str]) -> List[str]:
0008 |     seen = set()
0009 |     out: List[str] = []
0010 |     for x in xs:
0011 |         if x and x not in seen:
0012 |             seen.add(x)
0013 |             out.append(x)
0014 |     return out
0015 | 
0016 | 
0017 | class EmbeddingSeedFinder:
0018 |     """
0019 |     Responsible only for: query -> ranked seed node IDs (plus seed terms).
0020 |     """
0021 | 
0022 |     def __init__(
0023 |         self,
0024 |         *,
0025 |         search_entities_fn: Callable[[str, int], List[Tuple[str, float, Dict[str, Any]]]],
0026 |         detect_entities_fn: Callable[[str], List[Dict[str, Any]]],
0027 |         logger: Any = None,
0028 |     ) -> None:
0029 |         self.search_entities_fn = search_entities_fn
0030 |         self.detect_entities_fn = detect_entities_fn
0031 |         self.logger = logger
0032 | 
0033 |     def find_seeds(
0034 |         self,
0035 |         *,
0036 |         query: str,
0037 |         seed_k: int,
0038 |         per_entity_k: int,
0039 |         k_entities: int,
0040 |         min_seed_score: float,
0041 |         max_seeds: int,
0042 |     ) -> Dict[str, Any]:
0043 |         query = (query or "").strip()
0044 |         seed_terms: List[str] = []
0045 | 
0046 |         try:
0047 |             ents = self.detect_entities_fn(query)[: max(1, int(k_entities))]
0048 |             seed_terms = _dedupe_keep_order(
0049 |                 [e.get("text", "").strip() for e in ents if e.get("text")]
0050 |             )
0051 |         except Exception as ex:
0052 |             if self.logger:
0053 |                 self.logger.warning(f"EmbeddingSeedFinder: detect_entities failed: {ex}")
0054 | 
0055 |         scored: List[Tuple[str, float]] = []
0056 | 
0057 |         try:
0058 |             for node_id, score, _meta in self.search_entities_fn(query, k=int(seed_k)):
0059 |                 if float(score) >= float(min_seed_score):
0060 |                     scored.append((str(node_id), float(score)))
0061 | 
0062 |             for term in seed_terms[: int(k_entities)]:
0063 |                 for node_id, score, _meta in self.search_entities_fn(term, k=int(per_entity_k)):
0064 |                     if float(score) >= float(min_seed_score):
0065 |                         scored.append((str(node_id), float(score)))
0066 |         except Exception as ex:
0067 |             if self.logger:
0068 |                 self.logger.warning(f"EmbeddingSeedFinder: search_entities failed: {ex}")
0069 | 
0070 |         scored.sort(key=lambda x: x[1], reverse=True)
0071 | 
0072 |         seeds: List[str] = []
0073 |         seen = set()
0074 |         for nid, _ in scored:
0075 |             if nid and nid not in seen:
0076 |                 seeds.append(nid)
0077 |                 seen.add(nid)
0078 |                 if len(seeds) >= int(max_seeds):
0079 |                     break
0080 | 
0081 |         return {
0082 |             "seeds": seeds,
0083 |             "seed_terms": seed_terms,
0084 |             "seed_count": len(seeds),
0085 |         }
```


---

## F0012 — `subgraphs/subgraph_builder.py`

```text
FILE_ID: F0012
PATH: subgraphs/subgraph_builder.py
LANGUAGE: python
LINES: 235
BYTES_UTF8: 8471
SHA256: 90a58127a2a638329d57af20dec1769d3be6fa7f4a26fd128b59cd82d0f5b715
```

```python
0001 | # stephanie/services/knowledge_graph/subgraphs/subgraph_builder.py
0002 | from __future__ import annotations
0003 | 
0004 | from collections import Counter
0005 | from dataclasses import dataclass
0006 | from datetime import datetime, timezone
0007 | from typing import Any, Dict, List, Optional, Set, Tuple
0008 | 
0009 | from .edge_index import JSONLEdgeIndex, _rel_get
0010 | from .seed_finder import EmbeddingSeedFinder
0011 | 
0012 | 
0013 | @dataclass(frozen=True)
0014 | class SubgraphConfig:
0015 |     seed_k: int = 30
0016 |     per_entity_k: int = 12
0017 |     k_entities: int = 12
0018 | 
0019 |     max_hops: int = 2
0020 |     max_nodes: int = 200
0021 |     max_edges: int = 800
0022 | 
0023 |     # important safety knobs
0024 |     min_confidence: float = 0.75
0025 |     require_evidence: bool = True
0026 |     allowed_edge_types: Optional[List[str]] = None
0027 |     include_reverse: bool = True
0028 | 
0029 |     # seed control
0030 |     min_seed_score: float = 0.0
0031 | 
0032 |     # stop “hub explosion”
0033 |     max_incident_edges_per_node: int = 200
0034 | 
0035 | 
0036 | class SubgraphBuilder:
0037 |     """
0038 |     Responsible only for:
0039 |       - running bounded BFS expansion from seeds
0040 |       - applying filters (confidence/evidence/type)
0041 |       - producing stable output + stats
0042 | 
0043 |     It does NOT:
0044 |       - load edges (edge_index does)
0045 |       - decide seeds (seed_finder does)
0046 |     """
0047 | 
0048 |     def __init__(
0049 |         self,
0050 |         *,
0051 |         seed_finder: EmbeddingSeedFinder,
0052 |         edge_index: JSONLEdgeIndex,
0053 |         nexus_store: Any = None,
0054 |         logger: Any = None,
0055 |     ) -> None:
0056 |         self.seed_finder = seed_finder
0057 |         self.edge_index = edge_index
0058 |         self.nexus_store = nexus_store
0059 |         self.logger = logger
0060 | 
0061 |     def _node_label(self, node_id: str) -> str:
0062 |         if not self.nexus_store:
0063 |             return node_id
0064 |         try:
0065 |             n = self.nexus_store.get_node(node_id)
0066 |             if not n:
0067 |                 return node_id
0068 |             for attr in ("name", "title", "text"):
0069 |                 v = getattr(n, attr, None)
0070 |                 if isinstance(v, str) and v.strip():
0071 |                     return v.strip()
0072 |             payload = getattr(n, "payload", None)
0073 |             if isinstance(payload, dict):
0074 |                 nm = payload.get("name") or payload.get("title") or payload.get("text")
0075 |                 if isinstance(nm, str) and nm.strip():
0076 |                     return nm.strip()
0077 |         except Exception as ex:
0078 |             if self.logger:
0079 |                 self.logger.debug(f"SubgraphBuilder: label lookup failed for {node_id}: {ex}")
0080 |         return node_id
0081 | 
0082 |     def _has_evidence(self, e: Dict[str, Any]) -> bool:
0083 |         return bool(
0084 |             e.get("doc_hash")
0085 |             or e.get("evidence_type")
0086 |             or e.get("sentence_ix") is not None
0087 |             or e.get("scorable_id")
0088 |         )
0089 | 
0090 |     def _edge_key(self, s: str, t: str, r: str, e: Dict[str, Any]) -> Tuple[str, str, str, str]:
0091 |         """
0092 |         Key that dedupes edges but still distinguishes different evidence sources.
0093 |         If you later want “collapse evidence”, do it in post-processing.
0094 |         """
0095 |         doc_hash = str(e.get("doc_hash") or "")
0096 |         sent_ix = str(e.get("sentence_ix") if e.get("sentence_ix") is not None else "")
0097 |         return (s, t, r, f"{doc_hash}:{sent_ix}")
0098 | 
0099 |     def build(self, *, query: str, cfg: SubgraphConfig) -> Dict[str, Any]:
0100 |         query = (query or "").strip()
0101 |         if not query:
0102 |             return {"nodes": [], "edges": [], "meta": {"reason": "empty_query"}}
0103 | 
0104 |         seed_info = self.seed_finder.find_seeds(
0105 |             query=query,
0106 |             seed_k=cfg.seed_k,
0107 |             per_entity_k=cfg.per_entity_k,
0108 |             k_entities=cfg.k_entities,
0109 |             min_seed_score=cfg.min_seed_score,
0110 |             max_seeds=cfg.max_nodes,  # never seed > max_nodes
0111 |         )
0112 |         seeds: List[str] = seed_info["seeds"]
0113 |         seed_terms: List[str] = seed_info["seed_terms"]
0114 | 
0115 |         if not seeds:
0116 |             return {
0117 |                 "nodes": [],
0118 |                 "edges": [],
0119 |                 "meta": {
0120 |                     "query": query,
0121 |                     "seed_terms": seed_terms,
0122 |                     "seed_count": 0,
0123 |                     "reason": "no_seeds",
0124 |                     "ts": datetime.now(timezone.utc).isoformat(),
0125 |                 },
0126 |             }
0127 | 
0128 |         allowed = set(cfg.allowed_edge_types or [])
0129 | 
0130 |         kept_nodes: Set[str] = set(seeds)
0131 |         kept_edges: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
0132 | 
0133 |         frontier: List[str] = list(seeds)
0134 | 
0135 |         for _hop in range(int(cfg.max_hops)):
0136 |             if not frontier:
0137 |                 break
0138 | 
0139 |             next_frontier: List[str] = []
0140 | 
0141 |             for nid in frontier:
0142 |                 # Bound the explosion from hubs
0143 |                 incident = list(self.edge_index.neighbors(nid, include_reverse=cfg.include_reverse))
0144 |                 if len(incident) > int(cfg.max_incident_edges_per_node):
0145 |                     incident = incident[: int(cfg.max_incident_edges_per_node)]
0146 | 
0147 |                 for e in incident:
0148 |                     s = str(_rel_get(e, "source", "source_id", default=""))
0149 |                     t = str(_rel_get(e, "target", "target_id", default=""))
0150 |                     r = str(_rel_get(e, "type", "rel_type", default=""))
0151 | 
0152 |                     if not s or not t or not r:
0153 |                         continue
0154 | 
0155 |                     if allowed and r not in allowed:
0156 |                         continue
0157 | 
0158 |                     conf = float(_rel_get(e, "confidence", default=0.0))
0159 |                     if conf < float(cfg.min_confidence):
0160 |                         continue
0161 | 
0162 |                     if cfg.require_evidence and not self._has_evidence(e):
0163 |                         continue
0164 | 
0165 |                     key = self._edge_key(s, t, r, e)
0166 |                     if key not in kept_edges:
0167 |                         kept_edges[key] = {
0168 |                             "source": s,
0169 |                             "target": t,
0170 |                             "type": r,
0171 |                             "confidence": conf,
0172 |                             "doc_hash": e.get("doc_hash"),
0173 |                             "sentence_ix": e.get("sentence_ix"),
0174 |                             "scorable_id": e.get("scorable_id"),
0175 |                             "scorable_type": e.get("scorable_type"),
0176 |                             "evidence_type": e.get("evidence_type"),
0177 |                             "ts": e.get("ts"),
0178 |                         }
0179 |                         if len(kept_edges) >= int(cfg.max_edges):
0180 |                             break
0181 | 
0182 |                     # add nodes + expand
0183 |                     for other in (s, t):
0184 |                         if other and other not in kept_nodes and len(kept_nodes) < int(cfg.max_nodes):
0185 |                             kept_nodes.add(other)
0186 |                             next_frontier.append(other)
0187 | 
0188 |                 if len(kept_edges) >= int(cfg.max_edges):
0189 |                     break
0190 | 
0191 |             frontier = next_frontier
0192 |             if len(kept_nodes) >= int(cfg.max_nodes) or len(kept_edges) >= int(cfg.max_edges):
0193 |                 break
0194 | 
0195 |         # Stable output
0196 |         nodes = [{"id": nid, "label": self._node_label(nid)} for nid in sorted(kept_nodes)]
0197 |         edges = sorted(
0198 |             kept_edges.values(),
0199 |             key=lambda x: (x["source"], x["target"], x["type"], str(x.get("doc_hash") or ""), str(x.get("sentence_ix") or "")),
0200 |         )
0201 | 
0202 |         # Stats
0203 |         etypes = Counter(e["type"] for e in edges)
0204 |         evidence_count = sum(1 for e in edges if self._has_evidence(e))
0205 |         confidences = sorted(float(e.get("confidence", 0.0)) for e in edges)
0206 | 
0207 |         def q(p: float) -> float:
0208 |             if not confidences:
0209 |                 return 0.0
0210 |             i = int((len(confidences) - 1) * p)
0211 |             return float(confidences[max(0, min(i, len(confidences) - 1))])
0212 | 
0213 |         return {
0214 |             "nodes": nodes,
0215 |             "edges": edges,
0216 |             "meta": {
0217 |                 "query": query,
0218 |                 "seed_terms": seed_terms,
0219 |                 "seed_count": len(seeds),
0220 |                 "max_hops": cfg.max_hops,
0221 |                 "max_nodes": cfg.max_nodes,
0222 |                 "max_edges": cfg.max_edges,
0223 |                 "min_confidence": cfg.min_confidence,
0224 |                 "require_evidence": cfg.require_evidence,
0225 |                 "stats": {
0226 |                     "node_count": len(nodes),
0227 |                     "edge_count": len(edges),
0228 |                     "edge_types": dict(etypes),
0229 |                     "evidence_rate": evidence_count / max(1, len(edges)),
0230 |                     "confidence_p50": q(0.50),
0231 |                     "confidence_p90": q(0.90),
0232 |                 },
0233 |                 "ts": datetime.now(timezone.utc).isoformat(),
0234 |             },
0235 |         }
```
