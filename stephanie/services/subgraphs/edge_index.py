# stephanie/services/subgraphs/edge_index.py
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional


def _rel_get(rel: Dict[str, Any], *keys: str, default=None) -> Any:
    for k in keys:
        if k in rel:
            return rel[k]
    return default


class JSONLEdgeIndex:
    """
    Loads a JSONL relationship file once, builds adjacency maps, and serves neighbors fast.

    Expected edge schema (flexible):
      source / source_id
      target / target_id
      type   / rel_type
      confidence (optional)
      evidence fields (optional)
    """

    def __init__(self, *, rel_path: str | Path, logger: Any = None) -> None:
        self.rel_path = Path(rel_path)
        self.logger = logger

        self._loaded = False
        self._by_src: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._by_dst: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._edge_count = 0

    def ensure_loaded(self) -> None:
        if self._loaded:
            return

        if not self.rel_path.exists():
            if self.logger:
                self.logger.warning(f"JSONLEdgeIndex: rel_path missing: {self.rel_path}")
            self._loaded = True
            return

        try:
            with self.rel_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)

                    s = str(_rel_get(e, "source", "source_id", default="")).strip()
                    t = str(_rel_get(e, "target", "target_id", default="")).strip()
                    if not s or not t:
                        continue

                    self._by_src[s].append(e)
                    self._by_dst[t].append(e)
                    self._edge_count += 1

            self._loaded = True
            if self.logger:
                self.logger.info(
                    f"JSONLEdgeIndex loaded {self._edge_count} edges from {self.rel_path}"
                )
        except Exception as ex:
            self._loaded = True
            if self.logger:
                self.logger.warning(f"JSONLEdgeIndex: failed loading {self.rel_path}: {ex}")

    def neighbors(self, node_id: str, *, include_reverse: bool = True) -> Iterable[Dict[str, Any]]:
        self.ensure_loaded()
        node_id = str(node_id)
        if include_reverse:
            # yield in a stable order: outgoing then incoming
            yield from self._by_src.get(node_id, [])
            yield from self._by_dst.get(node_id, [])
        else:
            yield from self._by_src.get(node_id, [])

    @property
    def edge_count(self) -> int:
        self.ensure_loaded()
        return self._edge_count
