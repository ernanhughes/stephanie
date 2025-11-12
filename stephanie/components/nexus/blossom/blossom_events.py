# stephanie/components/nexus/app/blossom_events.py
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

@dataclass
class BlossomEvent:
    ts: float
    kind: str                     # "start"|"add_node"|"edge_add"|"reject"|"winner"|"progress"|"status"|"complete"
    blossom_id: int
    data: Dict[str, Any]

class BlossomEventWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # touch file so UIs can follow
        self.f = self.path.open("a", encoding="utf-8")
    def _emit(self, kind: str, blossom_id: int, **data):
        ev = BlossomEvent(ts=time.time(), kind=kind, blossom_id=blossom_id, data=data)
        self.f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n"); self.f.flush()
    def start(self, blossom_id: int, **meta):     self._emit("start", blossom_id, **meta)
    def add_node(self, blossom_id: int, **meta):  self._emit("add_node", blossom_id, **meta)   # bn_id, parent_id, role, accepted, score
    def edge_add(self, blossom_id: int, **meta):  self._emit("edge_add", blossom_id, **meta)   # src, dst, relation, score
    def reject(self, blossom_id: int, **meta):    self._emit("reject", blossom_id, **meta)     # bn_id
    def winner(self, blossom_id: int, **meta):    self._emit("winner", blossom_id, **meta)     # bn_id, score
    def status(self, blossom_id: int, **meta):    self._emit("status", blossom_id, **meta)     # text
    def progress(self, blossom_id: int, **meta):  self._emit("progress", blossom_id, **meta)   # made, accepted, rejected, total, pct
    def complete(self, blossom_id: int, **meta):  self._emit("complete", blossom_id, **meta)
    def close(self):                               self.f.close()
