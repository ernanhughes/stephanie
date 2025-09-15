# stephanie/knowledge/knowledge_bus.py
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class KBEvent:
    kind: str                    # "casebook.created", "case.added", "scorable.added", "trajectory.step", "decision.emitted"
    payload: Dict[str, Any]
    ts: float = time.time()
    id: str = uuid.uuid4().hex

class KnowledgeBus:
    """Append-only JSONL event log + simple query helpers."""
    def __init__(self, path: str = "./runs/knowledge/events.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def publish(self, kind: str, payload: Dict[str, Any]) -> str:
        evt = KBEvent(kind=kind, payload=payload, ts=time.time())
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(evt)) + "\n")
        return evt.id

    def query(self, kind: Optional[str] = None, where: Optional[Callable[[Dict[str, Any]], bool]] = None, limit: int = 500) -> List[Dict[str, Any]]:
        out = []
        if not self.path.exists(): return out
        with self.path.open() as f:
            for line in reversed(list(f)):
                try:
                    evt = json.loads(line)
                except Exception:
                    continue
                if kind and evt.get("kind") != kind:
                    continue
                if where and not where(evt):
                    continue
                out.append(evt)
                if len(out) >= limit: break
        return list(reversed(out))
