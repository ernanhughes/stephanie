# stephanie/components/nexus/events.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

@dataclass
class NodeAdded:
    node_id: str
    chat_id: str
    turn_id: str

@dataclass
class EdgesBuilt:
    node_id: str
    counts: dict  # {"knn_global": k, "lateint": m, ...}

@dataclass
class PathFound:
    path_id: str
    node_ids: List[str]
    score: float

@dataclass
class PathCommitted:
    path_id: str
    run_id: str
