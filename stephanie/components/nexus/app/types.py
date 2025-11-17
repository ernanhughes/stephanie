# stephanie/components/nexus/app/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import numpy as np

NodeId = str
EdgeType = Literal[
    "knn_global",
    "alternate_path",
    "temporal_next",
    "policy_shift",
    "anomaly_escape",
]


class NexusNodeType(str, Enum):
    SCORABLE = "scorable"
    ACTION = "action"

class ActionKind(str, Enum):
    SCORE = "score"
    GENERATE = "generate"
    VOTE = "vote"
    RED_FLAG = "red_flag"
    TOOL_CALL = "tool_call"
    PLAN_STEP = "plan_step"     # generic ExecutionStep
    MDAP_STEP = "mdap_step"     # if you want to tag MDAP micro-steps
    OTHER = "other"

@dataclass
class ActionNodeMeta:
    kind: ActionKind
    name: str
    agent: Optional[str] = None        # e.g. "MakerExecutionAgent", "NexusPollinator"
    protocol: Optional[str] = None     # e.g. "MDAPProtocol", "CoTProtocol"
    step_index: Optional[int] = None   # for PlanTrace
    trace_id: Optional[str] = None     # PlanTrace / run id
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NexusNode:
    node_id: NodeId                    # e.g. vpm://chat/abc/turn/123/path/7
    scorable_id: str                   # e.g. "12345" or memcube key for VPM
    scorable_type: str                 # e.g. "conversation_turn", "vpm", "goal"
    memcube_key: Optional[str] = None
    embed_global: Optional[np.ndarray] = None  # [d] embedding
    patchgrid_path: Optional[str] = None       # VPM path if applicable
    metrics: Dict[str, float] = field(default_factory=dict)
    policy: Optional[str] = None
    outcome: Optional[str] = None

@dataclass
class NexusEdge:
    src: NodeId
    dst: NodeId
    type: EdgeType
    weight: float
    channels: Optional[Dict[str, float]] = field(default=None)  # ← NEW
    extras: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, any]:
        d = {
            "src": self.src,
            "dst": self.dst,
            "type": self.type,
            "weight": self.weight,
        }
        if self.channels is not None:
            d["channels"] = self.channels
        if self.extras is not None:
            d["extras"] = self.extras
        return d


@dataclass
class NexusPath:
    path_id: str
    node_ids: List[NodeId]
    score: float
    constraints: Dict[str, float] = field(default_factory=dict)
