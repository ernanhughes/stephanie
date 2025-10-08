# stephanie/components/tree/solution_node.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import time
import uuid
from typing import Any, Dict, List, Optional

@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    has_submission_file: bool = False

@dataclass
class SolutionNode:
    plan: str
    code: Optional[str] = None
    metric: Optional[float] = None
    output: Optional[str] = None
    summary: Optional[str] = None
    parent_id: Optional[str] = None
    is_buggy: bool = False
    node_type: str = "draft"
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    children: List[str] = field(default_factory=list)
    tree_id: Optional[str] = None
    root_id: Optional[str] = None
    depth: int = 0
    sibling_index: int = 0
    path: str = "0"
    origin: Dict[str, Any] = field(default_factory=dict)
    lineage: List[str] = field(default_factory=list)
    plan_sha256: Optional[str] = None
    code_sha256: Optional[str] = None
    output_sha256: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.setdefault("prompt_text", self.plan)
        d.setdefault("compiled_prompt", self.plan)
        return d
