# co_ai/agents/compiler/reasoning_trace.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from uuid import uuid4


@dataclass
class ReasoningNode:
    id: str
    parent_id: Optional[str]
    goal: str
    thought: str
    action: str
    response: str
    score: float = 0.0
    children: List["ReasoningNode"] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

class ReasoningTree:
    def __init__(self):
        self.nodes: Dict[str, ReasoningNode] = {}
        self.root_id: Optional[str] = None

    def add_root(self, goal: str, thought: str, action: str) -> str:
        node = ReasoningNode(
            id=str(uuid4()),
            parent_id=None,
            goal=goal,
            thought=thought,
            action=action,
            response="",
            children=[]
        )
        self.nodes[node.id] = node
        self.root_id = node.id
        return node.id

    def add_child(self, parent_id: str, thought: str, action: str, response: str, score: float) -> str:
        node = ReasoningNode(
            id=str(uuid4()),
            parent_id=parent_id,
            goal=self.nodes[parent_id].goal,
            thought=thought,
            action=action,
            response=response,
            score=score,
            children=[],
            metadata={}
        )
        self.nodes[node.id] = node
        self.nodes[parent_id].children.append(node)
        return node.id

    def get_best_path(self, top_k: int = 1) -> List[ReasoningNode]:
        def dfs(node: ReasoningNode, path: List[ReasoningNode]):
            if not node.children:
                paths.append((sum(n.score for n in path), list(path)))
            for child in node.children:
                dfs(child, path + [child])

        paths = []
        root = self.nodes.get(self.root_id)
        if not root:
            return []

        dfs(root, [root])
        paths.sort(key=lambda x: x[0], reverse=True)
        return paths[0][1] if paths else []
