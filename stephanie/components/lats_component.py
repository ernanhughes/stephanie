# stephanie/components/lats_component.py
import math
import random
import uuid
from typing import Any, Callable, Dict, List, Optional


class LATSNode:
    def __init__(self, state, trace, parent=None):
        self.id = str(uuid.uuid4())
        self.state = state
        self.trace = trace
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.score = None
        self.dimension_scores = {}

    def is_leaf(self):
        return len(self.children) == 0


class LATSComponent:
    def __init__(
        self,
        expand_fn: Callable[[LATSNode, Dict[str, Any]], List[Dict]],
        score_fn: Callable[[LATSNode, Dict[str, Any]], Dict[str, Any]],
        is_terminal_fn: Optional[Callable[[LATSNode], bool]] = None,
        memory=None,
        logger: Optional[Any] = None,
        max_steps: int = 10,
        exploration_weight: float = 1.4,
    ):
        self.expand_fn = expand_fn
        self.score_fn = score_fn
        self.is_terminal_fn = is_terminal_fn or (lambda node: False)
        self.memory = memory
        self.logger = logger or (lambda *args, **kwargs: None)
        self.max_steps = max_steps
        self.exploration_weight = exploration_weight
        self.root = None

    def create_node(self, state, trace, parent=None):
        return LATSNode(state, trace, parent)

    def uct_score(self, parent_visits, child):
        if child.visits == 0:
            return float("inf")
        return child.reward / child.visits + self.exploration_weight * math.sqrt(
            math.log(parent_visits) / child.visits
        )

    def select_best_child(self, node):
        return max(node.children, key=lambda c: self.uct_score(node.visits, c))

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.reward += reward
            node = node.parent

    def simulate(self, node, context):
        if self.is_terminal_fn(node):
            return node.reward

        children = self.expand_fn(node, context)
        for child_info in children:
            child_node = self.create_node(
                child_info["state"], child_info["trace"], parent=node
            )
            node.children.append(child_node)

            score_result = self.score_fn(child_node, context)
            child_node.score = score_result.get("score", 0.0)
            child_node.dimension_scores = score_result.get("dimension_scores", {})
            reward = child_node.score or 0.0

            self.backpropagate(child_node, reward)

    def run(self, root_state, context):
        self.root = self.create_node(root_state, trace=[])

        for step in range(self.max_steps):
            node = self.root
            while not node.is_leaf():
                node = self.select_best_child(node)
            self.simulate(node, context)

        best = max(
            self.root.children,
            key=lambda c: c.reward / c.visits if c.visits > 0 else -1,
        )
        return best.trace, best.score, best.dimension_scores
