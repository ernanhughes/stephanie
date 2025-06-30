# co_ai/compiler/step_selector.py
from operator import attrgetter

from co_ai.agents.compiler.reasoning_trace import ReasoningNode, ReasoningTree


class StepSelector:
    def select_next_steps(self, tree: ReasoningTree, top_k: int = 3) -> list[dict]:
        # 1. Collect all leaf nodes
        leaf_nodes = [node for node in tree.nodes.values() if not node.children]

        # 2. Rank leaf nodes by score
        top_leafs = sorted(leaf_nodes, key=attrgetter("score"), reverse=True)[:top_k]

        # 3. Create dummy next steps based on top thoughts (can use a better prompt later)
        steps = []
        for leaf in top_leafs:
            steps.append({
                "parent_id": leaf.id,
                "thought": f"Refine step from: {leaf.thought}",
                "action": f"Expand on: {leaf.action}",
            })

        return steps

    def rank_paths(self, tree: ReasoningTree, metric="score") -> list[list[ReasoningNode]]:
        def dfs(node: ReasoningNode, path: list, all_paths: list):
            path.append(node)
            if not node.children:
                all_paths.append(list(path))
            else:
                for child in node.children:
                    dfs(child, path, all_paths)
            path.pop()

        # Start DFS from root
        all_paths = []
        root = tree.nodes.get(tree.root_id)
        if root:
            dfs(root, [], all_paths)

        # Sort paths by cumulative score
        ranked = sorted(all_paths, key=lambda path: sum(getattr(n, metric, 0.0) for n in path), reverse=True)
        return ranked
