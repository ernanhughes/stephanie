# stephanie/components/solution_tree.py
# --- components/solution_tree.py ---


class SolutionNode:
    def __init__(self, plan, code, metric, output, valid):
        self.plan = plan
        self.code = code
        self.metric = metric
        self.output = output
        self.valid = valid


class SolutionTree:
    def __init__(self):
        self.nodes = []

    def initialize(self, goal):
        self.goal = goal
        self.nodes.clear()

    def add_node(self, node):
        self.nodes.append(node)

    def get_best(self):
        valid_nodes = [n for n in self.nodes if n.valid]
        return max(valid_nodes, key=lambda n: n.metric, default=None)

    def get_buggy(self):
        return [n for n in self.nodes if not n.valid]

    def get_valid(self):
        return [n for n in self.nodes if n.valid]
