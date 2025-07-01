from stephanie.utils.graph_tools import compare_graphs


class SymbolicImpactAnalyzer:
    """
    Analyzes structural overlap and divergence between two graph representations (e.g., symbolic vs. LATS)
    and attributes score delta to divergent paths.
    """

    def __init__(self, score_lookup_fn):
        self.score_lookup_fn = (
            score_lookup_fn  # Function to get scores for a given node or trace
        )

    def analyze(self, graph1, graph2):
        matches, only_1, only_2 = compare_graphs(graph1, graph2)
        results = []

        for node in matches:
            score_1 = self.score_lookup_fn(node, source="graph1")
            score_2 = self.score_lookup_fn(node, source="graph2")
            results.append({
                "node": node,
                "type": "converged",
                "delta": score_2 - score_1
            })

        for node in only_1 + only_2:
            score = self.score_lookup_fn(node, source="graph1")
            results.append({
                "node": node,
                "type": "diverged",
                "score": score
            })

        return results