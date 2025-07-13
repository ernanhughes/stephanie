# stephanie/utils/graph_tools.py


def build_mermaid_graph(node, depth=0, max_depth=5, visited=None):
    """
    Recursively builds a Mermaid graph from a reasoning tree node.
    Returns a list of Mermaid-compatible graph lines.
    """
    if visited is None:
        visited = set()

    if depth > max_depth or id(node) in visited:
        return []

    visited.add(id(node))
    mermaid = []

    # Node ID
    node_id = f"N{node.get('id', id(node))}"

    # Extract score (fallback to reward or 0.0)
    score = node.get("score", node.get("reward", 0.0))

    # Extract last trace item (handle various types)
    trace = node.get("trace", [])
    if isinstance(trace, str):
        trace = trace.split("\n")
    elif not isinstance(trace, list):
        trace = []

    if trace:
        last_action = trace[-1][:30]
    else:
        state = node.get("state", {})
        goal_text = state.get("goal", "Root") if isinstance(state, dict) else "Root"
        last_action = goal_text[:30]

    # Label assembly
    label = f"{last_action[:20]}..."
    label += f" | Score: {score:.2f}"
    if node.get("is_terminal", False):
        label += " | TERMINAL"

    # Define the node visually
    mermaid.append(f'{node_id}["{label}"]')

    # Style node based on score
    if node.get("is_terminal", False):
        mermaid.append(f"style {node_id} fill:#e6ccff,stroke:#333")
    elif score > 0.8:
        mermaid.append(f"style {node_id} fill:#a8edc9,stroke:#333")
    elif score > 0.5:
        mermaid.append(f"style {node_id} fill:#f4f4f4,stroke:#333")
    else:
        mermaid.append(f"style {node_id} fill:#fddddd,stroke:#333")

    # Recursively add children
    children = node.get("children", [])
    for child in children[:3]:  # Limit to 3 branches
        child_lines = build_mermaid_graph(child, depth + 1, max_depth, visited)
        if child_lines:
            mermaid.extend(child_lines)
            child_id = f"N{child.get('id', id(child))}"
            mermaid.append(f"{node_id} --> {child_id}")

    return mermaid


def save_mermaid_to_file(diagram, filename="search_tree.mmd"):
    with open(filename, "w") as f:
        f.write("```mermaid\ngraph TD\n")
        f.write(diagram + "\n")
        f.write("```\n")


def compare_graphs(graph1, graph2):
    """
    Returns:
        matches: list of nodes present in both
        only_1: list of nodes only in graph1
        only_2: list of nodes only in graph2
    """
    # Ensure graphs contain full node dicts
    if not isinstance(graph1[0], dict) or not isinstance(graph2[0], dict):
        raise ValueError("compare_graphs() requires full node dicts")

    set1 = {n["id"]: n for n in graph1}
    set2 = {n["id"]: n for n in graph2}

    matches = [set1[k] for k in set1 if k in set2]
    only_1 = [n for n in graph1 if n["id"] not in set2]
    only_2 = [n for n in graph2 if n["id"] not in set1]

    return matches, only_1, only_2


def analyze_graph_impact(graph1, graph2, score_lookup_fn):
    """
    Returns a list of dictionaries summarizing overlap and score delta.
    """
    matches, only_1, only_2 = compare_graphs(graph1, graph2)
    results = []

    for node in matches:
        score1 = score_lookup_fn(node, source="graph1")
        score2 = score_lookup_fn(node, source="graph2")
        results.append({"node": node, "type": "match", "delta": score2 - score1})

    for node in only_1:
        results.append(
            {
                "node": node,
                "type": "only_graph1",
                "score": score_lookup_fn(node, source="graph1"),
            }
        )

    for node in only_2:
        results.append(
            {
                "node": node,
                "type": "only_graph2",
                "score": score_lookup_fn(node, source="graph2"),
            }
        )

    return results
