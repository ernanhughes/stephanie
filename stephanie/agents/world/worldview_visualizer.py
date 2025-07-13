# stephanie/agents/world/worldview_visualizer.py
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx

from stephanie.core.knowledge_cartridge import KnowledgeCartridge


class WorldviewVisualizer:
    """
    Visualizes key components of a worldview: belief systems, goal influence, cartridge scores,
    tuning changes over time, and belief network connectivity.
    """

    def __init__(self, worldview, logger=None):
        self.worldview = worldview
        self.logger = logger or self._default_logger()

    def plot_cartridge_scores(self):
        """Plot cartridge scores over time per domain"""
        cartridges = self.worldview.list_cartridges()
        domain_scores = defaultdict(list)

        for c in cartridges:
            domain = c.domain or "unknown"
            score = c.score or 0
            timestamp = c.timestamp or "n/a"
            domain_scores[domain].append((timestamp, score))

        for domain, scores in domain_scores.items():
            scores = sorted(scores, key=lambda x: x[0])
            timestamps, values = zip(*scores)
            plt.figure(figsize=(10, 4))
            plt.plot(timestamps, values, label=f"{domain} scores")
            plt.xticks(rotation=45)
            plt.title(f"Belief Score Over Time - {domain}")
            plt.xlabel("Timestamp")
            plt.ylabel("Score")
            plt.legend()
            plt.tight_layout()
            plt.show()

    def plot_belief_tuning_history(self):
        """Plot belief tuning history if available"""
        tuning_data = self.worldview.audit.get_tuning_history()
        if not tuning_data:
            print("No tuning history available.")
            return

        for belief_id, records in tuning_data.items():
            timestamps = [r["timestamp"] for r in records]
            values = [r["new_score"] for r in records]

            plt.figure(figsize=(8, 3))
            plt.plot(timestamps, values, marker="o")
            plt.title(f"Tuning History for Belief {belief_id}")
            plt.xlabel("Time")
            plt.ylabel("Score")
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.show()

    def draw_belief_influence_graph(self):
        """Draw a graph of how cartridges, beliefs, and goals are linked"""
        G = nx.DiGraph()

        for goal in self.worldview.list_goals():
            goal_id = f"goal:{goal['id']}"
            G.add_node(goal_id, label=goal["text"], type="goal")

        for cartridge in self.worldview.list_cartridges():
            cid = f"cart:{cartridge.id}"
            G.add_node(cid, label=cartridge.summary[:50], type="cartridge")

            # Link cartridge to goal
            if cartridge.goal_id:
                G.add_edge(f"goal:{cartridge.goal_id}", cid)

            # Link beliefs (if structured that way)
            for belief in cartridge.beliefs:
                bid = f"belief:{belief.get('id', belief.get('title', '')[:20])}"
                G.add_node(bid, label=belief.get("title", ""), type="belief")
                G.add_edge(cid, bid)

        pos = nx.spring_layout(G, k=0.5)
        plt.figure(figsize=(12, 8))

        colors = []
        for n in G.nodes(data=True):
            if n[1]["type"] == "goal":
                colors.append("lightblue")
            elif n[1]["type"] == "cartridge":
                colors.append("lightgreen")
            else:
                colors.append("lightcoral")

        nx.draw(G, pos, node_color=colors, with_labels=True, font_size=8, arrows=True)
        plt.title("Worldview Belief Influence Graph")
        plt.show()

    def _default_logger(self):
        class DummyLogger:
            def log(self, tag, payload):
                print(f"[{tag}] {payload}")

        return DummyLogger()
