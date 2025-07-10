import numpy as np
from sklearn.cluster import DBSCAN

from stephanie.core.knowledge_cartridge import KnowledgeCartridge


class CartridgeMerger:
    def __init__(self, cfg, memory, logger=None):
        self.cfg = cfg
        self.memory = memory  # Assumed to expose `.embed(texts: list[str]) -> list[np.array]`
        self.logger = logger
        self.similarity_threshold = getattr(cfg, "similarity_threshold", 0.85)

    def merge(self, cartridges):
        """Merge multiple cartridges into a master cartridge"""
        if not cartridges:
            return None

        master = KnowledgeCartridge(
            goal=cartridges[0].goal,
            generation=max(c.generation for c in cartridges) + 1
        )

        # Merge core thesis (could be improved with attention over top hypotheses)
        master.schema["core_thesis"] = self.fuse_core_thesis(cartridges)

        for category in ["supporting_evidence", "contradictions", "hypotheses"]:
            all_items = []
            for c in cartridges:
                for item in c.schema.get(category, []):
                    item["weight"] = c.quality_metrics.get("overall_score", 0.7)
                    all_items.append(item)

            clusters = self.cluster_items(all_items)

            for cluster in clusters:
                best_item = max(cluster, key=lambda x: x["confidence"] * x["weight"])
                master.add_finding(
                    category=category,
                    content=best_item["content"],
                    source=f"Merged from {len(cluster)} sources",
                    confidence=best_item["confidence"]
                )

        if self.logger:
            self.logger.log("CartridgeMerger", {
                "merged_goal": master.goal,
                "num_sources": len(cartridges),
                "num_clusters": sum(len(c) for c in clusters)
            })

        return master

    def cluster_items(self, items):
        """Cluster similar knowledge items using memory-based embedding"""
        if not items:
            return []

        contents = [item["content"] for item in items]
        embeddings = self.memory.embed(contents)

        clustering = DBSCAN(eps=0.5, min_samples=2).fit(embeddings)
        labels = clustering.labels_

        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(items[i])

        return list(clusters.values())

    def fuse_core_thesis(self, cartridges):
        """Placeholder — you can improve this using abstraction over top hypotheses"""
        if not cartridges:
            return "No core thesis available."
        return cartridges[0].schema.get("core_thesis", "No core thesis.")
