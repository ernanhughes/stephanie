# stephanie/components/nexus/service/nexus_service.py
from __future__ import annotations

from typing import Dict, Iterable, Tuple

from ..graph.pathfinder import NexusPathFinder
from ..index.indexer import NexusIndexer
from ..store.dict_store import NexusGraphStore


class NexusService:
    def __init__(self, cfg: Dict, memory=None) -> None:
        self.cfg = cfg
        self.memory = memory
        self.store = NexusGraphStore()
        self.indexer = NexusIndexer(self.store, cfg, memory)
        self.pathfinder = NexusPathFinder(self.store, cfg)

    def build_index_from_scorables(
        self,
        items: Iterable[Tuple[str, str, str, dict]],
    ) -> int:
        nodes = self.indexer.add_nodes_from_scorables(items)
        self.indexer.build_knn_edges(self.cfg["graph"]["knn_k"])
        return len(nodes)

    def build_index_from_vpms(self, items: Iterable[Tuple[str, dict]]) -> int:
        nodes = self.indexer.add_nodes_from_vpms(items)
        self.indexer.build_knn_edges(self.cfg["graph"]["knn_k"])
        return len(nodes)

    def find_path(self, start_node_id: str, goal_vec=None, goal_text: str = "") -> dict:
        scorer = self.pathfinder.make_goal_scorer(
            goal_vec,
            self.cfg["path"]["weights"],
            memory=self.memory,
            goal_text=goal_text,
        )
        p = self.pathfinder.find_path(start_node_id, scorer)
        return {"path_id": p.path_id, "nodes": p.node_ids, "score": p.score}
