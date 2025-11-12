# stephanie/components/nexus/blossum/nexus_merge.py
from __future__ import annotations


class NexusMergeWriter:
    def __init__(self, memory, logger):
        self.nexus = memory.nexus
        self.logger = logger

    def merge_winner(self, blossom_episode_id: str, winner: dict, meta: dict):
        path = winner["path"]
        prev = None
        for nid in path:
            plan = meta["tree"].nodes_by_id[nid].plan
            nx_node = self.nexus.upsert_node(text=plan, origin=f"blossom:{blossom_episode_id}")
            if prev:
                self.nexus.upsert_edge(prev, nx_node.id, kind="blossom_path")
            prev = nx_node.id
