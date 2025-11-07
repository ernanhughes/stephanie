from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np

from ..app.types import NexusNode, NexusPath
from ..stores.dict_store import NexusGraphStore


class NexusPathFinder:
    def __init__(self, store: NexusGraphStore, cfg: dict) -> None:
        self.store = store
        self.cfg = cfg

    def find_path(
        self,
        start_id: str,
        scorer: Callable[[NexusNode], float],
        steps_max: int | None = None,
        beam_size: int = 8,
    ) -> NexusPath:
        steps_max = steps_max or self.cfg["path"]["steps_max"]
        beams: List[Tuple[float, List[str]]] = [(0.0, [start_id])]
        best_score, best_path = -math.inf, [start_id]

        for _ in range(steps_max):
            new_beams: List[Tuple[float, List[str]]] = []
            for score, path in beams:
                last = path[-1]
                for e in self.store.neighbors(last):
                    nid = e.dst
                    if nid in path:
                        continue
                    n = self.store.get(nid)
                    s = score + scorer(n) - self._switch_penalty(path[-1], nid)
                    cand = (s, path + [nid])
                    new_beams.append(cand)
                    if s > best_score:
                        best_score, best_path = s, cand[1]

            new_beams.sort(key=lambda x: -x[0])
            beams = new_beams[:beam_size]
            if not beams:
                break

        return NexusPath(
            path_id=f"nexus-path-{best_path[0]}-{best_path[-1]}",
            node_ids=best_path,
            score=best_score,
        )

    def make_goal_scorer(
        self,
        goal_vec: np.ndarray | None,
        weight_cfg: dict,
        memory=None,
        goal_text: str = "",
    ) -> Callable[[NexusNode], float]:
        a = float(weight_cfg.get("alpha_text", 1.0))
        b = float(weight_cfg.get("beta_goal", 0.7))
        g = float(weight_cfg.get("gamma_stability", 0.6))
        z = float(weight_cfg.get("zeta_agreement", 0.4))
        d = float(weight_cfg.get("delta_domain", 0.0))     # defaulted
        e = float(weight_cfg.get("epsilon_entity", 0.0))   # defaulted

        def _sim_text(n: NexusNode) -> float:
            # Optional: if you later pass a live query embedding, score it here.
            return 0.0

        def _sim_goal(n: NexusNode) -> float:
            if n.embed_global is None or goal_vec is None:
                return 0.0
            return float(np.dot(n.embed_global, goal_vec))

        def _stability(n: NexusNode) -> float:
            hall = float(n.metrics.get("hall", 0.0))
            ent = float(n.metrics.get("uncertainty", 0.0))
            return -(hall + ent)

        def _agreement(n: NexusNode) -> float:
            disagree = float(n.metrics.get("disagree", 0.0))
            return -disagree

        def scorer(n: NexusNode) -> float:
            base = a * _sim_text(n) + b * _sim_goal(n) + g * _stability(n) + z * _agreement(n)

            if memory and d > 0:
                # domain boost via your annotation tables at query time
                domains = memory.scorable_domains.get_domains(n.scorable_id, n.scorable_type)
                if domains and any(dom.get("domain") == "evaluation" for dom in domains):
                    base += d

            if memory and e > 0:
                ner = memory.scorable_entities.get_by_scorable(n.scorable_id, n.scorable_type)
                if ner and any(ent.get("text") == "GRPO" for ent in ner):
                    base += e

            return base

        return scorer

    def _switch_penalty(self, src: str, dst: str) -> float:
        return float(self.cfg["path"]["weights"].get("kappa_switch", 0.2))