# stephanie/components/information/tasks/section_link_task.py
from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

from stephanie.components.information.data import (ConceptCluster,
                                                   PaperSection, SectionMatch)

log = logging.getLogger(__name__)



class SectionLinkTask:
    """
    Take a set of DocumentSection objects with embeddings and:

    - For each root section, find top-k similar sections in other papers.
    - Optionally build very simple concept clusters.

    This assumes that `section.embedding` is already populated (e.g. by
    SectionBuildTask + your embedding store).
    """

    def __init__(
        self,
        root_arxiv_id: str,
        top_k: int = 5,
        min_sim: float = 0.4,
    ) -> None:
        self.root_arxiv_id = root_arxiv_id
        self.top_k = top_k
        self.min_sim = min_sim

    # ------------------------------------------------------------------ #
    def run(
        self,
        sections: Sequence[PaperSection],
    ) -> Tuple[List[SectionMatch], List[ConceptCluster]]:
        root_secs = [s for s in sections if s.paper_arxiv_id == self.root_arxiv_id]
        other_secs = [s for s in sections if s.paper_arxiv_id != self.root_arxiv_id]

        # Build index for others
        other_vecs: List[np.ndarray] = []
        other_ids: List[str] = []

        for sec in other_secs:
            if sec.embedding is None:
                continue
            vec = np.asarray(sec.embedding, dtype=np.float32)
            if vec.ndim != 1:
                continue
            other_vecs.append(vec)
            other_ids.append(sec.id)

        if not other_vecs:
            log.warning("[SectionLinkTask] No embeddings for non-root sections")
            return [], []

        other_matrix = np.stack(other_vecs, axis=0)

        matches: List[SectionMatch] = []

        # For each root section, compute cosine similarities to all others
        for root_sec in root_secs:
            if root_sec.embedding is None:
                continue

            root_vec = np.asarray(root_sec.embedding, dtype=np.float32)
            if root_vec.ndim != 1:
                continue

            sims = other_matrix @ root_vec
            norms = np.linalg.norm(other_matrix, axis=1) * np.linalg.norm(root_vec)
            norms = norms + 1e-8
            sims = sims / norms

            # Take top-k
            order = np.argsort(-sims)  # descending
            rank = 0
            for idx in order[: self.top_k]:
                score = float(sims[idx])
                if score < self.min_sim:
                    continue

                target_id = other_ids[int(idx)]

                matches.append(
                    SectionMatch(
                        source_section_id=root_sec.id,
                        target_section_id=target_id,
                        score=score,
                        rank=rank,
                        reason=None,
                    )
                )
                rank += 1

        # Simple concept clusters: group by target section id and treat each
        # as a "concept" anchored on that target. You can replace this with
        # something more advanced later.
        clusters = self._build_concept_clusters(matches)

        log.info(
            "[SectionLinkTask] Built %d matches and %d clusters",
            len(matches),
            len(clusters),
        )

        return matches, clusters

    # ------------------------------------------------------------------ #
    def _build_concept_clusters(
        self,
        matches: Sequence[SectionMatch],
    ) -> List[ConceptCluster]:
        by_target: Dict[str, List[SectionMatch]] = {}
        for m in matches:
            by_target.setdefault(m.target_section_id, []).append(m)

        clusters: List[ConceptCluster] = []
        for idx, (target_id, ms) in enumerate(by_target.items()):
            cluster_id = f"concept-{idx}"
            section_ids = {m.source_section_id for m in ms}
            section_ids.add(target_id)

            avg_score = float(
                sum(m.score for m in ms) / max(len(ms), 1)
            )

            clusters.append(
                ConceptCluster(
                    cluster_id=cluster_id,
                    section_ids=sorted(section_ids),
                    score=avg_score,
                    label=None,
                    meta={"anchor_section_id": target_id},
                )
            )

        return clusters
