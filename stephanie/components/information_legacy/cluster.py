import math
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set

from stephanie.components.information_legacy.data import DocumentSection
from stephanie.utils.similarity_utils import cosine


class ConceptClusterTask:
    """
    Find clusters of sections whose title+summary embeddings are very similar.
    These clusters represent shared concepts across different papers/sections.
    """

    def __init__(
        self,
        strong_threshold: float = 0.8,  # “very strong similarity”
        min_sections: int = 2,          # at least 2 sections in a cluster
        min_papers: int = 2,            # at least 2 different papers
    ) -> None:
        self.strong_threshold = strong_threshold
        self.min_sections = min_sections
        self.min_papers = min_papers

    def _concept_embedding(self, sec: DocumentSection) -> Optional[list[float]]:
        """
        Combine title + summary embedding to represent the concept.
        """
        embs = []
        if sec.title_embedding is not None:
            embs.append(sec.title_embedding)
        if sec.summary_embedding is not None:
            embs.append(sec.summary_embedding)

        if not embs:
            return None

        dim = len(embs[0])
        avg = [0.0] * dim
        for e in embs:
            for i, v in enumerate(e):
                avg[i] += v
        n = float(len(embs))
        return [v / n for v in avg]

    def run(self, sections: List[DocumentSection]) -> Dict[str, ConceptCluster]:
        """
        Mutates sections in-place to add concept_cluster_id/strength.
        Returns cluster_id -> ConceptCluster.
        """
        # 1) Build embeddings list
        sec_list: List[tuple[int, DocumentSection, Optional[list[float]]]] = []
        for idx, sec in enumerate(sections):
            emb = self._concept_embedding(sec)
            if emb is not None:
                sec_list.append((idx, sec, emb))

        n = len(sec_list)
        if n == 0:
            return {}

        # 2) Build similarity graph (adjacency) with strong_threshold
        adjacency: Dict[int, List[int]] = defaultdict(list)

        for i in range(n):
            idx_i, sec_i, emb_i = sec_list[i]
            for j in range(i + 1, n):
                idx_j, sec_j, emb_j = sec_list[j]
                sim = cosine(emb_i, emb_j)
                if sim >= self.strong_threshold:
                    adjacency[idx_i].append(idx_j)
                    adjacency[idx_j].append(idx_i)

        # 3) DFS/Union-Find to get connected components
        visited: Set[int] = set()
        clusters: Dict[str, ConceptCluster] = {}

        for i in range(n):
            idx_i, sec_i, emb_i = sec_list[i]
            if idx_i in visited:
                continue

            stack = [idx_i]
            component_idxs: List[int] = []
            visited.add(idx_i)

            while stack:
                cur = stack.pop()
                component_idxs.append(cur)
                for neigh in adjacency.get(cur, []):
                    if neigh not in visited:
                        visited.add(neigh)
                        stack.append(neigh)

            if len(component_idxs) < self.min_sections:
                continue  # too small

            # 4) Build cluster stats
            section_ids: List[str] = []
            paper_ids: Set[Optional[str]] = set()
            sims: List[float] = []

            # compute pairwise sims inside the component for avg_similarity
            for idx_a in component_idxs:
                _, sec_a, emb_a = sec_list[idx_a]
                section_ids.append(sec_a.section_id)
                paper_ids.add(sec_a.paper_arxiv_id)

            # Only keep clusters that span multiple papers
            if len(paper_ids) < self.min_papers:
                continue

            # simple pairwise similarity average
            for i_pos in range(len(component_idxs)):
                idx_a = component_idxs[i_pos]
                _, _, emb_a = sec_list[idx_a]
                for j_pos in range(i_pos + 1, len(component_idxs)):
                    idx_b = component_idxs[j_pos]
                    _, _, emb_b = sec_list[idx_b]
                    sims.append(cosine(emb_a, emb_b))

            avg_sim = sum(sims) / len(sims) if sims else self.strong_threshold

            cluster_id = str(uuid.uuid4())
            clusters[cluster_id] = ConceptCluster(
                cluster_id=cluster_id,
                section_ids=section_ids,
                paper_ids=paper_ids,
                avg_similarity=avg_sim,
            )

        # 5) Push cluster info back into sections as “concept_cluster_id/strength”
        # strength = cluster avg_similarity * log(#papers) or similar
        for cluster in clusters.values():
            strength = cluster.avg_similarity * (1.0 + math.log10(len(cluster.paper_ids) + 1e-6))
            for sid in cluster.section_ids:
                for sec in sections:
                    if sec.section_id == sid:
                        sec.concept_cluster_id = cluster.cluster_id
                        sec.concept_cluster_strength = strength

        return clusters
