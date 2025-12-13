from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional

from stephanie.components.information_legacy.data import (DocumentSection,
                                                          SectionMatch)
from stephanie.utils.similarity_utils import cosine


class SectionMatcherTask:
    """
    For each root-paper section, find the most similar sections among
    all related papers, using summary/title embeddings.
    """

    def __init__(
        self,
        use_summary: bool = True,
        use_title: bool = True,
        top_k: int = 5,
        min_similarity: float = 0.4,
    ) -> None:
        self.use_summary = use_summary
        self.use_title = use_title
        self.top_k = top_k
        self.min_similarity = min_similarity

    def _build_query_embedding(self, sec: DocumentSection) -> Optional[list[float]]:
        """
        Combine title + summary embedding (if both available).
        Simple average for now.
        """
        embs: List[list[float]] = []

        if self.use_title and sec.title_embedding is not None:
            embs.append(sec.title_embedding)
        if self.use_summary and sec.summary_embedding is not None:
            embs.append(sec.summary_embedding)

        if not embs:
            return None

        # average
        dim = len(embs[0])
        avg = [0.0] * dim
        for e in embs:
            for i, v in enumerate(e):
                avg[i] += v
        n = float(len(embs))
        avg = [v / n for v in avg]
        return avg

    def run(
        self,
        all_sections: List[DocumentSection],
        root_arxiv_id: str,
    ) -> Dict[str, List[SectionMatch]]:
        """
        Returns:
            mapping: root_section_id -> list[SectionMatch] (sorted by similarity desc)
        """
        root_sections: List[DocumentSection] = []
        related_sections: List[DocumentSection] = []

        for sec in all_sections:
            if sec.paper_arxiv_id == root_arxiv_id:
                root_sections.append(sec)
            else:
                related_sections.append(sec)

        # Precompute candidate embeddings for related sections
        related_embs: List[tuple[DocumentSection, Optional[list[float]]]] = []
        for sec in related_sections:
            emb = self._build_query_embedding(sec)
            related_embs.append((sec, emb))

        matches_by_source: DefaultDict[str, List[SectionMatch]] = defaultdict(list)

        for src in root_sections:
            src_emb = self._build_query_embedding(src)
            if src_emb is None:
                continue

            scored: List[SectionMatch] = []
            for tgt, tgt_emb in related_embs:
                if tgt_emb is None:
                    continue
                sim = cosine(src_emb, tgt_emb)
                if sim < self.min_similarity:
                    continue
                scored.append(
                    SectionMatch(
                        source_section_id=src.section_id,
                        target_section_id=tgt.section_id,
                        similarity=sim,
                        target_paper_arxiv_id=tgt.paper_arxiv_id,
                        target_paper_role=tgt.paper_role,
                    )
                )

            scored.sort(key=lambda m: m.similarity, reverse=True)
            if self.top_k is not None:
                scored = scored[:self.top_k]

            matches_by_source[src.section_id] = scored

        return matches_by_source
