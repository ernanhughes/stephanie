# stephanie/memory/paper_store.py
from __future__ import annotations

import hashlib
from typing import Optional, List, Dict, Any

from sqlalchemy import desc
from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.paper import PaperORM, PaperReferenceORM, PaperSimilarORM


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class PaperStore(BaseSQLAlchemyStore):
    orm_model = PaperORM
    default_order_by = "updated_at"

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "papers"

    # ---------- Reads ----------

    def get_by_id(self, paper_id: str) -> Optional[PaperORM]:
        def op(s):
            return s.get(PaperORM, paper_id)
        return self._run(op)

    def get_by_url(self, url: str) -> Optional[PaperORM]:
        def op(s):
            return s.query(PaperORM).filter_by(url=url).first()
        return self._run(op)

    # ---------- Writes ----------

    def upsert_paper(self, paper_id: str, fields: Dict[str, Any]) -> PaperORM:
        """
        Upsert pattern consistent with your stores (query then set).
        """
        def op(s):
            row = s.get(PaperORM, paper_id)
            if row is None:
                row = PaperORM(id=paper_id, source=fields.get("source", "arxiv"))
                s.add(row)

            for k, v in fields.items():
                setattr(row, k, v)

            s.flush()
            return row
        return self._run(op)


class PaperReferenceStore(BaseSQLAlchemyStore):
    orm_model = PaperReferenceORM
    default_order_by = "id"

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "paper_references"

    def delete_for_paper(self, paper_id: str) -> int:
        def op(s):
            q = s.query(PaperReferenceORM).filter_by(paper_id=paper_id)
            n = q.count()
            q.delete(synchronize_session=False)
            return n
        return self._run(op)

    def replace_for_paper(self, paper_id: str, refs: List[Dict[str, Any]]) -> List[PaperReferenceORM]:
        """
        Simple and deterministic: delete then insert in one transaction.
        (Great for “ground truth is whatever the extractor produced”.)
        """
        def op(s):
            s.query(PaperReferenceORM).filter_by(paper_id=paper_id).delete(synchronize_session=False)

            rows: List[PaperReferenceORM] = []
            for idx, r in enumerate(refs):
                rows.append(
                    PaperReferenceORM(
                        paper_id=paper_id,
                        order_idx=r.get("order_idx", idx),
                        ref_arxiv_id=r.get("ref_arxiv_id"),
                        doi=r.get("doi"),
                        title=r.get("title"),
                        year=r.get("year"),
                        url=r.get("url"),
                        raw_citation=r.get("raw_citation"),
                        source=r.get("source", "parsed_pdf"),
                        raw=r.get("raw", {}),
                    )
                )

            s.add_all(rows)
            s.flush()
            return rows
        return self._run(op)

    def get_for_paper(self, paper_id: str, limit: int = 500) -> List[PaperReferenceORM]:
        def op(s):
            return (
                s.query(PaperReferenceORM)
                .filter_by(paper_id=paper_id)
                .order_by(PaperReferenceORM.order_idx.asc())
                .limit(limit)
                .all()
            )
        return self._run(op)


class PaperSimilarStore(BaseSQLAlchemyStore):
    orm_model = PaperSimilarORM
    default_order_by = "id"

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "paper_similar"

    def replace_for_paper(
        self,
        paper_id: str,
        similars: List[Dict[str, Any]],
        provider: str = "hf_similar",
    ) -> List[PaperSimilarORM]:
        def op(s):
            s.query(PaperSimilarORM).filter_by(paper_id=paper_id, provider=provider).delete(synchronize_session=False)

            rows: List[PaperSimilarORM] = []
            for idx, r in enumerate(similars):
                rows.append(
                    PaperSimilarORM(
                        paper_id=paper_id,
                        provider=provider,
                        rank=r.get("rank", idx),
                        score=r.get("score"),
                        similar_arxiv_id=r["similar_arxiv_id"],
                        url=r.get("url"),
                        title=r.get("title"),
                        raw=r.get("raw", {}),
                    )
                )

            s.add_all(rows)
            s.flush()
            return rows
        return self._run(op)

    def get_for_paper(self, paper_id: str, provider: str = "hf_similar", limit: int = 50) -> List[PaperSimilarORM]:
        def op(s):
            return (
                s.query(PaperSimilarORM)
                .filter_by(paper_id=paper_id, provider=provider)
                .order_by(PaperSimilarORM.rank.asc())
                .limit(limit)
                .all()
            )
        return self._run(op)
