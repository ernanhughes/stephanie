# stephanie/memory/paper_store.py
from __future__ import annotations

import hashlib
from typing import Optional, List, Dict, Any
import uuid

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.paper import PaperORM, PaperReferenceORM, PaperRunComparisonORM, PaperRunEventORM, PaperRunFeatureORM, PaperRunORM, PaperSimilarORM


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

class PaperStore(BaseSQLAlchemyStore):
    orm_model = PaperORM
    default_order_by = "updated_at"

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "papers"        
        self._refs = PaperReferenceStore(session_or_maker, logger)
        self._similar = PaperSimilarStore(session_or_maker, logger)

    # ---------- Reads ----------

    def get_many_by_id(self, paper_ids: list[str], chunk_size: int = 500) -> list[PaperORM]:
        def op(s):
            out: list[PaperORM] = []
            for i in range(0, len(paper_ids), chunk_size):
                chunk = paper_ids[i:i+chunk_size]
                out.extend(s.query(PaperORM).filter(PaperORM.id.in_(chunk)).all())
            return out
        return self._run(op)
    
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


    # ---------- References ----------

    def get_references(self, paper_id: str):
        return self._refs.get_for_paper(paper_id)

    def replace_references(self, paper_id: str, refs: list[dict]):
        return self._refs.replace_for_paper(paper_id, refs)

    # ---------- Similar papers ----------

    def get_similar(self, paper_id: str, provider="hf_similar", limit=50):
        return self._similar.get_for_paper(
            paper_id, provider=provider, limit=limit
        )

    def replace_similar(self, paper_id: str, similars: list[dict], provider="hf_similar"):
        return self._similar.replace_for_paper(
            paper_id, similars, provider=provider
        )

    def create_run(
        self,
        *,
        paper_id: str,
        run_type: str,
        config: dict,
        stats: dict | None = None,
        artifact_path: str | None = None,
    ) -> PaperRunORM:
        def op(s):
            run = PaperRunORM(
                id=uuid.uuid4().hex,
                paper_id=paper_id,
                run_type=run_type,
                config=config,
                stats=stats or {},
                artifact_path=artifact_path,
            )
            s.add(run)
            s.flush()
            return run
        return self._run(op)

    def record_ai_judgement(
        self,
        run_id: str,
        *,
        score: float,
        rationale: str,
        judge: str,
        prompt_hash: str,
    ) -> PaperRunORM:
        def op(s):
            run = s.get(PaperRunORM, run_id)
            if not run:
                return None
            run.ai_score = float(score)
            run.ai_rationale = rationale
            run.ai_judge = judge
            run.ai_prompt_hash = prompt_hash
            s.flush()
            return run
        return self._run(op)

    def get_last_run(
        self, paper_id: str, run_type: str | None = None
    ) -> PaperRunORM | None:
        def op(s):
            q = s.query(PaperRunORM).filter_by(paper_id=paper_id)
            if run_type:
                q = q.filter_by(run_type=run_type)
            return q.order_by(PaperRunORM.created_at.desc()).first()
        return self._run(op)

    def list_runs(
        self, paper_id: str, run_type: str | None = None, limit: int = 20
    ):
        def op(s):
            q = s.query(PaperRunORM).filter_by(paper_id=paper_id)
            if run_type:
                q = q.filter_by(run_type=run_type)
            return q.order_by(PaperRunORM.created_at.desc()).limit(limit).all()
        return self._run(op)


    def add_run_features(
        self,
        *,
        feature_id: str,
        run_id: str,
        features: dict,
        extractor: str | None = None,
    ):
        def op(s):
            row = PaperRunFeatureORM(
                id=feature_id,
                run_id=run_id,
                features=features,
                extractor=extractor,
            )
            s.add(row)
            s.flush()
            return row
        return self._run(op)


    def get_run_features(self, run_id: str):
        def op(s):
            return (
                s.query(PaperRunFeatureORM)
                .filter_by(run_id=run_id)
                .order_by(PaperRunFeatureORM.created_at.desc())
                .all()
            )
        return self._run(op)


    def add_run_comparison(
        self,
        *,
        comparison_id: str,
        paper_id: str,
        run_a_id: str,
        run_b_id: str,
        preference: float,
        judge: str | None = None,
        rationale: str | None = None,
    ):
        def op(s):
            row = PaperRunComparisonORM(
                id=comparison_id,
                paper_id=paper_id,
                run_a_id=run_a_id,
                run_b_id=run_b_id,
                preference=preference,
                judge=judge,
                rationale=rationale,
            )
            s.add(row)
            s.flush()
            return row
        return self._run(op)


    def get_comparisons_for_paper(self, paper_id: str):
        def op(s):
            return (
                s.query(PaperRunComparisonORM)
                .filter_by(paper_id=paper_id)
                .order_by(PaperRunComparisonORM.created_at.desc())
                .all()
            )
        return self._run(op)



    def add_run_event(
        self,
        *,
        event_id: str,
        run_id: str,
        event_type: str,
        payload: dict | None = None,
    ):
        def op(s):
            row = PaperRunEventORM(
                id=event_id,
                run_id=run_id,
                event_type=event_type,
                payload=payload,
            )
            s.add(row)
            s.flush()
            return row
        return self._run(op)


    def get_run_events(self, run_id: str, limit: int = 1000):
        def op(s):
            return (
                s.query(PaperRunEventORM)
                .filter_by(run_id=run_id)
                .order_by(PaperRunEventORM.created_at.asc())
                .limit(limit)
                .all()
            )
        return self._run(op)





# ---------------------------------------------------------------------------



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
