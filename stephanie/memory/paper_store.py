# stephanie/memory/paper_store.py
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import insert

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.paper import (PaperORM, PaperReferenceORM,
                                    PaperRunComparisonORM, PaperRunEventORM,
                                    PaperRunFeatureORM, PaperRunORM,
                                    PaperSectionORM, PaperSimilarORM)


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
        self._sections = PaperSectionStore(session_or_maker, logger)  # NEW

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

    # ---------- Sections ----------

    def get_sections_for_paper(self, paper_id: str) -> List[PaperSectionORM]:
        """Get all sections for a specific paper."""
        return self._sections.get_by_paper_id(paper_id)

    def replace_sections_for_paper(self, paper_id: str, sections: List[Dict[str, Any]]) -> List[PaperSectionORM]:
        """Replace all sections for a paper with new ones."""
        return self._sections.replace_for_paper(paper_id, sections)

    def upsert_section(self, section_data: Dict[str, Any]) -> PaperSectionORM:
        """Upsert a single section."""
        return self._sections.upsert(section_data)

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
        variant: str | None = None,
    ) -> PaperRunORM:
        def op(s):
            run = PaperRunORM(
                id=uuid.uuid4().hex,
                paper_id=paper_id,
                run_type=run_type,
                config=config,
                stats=stats or {},
                artifact_path=artifact_path,
                variant=variant,
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

    # ---------------------------
    # Runs
    # ---------------------------

    def mark_run_error(self, run_id: str, error: str) -> None:
        def op(s):
            r = s.get(PaperRunORM, run_id)
            if not r:
                return
            r.status = "error"
            r.error = error
        self._run(op)

    def set_run_judge(
        self,
        *,
        run_id: str,
        ai_score: float,
        ai_scores: Optional[Dict[str, Any]] = None,
        ai_rationale: Optional[str] = None,
        judge_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        def op(s):
            r = s.get(PaperRunORM, run_id)
            if not r:
                return
            r.ai_score = float(ai_score)
            r.ai_scores = ai_scores or {}
            r.ai_rationale = ai_rationale
            r.judge_meta = judge_meta or {}
        self._run(op)

    # ---------------------------
    # Run features
    # ---------------------------

    def upsert_run_features(self, run_id: str, features: Dict[str, Any]) -> None:
        def op(s):
            stmt = insert(PaperRunFeatureORM).values(run_id=run_id, features=features)
            stmt = stmt.on_conflict_do_update(
                index_elements=[PaperRunFeatureORM.run_id],
                set_={"features": stmt.excluded.features},
            )
            s.execute(stmt)
        self._run(op)

    # ---------------------------
    # Comparisons
    # ---------------------------

    def add_comparison(
        self,
        *,
        paper_id: str,
        left_run_id: str,
        right_run_id: str,
        winner_run_id: Optional[str],
        preference: str,
        rationale: Optional[str] = None,
        judge_source: str = "llm",
        scores: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        def op(s):
            obj = PaperRunComparisonORM(
                paper_id=paper_id,
                left_run_id=left_run_id,
                right_run_id=right_run_id,
                winner_run_id=winner_run_id,
                preference=preference,
                rationale=rationale,
                judge_source=judge_source,
                scores=scores or {},
                meta=meta or {},
            )
            s.add(obj)
        self._run(op)

    # ---------------------------
    # Events
    # ---------------------------

    def add_event(
        self,
        *,
        run_id: str,
        stage: str,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        def op(s):
            s.add(
                PaperRunEventORM(
                    run_id=run_id,
                    stage=stage,
                    event_type=event_type,
                    message=message,
                    data=data or {},
                )
            )
        self._run(op)

    # ---------------------------
    # Helpers
    # ---------------------------

    @staticmethod
    def hash_config(config: Dict[str, Any], keys: Optional[List[str]] = None) -> str:
        """
        Stable hash of the subset of config that matters.
        """
        if keys:
            payload = {k: config.get(k) for k in keys}
        else:
            payload = config
        s = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------


class PaperSectionStore(BaseSQLAlchemyStore):
    orm_model = PaperSectionORM
    default_order_by = "section_index"

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "paper_sections"

    def get_by_paper_id(self, paper_id: str) -> List[PaperSectionORM]:
        """Get all sections for a specific paper."""
        def op(s):
            return (
                s.query(PaperSectionORM)
                .filter_by(paper_id=paper_id)
                .order_by(PaperSectionORM.section_index.asc())
                .all()
            )
        return self._run(op)
    
    def delete_by_paper_id(self, paper_id: str) -> int:
        """Delete all sections for a paper."""
        def op(s):
            q = s.query(PaperSectionORM).filter_by(paper_id=paper_id)
            n = q.count()
            q.delete(synchronize_session=False)
            return n
        return self._run(op)
    
    def replace_for_paper(self, paper_id: str, sections: List[Dict[str, Any]]) -> List[PaperSectionORM]:
        """
        Replace all sections for a paper with new ones.
        First deletes existing sections, then inserts new ones.
        """
        def op(s):
            # Delete existing sections for this paper
            self.delete_by_paper_id(paper_id)
            
            # Insert new sections
            rows: List[PaperSectionORM] = []
            for idx, section in enumerate(sections):
                # Create a new PaperSectionORM instance
                row = PaperSectionORM(
                    id=section.get("id", f"{paper_id}::sec-{idx}"),
                    paper_id=paper_id,
                    section_index=section.get("section_index", idx),
                    start_char=section.get("start_char"),
                    end_char=section.get("end_char"),
                    start_page=section.get("start_page"),
                    end_page=section.get("end_page"),
                    text=section.get("text"),
                    title=section.get("title"),
                    summary=section.get("summary"),
                    meta=section.get("meta", {}),
                )
                rows.append(row)
            
            s.add_all(rows)
            s.flush()
            return rows
        return self._run(op)
    
    def upsert(self, section_data: Dict[str, Any]) -> PaperSectionORM:
        """
        Upsert a single section based on its id.
        If section with given id exists, update it; otherwise create new.
        """
        def op(s):
            section_id = section_data.get("id")
            if not section_id:
                raise ValueError("section_data must have an 'id' field")
            
            # Check if section exists
            row = s.query(PaperSectionORM).get(section_id)
            if row is None:
                row = PaperSectionORM(id=section_id)
                s.add(row)
            
            # Update fields
            for key, value in section_data.items():
                if key != "id":
                    setattr(row, key, value)
            
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
