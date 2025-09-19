# stephanie/memory/score_store.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import joinedload

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from stephanie.models.score_attribute import ScoreAttributeORM


class ScoreStore(BaseSQLAlchemyStore):
    orm_model = ScoreORM
    default_order_by = ScoreORM.id.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "scores"
        self.table_name = "scores"

    def add_score(self, score: ScoreORM) -> ScoreORM:
        def op(s):
            
            s.add(score)
            s.flush()
            s.refresh(score)
            return score
        return self._run(op)

    def add_scores_bulk(self, scores: List[ScoreORM]):
        def op(s):
            self._scope().add_all(scores)
        return self._run(op)

    def get_scores_for_evaluation(self, evaluation_id: int) -> List[ScoreORM]:
        def op(s):
            return (
                s.query(ScoreORM)
                .filter_by(evaluation_id=evaluation_id)
                .order_by(ScoreORM.dimension.asc())
                .all()
            )
        return self._run(op)

    def get_scores_for_hypothesis(self, hypothesis_id: int) -> List[ScoreORM]:
        def op(s):
            return (
                s.query(ScoreORM)
                .filter(ScoreORM.evaluation.has(hypothesis_id=hypothesis_id))
                .order_by(ScoreORM.dimension.asc())
                .all()
            )
        return self._run(op)

    def get_scores_by_dimension(self, dimension: str, top_k: int = 100) -> List[ScoreORM]:
        def op(s):
            return (
                s.query(ScoreORM)
                .filter_by(dimension=dimension)
                .order_by(ScoreORM.score.desc().nullslast())
                .limit(top_k)
                .all()
            )
        return self._run(op)

    def delete_scores_for_evaluation(self, evaluation_id: int):
        def op(s):
            s.query(ScoreORM).filter_by(evaluation_id=evaluation_id).delete()
        return self._run(op)

    def get_all(self, limit: Optional[int] = None) -> List[ScoreORM]:
        def op(s):
            q = s.query(ScoreORM).order_by(ScoreORM.id.desc())
            if limit:
                q = q.limit(limit)
            return q.all()
        return self._run(op)

    def get_by_id(self, score_id: int) -> Optional[ScoreORM]:
        def op(s):
            return s.query(ScoreORM).filter_by(id=score_id).first()
        return self._run(op)

    def get_by_evaluation_ids(self, evaluation_ids: list[int]) -> list[ScoreORM]:
        if not evaluation_ids:
            return []
        def op(s):
            return (
                s.query(ScoreORM)
                .filter(ScoreORM.evaluation_id.in_(evaluation_ids))
                .all()
            )
        return self._run(op)

    def get_score_by_prompt_hash(self, prompt_hash: str) -> Optional[ScoreORM]:
        def op(s):
            return (
                s.query(ScoreORM)
                .filter(ScoreORM.prompt_hash == prompt_hash, ScoreORM.score > 0)
                .order_by(ScoreORM.id.desc())
                .first()
            )
        return self._run(op)

    def get_scores_for_target(
        self, target_id: str, target_type: str, dimensions: Optional[list[str]] = None
    ) -> List[ScoreORM]:
        def op(s):
            q = (
                s.query(ScoreORM)
                .join(EvaluationORM, ScoreORM.evaluation_id == EvaluationORM.id)
                .options(joinedload(ScoreORM.evaluation))
                .filter(EvaluationORM.scorable_id == str(target_id))
                .filter(EvaluationORM.scorable_type == target_type)
            )
            if dimensions:
                q = q.filter(ScoreORM.dimension.in_(dimensions))
            return q.all()
        return self._run(op)

    def add_dimension_score(
        self,
        evaluation_id: int,
        dimension: str,
        score: float,
        *,
        weight: float = 1.0,
        source: Optional[str] = None,
        rationale: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> ScoreORM:
        def op(s):
            
            row = ScoreORM(
                evaluation_id=evaluation_id,
                dimension=dimension,
                score=score,
                weight=weight,
                source=source,
                rationale=rationale,
            )
            s.add(row)
            s.flush()
            if attributes:
                for k, v in attributes.items():
                    s.add(
                        ScoreAttributeORM(
                            score_id=row.id,
                            key=k,
                            value=json.dumps(v) if isinstance(v, (list, dict)) else str(v),
                            data_type=(
                                "json"
                                if isinstance(v, (list, dict))
                                else "float"
                                if isinstance(v, (int, float))
                                else "string"
                            ),
                        )
                    )
            s.flush()
            s.refresh(row)
            return row
        return self._run(op)

    def get_latest_for_target_dimension(
        self, scorable_id: str, scorable_type: str, dimension: str
    ) -> Optional[ScoreORM]:
        def op(s):
            return (
                s.query(ScoreORM)
                .join(EvaluationORM, ScoreORM.evaluation_id == EvaluationORM.id)
                .filter(EvaluationORM.scorable_id == str(scorable_id))
                .filter(EvaluationORM.scorable_type == str(scorable_type))
                .filter(ScoreORM.dimension == dimension)
                .order_by(EvaluationORM.created_at.desc(), ScoreORM.id.desc())
                .first()
            )
        return self._run(op)

    def get_value_signal_for_target(
        self,
        target_id: str,
        target_type: str,
        prefer: tuple[str, ...] = ("alignment",),
        *,
        normalize: bool = True,
        default: float = 0.0,
    ) -> float:
        for dim in prefer:
            row = self.get_latest_for_target_dimension(target_id, target_type, dim)
            if row:
                val = float(row.score)
                if normalize:
                    if val > 5.0:
                        val = min(val / 100.0, 1.0)
                    val = max(0.0, min(1.0, val))
                return val
        def op(s):
            eval_rec = (
                s.query(EvaluationORM)
                .filter(
                    EvaluationORM.scorable_id == str(target_id),
                    EvaluationORM.scorable_type == target_type,
                )
                .order_by(EvaluationORM.created_at.desc())
                .first()
            )
            if not eval_rec:
                return default
            raw = float((eval_rec.scores or {}).get("avg", default))
            if normalize and raw > 5.0:
                raw = min(raw / 100.0, 1.0)
            return max(0.0, min(1.0, raw))
        return self._run(op)

    def get_hrm_score(
        self,
        scorable_id: str,
        scorable_type: str,
        *,
        normalize: bool = True,
        prefer: tuple[str, ...] = ("hrm", "reasoning_quality", "HRM"),
        fallback_to_avg: bool = False,
    ) -> Optional[float]:
        for dim in prefer:
            row = self.get_latest_for_target_dimension(scorable_id, scorable_type, dim)
            if row is not None:
                val = float(row.score)
                if normalize:
                    if val > 5.0:
                        val = min(val / 100.0, 1.0)
                    val = max(0.0, min(1.0, val))
                return val
        if fallback_to_avg:
            def op(s):
                eval_rec = (
                    s.query(EvaluationORM)
                    .filter(
                        EvaluationORM.scorable_id == str(scorable_id),
                        EvaluationORM.scorable_type == str(scorable_type),
                    )
                    .order_by(EvaluationORM.created_at.desc())
                    .first()
                )
                if eval_rec and isinstance(eval_rec.scores, dict):
                    raw = eval_rec.scores.get("avg")
                    if raw is not None:
                        val = float(raw)
                        if normalize and val > 5.0:
                            val = min(val / 100.0, 1.0)
                        return max(0.0, min(1.0, val))
                return None
            return self._run(op)
        return None
