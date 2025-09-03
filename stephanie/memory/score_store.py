# stephanie/memory/score_store.py
import json
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session, joinedload

from stephanie.models.evaluation import EvaluationORM
from stephanie.models.score import ScoreORM
from stephanie.models.score_attribute import ScoreAttributeORM


class ScoreStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "scores"
        self.table_name = "scores"

    def add_score(self, score: ScoreORM) -> ScoreORM:
        self.session.add(score)
        self.session.commit()
        self.session.refresh(score)
        return score

    def add_scores_bulk(self, scores: List[ScoreORM]):
        self.session.add_all(scores)
        self.session.commit()

    def get_scores_for_evaluation(self, evaluation_id: int) -> List[ScoreORM]:
        return (
            self.session.query(ScoreORM)
            .filter_by(evaluation_id=evaluation_id)
            .order_by(ScoreORM.dimension.asc())
            .all()
        )

    def get_scores_for_hypothesis(self, hypothesis_id: int) -> List[ScoreORM]:
        return (
            self.session.query(ScoreORM)
            .filter(ScoreORM.evaluation.has(hypothesis_id=hypothesis_id))
            .order_by(ScoreORM.dimension.asc())
            .all()
        )

    def get_scores_by_dimension(
        self, dimension: str, top_k: int = 100
    ) -> List[ScoreORM]:
        return (
            self.session.query(ScoreORM)
            .filter_by(dimension=dimension)
            .order_by(ScoreORM.score.desc().nullslast())
            .limit(top_k)
            .all()
        )

    def delete_scores_for_evaluation(self, evaluation_id: int):
        self.session.query(ScoreORM).filter_by(
            evaluation_id=evaluation_id
        ).delete()
        self.session.commit()

    def get_all(self, limit: Optional[int] = None) -> List[ScoreORM]:
        query = self.session.query(ScoreORM).order_by(ScoreORM.id.desc())
        if limit:
            query = query.limit(limit)
        return query.all()

    def get_by_id(self, score_id: int) -> Optional[ScoreORM]:
        return self.session.query(ScoreORM).filter_by(id=score_id).first()

    def get_by_evaluation_ids(
        self, evaluation_ids: list[int]
    ) -> list[ScoreORM]:
        if not evaluation_ids:
            return []
        try:
            return (
                self.session.query(ScoreORM)
                .filter(ScoreORM.evaluation_id.in_(evaluation_ids))
                .all()
            )
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "GetByEvaluationError",
                    {
                        "method": "get_by_evaluation_ids",
                        "error": str(e),
                        "evaluation_ids": evaluation_ids,
                    },
                )
            return []

    def get_score_by_prompt_hash(self, prompt_hash: str) -> Optional[ScoreORM]:
        try:
            return (
                self.session.query(ScoreORM)
                .filter(
                    ScoreORM.prompt_hash == prompt_hash, ScoreORM.score > 0
                )
                .order_by(ScoreORM.id.desc())
                .first()
            )
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "GetScoreError",
                    {
                        "method": "get_score_by_prompt_hash",
                        "error": str(e),
                        "prompt_hash": prompt_hash,
                    },
                )
            return None

    def get_scores_for_target(
        self,
        target_id: str,
        target_type: str,
        dimensions: Optional[list[str]] = None,
    ) -> List[ScoreORM]:
        from stephanie.models.evaluation import EvaluationORM

        """
        Fetch all scores for a given (target_id, target_type).
        This works for documents, cartridges, theorems, triples, etc.

        Args:
            target_id (str): ID of the target (document_id, cartridge_id, etc.)
            target_type (str): Type of the target ("document", "cartridge", "theorem", ...)
            dimensions (list[str], optional): If provided, restrict to these dimensions.

        Returns:
            List[ScoreORM]: Score objects with linked evaluations.
        """
        try:
            q = (
                self.session.query(ScoreORM)
                .join(
                    EvaluationORM, ScoreORM.evaluation_id == EvaluationORM.id
                )
                .options(joinedload(ScoreORM.evaluation))
                .filter(EvaluationORM.scorable_id == str(target_id))
                .filter(EvaluationORM.scorable_type == target_type)
            )

            if dimensions:
                q = q.filter(ScoreORM.dimension.in_(dimensions))

            return q.all()

        except Exception as e:
            if self.logger:
                self.logger.log(
                    "GetScoresForTargetError",
                    {
                        "target_id": target_id,
                        "target_type": target_type,
                        "dimensions": dimensions,
                        "error": str(e),
                    },
                )
            return []

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
        """Convenience: add a single ScoreORM row (e.g., 'hrm', 'rank_score')."""
        row = ScoreORM(
            evaluation_id=evaluation_id,
            dimension=dimension,
            score=score,
            weight=weight,
            source=source,
            rationale=rationale,
        )
        self.session.add(row)
        self.session.flush()

        if attributes:
            for k, v in attributes.items():
                self.session.add(
                    ScoreAttributeORM(
                        score_id=row.id,
                        key=k,
                        value=json.dumps(v)
                        if isinstance(v, (list, dict))
                        else str(v),
                        data_type="json"
                        if isinstance(v, (list, dict))
                        else "float"
                        if isinstance(v, (int, float))
                        else "string",
                    )
                )

        self.session.commit()
        self.session.refresh(row)
        return row

    def get_latest_for_target_dimension(
        self, scorable_id: str, scorable_type: str, dimension: str
    ) -> Optional[ScoreORM]:
        """Return latest ScoreORM for (scorable_id, scorable_type, dimension)."""
        return (
            self.session.query(ScoreORM)
            .join(EvaluationORM, ScoreORM.evaluation_id == EvaluationORM.id)
            .filter(EvaluationORM.scorable_id == str(scorable_id))
            .filter(EvaluationORM.scorable_type == str(scorable_type))
            .filter(ScoreORM.dimension == dimension)
            .order_by(EvaluationORM.created_at.desc(), ScoreORM.id.desc())
            .first()
        )

    def get_value_signal_for_target(
        self,
        target_id: str,
        target_type: str,
        prefer: tuple[str, ...] = ("alignment"),
        *,
        normalize: bool = True,
        default: float = 0.0,
    ) -> float:
        """Return a scalar value in [0,1] for ranking.
        Prefers HRM, then rank_score; falls back to EvaluationORM.scores['avg'].
        """
        # try preferred dimensions in order
        for dim in prefer:
            row = self.get_latest_for_target_dimension(
                target_id, target_type, dim
            )
            if row:
                val = float(row.score)
                if normalize:
                    # legacy guard: if it looks like 0–100, squash to 0–1
                    if val > 5.0:
                        val = min(val / 100.0, 1.0)
                    # clamp
                    val = max(0.0, min(1.0, val))
                return val

        # fallback: EvaluationORM.scores["avg"]
        eval_rec = (
            self.session.query(EvaluationORM)
            .filter(
                EvaluationORM.scorable_id == str(target_id),
                EvaluationORM.scorable_type == target_type,
            )
            .order_by(EvaluationORM.created_at.desc())
            .first()
        )
        try:
            raw = (
                float((eval_rec.scores or {}).get("avg", default))
                if eval_rec
                else default
            )
            if normalize and raw > 5.0:
                raw = min(raw / 100.0, 1.0)
            return max(0.0, min(1.0, raw))
        except Exception:
            return default

    def get_hrm_score(
        self,
        scorable_id: str,
        scorable_type: str,
        *,
        normalize: bool = True,
        prefer: tuple[str, ...] = ("hrm", "reasoning_quality", "HRM"),
        fallback_to_avg: bool = False,
    ) -> Optional[float]:
        """
        Return the latest HRM-like scalar for (target_id, target_type).

        - Checks preferred dimensions in order (default: 'hrm', 'reasoning_quality', 'HRM').
        - Normalizes to [0,1] (legacy guard: if >5, assume 0–100 and divide by 100).
        - Returns None if not found (unless fallback_to_avg=True, then uses latest EvaluationORM.scores['avg']).

        Args:
            target_id: scorable id (e.g., plan_trace.trace_id)
            target_type: scorable type (e.g., "plan_trace")
            normalize: clamp/convert to [0,1]
            prefer: tuple of dimensions to look for
            fallback_to_avg: if True, fall back to Evaluation.scores['avg'] when no HRM-like row exists

        Returns:
            float in [0,1] or None
        """
        # 1) Try preferred dimensions
        for dim in prefer:
            row = self.get_latest_for_target_dimension(scorable_id, scorable_type, dim)
            if row is not None:
                val = float(row.score)
                if normalize:
                    if val > 5.0:  # legacy 0–100
                        val = min(val / 100.0, 1.0)
                    val = max(0.0, min(1.0, val))
                return val

        # 2) Optional fallback to latest Evaluation.avg
        if fallback_to_avg:
            from stephanie.models.evaluation import EvaluationORM
            eval_rec = (
                self.session.query(EvaluationORM)
                .filter(EvaluationORM.scorable_id == str(scorable_id),
                        EvaluationORM.scorable_type == str(scorable_type))
                .order_by(EvaluationORM.created_at.desc())
                .first()
            )
            if eval_rec and isinstance(eval_rec.scores, dict):
                raw = eval_rec.scores.get("avg")
                if raw is not None:
                    try:
                        val = float(raw)
                        if normalize:
                            if val > 5.0:
                                val = min(val / 100.0, 1.0)
                            val = max(0.0, min(1.0, val))
                        return val
                    except Exception:
                        pass

        return None
