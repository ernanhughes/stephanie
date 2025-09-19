# stephanie/memory/evaluation_attribute_store.py
from __future__ import annotations

from typing import Optional

from sqlalchemy import func

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.evaluation_attribute import EvaluationAttributeORM


class EvaluationAttributeStore(BaseSQLAlchemyStore):
    orm_model = EvaluationAttributeORM
    default_order_by = "created_at"

    def __init__(self, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "evaluation_attributes"
        self.table_name = "evaluation_attributes"

    def name(self) -> str:
        return self.name

    # -------------------
    # Insert
    # -------------------
    def insert(self, attribute: EvaluationAttributeORM) -> int:
        """Insert a single evaluation attribute with safe transaction handling."""
        def op(s):
            s.add(attribute)
            s.flush()
            return attribute.id

        attr_id = self._run(op)

        if self.logger:
            self.logger.log("AttributeStored", attribute.to_dict())
        return attr_id

    def bulk_insert(self, attributes: list[EvaluationAttributeORM]) -> list[int]:
        """Insert multiple attributes in a single transaction."""
        def op(s):
            ids = []
            for attr in attributes:
                s.add(attr)
                s.flush()
                ids.append(attr.id)
            return ids

        ids = self._run(op)

        if self.logger and ids:
            self.logger.log("BulkAttributesStored", {
                "count": len(ids),
                "evaluation_ids": list({a.evaluation_id for a in attributes}),
            })
        return ids

    # -------------------
    # Retrieval
    # -------------------
    def get_by_evaluation_id(self, evaluation_id: int) -> list[EvaluationAttributeORM]:
        return self._run(
            lambda s: s.query(EvaluationAttributeORM)
                      .filter_by(evaluation_id=evaluation_id)
                      .all()
        )

    def get_by_dimension(self, evaluation_id: int, dimension: str) -> list[EvaluationAttributeORM]:
        return self._run(
            lambda s: s.query(EvaluationAttributeORM)
                      .filter(
                          EvaluationAttributeORM.evaluation_id == evaluation_id,
                          EvaluationAttributeORM.dimension == dimension,
                      )
                      .all()
        )

    def get_by_source(self, evaluation_id: int, dimension: str, source: str) -> Optional[EvaluationAttributeORM]:
        return self._run(
            lambda s: s.query(EvaluationAttributeORM)
                      .filter(
                          EvaluationAttributeORM.evaluation_id == evaluation_id,
                          EvaluationAttributeORM.dimension == dimension,
                          EvaluationAttributeORM.source == source,
                      )
                      .first()
        )

    def get_by_source_and_dimension(self, source: str, dimension: str, limit: int = 100) -> list[EvaluationAttributeORM]:
        return self._run(
            lambda s: s.query(EvaluationAttributeORM)
                      .filter(
                          EvaluationAttributeORM.source == source,
                          EvaluationAttributeORM.dimension == dimension,
                      )
                      .order_by(EvaluationAttributeORM.created_at.desc())
                      .limit(limit)
                      .all()
        )

    def get_high_uncertainty_samples(self, threshold: float = 0.3, limit: int = 100) -> list[EvaluationAttributeORM]:
        return self._run(
            lambda s: s.query(EvaluationAttributeORM)
                      .filter(EvaluationAttributeORM.uncertainty >= threshold)
                      .order_by(EvaluationAttributeORM.uncertainty.desc())
                      .limit(limit)
                      .all()
        )

    # -------------------
    # Delete / Update
    # -------------------
    def delete_by_evaluation(self, evaluation_id: int) -> int:
        def op(s):
            return (
                s.query(EvaluationAttributeORM)
                .filter_by(evaluation_id=evaluation_id)
                .delete()
            )
        deleted = self._run(op)

        if self.logger:
            self.logger.log("AttributesDeleted", {"evaluation_id": evaluation_id, "count": deleted})
        return deleted

    def update_attributes(self, attributes: list[dict[str, any]]) -> None:
        def op(s):
            for attr_data in attributes:
                attr_id = attr_data.pop("id", None)
                if attr_id:
                    s.query(EvaluationAttributeORM).filter_by(id=attr_id).update(attr_data)

        self._run(op)

        if self.logger:
            self.logger.log("AttributesUpdated", {"count": len(attributes)})

    # -------------------
    # Metrics
    # -------------------
    def get_policy_logits(self, evaluation_id: int, dimension: str) -> Optional[list[float]]:
        attr = self._run(
            lambda s: s.query(EvaluationAttributeORM)
                      .filter(
                          EvaluationAttributeORM.evaluation_id == evaluation_id,
                          EvaluationAttributeORM.dimension == dimension,
                      )
                      .first()
        )
        return attr.policy_logits if attr and attr.policy_logits else None

    def get_dimension_stats(self, dimension: str) -> dict[str, float]:
        result = self._run(
            lambda s: s.query(
                          func.avg(EvaluationAttributeORM.uncertainty).label("avg_uncertainty"),
                          func.avg(EvaluationAttributeORM.entropy).label("avg_entropy"),
                          func.count().label("total"),
                      )
                      .filter(EvaluationAttributeORM.dimension == dimension)
                      .first()
        )
        return {
            "dimension": dimension,
            "avg_uncertainty": float(result.avg_uncertainty) if result.avg_uncertainty else 0.0,
            "avg_entropy": float(result.avg_entropy) if result.avg_entropy else 0.0,
            "total_samples": result.total or 0,
        }
