# stephanie/memory/evaluation_attribute_store.py

from typing import Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from stephanie.models.evaluation_attribute import EvaluationAttributeORM


class EvaluationAttributeStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "evaluation_attributes"
        self.table_name = "evaluation_attributes"

    def insert(self, attribute: EvaluationAttributeORM) -> int:
        """Insert a single evaluation attribute with enhanced error handling"""
        try:
            self.session.add(attribute)
            self.session.flush()  # Get ID before commit
            self.session.commit()
            
            # Log structured metrics
            if self.logger:
                self.logger.log("AttributeStored", {
                    "evaluation_id": attribute.evaluation_id,
                    "dimension": attribute.dimension,
                    "source": attribute.source,
                    "q_value": attribute.q_value,
                    "v_value": attribute.v_value,
                    "uncertainty": attribute.uncertainty,
                    "policy_logits": attribute.policy_logits,
                    "entropy": attribute.entropy
                })
            
            return attribute.id
            
        except SQLAlchemyError as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("AttributeInsertFailed", {
                    "error": str(e),
                    "attribute": attribute.to_dict()
                })
            raise

    def bulk_insert(self, attributes: list[EvaluationAttributeORM]) -> list[int]:
        """Insert multiple attributes in a single transaction"""
        try:
            ids = []
            for attr in attributes:
                self.session.add(attr)
                self.session.flush()
                ids.append(attr.id)
            
            self.session.commit()
            
            if self.logger and ids:
                self.logger.log("BulkAttributesStored", {
                    "count": len(ids),
                    "evaluation_ids": list(set(a.evaluation_id for a in attributes))
                })
            
            return ids
            
        except SQLAlchemyError as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("BulkInsertFailed", {
                    "error": str(e),
                    "attribute_count": len(attributes)
                })
            raise

    def get_by_evaluation_id(self, evaluation_id: int) -> list[EvaluationAttributeORM]:
        """Get all attributes for an evaluation"""
        return (
            self.session.query(EvaluationAttributeORM)
            .filter_by(evaluation_id=evaluation_id)
            .all()
        )

    def get_by_dimension(self, evaluation_id: int, dimension: str) -> list[EvaluationAttributeORM]:
        """Get attributes for specific dimension in an evaluation"""
        return (
            self.session.query(EvaluationAttributeORM)
            .filter(
                EvaluationAttributeORM.evaluation_id == evaluation_id,
                EvaluationAttributeORM.dimension == dimension
            )
            .all()
        )

    def get_by_source(self, evaluation_id: int, dimension: str, source: str) -> Optional[EvaluationAttributeORM]:
        """Get attributes by source and dimension"""
        return (
            self.session.query(EvaluationAttributeORM)
            .filter(
                EvaluationAttributeORM.evaluation_id == evaluation_id,
                EvaluationAttributeORM.dimension == dimension,
                EvaluationAttributeORM.source == source
            )
            .first()
        )

    def get_by_source_and_dimension(self, source: str, dimension: str, limit: int = 100) -> list[EvaluationAttributeORM]:
        """Get recent attributes by source and dimension"""
        return (
            self.session.query(EvaluationAttributeORM)
            .filter(
                EvaluationAttributeORM.source == source,
                EvaluationAttributeORM.dimension == dimension
            )
            .order_by(EvaluationAttributeORM.created_at.desc())
            .limit(limit)
            .all()
        )

    def get_high_uncertainty_samples(self, threshold: float = 0.3, limit: int = 100) -> list[EvaluationAttributeORM]:
        """Get attributes with high epistemic uncertainty"""
        return (
            self.session.query(EvaluationAttributeORM)
            .filter(EvaluationAttributeORM.uncertainty >= threshold)
            .order_by(EvaluationAttributeORM.uncertainty.desc())
            .limit(limit)
            .all()
        )

    def delete_by_evaluation(self, evaluation_id: int) -> int:
        """Delete attributes for an evaluation"""
        try:
            deleted = (
                self.session.query(EvaluationAttributeORM)
                .filter_by(evaluation_id=evaluation_id)
                .delete()
            )
            self.session.commit()
            return deleted
            
        except SQLAlchemyError as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("AttributeDeleteFailed", {
                    "error": str(e),
                    "evaluation_id": evaluation_id
                })
            raise

    def update_attributes(self, attributes: list[dict[str, any]]) -> None:
        """Update multiple attributes by ID"""
        try:
            for attr_data in attributes:
                attr_id = attr_data.pop('id', None)
                if attr_id:
                    self.session.query(EvaluationAttributeORM).filter_by(id=attr_id).update(attr_data)
            
            self.session.commit()
            
        except SQLAlchemyError as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("AttributeUpdateFailed", {
                    "error": str(e),
                    "attribute_count": len(attributes)
                })
            raise

    def get_policy_logits(self, evaluation_id: int, dimension: str) -> Optional[list[float]]:
        """Get policy logits for a specific evaluation and dimension"""
        attr = self.session.query(EvaluationAttributeORM).filter(
            EvaluationAttributeORM.evaluation_id == evaluation_id,
            EvaluationAttributeORM.dimension == dimension
        ).first()
        
        return attr.policy_logits if attr and attr.policy_logits else None

    def get_dimension_stats(self, dimension: str) -> dict[str, float]:
        """Get statistical metrics for a dimension"""
        from sqlalchemy import func
        
        results = (
            self.session.query(
                func.avg(EvaluationAttributeORM.uncertainty).label("avg_uncertainty"),
                func.avg(EvaluationAttributeORM.entropy).label("avg_entropy"),
                func.count().label("total")
            )
            .filter(EvaluationAttributeORM.dimension == dimension)
            .first()
        )
        
        return {
            "dimension": dimension,
            "avg_uncertainty": float(results.avg_uncertainty) if results.avg_uncertainty else 0.0,
            "avg_entropy": float(results.avg_entropy) if results.avg_entropy else 0.0,
            "total_samples": results.total or 0
        }