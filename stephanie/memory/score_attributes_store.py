# stephanie/memory/score_attribute_store.py
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.score import ScoreORM
from stephanie.models.score_attribute import ScoreAttributeORM


class ScoreAttributeStore(BaseSQLAlchemyStore):
    orm_model = ScoreAttributeORM
    default_order_by = ScoreAttributeORM.created_at.asc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "score_attributes"
        self.table_name = "score_attributes"

    def add_attribute(self, attribute: ScoreAttributeORM) -> ScoreAttributeORM:
        """Add a single score attribute to the database"""
        def op(s):
            
            s.add(attribute)
            s.flush()
            s.refresh(attribute)
            return attribute
        return self._run(op)

    def add_attributes_bulk(self, attributes: List[ScoreAttributeORM]):
        """Add multiple score attributes in a single transaction"""
        def op(s):
            
            s.add_all(attributes)
        return self._run(op)

    def get_attributes_for_score(self, score_id: int) -> List[ScoreAttributeORM]:
        """Get all attributes for a specific score"""
        def op(s):
            return (
                s.query(ScoreAttributeORM)
                .filter_by(score_id=score_id)
                .order_by(ScoreAttributeORM.key.asc())
                .all()
            )
        return self._run(op)

    def get_attributes_for_scores(self, score_ids: List[int]) -> List[ScoreAttributeORM]:
        """Get all attributes for multiple scores"""
        if not score_ids:
            return []
        def op(s):
            return (
                s.query(ScoreAttributeORM)
                .filter(ScoreAttributeORM.score_id.in_(score_ids))
                .order_by(ScoreAttributeORM.score_id, ScoreAttributeORM.key)
                .all()
            )
        return self._run(op)

    def get_attributes_by_key(
        self, key: str, score_ids: Optional[List[int]] = None, limit: int = 100
    ) -> List[ScoreAttributeORM]:
        """Get attributes by key, optionally filtered by score IDs"""
        def op(s):
            q = s.query(ScoreAttributeORM).filter_by(key=key)
            if score_ids:
                q = q.filter(ScoreAttributeORM.score_id.in_(score_ids))
            return q.order_by(ScoreAttributeORM.created_at.desc()).limit(limit).all()
        return self._run(op)

    def get_attributes_by_keys(
        self, keys: List[str], score_ids: Optional[List[int]] = None
    ) -> List[ScoreAttributeORM]:
        """Get attributes by multiple keys, optionally filtered by score IDs"""
        def op(s):
            q = s.query(ScoreAttributeORM).filter(
                ScoreAttributeORM.key.in_(keys)
            )
            if score_ids:
                q = q.filter(ScoreAttributeORM.score_id.in_(score_ids))
            return q.order_by(ScoreAttributeORM.score_id, ScoreAttributeORM.key).all()
        return self._run(op)

    def get_attribute_matrix(
        self, score_ids: List[int], keys: List[str]
    ) -> Dict[int, Dict[str, ScoreAttributeORM]]:
        """Get a matrix of attributes for multiple scores and keys"""
        if not score_ids or not keys:
            return {}
        attributes = self.get_attributes_by_keys(keys, score_ids)
        matrix: Dict[int, Dict[str, ScoreAttributeORM]] = {}
        for attr in attributes:
            matrix.setdefault(attr.score_id, {})[attr.key] = attr
        return matrix

    def get_attribute_values(
        self, score_ids: List[int], keys: List[str]
    ) -> Dict[int, Dict[str, any]]:
        """Get a matrix of attribute values for multiple scores and keys"""
        matrix = self.get_attribute_matrix(score_ids, keys)
        result: Dict[int, Dict[str, any]] = {}
        for score_id, attrs in matrix.items():
            result[score_id] = {}
            for key in keys:
                if key in attrs:
                    attr = attrs[key]
                    if attr.data_type == "float":
                        result[score_id][key] = float(attr.value)
                    elif attr.data_type == "json":
                        try:
                            result[score_id][key] = json.loads(attr.value)
                        except Exception:
                            result[score_id][key] = attr.value
                    else:
                        result[score_id][key] = attr.value
                else:
                    result[score_id][key] = None
        return result

    def get_score_attribute_tensor(
        self, score_ids: List[int], dimensions: List[str], scorers: List[str], metrics: List[str]
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, any]]]]:
        """
        Get a 4D tensor representation of score attributes:
        [scores × dimensions × scorers × metrics]
        """
        def op(s):
            scores = (
                s.query(ScoreORM)
                .filter(ScoreORM.id.in_(score_ids))
                .all()
            )
            score_by_dim_scorer = {}
            for score in scores:
                key = (score.dimension, score.source)
                score_by_dim_scorer.setdefault(key, []).append(score.id)

            all_score_ids = [s.id for s in scores]
            attributes = self.get_attributes_by_keys(metrics, all_score_ids)

            tensor: Dict[str, Dict[str, Dict[str, Dict[str, any]]]] = {}
            for dim in dimensions:
                tensor[dim] = {}
                for scorer in scorers:
                    key = (dim, scorer)
                    if key in score_by_dim_scorer:
                        tensor[dim][scorer] = {}
                        ids = score_by_dim_scorer[key]
                        for attr in attributes:
                            if attr.score_id in ids and attr.key in metrics:
                                tensor[dim][scorer].setdefault(attr.key, [])
                                if attr.data_type == "float":
                                    tensor[dim][scorer][attr.key].append(float(attr.value))
                                elif attr.data_type == "json":
                                    try:
                                        tensor[dim][scorer][attr.key].append(json.loads(attr.value))
                                    except Exception:
                                        tensor[dim][scorer][attr.key].append(attr.value)
                                else:
                                    tensor[dim][scorer][attr.key].append(attr.value)
            return tensor
        return self._run(op)

    def delete_attributes_for_score(self, score_id: int):
        """Delete all attributes for a specific score"""
        def op(s):
            s.query(ScoreAttributeORM).filter_by(score_id=score_id).delete()
        return self._run(op)

    def delete_attributes_for_scores(self, score_ids: List[int]):
        """Delete all attributes for multiple scores"""
        if not score_ids:
            return
        def op(s):
            s.query(ScoreAttributeORM).filter(
                ScoreAttributeORM.score_id.in_(score_ids)
            ).delete()
        return self._run(op)

    def get_all(self, limit: Optional[int] = None) -> List[ScoreAttributeORM]:
        """Get all attributes (with optional limit)"""
        def op(s):
            q = s.query(ScoreAttributeORM).order_by(ScoreAttributeORM.id.desc())
            if limit:
                q = q.limit(limit)
            return q.all()
        return self._run(op)

    def get_by_id(self, attribute_id: int) -> Optional[ScoreAttributeORM]:
        """Get a specific attribute by ID"""
        def op(s):
            return s.query(ScoreAttributeORM).filter_by(id=attribute_id).first()
        return self._run(op)

    def get_attributes_by_score_ids(self, score_ids: list[int]) -> list[ScoreAttributeORM]:
        """Get attributes for a list of score IDs"""
        if not score_ids:
            return []
        def op(s):
            return (
                s.query(ScoreAttributeORM)
                .filter(ScoreAttributeORM.score_id.in_(score_ids))
                .all()
            )
        return self._run(op)

    def get_attribute_stats(self, key: str, score_ids: Optional[List[int]] = None) -> Dict:
        """Get statistical summary for a specific attribute key"""
        def op(s):
            q = s.query(
                func.avg(ScoreAttributeORM.value.cast(func.FLOAT)).label("mean"),
                func.stddev_samp(ScoreAttributeORM.value.cast(func.FLOAT)).label("stddev"),
                func.min(ScoreAttributeORM.value.cast(func.FLOAT)).label("min"),
                func.max(ScoreAttributeORM.value.cast(func.FLOAT)).label("max"),
                func.count(ScoreAttributeORM.id).label("count")
            ).filter(
                ScoreAttributeORM.key == key,
                ScoreAttributeORM.data_type == "float"
            )
            if score_ids:
                q = q.filter(ScoreAttributeORM.score_id.in_(score_ids))
            return q.first()
        result = self._run(op)
        return {
            "key": key,
            "mean": float(result.mean) if result and result.mean is not None else None,
            "stddev": float(result.stddev) if result and result.stddev is not None else None,
            "min": float(result.min) if result and result.min is not None else None,
            "max": float(result.max) if result and result.max is not None else None,
            "count": result.count if result else 0
        }

    def get_metric_correlations(
        self, metrics: List[str], score_ids: Optional[List[int]] = None
    ) -> Dict[Tuple[str, str], float]:
        """Calculate correlation coefficients between different metrics"""
        if len(metrics) < 2:
            return {}
        attributes = self.get_attributes_by_keys(metrics, score_ids)
        score_metrics: Dict[int, Dict[str, float]] = {}
        for attr in attributes:
            if attr.data_type == "float":
                try:
                    score_metrics.setdefault(attr.score_id, {})[attr.key] = float(attr.value)
                except (TypeError, ValueError):
                    pass
        correlations: Dict[Tuple[str, str], float] = {}
        for i in range(len(metrics)):
            for j in range(i + 1, len(metrics)):
                m1, m2 = metrics[i], metrics[j]
                vals1, vals2 = [], []
                for metrics_dict in score_metrics.values():
                    if m1 in metrics_dict and m2 in metrics_dict:
                        vals1.append(metrics_dict[m1])
                        vals2.append(metrics_dict[m2])
                if len(vals1) > 1:
                    try:
                        from scipy import stats
                        corr, _ = stats.pearsonr(vals1, vals2)
                        correlations[(m1, m2)] = corr
                    except ImportError:
                        mean1 = sum(vals1) / len(vals1)
                        mean2 = sum(vals2) / len(vals2)
                        num = sum((x - mean1) * (y - mean2) for x, y in zip(vals1, vals2))
                        d1 = sum((x - mean1) ** 2 for x in vals1) ** 0.5
                        d2 = sum((y - mean2) ** 2 for y in vals2) ** 0.5
                        correlations[(m1, m2)] = num / (d1 * d2) if d1 > 0 and d2 > 0 else 0.0
                else:
                    correlations[(m1, m2)] = 0.0
        return correlations

    def __repr__(self):
        return f"<ScoreAttributeStore(session={self.session})>"
