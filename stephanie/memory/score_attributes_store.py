# stephanie/memory/score_attribute_store.py
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from stephanie.models.score import ScoreORM
from stephanie.models.score_attribute import ScoreAttributeORM

class ScoreAttributeStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "score_attributes"
        self.table_name = "score_attributes"

    def add_attribute(self, attribute: ScoreAttributeORM) -> ScoreAttributeORM:
        """Add a single score attribute to the database"""
        self.session.add(attribute)
        self.session.commit()
        self.session.refresh(attribute)
        return attribute

    def add_attributes_bulk(self, attributes: List[ScoreAttributeORM]):
        """Add multiple score attributes in a single transaction"""
        self.session.add_all(attributes)
        self.session.commit()

    def get_attributes_for_score(self, score_id: int) -> List[ScoreAttributeORM]:
        """Get all attributes for a specific score"""
        return (
            self.session.query(ScoreAttributeORM)
            .filter_by(score_id=score_id)
            .order_by(ScoreAttributeORM.key.asc())
            .all()
        )

    def get_attributes_for_scores(self, score_ids: List[int]) -> List[ScoreAttributeORM]:
        """Get all attributes for multiple scores"""
        if not score_ids:
            return []
        return (
            self.session.query(ScoreAttributeORM)
            .filter(ScoreAttributeORM.score_id.in_(score_ids))
            .order_by(ScoreAttributeORM.score_id, ScoreAttributeORM.key)
            .all()
        )

    def get_attributes_by_key(
        self, key: str, score_ids: Optional[List[int]] = None, limit: int = 100
    ) -> List[ScoreAttributeORM]:
        """Get attributes by key, optionally filtered by score IDs"""
        query = self.session.query(ScoreAttributeORM).filter_by(key=key)
        
        if score_ids:
            query = query.filter(ScoreAttributeORM.score_id.in_(score_ids))
            
        return query.order_by(ScoreAttributeORM.created_at.desc()).limit(limit).all()

    def get_attributes_by_keys(
        self, keys: List[str], score_ids: Optional[List[int]] = None
    ) -> List[ScoreAttributeORM]:
        """Get attributes by multiple keys, optionally filtered by score IDs"""
        query = self.session.query(ScoreAttributeORM).filter(ScoreAttributeORM.key.in_(keys))
        
        if score_ids:
            query = query.filter(ScoreAttributeORM.score_id.in_(score_ids))
            
        return query.order_by(ScoreAttributeORM.score_id, ScoreAttributeORM.key).all()

    def get_attribute_matrix(
        self, score_ids: List[int], keys: List[str]
    ) -> Dict[int, Dict[str, ScoreAttributeORM]]:
        """
        Get a matrix of attributes for multiple scores and keys
        Returns: {score_id: {key: attribute}}
        """
        if not score_ids or not keys:
            return {}
            
        attributes = self.get_attributes_by_keys(keys, score_ids)
        
        # Organize by score_id and key
        matrix = {}
        for attr in attributes:
            if attr.score_id not in matrix:
                matrix[attr.score_id] = {}
            matrix[attr.score_id][attr.key] = attr
            
        return matrix

    def get_attribute_values(
        self, score_ids: List[int], keys: List[str]
    ) -> Dict[int, Dict[str, any]]:
        """
        Get a matrix of attribute values for multiple scores and keys
        Returns: {score_id: {key: value}}
        """
        matrix = self.get_attribute_matrix(score_ids, keys)
        result = {}
        
        for score_id, attrs in matrix.items():
            result[score_id] = {}
            for key in keys:
                if key in attrs:
                    # Convert to appropriate type based on data_type
                    attr = attrs[key]
                    if attr.data_type == "float":
                        result[score_id][key] = float(attr.value)
                    elif attr.data_type == "json":
                        try:
                            import json
                            result[score_id][key] = json.loads(attr.value)
                        except:
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
        
        Returns a nested dictionary structure that can be converted to a tensor
        """
        # First get all relevant scores with their metadata
        scores = (
            self.session.query(ScoreORM)
            .filter(ScoreORM.id.in_(score_ids))
            .all()
        )
        
        # Organize scores by dimension and source
        score_by_dim_scorer = {}
        for score in scores:
            dim_scorer_key = (score.dimension, score.source)
            if dim_scorer_key not in score_by_dim_scorer:
                score_by_dim_scorer[dim_scorer_key] = []
            score_by_dim_scorer[dim_scorer_key].append(score.id)
        
        # Get all relevant attributes
        all_score_ids = [score.id for score in scores]
        attributes = self.get_attributes_by_keys(metrics, all_score_ids)
        
        # Build the tensor structure
        tensor = {}
        for dim in dimensions:
            tensor[dim] = {}
            for scorer in scorers:
                dim_scorer_key = (dim, scorer)
                if dim_scorer_key in score_by_dim_scorer:
                    tensor[dim][scorer] = {}
                    score_ids_for_dim_scorer = score_by_dim_scorer[dim_scorer_key]
                    
                    # Get attributes for these scores
                    for attr in attributes:
                        if attr.score_id in score_ids_for_dim_scorer and attr.key in metrics:
                            if attr.key not in tensor[dim][scorer]:
                                tensor[dim][scorer][attr.key] = []
                            
                            # Convert to appropriate type
                            if attr.data_type == "float":
                                tensor[dim][scorer][attr.key].append(float(attr.value))
                            elif attr.data_type == "json":
                                try:
                                    import json
                                    tensor[dim][scorer][attr.key].append(json.loads(attr.value))
                                except:
                                    tensor[dim][scorer][attr.key].append(attr.value)
                            else:
                                tensor[dim][scorer][attr.key].append(attr.value)
        
        return tensor

    def delete_attributes_for_score(self, score_id: int):
        """Delete all attributes for a specific score"""
        self.session.query(ScoreAttributeORM).filter_by(score_id=score_id).delete()
        self.session.commit()

    def delete_attributes_for_scores(self, score_ids: List[int]):
        """Delete all attributes for multiple scores"""
        if not score_ids:
            return
        self.session.query(ScoreAttributeORM).filter(
            ScoreAttributeORM.score_id.in_(score_ids)
        ).delete()
        self.session.commit()

    def get_all(self, limit: Optional[int] = None) -> List[ScoreAttributeORM]:
        """Get all attributes (with optional limit)"""
        query = self.session.query(ScoreAttributeORM).order_by(
            ScoreAttributeORM.id.desc()
        )
        if limit:
            query = query.limit(limit)
        return query.all()

    def get_by_id(self, attribute_id: int) -> Optional[ScoreAttributeORM]:
        """Get a specific attribute by ID"""
        return (
            self.session.query(ScoreAttributeORM)
            .filter_by(id=attribute_id)
            .first()
        )

    def get_attributes_by_score_ids(self, score_ids: list[int]) -> list[ScoreAttributeORM]:
        """Get attributes for a list of score IDs"""
        if not score_ids:
            return []
        try:
            return (
                self.session.query(ScoreAttributeORM)
                .filter(ScoreAttributeORM.score_id.in_(score_ids))
                .all()
            )
        except Exception as e:
            if self.logger:
                self.logger.log(
                    "GetByScoreError",
                    {
                        "method": "get_attributes_by_score_ids",
                        "error": str(e),
                        "score_ids": score_ids,
                    },
                )
            return []

    def get_attribute_stats(self, key: str, score_ids: Optional[List[int]] = None) -> Dict:
        """Get statistical summary for a specific attribute key"""
        query = self.session.query(
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
            query = query.filter(ScoreAttributeORM.score_id.in_(score_ids))
            
        result = query.first()
        
        return {
            "key": key,
            "mean": float(result.mean) if result.mean is not None else None,
            "stddev": float(result.stddev) if result.stddev is not None else None,
            "min": float(result.min) if result.min is not None else None,
            "max": float(result.max) if result.max is not None else None,
            "count": result.count
        }

    def get_metric_correlations(
        self, metrics: List[str], score_ids: Optional[List[int]] = None
    ) -> Dict[Tuple[str, str], float]:
        """
        Calculate correlation coefficients between different metrics
        Returns: {("metric1", "metric2"): correlation}
        """
        if len(metrics) < 2:
            return {}
            
        # Get all relevant attributes
        attributes = self.get_attributes_by_keys(metrics, score_ids)
        
        # Organize by score_id and metric
        score_metrics = {}
        for attr in attributes:
            if attr.score_id not in score_metrics:
                score_metrics[attr.score_id] = {}
            if attr.data_type == "float":
                try:
                    score_metrics[attr.score_id][attr.key] = float(attr.value)
                except (TypeError, ValueError):
                    pass
                    
        # Calculate correlations
        correlations = {}
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                metric1, metric2 = metrics[i], metrics[j]
                
                # Collect values for both metrics
                values1, values2 = [], []
                for metrics_dict in score_metrics.values():
                    if metric1 in metrics_dict and metric2 in metrics_dict:
                        values1.append(metrics_dict[metric1])
                        values2.append(metrics_dict[metric2])
                
                # Calculate correlation if we have enough data
                if len(values1) > 1:
                    try:
                        from scipy import stats
                        corr, _ = stats.pearsonr(values1, values2)
                        correlations[(metric1, metric2)] = corr
                    except ImportError:
                        # Fallback to simple correlation calculation
                        mean1 = sum(values1) / len(values1)
                        mean2 = sum(values2) / len(values2)
                        
                        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(values1, values2))
                        denom1 = sum((x - mean1) ** 2 for x in values1) ** 0.5
                        denom2 = sum((y - mean2) ** 2 for y in values2) ** 0.5
                        
                        if denom1 > 0 and denom2 > 0:
                            correlations[(metric1, metric2)] = numerator / (denom1 * denom2)
                        else:
                            correlations[(metric1, metric2)] = 0.0
                else:
                    correlations[(metric1, metric2)] = 0.0
                    
        return correlations

    def __repr__(self):
        return f"<ScoreAttributeStore(session={self.session})>"