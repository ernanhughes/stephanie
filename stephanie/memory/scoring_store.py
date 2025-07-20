# stephanie/memory/scoring_store.py

from dataclasses import dataclass
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
import pandas as pd

@dataclass
class GILDScoringExample:
    document_id: int
    dimension: str
    llm_score: Optional[float]
    hnet_ebt_score: Optional[float]
    huggingface_ebt_score: Optional[float]
    ollama_ebt_score: Optional[float]
    hnet_svm_score: Optional[float]
    huggingface_svm_score: Optional[float]
    ollama_svm_score: Optional[float]
    hnet_mrq_score: Optional[float]
    huggingface_mrq_score: Optional[float]
    ollama_mrq_score: Optional[float]
    created_at: Optional[str]

    @property
    def scores(self) -> dict:
        return {
            "hnet_ebt": self.hnet_ebt_score,
            "huggingface_ebt": self.huggingface_ebt_score,
            "ollama_ebt": self.ollama_ebt_score,
            "hnet_svm": self.hnet_svm_score,
            "huggingface_svm": self.huggingface_svm_score,
            "ollama_svm": self.ollama_svm_score,
            "hnet_mrq": self.hnet_mrq_score,
            "huggingface_mrq": self.huggingface_mrq_score,
            "ollama_mrq": self.ollama_mrq_score,
            "llm": self.llm_score,  # Include for filter logic
            "created_at": self.created_at
        }


class ScoringStore:
    def __init__(self, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "scoring"

    def load_gild_examples(self) -> List[GILDScoringExample]:
        try:
            query = text("""
                WITH ranked_scores AS (
                    SELECT
                        s.dimension,
                        s.score,
                        e.embedding_type,
                        e.target_id AS document_id,
                        e.evaluator_name,
                        e.created_at,
                        ROW_NUMBER() OVER (
                            PARTITION BY e.target_id, s.dimension, e.embedding_type, e.evaluator_name
                            ORDER BY e.created_at DESC
                        ) AS rank
                    FROM scores s
                    JOIN evaluations e ON e.id = s.evaluation_id
                    WHERE e.target_type = 'DOCUMENT'
                    AND (
                            e.evaluator_name = 'llm' OR
                            (e.evaluator_name IN ('ebt', 'svm', 'mrq') 
                            AND e.embedding_type IN ('hnet', 'huggingface', 'ollama'))
                    )
                )

                SELECT
                    document_id,
                    dimension,
                    MAX(created_at) AS created_at,  -- you could also use MIN or GROUP_CONCAT if you want all
                    MAX(CASE WHEN evaluator_name = 'llm' THEN score END) AS llm_score,
                    MAX(CASE WHEN evaluator_name = 'ebt' AND embedding_type = 'hnet' THEN score END) AS hnet_ebt_score,
                    MAX(CASE WHEN evaluator_name = 'ebt' AND embedding_type = 'huggingface' THEN score END) AS huggingface_ebt_score,
                    MAX(CASE WHEN evaluator_name = 'ebt' AND embedding_type = 'ollama' THEN score END) AS ollama_ebt_score,

                    MAX(CASE WHEN evaluator_name = 'svm' AND embedding_type = 'hnet' THEN score END) AS hnet_svm_score,
                    MAX(CASE WHEN evaluator_name = 'svm' AND embedding_type = 'huggingface' THEN score END) AS huggingface_svm_score,
                    MAX(CASE WHEN evaluator_name = 'svm' AND embedding_type = 'ollama' THEN score END) AS ollama_svm_score,

                    MAX(CASE WHEN evaluator_name = 'mrq' AND embedding_type = 'hnet' THEN score END) AS hnet_mrq_score,
                    MAX(CASE WHEN evaluator_name = 'mrq' AND embedding_type = 'huggingface' THEN score END) AS huggingface_mrq_score,
                    MAX(CASE WHEN evaluator_name = 'mrq' AND embedding_type = 'ollama' THEN score END) AS ollama_mrq_score

                FROM ranked_scores
                WHERE rank = 1
                GROUP BY document_id, dimension
                ORDER BY document_id, dimension;
            """)

            result = self.session.execute(query).fetchall()
            return [GILDScoringExample(**dict(row._mapping)) for row in result]

        except Exception as e:
            if self.logger:
                self.logger.log("GILDScoreLoadFailed", {"error": str(e)})
            else:
                print(f"[ScoringStore] Error: {e}")
            return []

    # In ScoringStore
    def get_scorer_stats(self):
        query = text("""
        SELECT 
            evaluator_name,
            embedding_type,
            COUNT(*) AS example_count,
            AVG(score) AS avg_score,
            STDDEV(score) AS std_score
        FROM evaluations
        JOIN scores ON evaluations.id = scores.evaluation_id
        WHERE evaluator_name != 'llm'
        GROUP BY evaluator_name, embedding_type
        """)
        return [dict(row._mapping) for row in self.session.execute(query).fetchall()]


    # In ScoringStore
    def generate_comparison_report(self, goal_id: int):
        query = text("""
        WITH scorer_comparison AS (
            SELECT 
                dimension,
                AVG(hnet_ebt_score - llm_score) AS hnet_bias,
                STDDEV(hnet_ebt_score - llm_score) AS hnet_var,
                AVG(huggingface_ebt_score - llm_score) AS hf_bias,
                STDDEV(huggingface_ebt_score - llm_score) AS hf_var
            FROM gild_scoring_examples
            WHERE goal_id = :goal_id
            GROUP BY dimension
        )
        SELECT dimension, 
            hnet_bias, hnet_var,
            hf_bias, hf_var
        FROM scorer_comparison
        """)
        
        results = self.session.execute(query, {"goal_id": goal_id}).fetchall()
        # Convert to pandas DataFrame for visualization
        return pd.DataFrame([dict(row._mapping) for row in results])