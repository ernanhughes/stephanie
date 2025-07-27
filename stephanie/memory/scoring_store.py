# stephanie/memory/scoring_store.py

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy.sql import text


@dataclass
class ScoringExample:
    document_id: int
    dimension: str
    llm_score: Optional[float]
    
    # EBT scores
    hnet_ebt_score: Optional[float]
    huggingface_ebt_score: Optional[float]
    ollama_ebt_score: Optional[float]
    
    # SVM scores
    hnet_svm_score: Optional[float]
    huggingface_svm_score: Optional[float]
    ollama_svm_score: Optional[float]
    
    # MRQ scores
    hnet_mrq_score: Optional[float]
    huggingface_mrq_score: Optional[float]
    ollama_mrq_score: Optional[float]
    
    # SICQL scores
    hnet_sicql_score: Optional[float]
    huggingface_sicql_score: Optional[float]
    ollama_sicql_score: Optional[float]
    
    # GILD-enhanced scores
    hnet_gild_score: Optional[float]
    huggingface_gild_score: Optional[float]
    ollama_gild_score: Optional[float]
    
    created_at: Optional[str]
    
    @property
    def scores(self) -> dict:
        return {
            "llm": self.llm_score,
            "hnet_ebt": self.hnet_ebt_score,
            "huggingface_ebt": self.huggingface_ebt_score,
            "ollama_ebt": self.ollama_ebt_score,
            "hnet_svm": self.hnet_svm_score,
            "huggingface_svm": self.huggingface_svm_score,
            "ollama_svm": self.ollama_svm_score,
            "hnet_mrq": self.hnet_mrq_score,
            "huggingface_mrq": self.huggingface_mrq_score,
            "ollama_mrq": self.ollama_mrq_score,
            "hnet_sicql": self.hnet_sicql_score,
            "huggingface_sicql": self.huggingface_sicql_score,
            "ollama_sicql": self.ollama_sicql_score,
            "hnet_gild": self.hnet_gild_score,
            "huggingface_gild": self.huggingface_gild_score,
            "ollama_gild": self.ollama_gild_score,
            "created_at": self.created_at
        }
    
    @property
    def filtered_scores(self) -> dict:
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

    def load_gild_examples(self) -> List[ScoringExample]:
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
                        e.evaluator_name IN ('llm', 'ebt', 'svm', 'mrq', 'sicql', 'gild')
                        AND e.embedding_type IN ('hnet', 'huggingface', 'ollama')
                    )
                )

                SELECT
                    document_id,
                    dimension,
                    MAX(CASE WHEN evaluator_name = 'llm' THEN score END) AS llm_score,
                    
                    -- EBT scores
                    MAX(CASE WHEN evaluator_name = 'ebt' AND embedding_type = 'hnet' THEN score END) AS hnet_ebt_score,
                    MAX(CASE WHEN evaluator_name = 'ebt' AND embedding_type = 'huggingface' THEN score END) AS huggingface_ebt_score,
                    MAX(CASE WHEN evaluator_name = 'ebt' AND embedding_type = 'ollama' THEN score END) AS ollama_ebt_score,
                    
                    -- SVM scores
                    MAX(CASE WHEN evaluator_name = 'svm' AND embedding_type = 'hnet' THEN score END) AS hnet_svm_score,
                    MAX(CASE WHEN evaluator_name = 'svm' AND embedding_type = 'huggingface' THEN score END) AS huggingface_svm_score,
                    MAX(CASE WHEN evaluator_name = 'svm' AND embedding_type = 'ollama' THEN score END) AS ollama_svm_score,
                    
                    -- MRQ scores
                    MAX(CASE WHEN evaluator_name = 'mrq' AND embedding_type = 'hnet' THEN score END) AS hnet_mrq_score,
                    MAX(CASE WHEN evaluator_name = 'mrq' AND embedding_type = 'huggingface' THEN score END) AS huggingface_mrq_score,
                    MAX(CASE WHEN evaluator_name = 'mrq' AND embedding_type = 'ollama' THEN score END) AS ollama_mrq_score,
                    
                    -- SICQL scores
                    MAX(CASE WHEN evaluator_name = 'sicql' AND embedding_type = 'hnet' THEN score END) AS hnet_sicql_score,
                    MAX(CASE WHEN evaluator_name = 'sicql' AND embedding_type = 'huggingface' THEN score END) AS huggingface_sicql_score,
                    MAX(CASE WHEN evaluator_name = 'sicql' AND embedding_type = 'ollama' THEN score END) AS ollama_sicql_score,
                    
                    -- GILD scores
                    MAX(CASE WHEN evaluator_name = 'gild' AND embedding_type = 'hnet' THEN score END) AS hnet_gild_score,
                    MAX(CASE WHEN evaluator_name = 'gild' AND embedding_type = 'huggingface' THEN score END) AS huggingface_gild_score,
                    MAX(CASE WHEN evaluator_name = 'gild' AND embedding_type = 'ollama' THEN score END) AS ollama_gild_score,

                    MAX(created_at) AS created_at
                FROM ranked_scores
                WHERE rank = 1
                GROUP BY document_id, dimension
                ORDER BY document_id, dimension;
            """)
            
            result = self.session.execute(query).fetchall()
            return [ScoringExample(**dict(row._mapping)) for row in result]
            
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
                AVG(hnet_ebt_score - llm_score) AS hnet_ebt_bias,
                STDDEV(hnet_ebt_score - llm_score) AS hnet_ebt_var,
                AVG(huggingface_ebt_score - llm_score) AS hf_ebt_bias,
                STDDEV(huggingface_ebt_score - llm_score) AS hf_ebt_var,
                AVG(ollama_ebt_score - llm_score) AS ollama_ebt_bias,
                STDDEV(ollama_ebt_score - llm_score) AS ollama_ebt_var,
                
                -- SVM comparison
                AVG(hnet_svm_score - llm_score) AS hnet_svm_bias,
                STDDEV(hnet_svm_score - llm_score) AS hnet_svm_var,
                AVG(huggingface_svm_score - llm_score) AS hf_svm_bias,
                STDDEV(huggingface_svm_score - llm_score) AS hf_svm_var,
                
                -- MRQ comparison
                AVG(hnet_mrq_score - llm_score) AS hnet_mrq_bias,
                STDDEV(hnet_mrq_score - llm_score) AS hnet_mrq_var,
                AVG(huggingface_mrq_score - llm_score) AS hf_mrq_bias,
                STDDEV(huggingface_mrq_score - llm_score) AS hf_mrq_var,
                
                -- SICQL comparison
                AVG(hnet_sicql_score - llm_score) AS hnet_sicql_bias,
                STDDEV(hnet_sicql_score - llm_score) AS hnet_sicql_var,
                AVG(huggingface_sicql_score - llm_score) AS hf_sicql_bias,
                STDDEV(huggingface_sicql_score - llm_score) AS hf_sicql_var,
                
                -- GILD comparison
                AVG(hnet_gild_score - llm_score) AS hnet_gild_bias,
                STDDEV(hnet_gild_score - llm_score) AS hnet_gild_var,
                AVG(huggingface_gild_score - llm_score) AS hf_gild_bias,
                STDDEV(huggingface_gild_score - llm_score) AS hf_gild_var
            FROM gild_scoring_examples
            WHERE goal_id = :goal_id
            GROUP BY dimension
        )
        SELECT 
            dimension,
            hnet_ebt_bias, hnet_ebt_var,
            hf_ebt_bias, hf_ebt_var,
            ollama_ebt_bias, ollama_ebt_var,
            hnet_svm_bias, hnet_svm_var,
            hf_svm_bias, hf_svm_var,
            hnet_mrq_bias, hnet_mrq_var,
            hf_mrq_bias, hf_mrq_var,
            hnet_sicql_bias, hnet_sicql_var,
            hf_sicql_bias, hf_sicql_var,
            hnet_gild_bias, hnet_gild_var,
            hf_gild_bias, hf_gild_var
        FROM scorer_comparison
        """)
        
        results = self.session.execute(query, {"goal_id": goal_id}).fetchall()
        return pd.DataFrame([dict(row._mapping) for row in results])

    def get_temporal_analysis(self, dimension: str):
        query = text("""
        SELECT
            DATE_TRUNC('week', created_at) AS week,
            AVG(hnet_ebt_score - llm_score) AS hnet_ebt_bias,
            AVG(hnet_svm_score - llm_score) AS hnet_svm_bias,
            AVG(hnet_mrq_score - llm_score) AS hnet_mrq_bias,
            AVG(hnet_sicql_score - llm_score) AS hnet_sicql_bias,
            AVG(hnet_gild_score - llm_score) AS hnet_gild_bias
        FROM gild_scoring_examples
        WHERE dimension = :dimension
        GROUP BY DATE_TRUNC('week', created_at)
        ORDER BY week
        """)
        
        results = self.session.execute(query, {"dimension": dimension}).fetchall()
        return pd.DataFrame([dict(row._mapping) for row in results])


    def get_scorer_correlation(self, scorer: str, embedding: str):
        query = text(f"""
        SELECT 
            {embedding}_{scorer}_score,
            llm_score
        FROM gild_scoring_examples
        WHERE {embedding}_{scorer}_score IS NOT NULL
        """)
        
        results = self.session.execute(query).fetchall()
        df = pd.DataFrame([dict(row._mapping) for row in results])
        return df.corr()


    def get_score_gaps(self, scorer: str, embedding: str):
        query = text(f"""
        SELECT
            dimension,
            AVG({embedding}_{scorer}_score - llm_score) AS avg_gap,
            STDDEV({embedding}_{scorer}_score - llm_score) AS std_gap
        FROM gild_scoring_examples
        WHERE {embedding}_{scorer}_score IS NOT NULL
        GROUP BY dimension
        """)
        
        return pd.DataFrame([dict(row._mapping) for row in self.session.execute(query).fetchall()])


    def get_gild_effectiveness(self):
        query = text("""
        SELECT
            dimension,
            AVG(hnet_gild_score - hnet_scorer_score) AS hnet_gain,
            AVG(huggingface_gild_score - huggingface_scorer_score) AS hf_gain,
            AVG(ollama_gild_score - ollama_scorer_score) AS ollama_gain
        FROM gild_scoring_examples
        GROUP BY dimension
        """)
        
        return pd.DataFrame([dict(row._mapping) for row in self.session.execute(query).fetchall()])
    

    def plot_scorer_comparison(self, dimension: str):
        import matplotlib.pyplot as plt
        report = self.generate_comparison_report(goal_id=123)
        data = report[report.dimension == dimension]
        
        plt.figure(figsize=(12, 6))
        plt.bar(data.columns, data.iloc[0])
        plt.title(f"Scorer Comparison: {dimension}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{dimension}_comparison.png")

    def get_gild_training_feedback(self):
        """Get feedback for GILD training"""
        query = text("""
        SELECT
            dimension,
            COUNT(*) AS sample_count,
            AVG(hnet_gild_score - llm_score) AS avg_deviation,
            AVG(hnet_gild_score - hnet_mrq_score) AS policy_improvement
        FROM gild_scoring_examples
        GROUP BY dimension
        """)
        return pd.DataFrame([dict(row._mapping) for row in self.session.execute(query).fetchall()])