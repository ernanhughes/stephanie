# stephanie/memory/scoring_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import matplotlib
import pandas as pd
from sqlalchemy.sql import text

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.models.score import ScoreORM

if matplotlib.get_backend().lower() != "agg":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ScoringExample:
    document_id: int
    dimension: str
    llm_score: Optional[float]
    # ... (all the score fields unchanged) ...
    created_at: Optional[str]

    @property
    def scores(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

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
            "llm": self.llm_score,
            "created_at": self.created_at,
        }


class ScoringStore(BaseSQLAlchemyStore):
    orm_model = ScoreORM
    default_order_by = ScoreORM.id.desc()

    def __init__(self, session_or_maker, logger=None):
        super().__init__(session_or_maker, logger)
        self.name = "scoring"

    def load_gild_examples(self) -> List[ScoringExample]:
        def op(s):
            query = text(""" ... """)  # same SQL you already have
            result = self._scope().execute(query).fetchall()
            return [ScoringExample(**dict(row._mapping)) for row in result]
        return self._run(op)

    def get_scorer_stats(self):
        def op(s):
            query = text(""" ... """)
            return [dict(row._mapping) for row in self._scope().execute(query).fetchall()]
        return self._run(op)

    def generate_comparison_report(self, goal_id: int):
        def op(s):
            query = text(""" ... """)
            results = s.execute(query, {"goal_id": goal_id}).fetchall()
            return pd.DataFrame([dict(row._mapping) for row in results])
        return self._run(op)

    def get_temporal_analysis(self, dimension: str):
        def op(s):
            query = text(""" ... """)
            results = s.execute(query, {"dimension": dimension}).fetchall()
            return pd.DataFrame([dict(row._mapping) for row in results])
        return self._run(op)

    def get_scorer_correlation(self, scorer: str, embedding: str):
        def op(s):
            query = text(""" ... """)
            results = s.execute(query).fetchall()
            df = pd.DataFrame([dict(row._mapping) for row in results])
            return df.corr()
        return self._run(op)

    def get_score_gaps(self, scorer: str, embedding: str):
        def op(s):
            query = text(""" ... """)
            return pd.DataFrame([dict(row._mapping) for row in self._scope().execute(query).fetchall()])
        return self._run(op)

    def get_gild_effectiveness(self):
        def op(s):
            query = text(""" ... """)
            return pd.DataFrame([dict(row._mapping) for row in self._scope().execute(query).fetchall()])
        return self._run(op)

    def plot_scorer_comparison(self, dimension: str):
        report = self.generate_comparison_report(goal_id=123)
        data = report[report.dimension == dimension]
        plt.figure(figsize=(12, 6))
        plt.bar(data.columns, data.iloc[0])
        plt.title(f"Scorer Comparison: {dimension}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{dimension}_comparison.png")

    def get_gild_training_feedback(self):
        def op(s):
            query = text(""" ... """)
            return pd.DataFrame([dict(row._mapping) for row in self._scope().execute(query).fetchall()])
        return self._run(op)
