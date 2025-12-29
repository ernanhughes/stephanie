# stephanie/memory/mrq_store.py
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import text

from stephanie.memory.base_store import BaseSQLAlchemyStore
from stephanie.orm import (MRQMemoryEntryORM, MRQPreferencePairORM,
                              ReflectionDeltaORM)


class MRQStore(BaseSQLAlchemyStore):
    orm_model = MRQMemoryEntryORM
    default_order_by = MRQMemoryEntryORM.created_at.desc()

    def __init__(self, cfg: dict, session_maker, logger=None):
        super().__init__(session_maker, logger)
        self.name = "mrq"
        self.cfg = cfg

    def log_evaluations(self) -> bool:
        return self.cfg.get("log_evaluations", True)

    # -------------------
    # Insert
    # -------------------
    def add(
        self,
        goal: str,
        strategy: str,
        prompt: str,
        response: str,
        reward: float,
        metadata: dict = None,
    ) -> int:
        """Add a new entry to MRQ memory."""
        def op(s):
            db_entry = MRQMemoryEntryORM(
                goal=goal,
                strategy=strategy,
                prompt=prompt,
                response=response,
                reward=reward,
                embedding=None,
                features=None,
                source="manual",
                run_id=(metadata or {}).get("run_id"),
                metadata_=json.dumps(metadata or {}),
                created_at=datetime.now(timezone.utc),
            )
            s.add(db_entry)
            s.flush()

            if self.logger:
                self.logger.log(
                    "MRQMemoryEntryInserted",
                    {
                        "goal_snippet": goal[:100],
                        "prompt_snippet": prompt[:100],
                        "strategy": strategy,
                        "reward": reward,
                        "timestamp": db_entry.created_at.isoformat(),
                    },
                )
            return db_entry.id
        return self._run(op)

    def add_preference_pair(
        self,
        goal: str,
        prompt: str,
        output_a: str,
        output_b: str,
        preferred: str,
        fmt_a: str,
        fmt_b: str,
        difficulty: str,
        source: str = "arm_dataloader",
        run_id: str = None,
    ) -> int:
        """Save preference pair to database."""
        def op(s):
            entry = MRQPreferencePairORM(
                goal=goal,
                prompt=prompt,
                output_a=output_a,
                output_b=output_b,
                preferred=preferred,
                fmt_a=fmt_a,
                fmt_b=fmt_b,
                difficulty=difficulty,
                source=source,
                run_id=run_id,
            )
            s.add(entry)
            s.flush()
            return entry.id
        return self._run(op)

    # -------------------
    # Retrieval
    # -------------------
    def get_similar_prompt(self, prompt: str, top_k: int = 5) -> List[MRQMemoryEntryORM]:
        """Naive substring search for similar prompts."""
        def op(s):
            return (
                s.query(MRQMemoryEntryORM)
                .filter(MRQMemoryEntryORM.prompt.ilike(f"%{prompt}%"))
                .limit(top_k)
                .all()
            )
        return self._run(op)

    def get_by_strategy(self, strategy: str, limit: int = 100) -> List[MRQMemoryEntryORM]:
        def op(s):
            return (
                s.query(MRQMemoryEntryORM)
                .filter_by(strategy=strategy)
                .limit(limit)
                .all()
            )
        return self._run(op)

    def get_all(self, limit: int = 100) -> List[MRQMemoryEntryORM]:
        def op(s):
            return (
                s.query(MRQMemoryEntryORM)
                .order_by(MRQMemoryEntryORM.created_at.desc())
                .limit(limit)
                .all()
            )
        return self._run(op)

    def get_training_preferece_pairs(self, goal: str, limit: int = 1000) -> List[dict]:
        def op(s):
            q = s.query(MRQPreferencePairORM).filter(
                MRQPreferencePairORM.goal == goal
            )
            results = q.limit(limit).all()
            return [
                {
                    "prompt": r.prompt,
                    "output_a": r.output_a,
                    "output_b": r.output_b,
                    "preferred": r.preferred,
                    "fmt_a": r.fmt_a,
                    "fmt_b": r.fmt_b,
                }
                for r in results
            ]
        return self._run(op)

    # -------------------
    # Training Helpers
    # -------------------
    def train_from_reflection_deltas(self):
        """Build examples from reflection deltas for symbolic training."""
        def op(s):
            deltas = s.query(ReflectionDeltaORM).all()
            examples = []
            for d in deltas:
                a, b = d.pipeline_a, d.pipeline_b
                sa, sb = d.score_a, d.score_b
                if not isinstance(a, list) or not isinstance(b, list):
                    continue
                if sa is None or sb is None or abs(sa - sb) < 0.05:
                    continue
                label = "b" if sb > sa else "a"
                examples.append(
                    {
                        "goal_text": d.goal.goal_text,
                        "pipeline_a": a,
                        "pipeline_b": b,
                        "value_a": sa,
                        "value_b": sb,
                        "label": label,
                    }
                )
            self.training_data = examples
            self.trained_ranker = self.symbolic_ranker()
            if self.logger:
                self.logger.log("MRQTrainingDataLoaded", {"count": len(examples)})
            return examples
        return self._run(op)

    def symbolic_ranker(self):
        """Simple hand-coded pipeline scorer."""
        def score_pipeline(pipeline: list):
            base = len(pipeline) * 0.3
            if "verifier" in pipeline:
                base += 1.5
            if "reviewer" in pipeline:
                base += 1.2
            if "retriever" in pipeline:
                base += 1.0
            if "cot_generator" in pipeline:
                base += 0.8
            return base
        return score_pipeline

    def get_training_pairs_by_dimension(
        self, goal: Optional[str] = None, limit: int = 10000
    ) -> dict:
        """Top/bottom scored prompt-response pairs per dimension."""
        def op(s):
            query = text(
                """
                WITH scored_prompts AS (
                    SELECT
                        s.dimension,
                        s.score,
                        e.pipeline_run_id,
                        p.id AS prompt_id,
                        p.prompt_text,
                        p.response_text,
                        ROW_NUMBER() OVER (
                            PARTITION BY s.dimension, p.id ORDER BY s.score DESC
                        ) AS rank_high,
                        ROW_NUMBER() OVER (
                            PARTITION BY s.dimension, p.id ORDER BY s.score ASC
                        ) AS rank_low
                    FROM scores s
                    JOIN evaluations e ON s.evaluation_id = e.id
                    JOIN prompts p ON e.pipeline_run_id = p.pipeline_run_id
                    WHERE s.score IS NOT NULL
                    {goal_filter}
                )
                SELECT
                    dimension,
                    prompt_text,
                    response_text,
                    score,
                    rank_type
                FROM (
                    SELECT
                        dimension,
                        prompt_text,
                        response_text,
                        score,
                        'top' AS rank_type,
                        prompt_id
                    FROM scored_prompts
                    WHERE rank_high = 1
                      AND prompt_text IS NOT NULL
                      AND response_text IS NOT NULL
                      AND prompt_text <> ''
                      AND response_text <> ''

                    UNION ALL

                    SELECT
                        dimension,
                        prompt_text,
                        response_text,
                        score,
                        'bottom' AS rank_type,
                        prompt_id
                    FROM scored_prompts
                    WHERE rank_low = 1
                ) AS ranked_pairs
                ORDER BY dimension, prompt_id
                LIMIT :limit
                """.replace("{goal_filter}", "AND p.goal_text = :goal" if goal else "")
            )
            params = {"limit": limit}
            if goal:
                params["goal"] = goal
            rows = s.execute(query, params).fetchall()

            grouped = defaultdict(dict)
            for row in rows:
                key = (row.dimension, row.prompt_text)
                grouped[key][row.rank_type] = row

            results_by_dimension = defaultdict(list)
            for (dimension, prompt_text), data in grouped.items():
                if "top" in data and "bottom" in data:
                    results_by_dimension[dimension].append(
                        {
                            "prompt": prompt_text,
                            "output_a": data["top"].response_text,
                            "output_b": data["bottom"].response_text,
                            "value_a": data["top"].score,
                            "value_b": data["bottom"].score,
                        }
                    )
            return dict(results_by_dimension)
        return self._run(op)
