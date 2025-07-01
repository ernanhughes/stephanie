# stores/mrq_store.py
import json
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session

from stephanie.models import (MRQMemoryEntryORM, MRQPreferencePairORM,
                          ReflectionDeltaORM)


class MRQStore:
    def __init__(self, cfg: dict, session: Session, logger=None):
        self.db = session
        self.logger = logger
        self.name = "mrq"
        self.cfg = cfg

    def log_evaluations(self):
        return self.cfg.get("log_evaluations", True)

    def add(
        self,
        goal: str,
        strategy: str,
        prompt: str,
        response: str,
        reward: float,
        metadata: dict = None,
    ):
        """
        Adds a new entry to MRQ memory for symbolic learning or training.
        """
        try:
            db_entry = MRQMemoryEntryORM(
                goal=goal,
                strategy=strategy,
                prompt=prompt,
                response=response,
                reward=reward,
                embedding=None,  # optional: compute from prompt/response
                features=None,  # optional: extract features from metadata
                source="manual",
                run_id=metadata.get("run_id") if metadata else None,
                metadata_=json.dumps(metadata or {}),
                created_at=datetime.now(timezone.utc),
            )

            self.db.add(db_entry)
            self.db.flush()  # Get ID before commit

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

        except Exception as e:
            self.db.rollback()
            if self.logger:
                self.logger.log("MRQMemoryInsertFailed", {"error": str(e)})
            raise

    def get_similar_prompt(self, prompt: str, top_k: int = 5) -> list:
        """
        Gets similar prompts based on text match.
        Future: can use vector similarity instead of trigram search.
        """
        try:
            results = (
                self.db.query(MRQMemoryEntryORM)
                .filter(MRQMemoryEntryORM.prompt.ilike(f"%{prompt}%"))
                .limit(top_k)
                .all()
            )

            return results

        except Exception as e:
            if self.logger:
                self.logger.log("MRQSimilarPromptSearchFailed", {"error": str(e)})
            return []

    def get_by_strategy(self, strategy: str, limit: int = 100) -> list:
        """Returns all entries generated using a specific strategy."""
        return (
            self.db.query(MRQMemoryEntryORM)
            .filter_by(strategy=strategy)
            .limit(limit)
            .all()
        )

    def get_all(self, limit: int = 100) -> list:
        """Returns most recent MRQ memory entries."""
        return (
            self.db.query(MRQMemoryEntryORM)
            .order_by(MRQMemoryEntryORM.created_at.desc())
            .limit(limit)
            .all()
        )

    def train_from_reflection_deltas(self):
        """Train ranker from reflection deltas (symbolic_ranker example)"""
        deltas = self.db.query(ReflectionDeltaORM).all()
        examples = []

        for d in deltas:
            a = d.pipeline_a
            b = d.pipeline_b
            score_a = d.score_a
            score_b = d.score_b

            if not isinstance(a, list) or not isinstance(b, list):
                continue
            if score_a is None or score_b is None:
                continue
            if abs(score_a - score_b) < 0.05:
                continue  # Skip small differences

            label = "b" if score_b > score_a else "a"
            examples.append(
                {
                    "goal_text": d.goal.goal_text,
                    "pipeline_a": a,
                    "pipeline_b": b,
                    "value_a": score_a,
                    "value_b": score_b,
                    "label": label,
                }
            )

        self.training_data = examples
        self.trained_ranker = self.symbolic_ranker()

        if self.logger:
            self.logger.log("MRQTrainingDataLoaded", {"count": len(examples)})

    def symbolic_ranker(self):
        """Simple rule-based ranker used until we train a learned one"""

        def score_pipeline(pipeline: list):
            base_score = len(pipeline) * 0.3
            if "verifier" in pipeline:
                base_score += 1.5
            if "reviewer" in pipeline:
                base_score += 1.2
            if "retriever" in pipeline:
                base_score += 1.0
            if "cot_generator" in pipeline:
                base_score += 0.8
            return base_score

        return score_pipeline

    def get_training_pairs(
        self, goal: str, limit: int = 100, agent_name="generation"
    ) -> list[dict]:
        try:
            sql = text("""
                WITH top_h AS (
                    SELECT DISTINCT ON (p.id)
                        p.id AS prompt_id,
                        g.goal_text AS goal,
                        p.prompt_text,
                        h.text AS output_a,
                        h.elo_rating AS value_a
                    FROM prompts p
                    JOIN goals g ON p.goal_id = g.id
                    JOIN hypotheses h ON h.prompt_id = p.id
                    WHERE h.enabled = TRUE
                    AND h.goal_id = g.id
                    AND p.agent_name = :agent_name
                    ORDER BY p.id, h.elo_rating DESC
                ),
                bottom_h AS (
                    SELECT DISTINCT ON (p.id)
                        p.id AS prompt_id,
                        h.text AS output_b,
                        h.elo_rating AS value_b
                    FROM prompts p
                    JOIN hypotheses h ON h.prompt_id = p.id
                    JOIN goals g ON p.goal_id = g.id
                    WHERE h.enabled = TRUE
                    AND h.goal_id = g.id
                    ORDER BY p.id, h.elo_rating ASC
                )
                SELECT 
                    top_h.prompt_id,
                    top_h.goal,
                    top_h.prompt_text,
                    top_h.output_a,
                    top_h.value_a,
                    bottom_h.output_b,
                    bottom_h.value_b
                FROM top_h
                JOIN bottom_h ON top_h.prompt_id = bottom_h.prompt_id
                WHERE top_h.value_a != bottom_h.value_b
                LIMIT :limit;
            """)

            result = self.db.execute(
                sql, {"goal": goal, "agent_name": agent_name, "limit": limit}
            )
            rows = result.fetchall()

            return [
                {
                    "prompt": row[2],
                    "output_a": row[3],
                    "output_b": row[5],
                    "preferred": "a" if row[4] > row[6] else "b",
                    "value_a": row[4],
                    "value_b": row[6],
                }
                for row in rows
            ]

        except Exception as e:
            self.db.rollback()
            if self.logger:
                self.logger.log(
                    "GetMRQTrainingPairsFailed", {"error": str(e), "goal": goal}
                )
            return []

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
        run_id: str = None
    ):
        """
        Save preference pair to database with precomputed embeddings.
        Args:
            goal: Task name or group key (e.g., "arm_dpo")
            prompt: Input question or instruction
            output_a: First response (chosen or rejected)
            output_b: Second response
            preferred: Either "a" or "b"
            prompt_emb: Precomputed embedding of the prompt
            output_a_emb: Precomputed embedding of output_a
            output_b_emb: Precomputed embedding of output_b
        """
        try:
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
            self.db.add(entry)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise RuntimeError(f"Failed to save preference pair: {str(e)}")
        finally:
            self.db.close()

    def get_training_preferece_pairs(self, goal: str, limit: int = 1000) -> list[dict]:
        try:
            query = self.db.query(MRQPreferencePairORM).filter(
                MRQPreferencePairORM.goal == goal
            )
            results = query.limit(limit).all()
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
        except Exception as e:
            raise RuntimeError(
                f"Failed to load preference pairs for goal '{goal}': {str(e)}"
            )
        finally:
            self.db.close()

    def get_training_pairs_by_dimension(self, goal: str = None, limit: int = 10000) -> dict:
        """
        Returns top and bottom scored prompt/response pairs per dimension,
        suitable for MR.Q training.
        """
        query = text("""
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
                  AND response_text  <> ''
                  
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
        """.replace("{goal_filter}", "AND p.goal_text = :goal" if goal else ""))

        params = {"limit": limit}
        if goal:
            params["goal"] = goal

        rows = self.db.execute(query, params).fetchall()

        # Group into pairs (top + bottom) by dimension and prompt_id
        from collections import defaultdict
        grouped = defaultdict(dict)
        for row in rows:
            key = (row.dimension, row.prompt_text)
            grouped[key][row.rank_type] = row
            results_by_dimension = defaultdict(list)
            for (dimension, prompt_text), data in grouped.items():
                if "top" in data and "bottom" in data:
                    results_by_dimension[dimension].append({
                        "prompt": prompt_text,
                        "output_a": data["top"].response_text,
                        "output_b": data["bottom"].response_text,
                        "value_a": data["top"].score,
                        "value_b": data["bottom"].score,
                    })
        return dict(results_by_dimension)