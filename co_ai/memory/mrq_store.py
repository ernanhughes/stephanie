# stores/mrq_store.py
import json
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from co_ai.models import MRQMemoryEntryORM, MRQPreferencePairORM, ReflectionDeltaORM


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
                created_at=datetime.utcnow(),
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
                    "score_a": score_a,
                    "score_b": score_b,
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
                        h.elo_rating AS rating_a
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
                        h.elo_rating AS rating_b
                    FROM prompts p
                    JOIN hypotheses h ON h.prompt_id = p.id
                    JOIN goals g ON p.goal_id = g.id
                    WHERE h.enabled = TRUE
                    AND h.goal_id = g.id
                    AND p.agent_name = :agent_name
                    ORDER BY p.id, h.elo_rating ASC
                )
                SELECT 
                    top_h.prompt_id,
                    top_h.goal,
                    top_h.prompt_text,
                    top_h.output_a,
                    top_h.rating_a,
                    bottom_h.output_b,
                    bottom_h.rating_b
                FROM top_h
                JOIN bottom_h ON top_h.prompt_id = bottom_h.prompt_id
                WHERE top_h.rating_a != bottom_h.rating_b
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
                    "rating_a": row[4],
                    "rating_b": row[6],
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
