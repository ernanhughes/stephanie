# stores/mrq_store.py
import json
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from co_ai.models import ReflectionDeltaORM
from co_ai.models.mrq_memory_entry import MRQMemoryEntryORM
from datetime import datetime


class MRQStore:
    def __init__(self, cfg:dict, session: Session, logger=None):
        self.session = session
        self.logger = logger
        self.name = "mrq"
        self.cfg = cfg

    def log_evaluations(self):
        return self.cfg.get("log_evaluations", True)

    def add(self, goal: str, strategy: str, prompt: str,
            response: str, reward: float, metadata: dict = None):
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
                features=None,    # optional: extract features from metadata
                source="manual",
                run_id=metadata.get("run_id") if metadata else None,
                metadata_=json.dumps(metadata or {}),
                created_at=datetime.utcnow()
            )

            self.session.add(db_entry)
            self.session.flush()  # Get ID before commit

            if self.logger:
                self.logger.log("MRQMemoryEntryInserted", {
                    "goal_snippet": goal[:100],
                    "prompt_snippet": prompt[:100],
                    "strategy": strategy,
                    "reward": reward,
                    "timestamp": db_entry.created_at.isoformat()
                })

            return db_entry.id

        except Exception as e:
            self.session.rollback()
            if self.logger:
                self.logger.log("MRQMemoryInsertFailed", {"error": str(e)})
            raise

    def get_similar_prompt(self, prompt: str, top_k: int = 5) -> list:
        """
        Gets similar prompts based on text match.
        Future: can use vector similarity instead of trigram search.
        """
        try:
            results = self.session.query(MRQMemoryEntryORM).filter(
                MRQMemoryEntryORM.prompt.ilike(f"%{prompt}%")
            ).limit(top_k).all()

            return results

        except Exception as e:
            if self.logger:
                self.logger.log("MRQSimilarPromptSearchFailed", {"error": str(e)})
            return []

    def get_by_strategy(self, strategy: str, limit: int = 100) -> list:
        """Returns all entries generated using a specific strategy."""
        return self.session.query(MRQMemoryEntryORM).filter_by(strategy=strategy).limit(limit).all()

    def get_all(self, limit: int = 100) -> list:
        """Returns most recent MRQ memory entries."""
        return self.session.query(MRQMemoryEntryORM).order_by(MRQMemoryEntryORM.created_at.desc()).limit(limit).all()

    def train_from_reflection_deltas(self):
        """Train ranker from reflection deltas (symbolic_ranker example)"""
        deltas = self.session.query(ReflectionDeltaORM).all()
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
            examples.append({
                "goal_text": d.goal.goal_text,
                "pipeline_a": a,
                "pipeline_b": b,
                "score_a": score_a,
                "score_b": score_b,
                "label": label
            })

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