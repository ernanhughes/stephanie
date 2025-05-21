from co_ai.memory import BaseStore
from co_ai.models.hypothesis import Hypothesis
from typing import Optional
from psycopg2.extras import Json
from datetime import datetime

class HypothesesStore(BaseStore):
    def __init__(self, db, embeddings, logger=None):
        self.db = db
        self.embeddings = embeddings
        self.logger = logger
        self.name = "hypotheses"
        self._rankings = {}

    def __repr__(self):
        return f"<{self.name} connected={self.db is not None}>"

    def name(self) -> str:
        return "hypotheses"

    def get_id_by_text(self, text: str) -> Optional[int]:
        hypothesis_id = None
        try:
            with self.db.cursor() as cur:
                cur.execute("SELECT id FROM hypotheses WHERE text = %s", (text,))
                row = cur.fetchone()
                if row:
                    hypothesis_id = row[0]
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log(
                    "HypothesesLookupFailed",
                    {"error": str(e), "hypotheses_snippet": text[:100]},
                )
        return hypothesis_id

    def get_prompt_id(self, prompt_text: str) -> int:
        prompt_id = None
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    "SELECT id FROM prompts WHERE prompt_text = %s ORDER BY timestamp DESC LIMIT 1",
                    (prompt_text,)
                )
                row = cur.fetchone()
                if row:
                    prompt_id = row[0]
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("PromptLookupFailed", {
                    "error": str(e),
                    "prompt_snippet": prompt_text[:100]
                })
        return prompt_id

    def get_or_create_goal(self, goal_text, goal_type=None, focus_area=None,
                           strategy=None, source="user"):
        """
        Looks up or inserts a goal in the `goals` table and returns the goal_id.
        Assumes `conn` is a psycopg2 or asyncpg connection or cursor.
        """
        try:
            with self.db.cursor() as cur:
                # Try to find existing goal
                cur.execute(
                    """
                    SELECT id FROM goals
                    WHERE goal_text = %s
                    LIMIT 1
                """,
                    (goal_text,),
                )
                result = cur.fetchone()
                if result:
                    return result[0]

                # Insert new goal
                cur.execute(
                    """
                    INSERT INTO goals (goal_text, goal_type, focus_area, strategy, source)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (goal_text, goal_type, focus_area, strategy, source),
                )
                new_id = cur.fetchone()[0]
                return new_id
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("GoalGetFailed", {"error": str(e)})
            return None



    def store(self, hypothesis: Hypothesis, prompt_text: Optional[str] = None):
        try:
            # Resolve prompt and goal
            embedding = self.embeddings.get_or_create(hypothesis.text)
            prompt_id = self.get_prompt_id(hypothesis.prompt)
            goal_id = self.get_or_create_goal(hypothesis.goal, hypothesis.goal_type)

            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO hypotheses (
                        goal_id, text, confidence, review, reflection, elo_rating,
                        embedding, features, prompt_id, source_hypothesis,
                        strategy_used, version, source, enabled, created_at, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        goal_id,
                        hypothesis.text,
                        hypothesis.confidence,
                        hypothesis.review,
                        hypothesis.reflection,
                        hypothesis.elo_rating,
                        embedding,
                        Json(hypothesis.features),
                        prompt_id,
                        hypothesis.source_hypothesis,
                        hypothesis.strategy_used,
                        hypothesis.version,
                        hypothesis.source,
                        hypothesis.enabled,
                        hypothesis.created_at,
                        hypothesis.updated_at,
                    )
                )

            if self.logger:
                self.logger.log("HypothesisStored", {
                    "text": hypothesis.text[:100],
                    "goal": hypothesis.goal,
                    "confidence": hypothesis.confidence
                })

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("HypothesisStoreFailed", {
                    "error": str(e),
                    "text": hypothesis.text[:100]
                })

    def store_review(self, hypothesis_text: str, review: dict):
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    UPDATE hypotheses
                    SET review = %s
                    WHERE text = %s
                    """,
                    (review, hypothesis_text)
                )

            if self.logger:
                self.logger.log("ReviewStored", {
                    "hypothesis_text": hypothesis_text[:200],
                    "review_snippet": review[:100]
                })
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("ReviewStoreFailed", {
                    "error": str(e),
                    "hypothesis_text": hypothesis_text[:200]
                })
            else:
                print(f"[VectorMemory] Failed to store review: {e}")

    def store_elo_ranking(self, hypothesis: str, score: float):
        try:
            if self.db:
                with self.db.cursor() as cur:
                    cur.execute(
                        """UPDATE hypotheses 
                        SET elo_rating = %s
                        WHERE text = %s""",
                        (score, hypothesis)
                    )
                    self.db.commit()
            else:
                self._rankings[hypothesis] = score
            if self.logger:
                self.logger.log("RankingStored", {
                    "score": score,
                    "hypotheses": hypothesis[:200]
                })
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("RankingStoreFailed", {
                    "error": str(e),
                    "hypotheses": hypothesis[:200]
                })
            else:
                print(f"[VectorMemory] Failed to store ranking: {e}")

    def get_similar(self, goal: str, top_k: int = 5) -> list[dict[str, any]]:
        """
        Retrieve most similar hypotheses based on current goal or hypothesis.
        
        Uses embeddings to compute cosine similarity.
        """
        try:
            # Get embedding for current goal
            goal_embedding = self.embeddings.get_or_create(goal)
            with self.db.cursor() as cur:
                cur.execute("""
                    SELECT text, embedding <-> %s::vector AS distance, source, elo_rating
                    FROM hypotheses
                    WHERE enabled IS NOT FALSE
                    ORDER BY distance ASC
                    LIMIT %s
                """, (goal_embedding, top_k))
                results = []
                for row in cur.fetchall():
                    results.append({
                        "text": row[0],
                        "similarity": 1 - float(row[1]),  # Convert distance → similarity
                        "source": row[2],
                        "elo_rating": row[3] or 1000
                    })
                return results
        
        except Exception as e:
            print(f"❌ Exception Failed to fetch similar hypotheses:: {type(e).__name__}: {e}")
            return []
        
    def get_ranked(self, goal: str, limit: int = 5) -> list[dict[str, any]]:
        """
        Get top-ranked hypotheses for the given goal.
        
        Args:
            goal: Research objective used to generate hypotheses
            limit: Number of hypotheses to retrieve
            
        Returns:
            List of dicts containing hypothesis + review + score
        """
        try:
            # Fetch top hypotheses sorted by Elo rating
            rows = []
            with self.db.cursor() as cur:
                cur.execute("""
                    SELECT 
                        h.text, 
                        h.reflection, 
                        h.review, 
                        h.elo_rating
                    FROM hypotheses h
                    JOIN goals g ON h.goal_id = g.id
                    WHERE g.goal_text = %s
                    ORDER BY h.elo_rating DESC
                    LIMIT %s;
                """, (goal, limit))
                rows = cur.fetchall()
            result = []
            for text, reflection, review, score in rows:
                result.append({
                    "goal": goal,
                    "hypotheses": text,
                    "reflection": reflection,
                    "score": score,
                    "review": review,
                    "elo_rating": score
                })
            return result
        except Exception as e:
            print(f"[Memory] Failed to load ranked hypotheses: {e}")
            return []

    def store_reflection(self, hypothesis_text: str, reflection: str):
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    UPDATE hypotheses
                    SET reflection = %s
                    WHERE text = %s
                    """,
                    (reflection, hypothesis_text)
                )

            if self.logger:
                self.logger.log("ReflectionStored", {
                    "hypothesis_text": hypothesis_text[:200],
                    "reflection_snippet": reflection[:100]
                })
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("ReflectionStoreFailed", {
                    "error": str(e),
                    "hypothesis_text": hypothesis_text[:200]
                })
            else:
                print(f"[VectorMemory] Failed to store reflection: {e}")

    def get_unreviewed(self, goal: str, limit: int = 10) -> list[dict[str, any]] | None:
        try:
            rows = []
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    SELECT h.text
                    FROM hypotheses h
                    JOIN goals g ON h.goal_id = g.id
                    WHERE h.review IS NULL
                    AND g.goal_text = %s
                    ORDER BY h.created_at DESC
                    LIMIT %s;
                    """,
                    (goal, limit),
                )
                rows = cur.fetchall()
            result = []
            for text in rows:
                result.append(text)
            return result
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log("GetUnReviewedFailed", {
                    "error": str(e),
                    "goal": goal
                })
            else:
                print(f"GetUnReviewedFailed: {e}")
        return None

    def get_unreflected(self, goal: str, limit: int = 10) -> list[dict[str, any]] | None:
        try:
            rows = []
            with self.db.cursor() as cur:
                cur.execute(
                    """
                        SELECT h.text
                        FROM hypotheses h
                        JOIN goals g ON h.goal_id = g.id
                        WHERE h.reflected IS NULL
                        AND g.goal_text = %s
                        ORDER BY h.created_at DESC
                        LIMIT %s;
                    """,
                    (goal, limit),
                )
                rows = cur.fetchall()
            result = []
            for text in rows:
                result.append(text)
            return result
        except Exception as e:
            if self.logger:
                self.logger.log("GetUnReflectedFailed", {
                    "error": str(e),
                    "goal": goal
                })
            else:
                print(f"GetUnReflectedFailed: {e}")
            return None

    def get_latest(self, goal: str, limit: int = 10) -> list[dict[str, any]] | None:
        try:
            rows = []
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    SELECT h.text
                    FROM hypotheses h
                    JOIN goals g ON h.goal_id = g.id
                    AND g.goal_text = %s
                    ORDER BY h.created_at DESC
                    LIMIT %s;
                    """,
                    (goal, limit),
                )
                rows = cur.fetchall()
            result = []
            for text in rows:
                result.append(text)
            return result
        except Exception as e:
            if self.logger:
                self.logger.log("GetLatestFailed", {
                    "error": str(e),
                    "goal": goal
                })
            else:
                print(f"GetUnReflectedFailed: {e}")
            return None

    def get_hypotheses_for_prompt(self, prompt_text: str, limit: int = 10) -> list[dict[str, any]] | None:
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    SELECT h.text, h.review
                    FROM hypotheses h
                    JOIN prompts p ON h.prompt_id = p.id
                    WHERE p.prompt_text = %s
                    ORDER BY h.created_at DESC
                    LIMIT %s;
                    """,
                    (prompt_text, limit),
                )
                rows = cur.fetchall()

            result = []
            for row in rows:
                result.append({
                    "hypothesis": row[0],
                    "review": row[1]
                })
            return result

        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log(
                    "GetHypothesesForPromptFailed",
                    {"error": str(e), "prompt_text": prompt_text},
                )
            else:
                print(f"GetHypothesesForPromptFailed: {e}")
            return None

    def store_pattern_stats(self, goal_id, hypothesis_id, patterns: list):
        """Insert a list of PatternStat dataclass instances into the cot_patterns table."""

        insert_query = """
               INSERT INTO cot_patterns (
                   goal_id,
                   hypothesis_id,
                   model_name,
                   agent_name,
                   dimension,
                   label,
                   confidence_score,
                   created_at
               ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
           """
        values = [
            (
                pattern.goal_id,
                pattern.hypothesis_id,
                pattern.model_name,
                pattern.agent_name,
                pattern.dimension,
                pattern.label,
                pattern.confidence_score,
                pattern.created_at or datetime.utcnow(),
            )
            for pattern in patterns
        ]
        try:
            with self.db.cursor() as cur:
                cur.executemany(insert_query, values)
        except Exception as e:
            print(f"❌ Exception: {type(e).__name__}: {e}")
            if self.logger:
                self.logger.log(
                    "StorePatternStatsFailed",
                    {"error": str(e), "hypothesis_id": hypothesis_id},
                )
