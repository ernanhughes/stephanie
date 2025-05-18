import json

from co_ai.memory import BaseStore


class HypothesesStore(BaseStore):
    def __init__(self, db, embeddings, logger=None):
        self.db = db
        self.embeddings = embeddings
        self.logger = logger
        self.name = "hypotheses"
        self._rankings = {}

    def name(self) -> str:
        return "hypotheses"

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
            if self.logger:
                self.logger.log("PromptLookupFailed", {
                    "error": str(e),
                    "prompt_snippet": prompt_text[:100]
                })
        return prompt_id

    def store(self, goal, text, confidence, review, features, prompt_text=None):
        embedding = self.embeddings.get_or_create(text)
        prompt_id = self.get_prompt_id(prompt_text)
        try:
            with self.db.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO hypotheses (goal, text, confidence, review, features, embedding, prompt_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (goal, text, confidence, review, features, embedding, prompt_id)
                )
            if self.logger:
                self.logger.log("HypothesisStored", {
                    "goal": goal,
                    "text": text[:100],
                    "confidence": confidence
                })
        except Exception as e:
            if self.logger:
                self.logger.log("HypothesisStoreFailed", {
                    "error": str(e),
                    "text": text[:100]
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
                    SELECT text, embedding <-> %s AS distance, source, elo_rating
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
            print(f"[VectorMemory] Failed to fetch similar hypotheses: {e}")
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
                        text, 
                        reflection, 
                        review, 
                        elo_rating
                    FROM hypotheses
                    WHERE goal = %s
                    ORDER BY elo_rating DESC
                    LIMIT %s
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
                    SELECT text FROM hypotheses
                    WHERE review IS NULL
                    AND goal = %s
                    ORDER BY created_at DESC
                    LIMIT %s
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
                    SELECT text FROM hypotheses
                    WHERE reflected IS NULL
                    AND goal = %s
                    ORDER BY created_at DESC
                    LIMIT %s
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
