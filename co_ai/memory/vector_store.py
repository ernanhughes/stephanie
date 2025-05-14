import json

import psycopg2

from co_ai.tools.embedding_tool import get_embedding


class VectorMemory:
    def __init__(self, cfg, logger=None):
        """
        Initialize PostgreSQL connection using Hydra config.
        
        Args:
            cfg: Hydra config object with database settings under cfg.db
            logger: JSONLogger instance for structured logging
        """
        db_config = cfg.db  # Load DB config from Hydra
        self.conn = psycopg2.connect(
            dbname=db_config.database,
            user=db_config.user,
            password=db_config.password,
            host=db_config.host,
            port=db_config.port,
        )
        self.conn.autocommit = True
        self.logger = logger
        self.cfg = cfg  # Store cfg if needed later
        self.rankings = {}

    def store_hypothesis(self, goal, text, confidence, review, features):
        try:
            embedding = get_embedding(text, self.cfg)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO hypotheses (goal, text, confidence, review, features, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        goal,
                        text,
                        confidence,
                        review,
                        features,
                        embedding
                    )
                )
            # Log the operation
            if self.logger:
                self.logger.log("HypothesisStored", {
                    "goal": goal,
                    "hypothesis_text": text[:200],
                    "confidence": confidence
                })
        except Exception as e:
            if self.logger:
                self.logger.log("HypothesisStoreFailed", {
                    "error": str(e),
                    "hypothesis_text": text[:200]
                })
            else:
                print(f"[VectorMemory] Failed to store hypothesis: {e}")

    def search_related(self, query: str, top_k: int = 5):
        try:
            embedding = get_embedding(query, self.cfg)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT text, goal, confidence, review
                    FROM hypotheses
                    ORDER BY embedding <-> %s
                    LIMIT %s;
                    """,
                    (embedding, top_k)
                )
                results = cur.fetchall()

            if self.logger:
                self.logger.log("HypothesesSearched", {
                    "query": query,
                    "top_k": top_k,
                    "result_count": len(results)
                })

            return results
        except Exception as e:
            if self.logger:
                self.logger.log("HypothesesSearchFailed", {
                    "error": str(e),
                    "query": query
                })
            else:
                print(f"[VectorMemory] Search failed: {e}")
            return []

    def store_review(self, hypothesis_text: str, review: dict):
        try:
            with self.conn.cursor() as cur:
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

    def store_ranking(self, hypothesis: str, score: float):
        try:
            print(f"[VectorMemory] Storing ranking: '{hypothesis[:60]}...' with score {score}")
            if hasattr(self, "db"):
                with self.db.cursor() as cur:
                    cur.execute(
                        "INSERT INTO rankings (hypothesis, score) VALUES (%s, %s) "
                        "ON CONFLICT (hypothesis) DO UPDATE SET score = EXCLUDED.score;",
                        (hypothesis, score)
                    )
            else:
                self.rankings[hypothesis] = score

            if self.logger:
                self.logger.log("RankingStored", {
                    "hypotheses": hypothesis[:200],
                    "score": score
                })
        except Exception as e:
            if self.logger:
                self.logger.log("RankingStoreFailed", {
                    "error": str(e),
                    "hypotheses": hypothesis[:200]
                })
            else:
                print(f"[VectorMemory] Failed to store ranking: {e}")

    def log_summary(self, summary: str):
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO summaries (text)
                    VALUES (%s);
                    """,
                    (summary,)
                )

            if self.logger:
                self.logger.log("SummaryLogged", {
                    "summary_snippet": summary[:100]
                })
        except Exception as e:
            if self.logger:
                self.logger.log("SummaryLogFailed", {
                    "error": str(e),
                    "summary_snippet": summary[:100]
                })
            else:
                print(f"[VectorMemory] Failed to log summary: {e}")

    def store_prompt(self, agent_name: str, prompt_key: str, prompt_text: str, 
                    response: str = None, source: str = "manual", version: int = 1, 
                    is_current: bool = True, strategy: str = "default", metadata: dict = None):
        """Store prompt with version control"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO prompts (
                        agent_name, prompt_key, prompt_text, response_text,
                        source, version, is_current, strategy, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    agent_name, prompt_key, prompt_text, response,
                    source, version, is_current, strategy, json.dumps(metadata or {})
                ))
                # Deactivate previous versions
                if is_current:
                    cur.execute("""
                        UPDATE prompts SET is_current = FALSE
                        WHERE agent_name = %s AND prompt_key = %s AND is_current IS TRUE
                    """, (agent_name, prompt_key))
            if self.logger:
                self.logger.log("Prompt", {
                    "agent_name": agent_name,
                    "prompt": prompt_text,
                    "response": response
                })
        except Exception as e:
            if self.logger:
                self.logger.log("PromptLogFailed", {
                    "error": str(e),
                    "prompt_snippet": prompt_text[:100]
                })
            else:
                print(f"[VectorMemory] Failed to log Prompt: {e}")

    def store_report(self, run_id: str, goal: str, summary: str, path: str):
        """Insert a report into the reports table."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO reports (run_id, goal, summary, path)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (run_id, goal, summary, path)
                )
        except Exception as e:
            if self.logger:
                self.logger.log("ReportLogFailed", {
                    "error": str(e),
                    "summary_snippet": summary[:100]
                })
            else:
                print(f"[VectorMemory] Failed to log Report: {e}")

    def get_similar(self, goal: str, top_k: int = 5) -> list[dict[str, any]]:
        """
        Retrieve most similar hypotheses based on current goal or hypothesis.
        
        Uses embeddings to compute cosine similarity.
        """
        try:
            # Get embedding for current goal
            goal_embedding = get_embedding(goal, self.cfg)
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT text, embedding <-> %s AS distance, source, elo_rating
                    FROM hypotheses
                    WHERE enabled IS NOT FALSE
                    ORDER BY distance ASC
                    LIMIT %s
                """, (str(goal_embedding), top_k))

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
        cur = self.conn.cursor()
        try:
            # Fetch top hypotheses sorted by Elo rating
            cur.execute("""
                 SELECT 
                    hypothesis, 
                    reflection, 
                    elo_rating,
                    metadata
                FROM hypothesis
                WHERE goal = %s
                ORDER BY elo_rating DESC
                LIMIT %s
            """, (goal[:200], limit))

            rows = cur.fetchall()
            result = []

            for hyp, reflection, score, meta in rows:
                reflection_dict = json.loads(reflection) if isinstance(reflection, str) else reflection or {}
                
                result.append({
                    "goal": goal,
                    "hypotheses": hyp,
                    "review": reflection_dict.get("full_review", ""),
                    "score": score,
                    "elo_rating": score,
                    "metadata": meta
                })

            return result
        except Exception as e:
            print(f"[Memory] Failed to load ranked hypotheses: {e}")
            return []
        finally:
            cur.close()

    def get_latest_context_version(self, run_id: str, stage: str) -> int:
        """Get the latest version of this context stage"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT MAX(version) FROM context_states
                    WHERE run_id = %s AND stage_name = %s
                """, (run_id, stage))

                result = cur.fetchone()[0]
                return result if result else 0
        except Exception as e:
            print(f"[VectorMemory] Failed to load version: {e}")
            return 0

    def save_context(self, run_id: str, stage: str, context: dict, preferences: dict = None, metadata: dict = None):
        """
        Save the current context state to the database.

        Args:
            run_id: Unique identifier for this pipeline run
            stage: Name of the current agent/stage
            context: The full context dict
            preferences: The full preferences dict
            metadata: The full metadata dict
        """
        try:
            # Deactivate previous version
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE context_states SET is_current = FALSE
                    WHERE run_id = %s AND stage_name = %s
                """, (run_id, stage))

                # Get next version number
                latest_version = self.get_latest_context_version(run_id, stage)
                new_version = latest_version + 1

                # Insert new context state
                cur.execute("""
                    INSERT INTO context_states (run_id, stage_name, version, context, preferences, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (run_id, stage, new_version, json.dumps(context), json.dumps(preferences  or {}), json.dumps(metadata or {})))

                self.conn.commit()
        except Exception as e:
            self.logger.log("ContextSaveFailed", {
                "run_id": run_id,
                "stage": stage,
                "error": str(e),
                "context_keys": list(context.keys())
            })

    def has_completed(self, run_id: str, stage_name: str) -> bool:
        """Check if this stage has already been run"""
        if not run_id or not stage_name:
            return False
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM context_states
                WHERE run_id = %s AND stage_name = %s
            """, (run_id, stage_name))
            return cur.fetchone()[0] > 0

    def load_context(self, run_id: str, stage: str = None) -> dict:
        """
        Load the latest saved context for a given run and optional stage.

        Args:
            run_id: Unique ID for the pipeline run
            stage: Optional stage name to resume from

        Returns:
            dict: The deserialized context
        """
        try:
            cur = self.conn.cursor()

            if stage:
                cur.execute("""
                    SELECT context FROM context_states
                    WHERE run_id = %s AND stage_name = %s
                    ORDER BY timestamp DESC LIMIT 1
                """, (run_id, stage))
            else:
                cur.execute("""
                    SELECT context FROM context_states
                    WHERE run_id = %s
                    ORDER BY timestamp ASC
                """, (run_id,))

            rows = cur.fetchall()
            if not rows:
                return {}

            # Reconstruct context by merging all stages up to this point
            result = {}
            for row in rows:
                partial_context =  row[0]
                result.update(partial_context)

            return result
        except Exception as e:
            self.logger.log("ContextLoadFailed", {"error": str(e)})
            return {}
        
    def get_top_ranked_hypotheses(self, goal: str, limit: int = 5) -> list[dict[str, any]]:
        """
        Retrieve top-ranked hypotheses based on Elo rating or review scores.
        
        Args:
            goal: Research objective used to filter hypotheses
            limit: Number of hypotheses to return
            
        Returns:
            List of dicts containing hypothesis + metadata
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        h.text AS hypothesis,
                        h.review,
                        h.elo_rating,
                        h.source
                    FROM hypotheses h
                    WHERE h.goal = %s
                    AND h.enabled IS NOT FALSE
                    ORDER BY h.elo_rating DESC
                    LIMIT %s
                """, (goal[:200], limit))

                rows = cur.fetchall()
                result = []

                for hyp, review, score, source, prompt_key, strategy in rows:
                    result.append({
                        "goal": goal,
                        "hypotheses": hyp,
                        "review": review or "",
                        "score": score or 1000,
                        "elo_rating": score or 1000,
                        "prompt_key": prompt_key,
                        "strategy_used": strategy,
                        "source": source
                    })

                return result

        except Exception as e:
            print(f"[VectorMemory] Failed to load ranked hypotheses: {e}")
            return []


    def get_embedding(self, text: str) -> list[float]|None:
        """
        Get the embedding for a given text.
        
        Args:
            text: The input text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT embedding FROM embeddings WHERE text = %s", (text,))
                row = cur.fetchone()
                if row:
                    print("🔁 Loaded from DB cache")
                    return row[0]
                return None
        except Exception as e:
            print(f"[VectorMemory] Failed to fetch embedding: {e}")
            return None


    def set_embedding(self, text: str, embedding):
        """
        Get the embedding for a given text.
        
        Args:
            text: The input text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("INSERT INTO embeddings (text, embedding) VALUES (%s, %s) ON CONFLICT (text) DO NOTHING",
                        (text, embedding))
        except Exception as e:
            print(f"[VectorMemory] Failed to fetch embedding: {e}")
            return None