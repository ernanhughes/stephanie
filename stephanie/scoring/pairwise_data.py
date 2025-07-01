# stephanie/scoring/pairwise_data.py

from collections import defaultdict

from sqlalchemy.sql import text


class PreferencePairDatasetBuilder:
    """
    Builds preference training pairs for MR.Q or reward model training
    by selecting top and bottom scored outputs per dimension.
    """

    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger

    def get_training_pairs_by_dimension(self, goal: str = None, limit: int = 10000) -> dict:
        """
        Returns top and bottom scored prompt/response pairs per dimension.
        
        Output:
        {
            "relevance": [
                {
                    "prompt": "...",
                    "output_a": "...",  # top
                    "output_b": "...",  # bottom
                    "value_a": 8.9,
                    "value_b": 3.1
                },
                ...
            ],
            ...
        }
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
                rank_type,
                prompt_id
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
        """.replace("{goal_filter}", "AND p.goal_text = :goal" if goal else ""))

        params = {"limit": limit}
        if goal:
            params["goal"] = goal

        try:
            rows = self.db.execute(query, params).fetchall()
        except Exception as e:
            if self.logger:
                self.logger.error("Failed to get training pairs", extra={"error": str(e)})
            return {}

        grouped = defaultdict(dict)
        results_by_dimension = defaultdict(list)

        for row in rows:
            key = (row.dimension, row.prompt_id)
            grouped[key][row.rank_type] = row

        for (dimension, _), data in grouped.items():
            if "top" in data and "bottom" in data:
                results_by_dimension[dimension].append({
                    "prompt": data["top"].prompt_text,
                    "output_a": data["top"].response_text,
                    "output_b": data["bottom"].response_text,
                    "value_a": float(data["top"].score),
                    "value_b": float(data["bottom"].score)
                })

        return dict(results_by_dimension)
