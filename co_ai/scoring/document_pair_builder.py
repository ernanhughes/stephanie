# co_ai/scoring/document_pair_builder.py

from collections import defaultdict

from sqlalchemy.sql import text


class DocumentPreferencePairBuilder:
    """
    Builds preference training pairs from scored documents per dimension.
    Designed for MR.Q or reward model training to rank research/document quality.
    """

    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger

    def get_training_pairs_by_dimension(self, goal: str = None, limit: int = 10000) -> dict:
        """
        Returns a dictionary of document preference pairs grouped by dimension.

        Output Format:
        {
            "relevance": [
                {
                    "title": "...",
                    "text_a": "...",     # preferred
                    "text_b": "...",     # less preferred
                    "value_a": 9.1,
                    "value_b": 5.3
                },
                ...
            ],
            ...
        }
        """
        query = text("""
            WITH scored_docs AS (
                SELECT
                    s.dimension,
                    s.score,
                    d.id AS doc_id,
                    d.title,
                    d.content,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.dimension, d.id ORDER BY s.score DESC
                    ) AS rank_high,
                    ROW_NUMBER() OVER (
                        PARTITION BY s.dimension, d.id ORDER BY s.score ASC
                    ) AS rank_low
                FROM scores s
                JOIN evaluations e ON s.evaluation_id = e.id
                JOIN documents d ON e.document_id = d.id
                WHERE s.score IS NOT NULL
            )
            SELECT
                dimension,
                title,
                content,
                score,
                rank_type,
                doc_id
            FROM (
                SELECT
                    dimension,
                    title,
                    content,
                    score,
                    'top' AS rank_type,
                    doc_id
                FROM scored_docs
                WHERE rank_high = 1
                  AND content IS NOT NULL
                  AND content <> ''
                  
                UNION ALL

                SELECT
                    dimension,
                    title,
                    content,
                    score,
                    'bottom' AS rank_type,
                    doc_id
                FROM scored_docs
                WHERE rank_low = 1
            ) AS ranked_pairs
            ORDER BY dimension, doc_id
            LIMIT :limit
        """)

        params = {"limit": limit}
        if goal:
            params["goal"] = goal

        try:
            rows = self.db.execute(query, params).fetchall()
            print(f"Fetched {len(rows)} rows from the database.")
        except Exception as e:
            if self.logger:
                self.logger.log("DocumentPairBuilderError", {"error": str(e)})
            self.db.rollback()
            return {}

        grouped = defaultdict(dict)
        results_by_dimension = defaultdict(list)

        for row in rows:
            key = (row.dimension, row.doc_id)
            grouped[key][row.rank_type] = row

        for (dimension, _), data in grouped.items():
            if "top" in data and "bottom" in data:
                results_by_dimension[dimension].append({
                    "title": data["top"].title,
                    "output_a": data["top"].content,
                    "output_b": data["bottom"].content,
                    "value_a": float(data["top"].score),
                    "value_b": float(data["bottom"].score)
                })

        return dict(results_by_dimension)
