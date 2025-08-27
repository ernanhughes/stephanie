# stephanie/scoring/mrq/preference_pair_builder.py

from collections import defaultdict

from sqlalchemy.sql import text


class PreferencePairBuilder:
    """
    Builds preference training pairs from scored documents per dimension.
    Designed for MR.Q or reward model training to rank research/document quality.
    """

    def __init__(self, db, logger=None):
        self.db = db
        self.logger = logger

    def get_training_pairs_by_dimension(self, goal=None, limit=100, dim=None):
        query = text(f"""
            WITH scored_docs AS (
                SELECT
                    s.dimension,
                    s.score,
                    d.id AS doc_id,
                    d.title,
                    d.text,
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
                {"AND s.dimension IN :dims" if dim else ""}
            )
            SELECT
                dimension,
                title,
                text,
                score,
                rank_type,
                doc_id
            FROM (
                SELECT
                    dimension,
                    title,
                    text,
                    score,
                    'top' AS rank_type,
                    doc_id
                FROM scored_docs
                WHERE rank_high = 1
                AND text IS NOT NULL
                AND text <> ''

                UNION ALL

                SELECT
                    dimension,
                    title,
                    text,
                    score,
                    'bottom' AS rank_type,
                    doc_id
                FROM scored_docs
                WHERE rank_low = 1
            ) AS ranked_pairs
            ORDER BY dimension, doc_id
            LIMIT :limit
        """)

        params = {
            "limit": limit or 100
        }
        if dim:
            params["dims"] = tuple(dim)
        if goal:
            params["goal"] = goal  # Currently unused unless you add it to the query.

        # Optional: print full SQL for debugging
        # compiled = query.compile(self.db.bind, compile_kwargs={"literal_binds": True})
        # self.logger.log("SQLQuery", {"query": str(compiled)})
        try:
            rows = self.db.execute(query, params).fetchall()
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
                results_by_dimension[dimension].append(
                    {
                        "title": data["top"].title,
                        "output_a": data["top"].text,
                        "output_b": data["bottom"].text,
                        "value_a": float(data["top"].score),
                        "value_b": float(data["bottom"].score),
                    }
                )

        return dict(results_by_dimension)
