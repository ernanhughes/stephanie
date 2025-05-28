DELETE FROM search_results
WHERE id IN (
    SELECT id
    FROM (
        SELECT
            id,
            ROW_NUMBER() OVER (
                PARTITION BY url
                ORDER BY created_at ASC
            ) AS rn
        FROM search_results
        WHERE url IS NOT NULL
    ) AS duplicates
    WHERE rn > 1
);