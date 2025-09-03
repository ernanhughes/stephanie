CREATE VIEW scorable_rankings AS
SELECT 
    e.id AS evaluation_id,
    e.created_at,
    e.query_type,
    e.query_id,
    e.target_type AS candidate_type,
    e.target_id AS candidate_id,
    e.scores->>'rank_score' AS rank_score,
    s.dimension,
    s.score AS component_score,
    s.weight
FROM evaluations e
JOIN scores s ON s.evaluation_id = e.id
WHERE e.evaluator_name = 'ScorableRanker';
