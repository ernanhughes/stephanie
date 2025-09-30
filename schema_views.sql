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


--- view to export scores and attributes along with evaluation and scorable details
CREATE OR REPLACE VIEW evaluation_export_view AS
SELECT 
    e.id AS evaluation_id,
    e.scorable_type,
    e.scorable_id,
    e.agent_name,
    e.model_name,
    e.evaluator_name,
    e.strategy,
    e.reasoning_strategy,
    e.embedding_type,
    e.source,
    e.pipeline_run_id,
    e.symbolic_rule_id,
    e.extra_data,
    e.created_at,
    
    g.goal_text AS goal_text,

    d.title AS document_title,
    d.text AS document_text,
    d.summary AS document_summary,
    d.url AS document_url,
    d.domains AS document_domains,

    ds.section_name,
    ds.section_text,
    ds.summary AS section_summary,
    ds.extra_data AS section_extra,

    COALESCE(
        jsonb_agg(
            jsonb_build_object(
                'dimension', s.dimension,
                'score', s.score,
                'weight', s.weight,
                'rationale', s.rationale,
                'source', s.source,
                'prompt_hash', s.prompt_hash
            )
        ) FILTER (WHERE s.id IS NOT NULL),
        '[]'::jsonb
    ) AS scores,

    COALESCE(
        jsonb_agg(
            jsonb_build_object(
                'dimension', a.dimension,
                'source', a.source,
                'raw_score', a.raw_score,
                'energy', a.energy,
                'q', a.q_value,
                'v', a.v_value,
                'advantage', a.advantage,
                'pi', a.pi_value,
                'entropy', a.entropy,
                'uncertainty', a.uncertainty,
                'td_error', a.td_error,
                'expected_return', a.expected_return,
                'policy_logits', a.policy_logits,
                'extra', a.extra
            )
        ) FILTER (WHERE a.id IS NOT NULL),
        '[]'::jsonb
    ) AS attributes

FROM evaluations e
LEFT JOIN goals g ON e.goal_id = g.id
LEFT JOIN documents d 
    ON e.scorable_type = 'document' 
   AND d.id = NULLIF(e.scorable_id, '')::int
LEFT JOIN document_sections ds 
    ON e.scorable_type = 'document_section' 
   AND ds.id = NULLIF(e.scorable_id, '')::int
LEFT JOIN scores s ON e.id = s.evaluation_id
LEFT JOIN evaluation_attributes a ON e.id = a.evaluation_id

GROUP BY 
    e.id, g.goal_text,
    d.id, d.title, d.text, d.summary, d.url,
    ds.id, ds.section_name, ds.section_text, ds.summary;
