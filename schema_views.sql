-- View: public.reasoning_samples_view

-- DROP VIEW public.reasoning_samples_view;

CREATE OR REPLACE VIEW public.reasoning_samples_view
 AS
 SELECT e.id AS evaluation_id,
    e.scorable_id,
    e.scorable_type,
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
    g.goal_text,
    COALESCE(ds.section_text, d.text, h.text, p.final_output_text, pro.response_text) AS scorable_text,
    d.title AS document_title,
    d.summary AS document_summary,
    d.url AS document_url,
    ds.section_name,
    ds.summary AS section_summary,
    COALESCE(jsonb_agg(jsonb_build_object('dimension', s.dimension, 'score', s.score, 'weight', s.weight, 'rationale', s.rationale, 'source', s.source, 'prompt_hash', s.prompt_hash)) FILTER (WHERE s.id IS NOT NULL), '[]'::jsonb) AS scores,
    COALESCE(jsonb_agg(jsonb_build_object('dimension', a.dimension, 'source', a.source, 'raw_score', a.raw_score, 'energy', a.energy, 'q', a.q_value, 'v', a.v_value, 'advantage', a.advantage, 'pi', a.pi_value, 'entropy', a.entropy, 'uncertainty', a.uncertainty, 'td_error', a.td_error, 'expected_return', a.expected_return, 'policy_logits', a.policy_logits, 'extra', a.extra)) FILTER (WHERE a.id IS NOT NULL), '[]'::jsonb) AS attributes
   FROM evaluations e
     LEFT JOIN goals g ON e.goal_id = g.id
     LEFT JOIN documents d ON e.scorable_type = 'document'::text AND e.scorable_id = d.id::text
     LEFT JOIN document_sections ds ON e.scorable_type = 'document_section'::text AND e.scorable_id = ds.id::text
     LEFT JOIN hypotheses h ON e.scorable_type = 'hypothesis'::text AND e.scorable_id = h.id::text
     LEFT JOIN plan_traces p ON e.scorable_type = 'plan_trace'::text AND e.scorable_id = p.id::text
     LEFT JOIN prompts pro ON (e.scorable_type = ANY (ARRAY['prompt'::text, 'response'::text])) AND e.scorable_id = pro.id::text
     LEFT JOIN scores s ON e.id = s.evaluation_id
     LEFT JOIN evaluation_attributes a ON e.id = a.evaluation_id
  GROUP BY e.id, e.pipeline_run_id, g.goal_text, d.id, d.title, d.text, d.summary, d.url, ds.id, ds.section_name, ds.section_text, ds.summary, h.id, p.id, pro.id;

ALTER TABLE public.reasoning_samples_view
    OWNER TO postgres;

