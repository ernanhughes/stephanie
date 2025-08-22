--
-- PostgreSQL database dump
--

-- Dumped from database version 16.8
-- Dumped by pg_dump version 16.8

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

ALTER TABLE IF EXISTS ONLY public.training_stats DROP CONSTRAINT IF EXISTS training_stats_model_version_id_fkey;
ALTER TABLE IF EXISTS ONLY public.training_stats DROP CONSTRAINT IF EXISTS training_stats_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.theorems DROP CONSTRAINT IF EXISTS theorems_embedding_id_fkey;
ALTER TABLE IF EXISTS ONLY public.theorem_cartridges DROP CONSTRAINT IF EXISTS theorem_cartridges_theorem_id_fkey;
ALTER TABLE IF EXISTS ONLY public.theorem_cartridges DROP CONSTRAINT IF EXISTS theorem_cartridges_cartridge_id_fkey;
ALTER TABLE IF EXISTS ONLY public.symbolic_rules DROP CONSTRAINT IF EXISTS symbolic_rules_prompt_id_fkey;
ALTER TABLE IF EXISTS ONLY public.symbolic_rules DROP CONSTRAINT IF EXISTS symbolic_rules_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.sharpening_predictions DROP CONSTRAINT IF EXISTS sharpening_predictions_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.search_results DROP CONSTRAINT IF EXISTS search_results_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.scoring_history DROP CONSTRAINT IF EXISTS scoring_history_pipeline_run_id_fkey;
ALTER TABLE IF EXISTS ONLY public.scoring_history DROP CONSTRAINT IF EXISTS scoring_history_model_version_id_fkey;
ALTER TABLE IF EXISTS ONLY public.scoring_dimensions DROP CONSTRAINT IF EXISTS scoring_dimensions_event_id_fkey;
ALTER TABLE IF EXISTS ONLY public.evaluations DROP CONSTRAINT IF EXISTS scores_hypothesis_id_fkey;
ALTER TABLE IF EXISTS ONLY public.evaluations DROP CONSTRAINT IF EXISTS scores_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.scores DROP CONSTRAINT IF EXISTS scores_evaluation_id_fkey;
ALTER TABLE IF EXISTS ONLY public.evaluation_rule_links DROP CONSTRAINT IF EXISTS score_rule_links_rule_application_id_fkey;
ALTER TABLE IF EXISTS ONLY public.score_attributes DROP CONSTRAINT IF EXISTS score_attributes_score_id_fkey;
ALTER TABLE IF EXISTS ONLY public.rule_applications DROP CONSTRAINT IF EXISTS rule_applications_pipeline_run_id_fkey;
ALTER TABLE IF EXISTS ONLY public.rule_applications DROP CONSTRAINT IF EXISTS rule_applications_hypothesis_id_fkey;
ALTER TABLE IF EXISTS ONLY public.rule_applications DROP CONSTRAINT IF EXISTS rule_applications_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.reflection_deltas DROP CONSTRAINT IF EXISTS reflection_deltas_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.prompts DROP CONSTRAINT IF EXISTS prompts_embedding_id_fkey;
ALTER TABLE IF EXISTS ONLY public.prompt_versions DROP CONSTRAINT IF EXISTS prompt_versions_previous_prompt_id_fkey;
ALTER TABLE IF EXISTS ONLY public.prompt_programs DROP CONSTRAINT IF EXISTS prompt_programs_prompt_id_fkey;
ALTER TABLE IF EXISTS ONLY public.prompt_programs DROP CONSTRAINT IF EXISTS prompt_programs_pipeline_run_id_fkey;
ALTER TABLE IF EXISTS ONLY public.prompt_programs DROP CONSTRAINT IF EXISTS prompt_programs_parent_id_fkey;
ALTER TABLE IF EXISTS ONLY public.prompt_history DROP CONSTRAINT IF EXISTS prompt_history_original_prompt_id_fkey;
ALTER TABLE IF EXISTS ONLY public.prompt_evaluations DROP CONSTRAINT IF EXISTS prompt_evaluations_prompt_id_fkey;
ALTER TABLE IF EXISTS ONLY public.plan_traces DROP CONSTRAINT IF EXISTS plan_traces_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.pipeline_stages DROP CONSTRAINT IF EXISTS pipeline_stages_pipeline_run_id_fkey;
ALTER TABLE IF EXISTS ONLY public.pipeline_stages DROP CONSTRAINT IF EXISTS pipeline_stages_parent_stage_id_fkey;
ALTER TABLE IF EXISTS ONLY public.pipeline_stages DROP CONSTRAINT IF EXISTS pipeline_stages_output_context_id_fkey;
ALTER TABLE IF EXISTS ONLY public.pipeline_stages DROP CONSTRAINT IF EXISTS pipeline_stages_input_context_id_fkey;
ALTER TABLE IF EXISTS ONLY public.pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.method_plans DROP CONSTRAINT IF EXISTS method_plans_parent_plan_id_fkey;
ALTER TABLE IF EXISTS ONLY public.method_plans DROP CONSTRAINT IF EXISTS method_plans_idea_id_fkey;
ALTER TABLE IF EXISTS ONLY public.method_plans DROP CONSTRAINT IF EXISTS method_plans_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.lookaheads DROP CONSTRAINT IF EXISTS lookaheads_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.knowledge_sections DROP CONSTRAINT IF EXISTS knowledge_sections_document_id_fkey;
ALTER TABLE IF EXISTS ONLY public.knowledge_documents DROP CONSTRAINT IF EXISTS knowledge_documents_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.ideas DROP CONSTRAINT IF EXISTS ideas_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.hypotheses DROP CONSTRAINT IF EXISTS hypotheses_source_hypothesis_fkey;
ALTER TABLE IF EXISTS ONLY public.hypotheses DROP CONSTRAINT IF EXISTS hypotheses_prompt_id_fkey;
ALTER TABLE IF EXISTS ONLY public.goal_dimensions DROP CONSTRAINT IF EXISTS goal_dimensions_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.evaluations DROP CONSTRAINT IF EXISTS fk_scores_pipeline_run;
ALTER TABLE IF EXISTS ONLY public.evaluations DROP CONSTRAINT IF EXISTS fk_rule_application;
ALTER TABLE IF EXISTS ONLY public.hypotheses DROP CONSTRAINT IF EXISTS fk_prompt;
ALTER TABLE IF EXISTS ONLY public.context_states DROP CONSTRAINT IF EXISTS fk_pipeline_run;
ALTER TABLE IF EXISTS ONLY public.prompts DROP CONSTRAINT IF EXISTS fk_goal_prompt;
ALTER TABLE IF EXISTS ONLY public.hypotheses DROP CONSTRAINT IF EXISTS fk_goal_hypothesis;
ALTER TABLE IF EXISTS ONLY public.context_states DROP CONSTRAINT IF EXISTS fk_goal;
ALTER TABLE IF EXISTS ONLY public.hypotheses DROP CONSTRAINT IF EXISTS fk_goal;
ALTER TABLE IF EXISTS ONLY public.documents DROP CONSTRAINT IF EXISTS fk_documents_goal_id_goals;
ALTER TABLE IF EXISTS ONLY public.documents DROP CONSTRAINT IF EXISTS fk_documents_embedding;
ALTER TABLE IF EXISTS ONLY public.evaluations DROP CONSTRAINT IF EXISTS fk_document;
ALTER TABLE IF EXISTS ONLY public.execution_steps DROP CONSTRAINT IF EXISTS execution_steps_plan_trace_id_fkey;
ALTER TABLE IF EXISTS ONLY public.execution_steps DROP CONSTRAINT IF EXISTS execution_steps_evaluation_id_fkey;
ALTER TABLE IF EXISTS ONLY public.evaluation_rule_links DROP CONSTRAINT IF EXISTS evaluation_rule_links_evaluation_id_fkey;
ALTER TABLE IF EXISTS ONLY public.evaluation_attributes DROP CONSTRAINT IF EXISTS evaluation_attributes_evaluation_id_fkey;
ALTER TABLE IF EXISTS ONLY public.document_sections DROP CONSTRAINT IF EXISTS document_sections_document_id_fkey;
ALTER TABLE IF EXISTS ONLY public.document_section_domains DROP CONSTRAINT IF EXISTS document_section_domains_document_section_id_fkey;
ALTER TABLE IF EXISTS ONLY public.document_scores DROP CONSTRAINT IF EXISTS document_scores_evaluation_id_fkey;
ALTER TABLE IF EXISTS ONLY public.document_evaluations DROP CONSTRAINT IF EXISTS document_evaluations_document_id_fkey;
ALTER TABLE IF EXISTS ONLY public.document_domains DROP CONSTRAINT IF EXISTS document_domains_document_id_fkey;
ALTER TABLE IF EXISTS ONLY public.cot_patterns DROP CONSTRAINT IF EXISTS cot_patterns_hypothesis_id_fkey;
ALTER TABLE IF EXISTS ONLY public.cot_patterns DROP CONSTRAINT IF EXISTS cot_patterns_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.cot_pattern_stats DROP CONSTRAINT IF EXISTS cot_pattern_stats_hypothesis_id_fkey;
ALTER TABLE IF EXISTS ONLY public.cot_pattern_stats DROP CONSTRAINT IF EXISTS cot_pattern_stats_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.component_interfaces DROP CONSTRAINT IF EXISTS component_interfaces_component_id_fkey;
ALTER TABLE IF EXISTS ONLY public.cartridges DROP CONSTRAINT IF EXISTS cartridges_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.cartridges DROP CONSTRAINT IF EXISTS cartridges_embedding_id_fkey;
ALTER TABLE IF EXISTS ONLY public.cartridge_triples DROP CONSTRAINT IF EXISTS cartridge_triples_cartridge_id_fkey;
ALTER TABLE IF EXISTS ONLY public.cartridge_domains DROP CONSTRAINT IF EXISTS cartridge_domains_cartridge_id_fkey;
ALTER TABLE IF EXISTS ONLY public.belief_cartridges DROP CONSTRAINT IF EXISTS belief_cartridges_goal_id_fkey;
ALTER TABLE IF EXISTS ONLY public.belief_cartridges DROP CONSTRAINT IF EXISTS belief_cartridges_document_id_fkey;
DROP INDEX IF EXISTS public.unique_text_hash;
DROP INDEX IF EXISTS public.idx_training_stats_version;
DROP INDEX IF EXISTS public.idx_training_stats_model;
DROP INDEX IF EXISTS public.idx_training_stats_embedding;
DROP INDEX IF EXISTS public.idx_training_stats_dimension;
DROP INDEX IF EXISTS public.idx_theorem_cartridges_theorem_id;
DROP INDEX IF EXISTS public.idx_theorem_cartridges_cartridge_id;
DROP INDEX IF EXISTS public.idx_prompt_version;
DROP INDEX IF EXISTS public.idx_prompt_strategy;
DROP INDEX IF EXISTS public.idx_prompt_agent;
DROP INDEX IF EXISTS public.idx_plan_traces_trace_id;
DROP INDEX IF EXISTS public.idx_plan_traces_goal_id;
DROP INDEX IF EXISTS public.idx_plan_traces_created_at;
DROP INDEX IF EXISTS public.idx_pipeline_stages_status;
DROP INDEX IF EXISTS public.idx_pipeline_stages_run_id;
DROP INDEX IF EXISTS public.idx_pipeline_stages_parent;
DROP INDEX IF EXISTS public.idx_pipeline_stages_output_context;
DROP INDEX IF EXISTS public.idx_pipeline_stages_input_context;
DROP INDEX IF EXISTS public.idx_pipeline_stages_goal_id;
DROP INDEX IF EXISTS public.idx_nodes_stage_name;
DROP INDEX IF EXISTS public.idx_nodes_pipeline_run_id;
DROP INDEX IF EXISTS public.idx_nodes_goal_id;
DROP INDEX IF EXISTS public.idx_measurements_value_gin;
DROP INDEX IF EXISTS public.idx_measurements_entity_metric;
DROP INDEX IF EXISTS public.idx_measurements_created_at;
DROP INDEX IF EXISTS public.idx_hf_embedding_vector;
DROP INDEX IF EXISTS public.idx_execution_steps_step_order;
DROP INDEX IF EXISTS public.idx_execution_steps_plan_trace_id;
DROP INDEX IF EXISTS public.idx_execution_steps_evaluation_id;
DROP INDEX IF EXISTS public.idx_evaluation_attributes_start_time;
DROP INDEX IF EXISTS public.idx_evaluation_attributes_output_size;
DROP INDEX IF EXISTS public.idx_evaluation_attributes_duration;
ALTER TABLE IF EXISTS ONLY public.worldviews DROP CONSTRAINT IF EXISTS worldviews_pkey;
ALTER TABLE IF EXISTS ONLY public.worldviews DROP CONSTRAINT IF EXISTS worldviews_name_key;
ALTER TABLE IF EXISTS ONLY public.hnet_embeddings DROP CONSTRAINT IF EXISTS unique_text_hash_hnet;
ALTER TABLE IF EXISTS ONLY public.hf_embeddings DROP CONSTRAINT IF EXISTS unique_text_hash_hf;
ALTER TABLE IF EXISTS ONLY public.cartridges DROP CONSTRAINT IF EXISTS unique_source;
ALTER TABLE IF EXISTS ONLY public.document_section_domains DROP CONSTRAINT IF EXISTS unique_document_section_domain;
ALTER TABLE IF EXISTS ONLY public.document_domains DROP CONSTRAINT IF EXISTS unique_document_domain;
ALTER TABLE IF EXISTS ONLY public.unified_mrq_models DROP CONSTRAINT IF EXISTS unified_mrq_models_pkey;
ALTER TABLE IF EXISTS ONLY public.training_stats DROP CONSTRAINT IF EXISTS training_stats_pkey;
ALTER TABLE IF EXISTS ONLY public.theorems DROP CONSTRAINT IF EXISTS theorems_pkey;
ALTER TABLE IF EXISTS ONLY public.theorem_cartridges DROP CONSTRAINT IF EXISTS theorem_cartridges_pkey;
ALTER TABLE IF EXISTS ONLY public.theorem_applications DROP CONSTRAINT IF EXISTS theorem_applications_pkey;
ALTER TABLE IF EXISTS ONLY public.symbolic_rules DROP CONSTRAINT IF EXISTS symbolic_rules_pkey;
ALTER TABLE IF EXISTS ONLY public.summaries DROP CONSTRAINT IF EXISTS summaries_pkey;
ALTER TABLE IF EXISTS ONLY public.sharpening_results DROP CONSTRAINT IF EXISTS sharpening_results_pkey;
ALTER TABLE IF EXISTS ONLY public.sharpening_predictions DROP CONSTRAINT IF EXISTS sharpening_predictions_pkey;
ALTER TABLE IF EXISTS ONLY public.search_results DROP CONSTRAINT IF EXISTS search_results_pkey;
ALTER TABLE IF EXISTS ONLY public.scoring_history DROP CONSTRAINT IF EXISTS scoring_history_pkey;
ALTER TABLE IF EXISTS ONLY public.scoring_events DROP CONSTRAINT IF EXISTS scoring_events_pkey;
ALTER TABLE IF EXISTS ONLY public.scoring_dimensions DROP CONSTRAINT IF EXISTS scoring_dimensions_pkey;
ALTER TABLE IF EXISTS ONLY public.scores DROP CONSTRAINT IF EXISTS scores_pkey1;
ALTER TABLE IF EXISTS ONLY public.evaluations DROP CONSTRAINT IF EXISTS scores_pkey;
ALTER TABLE IF EXISTS ONLY public.evaluation_rule_links DROP CONSTRAINT IF EXISTS score_rule_links_pkey;
ALTER TABLE IF EXISTS ONLY public.score_attributes DROP CONSTRAINT IF EXISTS score_attributes_pkey;
ALTER TABLE IF EXISTS ONLY public.rule_applications DROP CONSTRAINT IF EXISTS rule_applications_pkey;
ALTER TABLE IF EXISTS ONLY public.reports DROP CONSTRAINT IF EXISTS reports_pkey;
ALTER TABLE IF EXISTS ONLY public.reflection_deltas DROP CONSTRAINT IF EXISTS reflection_deltas_pkey;
ALTER TABLE IF EXISTS ONLY public.refinement_events DROP CONSTRAINT IF EXISTS refinement_events_pkey;
ALTER TABLE IF EXISTS ONLY public.ranking_trace DROP CONSTRAINT IF EXISTS ranking_trace_pkey;
ALTER TABLE IF EXISTS ONLY public.protocols DROP CONSTRAINT IF EXISTS protocols_pkey;
ALTER TABLE IF EXISTS ONLY public.prompts DROP CONSTRAINT IF EXISTS prompts_pkey;
ALTER TABLE IF EXISTS ONLY public.prompt_versions DROP CONSTRAINT IF EXISTS prompt_versions_pkey;
ALTER TABLE IF EXISTS ONLY public.prompt_programs DROP CONSTRAINT IF EXISTS prompt_programs_pkey;
ALTER TABLE IF EXISTS ONLY public.prompt_history DROP CONSTRAINT IF EXISTS prompt_history_pkey;
ALTER TABLE IF EXISTS ONLY public.prompt_evaluations DROP CONSTRAINT IF EXISTS prompt_evaluations_pkey;
ALTER TABLE IF EXISTS ONLY public.plan_traces DROP CONSTRAINT IF EXISTS plan_traces_trace_id_key;
ALTER TABLE IF EXISTS ONLY public.plan_traces DROP CONSTRAINT IF EXISTS plan_traces_pkey;
ALTER TABLE IF EXISTS ONLY public.pipeline_stages DROP CONSTRAINT IF EXISTS pipeline_stages_pkey;
ALTER TABLE IF EXISTS ONLY public.pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_run_id_key;
ALTER TABLE IF EXISTS ONLY public.pipeline_runs DROP CONSTRAINT IF EXISTS pipeline_runs_pkey;
ALTER TABLE IF EXISTS ONLY public.nodes DROP CONSTRAINT IF EXISTS nodes_pkey;
ALTER TABLE IF EXISTS ONLY public.mrq_preference_pairs DROP CONSTRAINT IF EXISTS mrq_preference_pairs_pkey;
ALTER TABLE IF EXISTS ONLY public.mrq_memory DROP CONSTRAINT IF EXISTS mrq_memory_pkey;
ALTER TABLE IF EXISTS ONLY public.mrq_evaluations DROP CONSTRAINT IF EXISTS mrq_evaluations_pkey;
ALTER TABLE IF EXISTS ONLY public.model_versions DROP CONSTRAINT IF EXISTS model_versions_pkey;
ALTER TABLE IF EXISTS ONLY public.model_performance DROP CONSTRAINT IF EXISTS model_performance_pkey;
ALTER TABLE IF EXISTS ONLY public.method_plans DROP CONSTRAINT IF EXISTS method_plans_pkey;
ALTER TABLE IF EXISTS ONLY public.memcubes DROP CONSTRAINT IF EXISTS memcubes_pkey;
ALTER TABLE IF EXISTS ONLY public.memcube_versions DROP CONSTRAINT IF EXISTS memcube_versions_pkey;
ALTER TABLE IF EXISTS ONLY public.memcube_transformations DROP CONSTRAINT IF EXISTS memcube_transformations_pkey;
ALTER TABLE IF EXISTS ONLY public.mem_cubes DROP CONSTRAINT IF EXISTS mem_cubes_pkey;
ALTER TABLE IF EXISTS ONLY public.measurements DROP CONSTRAINT IF EXISTS measurements_pkey;
ALTER TABLE IF EXISTS ONLY public.lookaheads DROP CONSTRAINT IF EXISTS lookaheads_pkey;
ALTER TABLE IF EXISTS ONLY public.knowledge_sections DROP CONSTRAINT IF EXISTS knowledge_sections_pkey;
ALTER TABLE IF EXISTS ONLY public.knowledge_documents DROP CONSTRAINT IF EXISTS knowledge_documents_pkey;
ALTER TABLE IF EXISTS ONLY public.ideas DROP CONSTRAINT IF EXISTS ideas_pkey;
ALTER TABLE IF EXISTS ONLY public.hypotheses DROP CONSTRAINT IF EXISTS hypotheses_pkey;
ALTER TABLE IF EXISTS ONLY public.hnet_embeddings DROP CONSTRAINT IF EXISTS hnet_embeddings_pkey;
ALTER TABLE IF EXISTS ONLY public.hf_embeddings DROP CONSTRAINT IF EXISTS hf_embeddings_pkey;
ALTER TABLE IF EXISTS ONLY public.goals DROP CONSTRAINT IF EXISTS goals_pkey;
ALTER TABLE IF EXISTS ONLY public.goal_dimensions DROP CONSTRAINT IF EXISTS goal_dimensions_pkey;
ALTER TABLE IF EXISTS ONLY public.execution_steps DROP CONSTRAINT IF EXISTS execution_steps_pkey;
ALTER TABLE IF EXISTS ONLY public.execution_steps DROP CONSTRAINT IF EXISTS execution_steps_evaluation_id_key;
ALTER TABLE IF EXISTS ONLY public.events DROP CONSTRAINT IF EXISTS events_pkey;
ALTER TABLE IF EXISTS ONLY public.evaluation_attributes DROP CONSTRAINT IF EXISTS evaluation_attributes_pkey;
ALTER TABLE IF EXISTS ONLY public.embeddings DROP CONSTRAINT IF EXISTS embeddings_pkey;
ALTER TABLE IF EXISTS ONLY public.elo_ranking_log DROP CONSTRAINT IF EXISTS elo_ranking_log_pkey;
ALTER TABLE IF EXISTS ONLY public.documents DROP CONSTRAINT IF EXISTS documents_pkey;
ALTER TABLE IF EXISTS ONLY public.document_sections DROP CONSTRAINT IF EXISTS document_sections_pkey;
ALTER TABLE IF EXISTS ONLY public.document_sections DROP CONSTRAINT IF EXISTS document_sections_document_id_section_name_key;
ALTER TABLE IF EXISTS ONLY public.document_section_domains DROP CONSTRAINT IF EXISTS document_section_domains_pkey;
ALTER TABLE IF EXISTS ONLY public.document_scores DROP CONSTRAINT IF EXISTS document_scores_pkey;
ALTER TABLE IF EXISTS ONLY public.document_evaluations DROP CONSTRAINT IF EXISTS document_evaluations_pkey;
ALTER TABLE IF EXISTS ONLY public.document_domains DROP CONSTRAINT IF EXISTS document_domains_pkey;
ALTER TABLE IF EXISTS ONLY public.cot_patterns DROP CONSTRAINT IF EXISTS cot_patterns_pkey;
ALTER TABLE IF EXISTS ONLY public.cot_pattern_stats DROP CONSTRAINT IF EXISTS cot_pattern_stats_pkey;
ALTER TABLE IF EXISTS ONLY public.context_states DROP CONSTRAINT IF EXISTS context_states_pkey;
ALTER TABLE IF EXISTS ONLY public.component_versions DROP CONSTRAINT IF EXISTS component_versions_pkey;
ALTER TABLE IF EXISTS ONLY public.comparison_preferences DROP CONSTRAINT IF EXISTS comparison_preferences_pkey;
ALTER TABLE IF EXISTS ONLY public.cartridges DROP CONSTRAINT IF EXISTS cartridges_pkey;
ALTER TABLE IF EXISTS ONLY public.cartridge_triples DROP CONSTRAINT IF EXISTS cartridge_triples_pkey;
ALTER TABLE IF EXISTS ONLY public.cartridge_domains DROP CONSTRAINT IF EXISTS cartridge_domains_pkey;
ALTER TABLE IF EXISTS ONLY public.belief_graph_versions DROP CONSTRAINT IF EXISTS belief_graph_versions_pkey;
ALTER TABLE IF EXISTS ONLY public.belief_cartridges DROP CONSTRAINT IF EXISTS belief_cartridges_pkey;
ALTER TABLE IF EXISTS public.worldviews ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.unified_mrq_models ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.training_stats ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.theorems ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.theorem_applications ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.symbolic_rules ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.summaries ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.sharpening_results ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.sharpening_predictions ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.search_results ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.scoring_history ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.scoring_events ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.scores ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.score_attributes ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.rule_applications ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.reports ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.reflection_deltas ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.refinement_events ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.ranking_trace ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.prompts ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.prompt_versions ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.prompt_history ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.prompt_evaluations ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.plan_traces ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.pipeline_stages ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.pipeline_runs ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.nodes ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.mrq_preference_pairs ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.mrq_memory ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.mrq_evaluations ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.model_versions ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.model_performance ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.method_plans ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.memcube_transformations ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.mem_cubes ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.measurements ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.lookaheads ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.knowledge_sections ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.knowledge_documents ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.ideas ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.hypotheses ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.hnet_embeddings ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.hf_embeddings ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.goals ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.goal_dimensions ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.execution_steps ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.events ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.evaluations ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.evaluation_rule_links ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.evaluation_attributes ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.embeddings ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.elo_ranking_log ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.documents ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.document_sections ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.document_section_domains ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.document_scores ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.document_evaluations ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.document_domains ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.cot_patterns ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.cot_pattern_stats ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.context_states ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.cartridges ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.cartridge_triples ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.cartridge_domains ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.belief_graph_versions ALTER COLUMN id DROP DEFAULT;
DROP SEQUENCE IF EXISTS public.worldviews_id_seq;
DROP TABLE IF EXISTS public.worldviews;
DROP SEQUENCE IF EXISTS public.unified_mrq_models_id_seq;
DROP TABLE IF EXISTS public.unified_mrq_models;
DROP SEQUENCE IF EXISTS public.training_stats_id_seq;
DROP TABLE IF EXISTS public.training_stats;
DROP SEQUENCE IF EXISTS public.theorems_id_seq;
DROP TABLE IF EXISTS public.theorems;
DROP TABLE IF EXISTS public.theorem_cartridges;
DROP SEQUENCE IF EXISTS public.theorem_applications_id_seq;
DROP TABLE IF EXISTS public.theorem_applications;
DROP SEQUENCE IF EXISTS public.symbolic_rules_id_seq;
DROP TABLE IF EXISTS public.symbolic_rules;
DROP SEQUENCE IF EXISTS public.summaries_id_seq;
DROP TABLE IF EXISTS public.summaries;
DROP SEQUENCE IF EXISTS public.sharpening_results_id_seq;
DROP TABLE IF EXISTS public.sharpening_results;
DROP SEQUENCE IF EXISTS public.sharpening_predictions_id_seq;
DROP TABLE IF EXISTS public.sharpening_predictions;
DROP SEQUENCE IF EXISTS public.search_results_id_seq;
DROP TABLE IF EXISTS public.search_results;
DROP SEQUENCE IF EXISTS public.scoring_history_id_seq;
DROP TABLE IF EXISTS public.scoring_history;
DROP SEQUENCE IF EXISTS public.scoring_events_id_seq;
DROP TABLE IF EXISTS public.scoring_events;
DROP TABLE IF EXISTS public.scoring_dimensions;
DROP SEQUENCE IF EXISTS public.scores_id_seq1;
DROP SEQUENCE IF EXISTS public.scores_id_seq;
DROP TABLE IF EXISTS public.scores;
DROP SEQUENCE IF EXISTS public.score_rule_links_id_seq;
DROP SEQUENCE IF EXISTS public.score_attributes_id_seq;
DROP TABLE IF EXISTS public.score_attributes;
DROP SEQUENCE IF EXISTS public.rule_applications_id_seq;
DROP TABLE IF EXISTS public.rule_applications;
DROP SEQUENCE IF EXISTS public.reports_id_seq;
DROP TABLE IF EXISTS public.reports;
DROP SEQUENCE IF EXISTS public.reflection_deltas_id_seq;
DROP TABLE IF EXISTS public.reflection_deltas;
DROP SEQUENCE IF EXISTS public.refinement_events_id_seq;
DROP TABLE IF EXISTS public.refinement_events;
DROP SEQUENCE IF EXISTS public.ranking_trace_id_seq;
DROP TABLE IF EXISTS public.ranking_trace;
DROP TABLE IF EXISTS public.protocols;
DROP SEQUENCE IF EXISTS public.prompts_id_seq;
DROP TABLE IF EXISTS public.prompts;
DROP SEQUENCE IF EXISTS public.prompt_versions_id_seq;
DROP TABLE IF EXISTS public.prompt_versions;
DROP TABLE IF EXISTS public.prompt_programs;
DROP SEQUENCE IF EXISTS public.prompt_history_id_seq;
DROP TABLE IF EXISTS public.prompt_history;
DROP SEQUENCE IF EXISTS public.prompt_evaluations_id_seq;
DROP TABLE IF EXISTS public.prompt_evaluations;
DROP SEQUENCE IF EXISTS public.plan_traces_id_seq;
DROP TABLE IF EXISTS public.plan_traces;
DROP SEQUENCE IF EXISTS public.pipeline_stages_id_seq;
DROP TABLE IF EXISTS public.pipeline_stages;
DROP SEQUENCE IF EXISTS public.pipeline_runs_id_seq;
DROP TABLE IF EXISTS public.pipeline_runs;
DROP SEQUENCE IF EXISTS public.nodes_id_seq;
DROP TABLE IF EXISTS public.nodes;
DROP SEQUENCE IF EXISTS public.mrq_preference_pairs_id_seq;
DROP TABLE IF EXISTS public.mrq_preference_pairs;
DROP SEQUENCE IF EXISTS public.mrq_memory_id_seq;
DROP TABLE IF EXISTS public.mrq_memory;
DROP SEQUENCE IF EXISTS public.mrq_evaluations_id_seq;
DROP TABLE IF EXISTS public.mrq_evaluations;
DROP SEQUENCE IF EXISTS public.model_versions_id_seq;
DROP TABLE IF EXISTS public.model_versions;
DROP SEQUENCE IF EXISTS public.model_performance_id_seq;
DROP TABLE IF EXISTS public.model_performance;
DROP SEQUENCE IF EXISTS public.method_plans_id_seq;
DROP TABLE IF EXISTS public.method_plans;
DROP TABLE IF EXISTS public.memcubes;
DROP TABLE IF EXISTS public.memcube_versions;
DROP SEQUENCE IF EXISTS public.memcube_transformations_id_seq;
DROP TABLE IF EXISTS public.memcube_transformations;
DROP SEQUENCE IF EXISTS public.mem_cubes_id_seq;
DROP TABLE IF EXISTS public.mem_cubes;
DROP SEQUENCE IF EXISTS public.measurements_id_seq;
DROP TABLE IF EXISTS public.measurements;
DROP SEQUENCE IF EXISTS public.lookaheads_id_seq;
DROP TABLE IF EXISTS public.lookaheads;
DROP SEQUENCE IF EXISTS public.knowledge_sections_id_seq;
DROP TABLE IF EXISTS public.knowledge_sections;
DROP SEQUENCE IF EXISTS public.knowledge_documents_id_seq;
DROP TABLE IF EXISTS public.knowledge_documents;
DROP SEQUENCE IF EXISTS public.ideas_id_seq;
DROP TABLE IF EXISTS public.ideas;
DROP SEQUENCE IF EXISTS public.hypotheses_id_seq;
DROP TABLE IF EXISTS public.hypotheses;
DROP SEQUENCE IF EXISTS public.hnet_embeddings_id_seq;
DROP TABLE IF EXISTS public.hnet_embeddings;
DROP SEQUENCE IF EXISTS public.hf_embeddings_id_seq;
DROP TABLE IF EXISTS public.hf_embeddings;
DROP SEQUENCE IF EXISTS public.goals_id_seq;
DROP TABLE IF EXISTS public.goals;
DROP SEQUENCE IF EXISTS public.goal_dimensions_id_seq;
DROP TABLE IF EXISTS public.goal_dimensions;
DROP SEQUENCE IF EXISTS public.execution_steps_id_seq;
DROP TABLE IF EXISTS public.execution_steps;
DROP SEQUENCE IF EXISTS public.events_id_seq;
DROP TABLE IF EXISTS public.events;
DROP TABLE IF EXISTS public.evaluations;
DROP TABLE IF EXISTS public.evaluation_rule_links;
DROP SEQUENCE IF EXISTS public.evaluation_attributes_id_seq;
DROP TABLE IF EXISTS public.evaluation_attributes;
DROP SEQUENCE IF EXISTS public.embeddings_id_seq;
DROP TABLE IF EXISTS public.embeddings;
DROP SEQUENCE IF EXISTS public.elo_ranking_log_id_seq;
DROP TABLE IF EXISTS public.elo_ranking_log;
DROP SEQUENCE IF EXISTS public.documents_id_seq;
DROP TABLE IF EXISTS public.documents;
DROP SEQUENCE IF EXISTS public.document_sections_id_seq;
DROP TABLE IF EXISTS public.document_sections;
DROP SEQUENCE IF EXISTS public.document_section_domains_id_seq;
DROP TABLE IF EXISTS public.document_section_domains;
DROP SEQUENCE IF EXISTS public.document_scores_id_seq;
DROP TABLE IF EXISTS public.document_scores;
DROP SEQUENCE IF EXISTS public.document_evaluations_id_seq;
DROP TABLE IF EXISTS public.document_evaluations;
DROP SEQUENCE IF EXISTS public.document_domains_id_seq;
DROP TABLE IF EXISTS public.document_domains;
DROP SEQUENCE IF EXISTS public.cot_patterns_id_seq;
DROP TABLE IF EXISTS public.cot_patterns;
DROP SEQUENCE IF EXISTS public.cot_pattern_stats_id_seq;
DROP TABLE IF EXISTS public.cot_pattern_stats;
DROP SEQUENCE IF EXISTS public.context_states_id_seq;
DROP TABLE IF EXISTS public.context_states;
DROP TABLE IF EXISTS public.component_versions;
DROP TABLE IF EXISTS public.component_interfaces;
DROP TABLE IF EXISTS public.comparison_preferences;
DROP SEQUENCE IF EXISTS public.cartridges_id_seq;
DROP TABLE IF EXISTS public.cartridges;
DROP SEQUENCE IF EXISTS public.cartridge_triples_id_seq;
DROP TABLE IF EXISTS public.cartridge_triples;
DROP SEQUENCE IF EXISTS public.cartridge_domains_id_seq;
DROP TABLE IF EXISTS public.cartridge_domains;
DROP SEQUENCE IF EXISTS public.belief_graph_versions_id_seq;
DROP TABLE IF EXISTS public.belief_graph_versions;
DROP TABLE IF EXISTS public.belief_cartridges;
DROP EXTENSION IF EXISTS vector;
DROP EXTENSION IF EXISTS pgcrypto;
--
-- Name: pgcrypto; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public;


--
-- Name: EXTENSION pgcrypto; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: belief_cartridges; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.belief_cartridges (
    id text NOT NULL,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now(),
    source_id text,
    source_type text NOT NULL,
    source_url text,
    markdown_content text NOT NULL,
    is_active boolean DEFAULT true,
    idea_payload jsonb,
    goal_tags text[] DEFAULT ARRAY[]::text[],
    domain_tags text[] DEFAULT ARRAY[]::text[],
    derived_from jsonb DEFAULT '[]'::jsonb,
    applied_in jsonb DEFAULT '[]'::jsonb,
    version integer DEFAULT 1,
    memcube_id text,
    goal_id integer,
    document_id integer,
    CONSTRAINT belief_cartridges_source_type_check CHECK ((source_type = ANY (ARRAY['paper'::text, 'blog'::text, 'experiment'::text, 'pipeline'::text, 'manual'::text])))
);


--
-- Name: belief_graph_versions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.belief_graph_versions (
    id integer NOT NULL,
    goal text NOT NULL,
    node_count integer,
    edge_count integer,
    avg_strength double precision,
    avg_relevance double precision,
    contradictions integer,
    theorems integer,
    created_at timestamp without time zone DEFAULT now(),
    model_path text
);


--
-- Name: belief_graph_versions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.belief_graph_versions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: belief_graph_versions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.belief_graph_versions_id_seq OWNED BY public.belief_graph_versions.id;


--
-- Name: cartridge_domains; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cartridge_domains (
    id integer NOT NULL,
    cartridge_id integer NOT NULL,
    domain character varying NOT NULL,
    score double precision NOT NULL
);


--
-- Name: cartridge_domains_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cartridge_domains_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: cartridge_domains_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cartridge_domains_id_seq OWNED BY public.cartridge_domains.id;


--
-- Name: cartridge_triples; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cartridge_triples (
    id integer NOT NULL,
    cartridge_id integer NOT NULL,
    subject text NOT NULL,
    predicate text NOT NULL,
    object text NOT NULL,
    confidence double precision DEFAULT 1.0,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: cartridge_triples_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cartridge_triples_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: cartridge_triples_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cartridge_triples_id_seq OWNED BY public.cartridge_triples.id;


--
-- Name: cartridges; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cartridges (
    id integer NOT NULL,
    goal_id integer,
    source_type text NOT NULL,
    source_uri text,
    markdown_content text NOT NULL,
    embedding_id integer,
    title text,
    summary text,
    sections jsonb,
    triples jsonb,
    domain_tags jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: cartridges_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cartridges_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: cartridges_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cartridges_id_seq OWNED BY public.cartridges.id;


--
-- Name: comparison_preferences; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.comparison_preferences (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    goal_id integer NOT NULL,
    preferred_tag text NOT NULL,
    rejected_tag text NOT NULL,
    preferred_run_id uuid NOT NULL,
    rejected_run_id uuid NOT NULL,
    preferred_score double precision,
    rejected_score double precision,
    dimension_scores jsonb,
    reason text,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    source text
);


--
-- Name: component_interfaces; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.component_interfaces (
    component_id text,
    protocol text NOT NULL,
    implemented boolean DEFAULT true,
    last_checked timestamp without time zone DEFAULT now()
);


--
-- Name: component_versions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.component_versions (
    id text NOT NULL,
    name text NOT NULL,
    protocol text NOT NULL,
    class_path text NOT NULL,
    version text NOT NULL,
    config jsonb,
    performance jsonb,
    active boolean DEFAULT true,
    sensitivity text,
    created_at timestamp without time zone DEFAULT now(),
    last_used timestamp without time zone,
    usage_count integer DEFAULT 0,
    metadata jsonb,
    CONSTRAINT component_versions_sensitivity_check CHECK ((sensitivity = ANY (ARRAY['public'::text, 'internal'::text, 'confidential'::text, 'restricted'::text])))
);


--
-- Name: context_states; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.context_states (
    id integer NOT NULL,
    run_id text NOT NULL,
    stage_name text NOT NULL,
    version integer DEFAULT 1,
    context jsonb NOT NULL,
    preferences jsonb,
    feedback jsonb,
    extra_data jsonb DEFAULT '{}'::jsonb,
    "timestamp" timestamp with time zone DEFAULT now(),
    is_current boolean DEFAULT true,
    pipeline_run_id integer,
    goal_id integer,
    trace jsonb,
    token_count integer
);


--
-- Name: context_states_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.context_states_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: context_states_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.context_states_id_seq OWNED BY public.context_states.id;


--
-- Name: cot_pattern_stats; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cot_pattern_stats (
    id integer NOT NULL,
    goal_id integer,
    hypothesis_id integer,
    model_name text NOT NULL,
    agent_name text NOT NULL,
    dimension text NOT NULL,
    label text NOT NULL,
    confidence_score double precision,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: cot_pattern_stats_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cot_pattern_stats_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: cot_pattern_stats_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cot_pattern_stats_id_seq OWNED BY public.cot_pattern_stats.id;


--
-- Name: cot_patterns; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cot_patterns (
    id integer NOT NULL,
    goal_id integer,
    hypothesis_id integer,
    model_name text,
    agent_name text,
    dimension text,
    label text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: cot_patterns_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cot_patterns_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: cot_patterns_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cot_patterns_id_seq OWNED BY public.cot_patterns.id;


--
-- Name: document_domains; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_domains (
    id integer NOT NULL,
    document_id integer NOT NULL,
    domain text NOT NULL,
    score double precision NOT NULL,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: document_domains_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.document_domains_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: document_domains_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.document_domains_id_seq OWNED BY public.document_domains.id;


--
-- Name: document_evaluations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_evaluations (
    id integer NOT NULL,
    document_id integer NOT NULL,
    agent_name text,
    model_name text,
    evaluator_name text,
    strategy text,
    scores jsonb DEFAULT '{}'::jsonb,
    extra_data jsonb,
    created_at timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: document_evaluations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.document_evaluations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: document_evaluations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.document_evaluations_id_seq OWNED BY public.document_evaluations.id;


--
-- Name: document_scores; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_scores (
    id integer NOT NULL,
    evaluation_id integer NOT NULL,
    dimension character varying NOT NULL,
    score double precision,
    weight double precision,
    rationale text
);


--
-- Name: document_scores_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.document_scores_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: document_scores_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.document_scores_id_seq OWNED BY public.document_scores.id;


--
-- Name: document_section_domains; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_section_domains (
    id integer NOT NULL,
    document_section_id integer NOT NULL,
    domain text NOT NULL,
    score double precision NOT NULL
);


--
-- Name: document_section_domains_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.document_section_domains_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: document_section_domains_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.document_section_domains_id_seq OWNED BY public.document_section_domains.id;


--
-- Name: document_sections; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_sections (
    id integer NOT NULL,
    document_id integer NOT NULL,
    section_name text NOT NULL,
    section_text text NOT NULL,
    source text DEFAULT 'unstructured+llm'::text,
    summary text,
    embedding json,
    extra_data json,
    domains text[]
);


--
-- Name: document_sections_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.document_sections_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: document_sections_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.document_sections_id_seq OWNED BY public.document_sections.id;


--
-- Name: documents; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.documents (
    id integer NOT NULL,
    title text NOT NULL,
    source text NOT NULL,
    external_id text,
    url text,
    content text,
    date_added timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    summary text,
    goal_id integer,
    domain_label text,
    domains text[],
    embedding_id integer
);


--
-- Name: documents_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.documents_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: documents_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.documents_id_seq OWNED BY public.documents.id;


--
-- Name: elo_ranking_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.elo_ranking_log (
    id integer NOT NULL,
    run_id text,
    hypothesis text,
    prompt_version integer,
    prompt_strategy text,
    score integer,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: elo_ranking_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.elo_ranking_log_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: elo_ranking_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.elo_ranking_log_id_seq OWNED BY public.elo_ranking_log.id;


--
-- Name: embeddings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.embeddings (
    id integer NOT NULL,
    text text,
    embedding public.vector(1024),
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    text_hash text
);


--
-- Name: embeddings_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.embeddings_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: embeddings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.embeddings_id_seq OWNED BY public.embeddings.id;


--
-- Name: evaluation_attributes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.evaluation_attributes (
    id integer NOT NULL,
    evaluation_id integer NOT NULL,
    dimension text NOT NULL,
    source text NOT NULL,
    raw_score real,
    energy real,
    uncertainty real,
    advantage real,
    pi_value real,
    q_value real,
    v_value real,
    extra json,
    entropy double precision,
    td_error double precision,
    expected_return double precision,
    policy_logits json,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    start_time double precision,
    end_time double precision,
    duration double precision,
    error jsonb,
    output_keys jsonb,
    output_size integer,
    zsa jsonb,
    CONSTRAINT valid_timing CHECK ((((start_time IS NULL) AND (end_time IS NULL)) OR ((start_time IS NOT NULL) AND (end_time IS NULL)) OR ((start_time IS NULL) AND (end_time IS NOT NULL)) OR (start_time <= end_time)))
);


--
-- Name: COLUMN evaluation_attributes.start_time; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.evaluation_attributes.start_time IS 'Timestamp (seconds since epoch) when step execution started';


--
-- Name: COLUMN evaluation_attributes.end_time; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.evaluation_attributes.end_time IS 'Timestamp (seconds since epoch) when step execution ended';


--
-- Name: COLUMN evaluation_attributes.duration; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.evaluation_attributes.duration IS 'Execution duration in seconds';


--
-- Name: COLUMN evaluation_attributes.error; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.evaluation_attributes.error IS 'Error information when step fails (type, message, traceback)';


--
-- Name: COLUMN evaluation_attributes.output_keys; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.evaluation_attributes.output_keys IS 'List of context keys produced by this step';


--
-- Name: COLUMN evaluation_attributes.output_size; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.evaluation_attributes.output_size IS 'Size of output context in bytes';


--
-- Name: COLUMN evaluation_attributes.zsa; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.evaluation_attributes.zsa IS 'ZSA representation from SICQL model';


--
-- Name: evaluation_attributes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.evaluation_attributes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: evaluation_attributes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.evaluation_attributes_id_seq OWNED BY public.evaluation_attributes.id;


--
-- Name: evaluation_rule_links; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.evaluation_rule_links (
    id integer NOT NULL,
    evaluation_id integer NOT NULL,
    rule_application_id integer NOT NULL,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: evaluations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.evaluations (
    id integer NOT NULL,
    goal_id integer,
    hypothesis_id integer,
    agent_name text NOT NULL,
    model_name text NOT NULL,
    evaluator_name text NOT NULL,
    strategy text,
    reasoning_strategy text,
    run_id text,
    extra_data jsonb DEFAULT '{}'::jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    symbolic_rule_id integer,
    rule_application_id integer,
    pipeline_run_id integer,
    scores json DEFAULT '{}'::json,
    document_id integer,
    target_type text,
    target_id text,
    belief_cartridge_id text,
    embedding_type text,
    source text
);


--
-- Name: events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.events (
    id integer NOT NULL,
    event_type text NOT NULL,
    icon character varying(4) DEFAULT '📦'::character varying,
    data text NOT NULL,
    embedding public.vector(1024),
    hidden boolean DEFAULT false,
    "timestamp" timestamp with time zone DEFAULT now()
);


--
-- Name: events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.events_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.events_id_seq OWNED BY public.events.id;


--
-- Name: execution_steps; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.execution_steps (
    id integer NOT NULL,
    plan_trace_id integer NOT NULL,
    step_order integer NOT NULL,
    step_id text NOT NULL,
    description text NOT NULL,
    output_text text NOT NULL,
    output_embedding_id integer,
    evaluation_id integer,
    meta jsonb,
    created_at timestamp without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    step_type text DEFAULT 'action'::text,
    input_text text
);


--
-- Name: execution_steps_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.execution_steps_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: execution_steps_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.execution_steps_id_seq OWNED BY public.execution_steps.id;


--
-- Name: goal_dimensions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.goal_dimensions (
    id integer NOT NULL,
    goal_id integer NOT NULL,
    dimension text NOT NULL,
    rank integer DEFAULT 0,
    source text DEFAULT 'llm'::text,
    similarity_score double precision,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: goal_dimensions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.goal_dimensions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: goal_dimensions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.goal_dimensions_id_seq OWNED BY public.goal_dimensions.id;


--
-- Name: goals; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.goals (
    id integer NOT NULL,
    goal_text text NOT NULL,
    goal_type text,
    focus_area text,
    strategy text,
    llm_suggested_strategy text,
    source text DEFAULT 'user'::text,
    created_at timestamp without time zone DEFAULT now(),
    goal_category character varying DEFAULT 'analyze'::character varying,
    difficulty character varying DEFAULT 'medium'::character varying
);


--
-- Name: goals_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.goals_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: goals_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.goals_id_seq OWNED BY public.goals.id;


--
-- Name: hf_embeddings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.hf_embeddings (
    id integer NOT NULL,
    text text,
    embedding public.vector(1024),
    created_at timestamp with time zone DEFAULT now(),
    text_hash text
);


--
-- Name: hf_embeddings_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.hf_embeddings_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: hf_embeddings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.hf_embeddings_id_seq OWNED BY public.hf_embeddings.id;


--
-- Name: hnet_embeddings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.hnet_embeddings (
    id integer NOT NULL,
    text text,
    embedding public.vector(1024),
    created_at timestamp with time zone DEFAULT now(),
    text_hash text
);


--
-- Name: hnet_embeddings_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.hnet_embeddings_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: hnet_embeddings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.hnet_embeddings_id_seq OWNED BY public.hnet_embeddings.id;


--
-- Name: hypotheses; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.hypotheses (
    id integer NOT NULL,
    text text NOT NULL,
    confidence double precision DEFAULT 0.0,
    review text,
    elo_rating double precision DEFAULT 1000.0,
    embedding public.vector(1024),
    features jsonb,
    prompt_id integer,
    source_hypothesis_id integer,
    strategy text,
    version integer DEFAULT 1,
    source text,
    enabled boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    reflection text,
    goal_id integer,
    pipeline_signature text,
    pipeline_run_id integer
);


--
-- Name: hypotheses_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.hypotheses_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: hypotheses_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.hypotheses_id_seq OWNED BY public.hypotheses.id;


--
-- Name: ideas; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ideas (
    id integer NOT NULL,
    idea_text text NOT NULL,
    parent_goal text,
    focus_area text,
    strategy text,
    source text,
    origin text,
    extra_data json,
    goal_id integer,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: ideas_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ideas_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ideas_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ideas_id_seq OWNED BY public.ideas.id;


--
-- Name: knowledge_documents; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.knowledge_documents (
    id integer NOT NULL,
    title text NOT NULL,
    summary text,
    text text,
    url text,
    external_id text,
    source text,
    goal_id integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    domain_label text,
    content text
);


--
-- Name: knowledge_documents_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.knowledge_documents_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: knowledge_documents_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.knowledge_documents_id_seq OWNED BY public.knowledge_documents.id;


--
-- Name: knowledge_sections; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.knowledge_sections (
    id integer NOT NULL,
    document_id integer,
    section_title text,
    section_text text NOT NULL,
    embedding public.vector,
    domain text,
    domain_score double precision,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: knowledge_sections_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.knowledge_sections_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: knowledge_sections_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.knowledge_sections_id_seq OWNED BY public.knowledge_sections.id;


--
-- Name: lookaheads; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.lookaheads (
    id integer NOT NULL,
    goal_id integer,
    agent_name text NOT NULL,
    model_name text NOT NULL,
    input_pipeline text[],
    suggested_pipeline text[],
    rationale text,
    reflection text,
    backup_plans text[],
    extra_data jsonb DEFAULT '{}'::jsonb,
    run_id text,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: lookaheads_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.lookaheads_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: lookaheads_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.lookaheads_id_seq OWNED BY public.lookaheads.id;


--
-- Name: measurements; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.measurements (
    id integer NOT NULL,
    entity_type text NOT NULL,
    entity_id integer NOT NULL,
    metric_name text NOT NULL,
    value jsonb NOT NULL,
    context jsonb,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: measurements_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.measurements_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: measurements_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.measurements_id_seq OWNED BY public.measurements.id;


--
-- Name: mem_cubes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mem_cubes (
    id integer NOT NULL,
    scorable_type text NOT NULL,
    scorable_id integer NOT NULL,
    state text DEFAULT 'raw'::text,
    tags text[],
    notes text,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now()
);


--
-- Name: mem_cubes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.mem_cubes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: mem_cubes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.mem_cubes_id_seq OWNED BY public.mem_cubes.id;


--
-- Name: memcube_transformations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.memcube_transformations (
    id integer NOT NULL,
    source_cube_id text NOT NULL,
    target_cube_id text NOT NULL,
    transformation_type text,
    confidence double precision,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: memcube_transformations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.memcube_transformations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: memcube_transformations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.memcube_transformations_id_seq OWNED BY public.memcube_transformations.id;


--
-- Name: memcube_versions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.memcube_versions (
    id text NOT NULL,
    cube_id text NOT NULL,
    scorable_type text NOT NULL,
    content_hash text,
    version text NOT NULL,
    created_at timestamp without time zone DEFAULT now(),
    last_modified timestamp without time zone DEFAULT now(),
    sensitivity text,
    usage_count integer DEFAULT 0,
    extra_data jsonb,
    CONSTRAINT memcube_versions_sensitivity_check CHECK ((sensitivity = ANY (ARRAY['public'::text, 'internal'::text, 'confidential'::text, 'restricted'::text, 'archived'::text])))
);


--
-- Name: memcubes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.memcubes (
    id text NOT NULL,
    scorable_id bigint NOT NULL,
    scorable_type text NOT NULL,
    content text NOT NULL,
    dimension text,
    original_score double precision,
    refined_score double precision,
    refined_content text,
    version text NOT NULL,
    source text,
    model text,
    priority integer DEFAULT 5,
    sensitivity text DEFAULT 'public'::text,
    ttl integer,
    usage_count integer DEFAULT 0,
    extra_data jsonb,
    created_at timestamp without time zone DEFAULT now(),
    last_modified timestamp without time zone DEFAULT now()
);


--
-- Name: method_plans; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.method_plans (
    id integer NOT NULL,
    idea_text text NOT NULL,
    idea_id integer,
    goal_id integer NOT NULL,
    research_objective text NOT NULL,
    key_components jsonb,
    experimental_plan text,
    hypothesis_mapping jsonb,
    search_strategy text,
    knowledge_gaps text,
    next_steps text,
    task_description text,
    baseline_method text,
    literature_summary text,
    code_plan text,
    focus_area text,
    strategy text,
    score_novelty double precision,
    score_feasibility double precision,
    score_impact double precision,
    score_alignment double precision,
    evolution_level integer DEFAULT 0,
    parent_plan_id integer,
    is_refinement boolean DEFAULT false,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: method_plans_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.method_plans_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: method_plans_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.method_plans_id_seq OWNED BY public.method_plans.id;


--
-- Name: model_performance; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_performance (
    id integer NOT NULL,
    model_name text NOT NULL,
    task_type text NOT NULL,
    prompt_strategy text NOT NULL,
    preference_used text[],
    reward double precision NOT NULL,
    confidence_score double precision,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: model_performance_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.model_performance_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: model_performance_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.model_performance_id_seq OWNED BY public.model_performance.id;


--
-- Name: model_versions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_versions (
    id integer NOT NULL,
    model_type text NOT NULL,
    target_type text NOT NULL,
    dimension text NOT NULL,
    version text NOT NULL,
    trained_on jsonb,
    performance jsonb,
    created_at timestamp without time zone DEFAULT now(),
    active boolean DEFAULT true,
    extra_data jsonb,
    model_path text,
    encoder_path text,
    tuner_path text,
    scaler_path text,
    meta_path text,
    description text,
    source text DEFAULT 'user'::text
);


--
-- Name: model_versions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.model_versions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: model_versions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.model_versions_id_seq OWNED BY public.model_versions.id;


--
-- Name: mrq_evaluations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mrq_evaluations (
    id integer NOT NULL,
    goal text NOT NULL,
    prompt text NOT NULL,
    output_a text NOT NULL,
    output_b text NOT NULL,
    winner text NOT NULL,
    score_a double precision NOT NULL,
    score_b double precision NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: mrq_evaluations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.mrq_evaluations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: mrq_evaluations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.mrq_evaluations_id_seq OWNED BY public.mrq_evaluations.id;


--
-- Name: mrq_memory; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mrq_memory (
    id integer NOT NULL,
    goal text NOT NULL,
    strategy text NOT NULL,
    prompt text NOT NULL,
    response text NOT NULL,
    reward double precision NOT NULL,
    embedding public.vector(1024),
    extra_data jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: mrq_memory_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.mrq_memory_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: mrq_memory_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.mrq_memory_id_seq OWNED BY public.mrq_memory.id;


--
-- Name: mrq_preference_pairs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mrq_preference_pairs (
    id integer NOT NULL,
    goal text NOT NULL,
    prompt text NOT NULL,
    output_a text NOT NULL,
    output_b text NOT NULL,
    preferred text NOT NULL,
    fmt_a text,
    fmt_b text,
    difficulty text,
    source text,
    run_id text,
    features jsonb,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: mrq_preference_pairs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.mrq_preference_pairs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: mrq_preference_pairs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.mrq_preference_pairs_id_seq OWNED BY public.mrq_preference_pairs.id;


--
-- Name: nodes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.nodes (
    id integer NOT NULL,
    goal_id character varying NOT NULL,
    pipeline_run_id integer,
    stage_name character varying,
    config json,
    hypothesis text,
    metric double precision,
    valid boolean DEFAULT true,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: nodes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.nodes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: nodes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.nodes_id_seq OWNED BY public.nodes.id;


--
-- Name: pipeline_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pipeline_runs (
    id integer NOT NULL,
    goal_id integer,
    run_id text NOT NULL,
    pipeline jsonb,
    strategy text,
    model_name text,
    run_config jsonb,
    lookahead_context jsonb,
    symbolic_suggestion jsonb,
    extra_data jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    pipeline_text_backup text,
    name text,
    tag text,
    description text,
    embedding_type text,
    embedding_dimensions integer
);


--
-- Name: pipeline_runs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.pipeline_runs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: pipeline_runs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.pipeline_runs_id_seq OWNED BY public.pipeline_runs.id;


--
-- Name: pipeline_stages; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pipeline_stages (
    id integer NOT NULL,
    stage_name character varying NOT NULL,
    agent_class character varying NOT NULL,
    protocol_used character varying NOT NULL,
    goal_id character varying,
    run_id character varying NOT NULL,
    pipeline_run_id integer,
    parent_stage_id integer,
    input_context_id integer,
    output_context_id integer,
    "timestamp" timestamp without time zone DEFAULT now() NOT NULL,
    status character varying NOT NULL,
    score numeric,
    confidence numeric,
    symbols_applied jsonb,
    extra_data jsonb,
    exportable boolean,
    reusable boolean,
    invalidated boolean
);


--
-- Name: TABLE pipeline_stages; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON TABLE public.pipeline_stages IS 'Records each step in Stephanie’s reasoning process with full traceability.';


--
-- Name: COLUMN pipeline_stages.stage_name; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.pipeline_stages.stage_name IS 'Name of this pipeline stage (e.g., "generation", "judge")';


--
-- Name: COLUMN pipeline_stages.agent_class; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.pipeline_stages.agent_class IS 'Fully qualified name of the agent used';


--
-- Name: COLUMN pipeline_stages.protocol_used; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.pipeline_stages.protocol_used IS 'Protocol type used (e.g., "g3ps_search", "cot")';


--
-- Name: COLUMN pipeline_stages.goal_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.pipeline_stages.goal_id IS 'Optional link to the associated goal ID';


--
-- Name: COLUMN pipeline_stages.run_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.pipeline_stages.run_id IS 'Unique identifier for the current pipeline run';


--
-- Name: COLUMN pipeline_stages.pipeline_run_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.pipeline_stages.pipeline_run_id IS 'Foreign key to pipeline_runs table';


--
-- Name: COLUMN pipeline_stages.parent_stage_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.pipeline_stages.parent_stage_id IS 'Reference to prior stage for tracing reasoning paths';


--
-- Name: COLUMN pipeline_stages.input_context_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.pipeline_stages.input_context_id IS 'Context before running this stage';


--
-- Name: COLUMN pipeline_stages.output_context_id; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.pipeline_stages.output_context_id IS 'Context after running this stage';


--
-- Name: COLUMN pipeline_stages.status; Type: COMMENT; Schema: public; Owner: -
--

COMMENT ON COLUMN public.pipeline_stages.status IS 'Stage outcome: accepted, rejected, retry, partial, pending';


--
-- Name: pipeline_stages_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.pipeline_stages_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: pipeline_stages_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.pipeline_stages_id_seq OWNED BY public.pipeline_stages.id;


--
-- Name: plan_traces; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.plan_traces (
    id integer NOT NULL,
    trace_id text NOT NULL,
    goal_id integer,
    goal_embedding_id integer,
    plan_signature text NOT NULL,
    final_output_text text NOT NULL,
    final_output_embedding_id integer,
    target_epistemic_quality double precision,
    target_epistemic_quality_source text,
    meta jsonb,
    created_at timestamp without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL,
    updated_at timestamp without time zone DEFAULT (now() AT TIME ZONE 'utc'::text) NOT NULL
);


--
-- Name: plan_traces_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.plan_traces_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: plan_traces_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.plan_traces_id_seq OWNED BY public.plan_traces.id;


--
-- Name: prompt_evaluations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_evaluations (
    id integer NOT NULL,
    prompt_id integer NOT NULL,
    benchmark_name text NOT NULL,
    score double precision,
    metrics jsonb DEFAULT '{}'::jsonb,
    dataset_hash text,
    evaluator text DEFAULT 'auto'::text,
    notes text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: prompt_evaluations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompt_evaluations_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: prompt_evaluations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompt_evaluations_id_seq OWNED BY public.prompt_evaluations.id;


--
-- Name: prompt_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_history (
    id integer NOT NULL,
    original_prompt_id integer,
    prompt_text text NOT NULL,
    agent_name text NOT NULL,
    strategy text NOT NULL,
    prompt_key text NOT NULL,
    output_key text,
    input_key text,
    extraction_regex text,
    version integer DEFAULT 1,
    source text,
    is_current boolean DEFAULT false,
    config jsonb DEFAULT '{}'::jsonb,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: prompt_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompt_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: prompt_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompt_history_id_seq OWNED BY public.prompt_history.id;


--
-- Name: prompt_programs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_programs (
    id text NOT NULL,
    goal text NOT NULL,
    template text NOT NULL,
    inputs json DEFAULT '{}'::json,
    version integer DEFAULT 1,
    parent_id text,
    prompt_id integer,
    pipeline_run_id integer,
    strategy text DEFAULT 'default'::text,
    prompt_text text,
    hypothesis text,
    score double precision,
    rationale text,
    mutation_type text,
    execution_trace text,
    extra_data json DEFAULT '{}'::json
);


--
-- Name: prompt_versions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_versions (
    id integer NOT NULL,
    agent_name text NOT NULL,
    prompt_key text NOT NULL,
    prompt_text text NOT NULL,
    previous_prompt_id integer,
    strategy text,
    version integer NOT NULL,
    source text,
    score_improvement double precision,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: prompt_versions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompt_versions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: prompt_versions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompt_versions_id_seq OWNED BY public.prompt_versions.id;


--
-- Name: prompts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompts (
    id integer NOT NULL,
    agent_name text NOT NULL,
    prompt_key text NOT NULL,
    prompt_text text NOT NULL,
    response_text text,
    source text,
    version integer DEFAULT 1,
    is_current boolean DEFAULT false,
    strategy text,
    extra_data jsonb DEFAULT '{}'::jsonb,
    "timestamp" timestamp with time zone DEFAULT now(),
    goal_id integer,
    pipeline_run_id integer,
    embedding_id integer
);


--
-- Name: prompts_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: prompts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompts_id_seq OWNED BY public.prompts.id;


--
-- Name: protocols; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.protocols (
    name text NOT NULL,
    description text,
    input_format jsonb,
    output_format jsonb,
    failure_modes jsonb,
    depends_on jsonb,
    tags jsonb,
    capability text,
    preferred_for jsonb,
    avoid_for jsonb
);


--
-- Name: ranking_trace; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ranking_trace (
    id integer NOT NULL,
    run_id text,
    prompt_version integer,
    prompt_strategy text,
    winner text,
    loser text,
    explanation text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: ranking_trace_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ranking_trace_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: ranking_trace_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ranking_trace_id_seq OWNED BY public.ranking_trace.id;


--
-- Name: refinement_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.refinement_events (
    id integer NOT NULL,
    context text NOT NULL,
    original text NOT NULL,
    refined text NOT NULL,
    context_hash text NOT NULL,
    original_hash text NOT NULL,
    refined_hash text NOT NULL,
    original_score double precision,
    refined_score double precision,
    dimension text NOT NULL,
    improvement double precision,
    energy_before double precision,
    energy_after double precision,
    steps_used integer,
    source text DEFAULT 'auto'::text,
    created_at timestamp without time zone DEFAULT now()
);


--
-- Name: refinement_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.refinement_events_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: refinement_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.refinement_events_id_seq OWNED BY public.refinement_events.id;


--
-- Name: reflection_deltas; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.reflection_deltas (
    id integer NOT NULL,
    goal_id integer,
    run_id_a text NOT NULL,
    run_id_b text NOT NULL,
    score_a double precision,
    score_b double precision,
    score_delta double precision,
    pipeline_diff jsonb,
    strategy_diff boolean,
    model_diff boolean,
    rationale_diff jsonb,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    pipeline_a jsonb,
    pipeline_b jsonb
);


--
-- Name: reflection_deltas_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.reflection_deltas_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: reflection_deltas_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.reflection_deltas_id_seq OWNED BY public.reflection_deltas.id;


--
-- Name: reports; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.reports (
    id integer NOT NULL,
    run_id text NOT NULL,
    goal text,
    summary text,
    path text NOT NULL,
    "timestamp" timestamp with time zone DEFAULT now()
);


--
-- Name: reports_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.reports_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: reports_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.reports_id_seq OWNED BY public.reports.id;


--
-- Name: rule_applications; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.rule_applications (
    id integer NOT NULL,
    rule_id integer,
    goal_id integer,
    pipeline_run_id integer,
    hypothesis_id integer,
    applied_at timestamp without time zone DEFAULT now(),
    agent_name text,
    change_type text,
    details jsonb,
    post_score double precision,
    pre_score double precision,
    delta_score double precision,
    evaluator_name text,
    rationale text,
    notes text,
    context_hash text,
    stage_details json
);


--
-- Name: rule_applications_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.rule_applications_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: rule_applications_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.rule_applications_id_seq OWNED BY public.rule_applications.id;


--
-- Name: score_attributes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.score_attributes (
    id integer NOT NULL,
    score_id integer NOT NULL,
    key text NOT NULL,
    value text NOT NULL,
    data_type character varying(32) NOT NULL,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL
);


--
-- Name: score_attributes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.score_attributes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: score_attributes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.score_attributes_id_seq OWNED BY public.score_attributes.id;


--
-- Name: score_rule_links_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.score_rule_links_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: score_rule_links_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.score_rule_links_id_seq OWNED BY public.evaluation_rule_links.id;


--
-- Name: scores; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scores (
    id integer NOT NULL,
    evaluation_id integer,
    dimension text NOT NULL,
    score double precision,
    weight double precision,
    rationale text,
    source text,
    prompt_hash text,
    uncertainty double precision,
    energy double precision
);


--
-- Name: scores_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scores_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: scores_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scores_id_seq OWNED BY public.evaluations.id;


--
-- Name: scores_id_seq1; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scores_id_seq1
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: scores_id_seq1; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scores_id_seq1 OWNED BY public.scores.id;


--
-- Name: scoring_dimensions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scoring_dimensions (
    event_id integer NOT NULL,
    dimension text NOT NULL,
    mrq_score double precision,
    ebt_energy double precision,
    uncertainty_score double precision,
    final_score double precision
);


--
-- Name: scoring_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scoring_events (
    id integer NOT NULL,
    document_id integer NOT NULL,
    goal_text text NOT NULL,
    original_text text,
    refined_text text,
    final_source text NOT NULL,
    used_refinement boolean DEFAULT false,
    refinement_steps integer,
    used_llm_fallback boolean DEFAULT false,
    created_at timestamp without time zone DEFAULT now(),
    memcube_id text,
    version text,
    sensitivity text,
    source text,
    model text
);


--
-- Name: scoring_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scoring_events_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: scoring_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scoring_events_id_seq OWNED BY public.scoring_events.id;


--
-- Name: scoring_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scoring_history (
    id integer NOT NULL,
    model_version_id integer,
    goal_id integer,
    target_id integer NOT NULL,
    target_type text NOT NULL,
    dimension text NOT NULL,
    raw_score double precision,
    transformed_score double precision,
    uncertainty_score double precision,
    method text NOT NULL,
    source text,
    created_at timestamp without time zone DEFAULT now(),
    pipeline_run_id integer,
    model_type text
);


--
-- Name: scoring_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scoring_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: scoring_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scoring_history_id_seq OWNED BY public.scoring_history.id;


--
-- Name: search_results; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.search_results (
    id integer NOT NULL,
    query text NOT NULL,
    source text NOT NULL,
    result_type text,
    title text,
    summary text,
    url text,
    author text,
    published_at timestamp without time zone,
    tags text[],
    goal_id integer,
    parent_goal text,
    strategy text,
    focus_area text,
    extra_data jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now(),
    key_concepts text[],
    technical_insights text[],
    relevance_score integer,
    novelty_score integer,
    related_ideas text[],
    refined_summary text,
    extracted_methods text[],
    domain_knowledge_tags text[],
    critique_notes text
);


--
-- Name: search_results_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.search_results_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: search_results_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.search_results_id_seq OWNED BY public.search_results.id;


--
-- Name: sharpening_predictions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sharpening_predictions (
    id integer NOT NULL,
    goal_id integer,
    prompt_text text NOT NULL,
    output_a text NOT NULL,
    output_b text NOT NULL,
    preferred character(1),
    predicted character(1),
    value_a double precision,
    value_b double precision,
    created_at timestamp without time zone DEFAULT now(),
    CONSTRAINT sharpening_predictions_predicted_check CHECK ((predicted = ANY (ARRAY['a'::bpchar, 'b'::bpchar]))),
    CONSTRAINT sharpening_predictions_preferred_check CHECK ((preferred = ANY (ARRAY['a'::bpchar, 'b'::bpchar])))
);


--
-- Name: sharpening_predictions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.sharpening_predictions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: sharpening_predictions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.sharpening_predictions_id_seq OWNED BY public.sharpening_predictions.id;


--
-- Name: sharpening_results; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sharpening_results (
    id integer NOT NULL,
    goal text NOT NULL,
    prompt text NOT NULL,
    template text NOT NULL,
    original_output text NOT NULL,
    sharpened_output text NOT NULL,
    preferred_output text NOT NULL,
    winner text NOT NULL,
    improved boolean NOT NULL,
    comparison text NOT NULL,
    score_a double precision NOT NULL,
    score_b double precision NOT NULL,
    score_diff double precision NOT NULL,
    best_score double precision NOT NULL,
    prompt_template text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: sharpening_results_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.sharpening_results_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: sharpening_results_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.sharpening_results_id_seq OWNED BY public.sharpening_results.id;


--
-- Name: summaries; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.summaries (
    id integer NOT NULL,
    text text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: summaries_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.summaries_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: summaries_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.summaries_id_seq OWNED BY public.summaries.id;


--
-- Name: symbolic_rules; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.symbolic_rules (
    id integer NOT NULL,
    target text NOT NULL,
    rule_text text,
    source text,
    attributes jsonb,
    filter jsonb,
    context_hash text,
    score double precision,
    goal_id integer,
    pipeline_run_id integer,
    prompt_id integer,
    agent_name text,
    goal_type text,
    goal_category text,
    difficulty text,
    focus_area text,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now()
);


--
-- Name: symbolic_rules_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.symbolic_rules_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: symbolic_rules_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.symbolic_rules_id_seq OWNED BY public.symbolic_rules.id;


--
-- Name: theorem_applications; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.theorem_applications (
    id integer NOT NULL,
    theorem_id text NOT NULL,
    context text,
    result text,
    success boolean,
    energy double precision,
    uncertainty double precision,
    applied_at timestamp without time zone DEFAULT now()
);


--
-- Name: theorem_applications_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.theorem_applications_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: theorem_applications_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.theorem_applications_id_seq OWNED BY public.theorem_applications.id;


--
-- Name: theorem_cartridges; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.theorem_cartridges (
    theorem_id integer NOT NULL,
    cartridge_id integer NOT NULL
);


--
-- Name: theorems; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.theorems (
    id integer NOT NULL,
    statement text NOT NULL,
    proof text,
    embedding_id integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: theorems_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.theorems_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: theorems_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.theorems_id_seq OWNED BY public.theorems.id;


--
-- Name: training_stats; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.training_stats (
    id integer NOT NULL,
    model_type character varying NOT NULL,
    target_type character varying NOT NULL,
    dimension character varying NOT NULL,
    version character varying NOT NULL,
    embedding_type character varying NOT NULL,
    q_loss double precision,
    v_loss double precision,
    pi_loss double precision,
    avg_q_loss double precision,
    avg_v_loss double precision,
    avg_pi_loss double precision,
    policy_entropy double precision,
    policy_stability double precision,
    policy_logits jsonb,
    config jsonb,
    sample_count integer DEFAULT 0,
    valid_samples integer DEFAULT 0,
    invalid_samples integer DEFAULT 0,
    start_time timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    end_time timestamp without time zone,
    goal_id integer,
    model_version_id integer
);


--
-- Name: training_stats_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.training_stats_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: training_stats_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.training_stats_id_seq OWNED BY public.training_stats.id;


--
-- Name: unified_mrq_models; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.unified_mrq_models (
    id integer NOT NULL,
    dimension text NOT NULL,
    model_path text NOT NULL,
    trained_on timestamp without time zone DEFAULT now(),
    pair_count integer,
    trainer_version text,
    notes text,
    context jsonb
);


--
-- Name: unified_mrq_models_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.unified_mrq_models_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: unified_mrq_models_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.unified_mrq_models_id_seq OWNED BY public.unified_mrq_models.id;


--
-- Name: worldviews; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.worldviews (
    id integer NOT NULL,
    name text NOT NULL,
    description text,
    goal text,
    created_at timestamp without time zone DEFAULT now(),
    updated_at timestamp without time zone DEFAULT now(),
    extra_data json,
    db_path text,
    active boolean DEFAULT true
);


--
-- Name: worldviews_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.worldviews_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: worldviews_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.worldviews_id_seq OWNED BY public.worldviews.id;


--
-- Name: belief_graph_versions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.belief_graph_versions ALTER COLUMN id SET DEFAULT nextval('public.belief_graph_versions_id_seq'::regclass);


--
-- Name: cartridge_domains id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridge_domains ALTER COLUMN id SET DEFAULT nextval('public.cartridge_domains_id_seq'::regclass);


--
-- Name: cartridge_triples id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridge_triples ALTER COLUMN id SET DEFAULT nextval('public.cartridge_triples_id_seq'::regclass);


--
-- Name: cartridges id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridges ALTER COLUMN id SET DEFAULT nextval('public.cartridges_id_seq'::regclass);


--
-- Name: context_states id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_states ALTER COLUMN id SET DEFAULT nextval('public.context_states_id_seq'::regclass);


--
-- Name: cot_pattern_stats id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_pattern_stats ALTER COLUMN id SET DEFAULT nextval('public.cot_pattern_stats_id_seq'::regclass);


--
-- Name: cot_patterns id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_patterns ALTER COLUMN id SET DEFAULT nextval('public.cot_patterns_id_seq'::regclass);


--
-- Name: document_domains id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_domains ALTER COLUMN id SET DEFAULT nextval('public.document_domains_id_seq'::regclass);


--
-- Name: document_evaluations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_evaluations ALTER COLUMN id SET DEFAULT nextval('public.document_evaluations_id_seq'::regclass);


--
-- Name: document_scores id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_scores ALTER COLUMN id SET DEFAULT nextval('public.document_scores_id_seq'::regclass);


--
-- Name: document_section_domains id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_section_domains ALTER COLUMN id SET DEFAULT nextval('public.document_section_domains_id_seq'::regclass);


--
-- Name: document_sections id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_sections ALTER COLUMN id SET DEFAULT nextval('public.document_sections_id_seq'::regclass);


--
-- Name: documents id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.documents ALTER COLUMN id SET DEFAULT nextval('public.documents_id_seq'::regclass);


--
-- Name: elo_ranking_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.elo_ranking_log ALTER COLUMN id SET DEFAULT nextval('public.elo_ranking_log_id_seq'::regclass);


--
-- Name: embeddings id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embeddings ALTER COLUMN id SET DEFAULT nextval('public.embeddings_id_seq'::regclass);


--
-- Name: evaluation_attributes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_attributes ALTER COLUMN id SET DEFAULT nextval('public.evaluation_attributes_id_seq'::regclass);


--
-- Name: evaluation_rule_links id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_rule_links ALTER COLUMN id SET DEFAULT nextval('public.score_rule_links_id_seq'::regclass);


--
-- Name: evaluations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluations ALTER COLUMN id SET DEFAULT nextval('public.scores_id_seq'::regclass);


--
-- Name: events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.events ALTER COLUMN id SET DEFAULT nextval('public.events_id_seq'::regclass);


--
-- Name: execution_steps id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.execution_steps ALTER COLUMN id SET DEFAULT nextval('public.execution_steps_id_seq'::regclass);


--
-- Name: goal_dimensions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.goal_dimensions ALTER COLUMN id SET DEFAULT nextval('public.goal_dimensions_id_seq'::regclass);


--
-- Name: goals id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.goals ALTER COLUMN id SET DEFAULT nextval('public.goals_id_seq'::regclass);


--
-- Name: hf_embeddings id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hf_embeddings ALTER COLUMN id SET DEFAULT nextval('public.hf_embeddings_id_seq'::regclass);


--
-- Name: hnet_embeddings id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hnet_embeddings ALTER COLUMN id SET DEFAULT nextval('public.hnet_embeddings_id_seq'::regclass);


--
-- Name: hypotheses id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses ALTER COLUMN id SET DEFAULT nextval('public.hypotheses_id_seq'::regclass);


--
-- Name: ideas id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ideas ALTER COLUMN id SET DEFAULT nextval('public.ideas_id_seq'::regclass);


--
-- Name: knowledge_documents id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.knowledge_documents ALTER COLUMN id SET DEFAULT nextval('public.knowledge_documents_id_seq'::regclass);


--
-- Name: knowledge_sections id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.knowledge_sections ALTER COLUMN id SET DEFAULT nextval('public.knowledge_sections_id_seq'::regclass);


--
-- Name: lookaheads id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lookaheads ALTER COLUMN id SET DEFAULT nextval('public.lookaheads_id_seq'::regclass);


--
-- Name: measurements id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.measurements ALTER COLUMN id SET DEFAULT nextval('public.measurements_id_seq'::regclass);


--
-- Name: mem_cubes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_cubes ALTER COLUMN id SET DEFAULT nextval('public.mem_cubes_id_seq'::regclass);


--
-- Name: memcube_transformations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memcube_transformations ALTER COLUMN id SET DEFAULT nextval('public.memcube_transformations_id_seq'::regclass);


--
-- Name: method_plans id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.method_plans ALTER COLUMN id SET DEFAULT nextval('public.method_plans_id_seq'::regclass);


--
-- Name: model_performance id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_performance ALTER COLUMN id SET DEFAULT nextval('public.model_performance_id_seq'::regclass);


--
-- Name: model_versions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_versions ALTER COLUMN id SET DEFAULT nextval('public.model_versions_id_seq'::regclass);


--
-- Name: mrq_evaluations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mrq_evaluations ALTER COLUMN id SET DEFAULT nextval('public.mrq_evaluations_id_seq'::regclass);


--
-- Name: mrq_memory id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mrq_memory ALTER COLUMN id SET DEFAULT nextval('public.mrq_memory_id_seq'::regclass);


--
-- Name: mrq_preference_pairs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mrq_preference_pairs ALTER COLUMN id SET DEFAULT nextval('public.mrq_preference_pairs_id_seq'::regclass);


--
-- Name: nodes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.nodes ALTER COLUMN id SET DEFAULT nextval('public.nodes_id_seq'::regclass);


--
-- Name: pipeline_runs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_runs ALTER COLUMN id SET DEFAULT nextval('public.pipeline_runs_id_seq'::regclass);


--
-- Name: pipeline_stages id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_stages ALTER COLUMN id SET DEFAULT nextval('public.pipeline_stages_id_seq'::regclass);


--
-- Name: plan_traces id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_traces ALTER COLUMN id SET DEFAULT nextval('public.plan_traces_id_seq'::regclass);


--
-- Name: prompt_evaluations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_evaluations ALTER COLUMN id SET DEFAULT nextval('public.prompt_evaluations_id_seq'::regclass);


--
-- Name: prompt_history id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_history ALTER COLUMN id SET DEFAULT nextval('public.prompt_history_id_seq'::regclass);


--
-- Name: prompt_versions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_versions ALTER COLUMN id SET DEFAULT nextval('public.prompt_versions_id_seq'::regclass);


--
-- Name: prompts id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompts ALTER COLUMN id SET DEFAULT nextval('public.prompts_id_seq'::regclass);


--
-- Name: ranking_trace id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ranking_trace ALTER COLUMN id SET DEFAULT nextval('public.ranking_trace_id_seq'::regclass);


--
-- Name: refinement_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.refinement_events ALTER COLUMN id SET DEFAULT nextval('public.refinement_events_id_seq'::regclass);


--
-- Name: reflection_deltas id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.reflection_deltas ALTER COLUMN id SET DEFAULT nextval('public.reflection_deltas_id_seq'::regclass);


--
-- Name: reports id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.reports ALTER COLUMN id SET DEFAULT nextval('public.reports_id_seq'::regclass);


--
-- Name: rule_applications id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_applications ALTER COLUMN id SET DEFAULT nextval('public.rule_applications_id_seq'::regclass);


--
-- Name: score_attributes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.score_attributes ALTER COLUMN id SET DEFAULT nextval('public.score_attributes_id_seq'::regclass);


--
-- Name: scores id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scores ALTER COLUMN id SET DEFAULT nextval('public.scores_id_seq1'::regclass);


--
-- Name: scoring_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_events ALTER COLUMN id SET DEFAULT nextval('public.scoring_events_id_seq'::regclass);


--
-- Name: scoring_history id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_history ALTER COLUMN id SET DEFAULT nextval('public.scoring_history_id_seq'::regclass);


--
-- Name: search_results id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.search_results ALTER COLUMN id SET DEFAULT nextval('public.search_results_id_seq'::regclass);


--
-- Name: sharpening_predictions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sharpening_predictions ALTER COLUMN id SET DEFAULT nextval('public.sharpening_predictions_id_seq'::regclass);


--
-- Name: sharpening_results id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sharpening_results ALTER COLUMN id SET DEFAULT nextval('public.sharpening_results_id_seq'::regclass);


--
-- Name: summaries id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.summaries ALTER COLUMN id SET DEFAULT nextval('public.summaries_id_seq'::regclass);


--
-- Name: symbolic_rules id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.symbolic_rules ALTER COLUMN id SET DEFAULT nextval('public.symbolic_rules_id_seq'::regclass);


--
-- Name: theorem_applications id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorem_applications ALTER COLUMN id SET DEFAULT nextval('public.theorem_applications_id_seq'::regclass);


--
-- Name: theorems id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorems ALTER COLUMN id SET DEFAULT nextval('public.theorems_id_seq'::regclass);


--
-- Name: training_stats id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_stats ALTER COLUMN id SET DEFAULT nextval('public.training_stats_id_seq'::regclass);


--
-- Name: unified_mrq_models id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.unified_mrq_models ALTER COLUMN id SET DEFAULT nextval('public.unified_mrq_models_id_seq'::regclass);


--
-- Name: worldviews id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.worldviews ALTER COLUMN id SET DEFAULT nextval('public.worldviews_id_seq'::regclass);


--
-- Name: belief_cartridges belief_cartridges_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.belief_cartridges
    ADD CONSTRAINT belief_cartridges_pkey PRIMARY KEY (id);


--
-- Name: belief_graph_versions belief_graph_versions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.belief_graph_versions
    ADD CONSTRAINT belief_graph_versions_pkey PRIMARY KEY (id);


--
-- Name: cartridge_domains cartridge_domains_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridge_domains
    ADD CONSTRAINT cartridge_domains_pkey PRIMARY KEY (id);


--
-- Name: cartridge_triples cartridge_triples_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridge_triples
    ADD CONSTRAINT cartridge_triples_pkey PRIMARY KEY (id);


--
-- Name: cartridges cartridges_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridges
    ADD CONSTRAINT cartridges_pkey PRIMARY KEY (id);


--
-- Name: comparison_preferences comparison_preferences_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.comparison_preferences
    ADD CONSTRAINT comparison_preferences_pkey PRIMARY KEY (id);


--
-- Name: component_versions component_versions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.component_versions
    ADD CONSTRAINT component_versions_pkey PRIMARY KEY (id);


--
-- Name: context_states context_states_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_states
    ADD CONSTRAINT context_states_pkey PRIMARY KEY (id);


--
-- Name: cot_pattern_stats cot_pattern_stats_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_pattern_stats
    ADD CONSTRAINT cot_pattern_stats_pkey PRIMARY KEY (id);


--
-- Name: cot_patterns cot_patterns_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_patterns
    ADD CONSTRAINT cot_patterns_pkey PRIMARY KEY (id);


--
-- Name: document_domains document_domains_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_domains
    ADD CONSTRAINT document_domains_pkey PRIMARY KEY (id);


--
-- Name: document_evaluations document_evaluations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_evaluations
    ADD CONSTRAINT document_evaluations_pkey PRIMARY KEY (id);


--
-- Name: document_scores document_scores_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_scores
    ADD CONSTRAINT document_scores_pkey PRIMARY KEY (id);


--
-- Name: document_section_domains document_section_domains_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_section_domains
    ADD CONSTRAINT document_section_domains_pkey PRIMARY KEY (id);


--
-- Name: document_sections document_sections_document_id_section_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_sections
    ADD CONSTRAINT document_sections_document_id_section_name_key UNIQUE (document_id, section_name);


--
-- Name: document_sections document_sections_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_sections
    ADD CONSTRAINT document_sections_pkey PRIMARY KEY (id);


--
-- Name: documents documents_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_pkey PRIMARY KEY (id);


--
-- Name: elo_ranking_log elo_ranking_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.elo_ranking_log
    ADD CONSTRAINT elo_ranking_log_pkey PRIMARY KEY (id);


--
-- Name: embeddings embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embeddings
    ADD CONSTRAINT embeddings_pkey PRIMARY KEY (id);


--
-- Name: evaluation_attributes evaluation_attributes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_attributes
    ADD CONSTRAINT evaluation_attributes_pkey PRIMARY KEY (id);


--
-- Name: events events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.events
    ADD CONSTRAINT events_pkey PRIMARY KEY (id);


--
-- Name: execution_steps execution_steps_evaluation_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.execution_steps
    ADD CONSTRAINT execution_steps_evaluation_id_key UNIQUE (evaluation_id);


--
-- Name: execution_steps execution_steps_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.execution_steps
    ADD CONSTRAINT execution_steps_pkey PRIMARY KEY (id);


--
-- Name: goal_dimensions goal_dimensions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.goal_dimensions
    ADD CONSTRAINT goal_dimensions_pkey PRIMARY KEY (id);


--
-- Name: goals goals_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.goals
    ADD CONSTRAINT goals_pkey PRIMARY KEY (id);


--
-- Name: hf_embeddings hf_embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hf_embeddings
    ADD CONSTRAINT hf_embeddings_pkey PRIMARY KEY (id);


--
-- Name: hnet_embeddings hnet_embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hnet_embeddings
    ADD CONSTRAINT hnet_embeddings_pkey PRIMARY KEY (id);


--
-- Name: hypotheses hypotheses_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses
    ADD CONSTRAINT hypotheses_pkey PRIMARY KEY (id);


--
-- Name: ideas ideas_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ideas
    ADD CONSTRAINT ideas_pkey PRIMARY KEY (id);


--
-- Name: knowledge_documents knowledge_documents_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.knowledge_documents
    ADD CONSTRAINT knowledge_documents_pkey PRIMARY KEY (id);


--
-- Name: knowledge_sections knowledge_sections_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.knowledge_sections
    ADD CONSTRAINT knowledge_sections_pkey PRIMARY KEY (id);


--
-- Name: lookaheads lookaheads_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lookaheads
    ADD CONSTRAINT lookaheads_pkey PRIMARY KEY (id);


--
-- Name: measurements measurements_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.measurements
    ADD CONSTRAINT measurements_pkey PRIMARY KEY (id);


--
-- Name: mem_cubes mem_cubes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_cubes
    ADD CONSTRAINT mem_cubes_pkey PRIMARY KEY (id);


--
-- Name: memcube_transformations memcube_transformations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memcube_transformations
    ADD CONSTRAINT memcube_transformations_pkey PRIMARY KEY (id);


--
-- Name: memcube_versions memcube_versions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memcube_versions
    ADD CONSTRAINT memcube_versions_pkey PRIMARY KEY (id);


--
-- Name: memcubes memcubes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memcubes
    ADD CONSTRAINT memcubes_pkey PRIMARY KEY (id);


--
-- Name: method_plans method_plans_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.method_plans
    ADD CONSTRAINT method_plans_pkey PRIMARY KEY (id);


--
-- Name: model_performance model_performance_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_performance
    ADD CONSTRAINT model_performance_pkey PRIMARY KEY (id);


--
-- Name: model_versions model_versions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_versions
    ADD CONSTRAINT model_versions_pkey PRIMARY KEY (id);


--
-- Name: mrq_evaluations mrq_evaluations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mrq_evaluations
    ADD CONSTRAINT mrq_evaluations_pkey PRIMARY KEY (id);


--
-- Name: mrq_memory mrq_memory_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mrq_memory
    ADD CONSTRAINT mrq_memory_pkey PRIMARY KEY (id);


--
-- Name: mrq_preference_pairs mrq_preference_pairs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mrq_preference_pairs
    ADD CONSTRAINT mrq_preference_pairs_pkey PRIMARY KEY (id);


--
-- Name: nodes nodes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.nodes
    ADD CONSTRAINT nodes_pkey PRIMARY KEY (id);


--
-- Name: pipeline_runs pipeline_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_runs
    ADD CONSTRAINT pipeline_runs_pkey PRIMARY KEY (id);


--
-- Name: pipeline_runs pipeline_runs_run_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_runs
    ADD CONSTRAINT pipeline_runs_run_id_key UNIQUE (run_id);


--
-- Name: pipeline_stages pipeline_stages_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_stages
    ADD CONSTRAINT pipeline_stages_pkey PRIMARY KEY (id);


--
-- Name: plan_traces plan_traces_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_traces
    ADD CONSTRAINT plan_traces_pkey PRIMARY KEY (id);


--
-- Name: plan_traces plan_traces_trace_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_traces
    ADD CONSTRAINT plan_traces_trace_id_key UNIQUE (trace_id);


--
-- Name: prompt_evaluations prompt_evaluations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_evaluations
    ADD CONSTRAINT prompt_evaluations_pkey PRIMARY KEY (id);


--
-- Name: prompt_history prompt_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_history
    ADD CONSTRAINT prompt_history_pkey PRIMARY KEY (id);


--
-- Name: prompt_programs prompt_programs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_programs
    ADD CONSTRAINT prompt_programs_pkey PRIMARY KEY (id);


--
-- Name: prompt_versions prompt_versions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_versions
    ADD CONSTRAINT prompt_versions_pkey PRIMARY KEY (id);


--
-- Name: prompts prompts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompts
    ADD CONSTRAINT prompts_pkey PRIMARY KEY (id);


--
-- Name: protocols protocols_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.protocols
    ADD CONSTRAINT protocols_pkey PRIMARY KEY (name);


--
-- Name: ranking_trace ranking_trace_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ranking_trace
    ADD CONSTRAINT ranking_trace_pkey PRIMARY KEY (id);


--
-- Name: refinement_events refinement_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.refinement_events
    ADD CONSTRAINT refinement_events_pkey PRIMARY KEY (id);


--
-- Name: reflection_deltas reflection_deltas_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.reflection_deltas
    ADD CONSTRAINT reflection_deltas_pkey PRIMARY KEY (id);


--
-- Name: reports reports_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.reports
    ADD CONSTRAINT reports_pkey PRIMARY KEY (id);


--
-- Name: rule_applications rule_applications_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_applications
    ADD CONSTRAINT rule_applications_pkey PRIMARY KEY (id);


--
-- Name: score_attributes score_attributes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.score_attributes
    ADD CONSTRAINT score_attributes_pkey PRIMARY KEY (id);


--
-- Name: evaluation_rule_links score_rule_links_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_rule_links
    ADD CONSTRAINT score_rule_links_pkey PRIMARY KEY (id);


--
-- Name: evaluations scores_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluations
    ADD CONSTRAINT scores_pkey PRIMARY KEY (id);


--
-- Name: scores scores_pkey1; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scores
    ADD CONSTRAINT scores_pkey1 PRIMARY KEY (id);


--
-- Name: scoring_dimensions scoring_dimensions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_dimensions
    ADD CONSTRAINT scoring_dimensions_pkey PRIMARY KEY (event_id, dimension);


--
-- Name: scoring_events scoring_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_events
    ADD CONSTRAINT scoring_events_pkey PRIMARY KEY (id);


--
-- Name: scoring_history scoring_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_history
    ADD CONSTRAINT scoring_history_pkey PRIMARY KEY (id);


--
-- Name: search_results search_results_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.search_results
    ADD CONSTRAINT search_results_pkey PRIMARY KEY (id);


--
-- Name: sharpening_predictions sharpening_predictions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sharpening_predictions
    ADD CONSTRAINT sharpening_predictions_pkey PRIMARY KEY (id);


--
-- Name: sharpening_results sharpening_results_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sharpening_results
    ADD CONSTRAINT sharpening_results_pkey PRIMARY KEY (id);


--
-- Name: summaries summaries_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.summaries
    ADD CONSTRAINT summaries_pkey PRIMARY KEY (id);


--
-- Name: symbolic_rules symbolic_rules_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.symbolic_rules
    ADD CONSTRAINT symbolic_rules_pkey PRIMARY KEY (id);


--
-- Name: theorem_applications theorem_applications_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorem_applications
    ADD CONSTRAINT theorem_applications_pkey PRIMARY KEY (id);


--
-- Name: theorem_cartridges theorem_cartridges_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorem_cartridges
    ADD CONSTRAINT theorem_cartridges_pkey PRIMARY KEY (theorem_id, cartridge_id);


--
-- Name: theorems theorems_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorems
    ADD CONSTRAINT theorems_pkey PRIMARY KEY (id);


--
-- Name: training_stats training_stats_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_stats
    ADD CONSTRAINT training_stats_pkey PRIMARY KEY (id);


--
-- Name: unified_mrq_models unified_mrq_models_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.unified_mrq_models
    ADD CONSTRAINT unified_mrq_models_pkey PRIMARY KEY (id);


--
-- Name: document_domains unique_document_domain; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_domains
    ADD CONSTRAINT unique_document_domain UNIQUE (document_id, domain);


--
-- Name: document_section_domains unique_document_section_domain; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_section_domains
    ADD CONSTRAINT unique_document_section_domain UNIQUE (document_section_id, domain);


--
-- Name: cartridges unique_source; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridges
    ADD CONSTRAINT unique_source UNIQUE (source_type, source_uri);


--
-- Name: hf_embeddings unique_text_hash_hf; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hf_embeddings
    ADD CONSTRAINT unique_text_hash_hf UNIQUE (text_hash);


--
-- Name: hnet_embeddings unique_text_hash_hnet; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hnet_embeddings
    ADD CONSTRAINT unique_text_hash_hnet UNIQUE (text_hash);


--
-- Name: worldviews worldviews_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.worldviews
    ADD CONSTRAINT worldviews_name_key UNIQUE (name);


--
-- Name: worldviews worldviews_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.worldviews
    ADD CONSTRAINT worldviews_pkey PRIMARY KEY (id);


--
-- Name: idx_evaluation_attributes_duration; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_evaluation_attributes_duration ON public.evaluation_attributes USING btree (duration);


--
-- Name: idx_evaluation_attributes_output_size; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_evaluation_attributes_output_size ON public.evaluation_attributes USING btree (output_size);


--
-- Name: idx_evaluation_attributes_start_time; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_evaluation_attributes_start_time ON public.evaluation_attributes USING btree (start_time);


--
-- Name: idx_execution_steps_evaluation_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_execution_steps_evaluation_id ON public.execution_steps USING btree (evaluation_id);


--
-- Name: idx_execution_steps_plan_trace_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_execution_steps_plan_trace_id ON public.execution_steps USING btree (plan_trace_id);


--
-- Name: idx_execution_steps_step_order; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_execution_steps_step_order ON public.execution_steps USING btree (plan_trace_id, step_order);


--
-- Name: idx_hf_embedding_vector; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_hf_embedding_vector ON public.hf_embeddings USING ivfflat (embedding public.vector_cosine_ops);


--
-- Name: idx_measurements_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_measurements_created_at ON public.measurements USING btree (created_at);


--
-- Name: idx_measurements_entity_metric; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_measurements_entity_metric ON public.measurements USING btree (entity_type, entity_id, metric_name);


--
-- Name: idx_measurements_value_gin; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_measurements_value_gin ON public.measurements USING gin (value);


--
-- Name: idx_nodes_goal_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_nodes_goal_id ON public.nodes USING btree (goal_id);


--
-- Name: idx_nodes_pipeline_run_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_nodes_pipeline_run_id ON public.nodes USING btree (pipeline_run_id);


--
-- Name: idx_nodes_stage_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_nodes_stage_name ON public.nodes USING btree (stage_name);


--
-- Name: idx_pipeline_stages_goal_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pipeline_stages_goal_id ON public.pipeline_stages USING btree (goal_id);


--
-- Name: idx_pipeline_stages_input_context; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pipeline_stages_input_context ON public.pipeline_stages USING btree (input_context_id);


--
-- Name: idx_pipeline_stages_output_context; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pipeline_stages_output_context ON public.pipeline_stages USING btree (output_context_id);


--
-- Name: idx_pipeline_stages_parent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pipeline_stages_parent ON public.pipeline_stages USING btree (parent_stage_id);


--
-- Name: idx_pipeline_stages_run_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pipeline_stages_run_id ON public.pipeline_stages USING btree (run_id);


--
-- Name: idx_pipeline_stages_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pipeline_stages_status ON public.pipeline_stages USING btree (status);


--
-- Name: idx_plan_traces_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_plan_traces_created_at ON public.plan_traces USING btree (created_at);


--
-- Name: idx_plan_traces_goal_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_plan_traces_goal_id ON public.plan_traces USING btree (goal_id);


--
-- Name: idx_plan_traces_trace_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_plan_traces_trace_id ON public.plan_traces USING btree (trace_id);


--
-- Name: idx_prompt_agent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prompt_agent ON public.prompts USING btree (source);


--
-- Name: idx_prompt_strategy; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prompt_strategy ON public.prompts USING btree (strategy);


--
-- Name: idx_prompt_version; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prompt_version ON public.prompts USING btree (version);


--
-- Name: idx_theorem_cartridges_cartridge_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_theorem_cartridges_cartridge_id ON public.theorem_cartridges USING btree (cartridge_id);


--
-- Name: idx_theorem_cartridges_theorem_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_theorem_cartridges_theorem_id ON public.theorem_cartridges USING btree (theorem_id);


--
-- Name: idx_training_stats_dimension; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_training_stats_dimension ON public.training_stats USING btree (dimension);


--
-- Name: idx_training_stats_embedding; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_training_stats_embedding ON public.training_stats USING btree (embedding_type);


--
-- Name: idx_training_stats_model; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_training_stats_model ON public.training_stats USING btree (model_type);


--
-- Name: idx_training_stats_version; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_training_stats_version ON public.training_stats USING btree (version);


--
-- Name: unique_text_hash; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX unique_text_hash ON public.embeddings USING btree (text_hash);


--
-- Name: belief_cartridges belief_cartridges_document_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.belief_cartridges
    ADD CONSTRAINT belief_cartridges_document_id_fkey FOREIGN KEY (document_id) REFERENCES public.documents(id) ON DELETE SET NULL;


--
-- Name: belief_cartridges belief_cartridges_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.belief_cartridges
    ADD CONSTRAINT belief_cartridges_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE SET NULL;


--
-- Name: cartridge_domains cartridge_domains_cartridge_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridge_domains
    ADD CONSTRAINT cartridge_domains_cartridge_id_fkey FOREIGN KEY (cartridge_id) REFERENCES public.cartridges(id) ON DELETE CASCADE;


--
-- Name: cartridge_triples cartridge_triples_cartridge_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridge_triples
    ADD CONSTRAINT cartridge_triples_cartridge_id_fkey FOREIGN KEY (cartridge_id) REFERENCES public.cartridges(id) ON DELETE CASCADE;


--
-- Name: cartridges cartridges_embedding_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridges
    ADD CONSTRAINT cartridges_embedding_id_fkey FOREIGN KEY (embedding_id) REFERENCES public.embeddings(id) ON DELETE SET NULL;


--
-- Name: cartridges cartridges_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridges
    ADD CONSTRAINT cartridges_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE SET NULL;


--
-- Name: component_interfaces component_interfaces_component_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.component_interfaces
    ADD CONSTRAINT component_interfaces_component_id_fkey FOREIGN KEY (component_id) REFERENCES public.component_versions(id);


--
-- Name: cot_pattern_stats cot_pattern_stats_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_pattern_stats
    ADD CONSTRAINT cot_pattern_stats_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE CASCADE;


--
-- Name: cot_pattern_stats cot_pattern_stats_hypothesis_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_pattern_stats
    ADD CONSTRAINT cot_pattern_stats_hypothesis_id_fkey FOREIGN KEY (hypothesis_id) REFERENCES public.hypotheses(id) ON DELETE CASCADE;


--
-- Name: cot_patterns cot_patterns_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_patterns
    ADD CONSTRAINT cot_patterns_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id);


--
-- Name: cot_patterns cot_patterns_hypothesis_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_patterns
    ADD CONSTRAINT cot_patterns_hypothesis_id_fkey FOREIGN KEY (hypothesis_id) REFERENCES public.hypotheses(id);


--
-- Name: document_domains document_domains_document_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_domains
    ADD CONSTRAINT document_domains_document_id_fkey FOREIGN KEY (document_id) REFERENCES public.documents(id) ON DELETE CASCADE;


--
-- Name: document_evaluations document_evaluations_document_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_evaluations
    ADD CONSTRAINT document_evaluations_document_id_fkey FOREIGN KEY (document_id) REFERENCES public.documents(id) ON DELETE CASCADE;


--
-- Name: document_scores document_scores_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_scores
    ADD CONSTRAINT document_scores_evaluation_id_fkey FOREIGN KEY (evaluation_id) REFERENCES public.document_evaluations(id) ON DELETE CASCADE;


--
-- Name: document_section_domains document_section_domains_document_section_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_section_domains
    ADD CONSTRAINT document_section_domains_document_section_id_fkey FOREIGN KEY (document_section_id) REFERENCES public.document_sections(id) ON DELETE CASCADE;


--
-- Name: document_sections document_sections_document_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_sections
    ADD CONSTRAINT document_sections_document_id_fkey FOREIGN KEY (document_id) REFERENCES public.documents(id) ON DELETE CASCADE;


--
-- Name: evaluation_attributes evaluation_attributes_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_attributes
    ADD CONSTRAINT evaluation_attributes_evaluation_id_fkey FOREIGN KEY (evaluation_id) REFERENCES public.evaluations(id) ON DELETE CASCADE;


--
-- Name: evaluation_rule_links evaluation_rule_links_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_rule_links
    ADD CONSTRAINT evaluation_rule_links_evaluation_id_fkey FOREIGN KEY (evaluation_id) REFERENCES public.evaluations(id) ON DELETE CASCADE;


--
-- Name: execution_steps execution_steps_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.execution_steps
    ADD CONSTRAINT execution_steps_evaluation_id_fkey FOREIGN KEY (evaluation_id) REFERENCES public.evaluations(id) ON DELETE SET NULL;


--
-- Name: execution_steps execution_steps_plan_trace_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.execution_steps
    ADD CONSTRAINT execution_steps_plan_trace_id_fkey FOREIGN KEY (plan_trace_id) REFERENCES public.plan_traces(id) ON DELETE CASCADE;


--
-- Name: evaluations fk_document; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluations
    ADD CONSTRAINT fk_document FOREIGN KEY (document_id) REFERENCES public.documents(id) ON DELETE SET NULL;


--
-- Name: documents fk_documents_embedding; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT fk_documents_embedding FOREIGN KEY (embedding_id) REFERENCES public.embeddings(id) ON DELETE SET NULL;


--
-- Name: documents fk_documents_goal_id_goals; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT fk_documents_goal_id_goals FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE SET NULL;


--
-- Name: hypotheses fk_goal; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses
    ADD CONSTRAINT fk_goal FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE CASCADE;


--
-- Name: context_states fk_goal; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_states
    ADD CONSTRAINT fk_goal FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE SET NULL;


--
-- Name: hypotheses fk_goal_hypothesis; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses
    ADD CONSTRAINT fk_goal_hypothesis FOREIGN KEY (goal_id) REFERENCES public.goals(id);


--
-- Name: prompts fk_goal_prompt; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompts
    ADD CONSTRAINT fk_goal_prompt FOREIGN KEY (goal_id) REFERENCES public.goals(id);


--
-- Name: context_states fk_pipeline_run; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_states
    ADD CONSTRAINT fk_pipeline_run FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id) ON DELETE SET NULL;


--
-- Name: hypotheses fk_prompt; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses
    ADD CONSTRAINT fk_prompt FOREIGN KEY (prompt_id) REFERENCES public.prompts(id);


--
-- Name: evaluations fk_rule_application; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluations
    ADD CONSTRAINT fk_rule_application FOREIGN KEY (rule_application_id) REFERENCES public.rule_applications(id) ON DELETE SET NULL;


--
-- Name: evaluations fk_scores_pipeline_run; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluations
    ADD CONSTRAINT fk_scores_pipeline_run FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id) ON DELETE SET NULL;


--
-- Name: goal_dimensions goal_dimensions_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.goal_dimensions
    ADD CONSTRAINT goal_dimensions_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE CASCADE;


--
-- Name: hypotheses hypotheses_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses
    ADD CONSTRAINT hypotheses_prompt_id_fkey FOREIGN KEY (prompt_id) REFERENCES public.prompts(id);


--
-- Name: hypotheses hypotheses_source_hypothesis_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses
    ADD CONSTRAINT hypotheses_source_hypothesis_fkey FOREIGN KEY (source_hypothesis_id) REFERENCES public.hypotheses(id);


--
-- Name: ideas ideas_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ideas
    ADD CONSTRAINT ideas_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id);


--
-- Name: knowledge_documents knowledge_documents_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.knowledge_documents
    ADD CONSTRAINT knowledge_documents_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE SET NULL;


--
-- Name: knowledge_sections knowledge_sections_document_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.knowledge_sections
    ADD CONSTRAINT knowledge_sections_document_id_fkey FOREIGN KEY (document_id) REFERENCES public.knowledge_documents(id) ON DELETE CASCADE;


--
-- Name: lookaheads lookaheads_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lookaheads
    ADD CONSTRAINT lookaheads_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE CASCADE;


--
-- Name: method_plans method_plans_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.method_plans
    ADD CONSTRAINT method_plans_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE CASCADE;


--
-- Name: method_plans method_plans_idea_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.method_plans
    ADD CONSTRAINT method_plans_idea_id_fkey FOREIGN KEY (idea_id) REFERENCES public.ideas(id) ON DELETE SET NULL;


--
-- Name: method_plans method_plans_parent_plan_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.method_plans
    ADD CONSTRAINT method_plans_parent_plan_id_fkey FOREIGN KEY (parent_plan_id) REFERENCES public.method_plans(id) ON DELETE SET NULL;


--
-- Name: pipeline_runs pipeline_runs_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_runs
    ADD CONSTRAINT pipeline_runs_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE CASCADE;


--
-- Name: pipeline_stages pipeline_stages_input_context_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_stages
    ADD CONSTRAINT pipeline_stages_input_context_id_fkey FOREIGN KEY (input_context_id) REFERENCES public.context_states(id);


--
-- Name: pipeline_stages pipeline_stages_output_context_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_stages
    ADD CONSTRAINT pipeline_stages_output_context_id_fkey FOREIGN KEY (output_context_id) REFERENCES public.context_states(id);


--
-- Name: pipeline_stages pipeline_stages_parent_stage_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_stages
    ADD CONSTRAINT pipeline_stages_parent_stage_id_fkey FOREIGN KEY (parent_stage_id) REFERENCES public.pipeline_stages(id);


--
-- Name: pipeline_stages pipeline_stages_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_stages
    ADD CONSTRAINT pipeline_stages_pipeline_run_id_fkey FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id);


--
-- Name: plan_traces plan_traces_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_traces
    ADD CONSTRAINT plan_traces_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE CASCADE;


--
-- Name: prompt_evaluations prompt_evaluations_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_evaluations
    ADD CONSTRAINT prompt_evaluations_prompt_id_fkey FOREIGN KEY (prompt_id) REFERENCES public.prompts(id) ON DELETE CASCADE;


--
-- Name: prompt_history prompt_history_original_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_history
    ADD CONSTRAINT prompt_history_original_prompt_id_fkey FOREIGN KEY (original_prompt_id) REFERENCES public.prompts(id);


--
-- Name: prompt_programs prompt_programs_parent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_programs
    ADD CONSTRAINT prompt_programs_parent_id_fkey FOREIGN KEY (parent_id) REFERENCES public.prompt_programs(id);


--
-- Name: prompt_programs prompt_programs_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_programs
    ADD CONSTRAINT prompt_programs_pipeline_run_id_fkey FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id);


--
-- Name: prompt_programs prompt_programs_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_programs
    ADD CONSTRAINT prompt_programs_prompt_id_fkey FOREIGN KEY (prompt_id) REFERENCES public.prompts(id);


--
-- Name: prompt_versions prompt_versions_previous_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_versions
    ADD CONSTRAINT prompt_versions_previous_prompt_id_fkey FOREIGN KEY (previous_prompt_id) REFERENCES public.prompts(id);


--
-- Name: prompts prompts_embedding_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompts
    ADD CONSTRAINT prompts_embedding_id_fkey FOREIGN KEY (embedding_id) REFERENCES public.embeddings(id) ON DELETE SET NULL;


--
-- Name: reflection_deltas reflection_deltas_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.reflection_deltas
    ADD CONSTRAINT reflection_deltas_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE CASCADE;


--
-- Name: rule_applications rule_applications_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_applications
    ADD CONSTRAINT rule_applications_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE CASCADE;


--
-- Name: rule_applications rule_applications_hypothesis_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_applications
    ADD CONSTRAINT rule_applications_hypothesis_id_fkey FOREIGN KEY (hypothesis_id) REFERENCES public.hypotheses(id);


--
-- Name: rule_applications rule_applications_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_applications
    ADD CONSTRAINT rule_applications_pipeline_run_id_fkey FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id) ON DELETE CASCADE;


--
-- Name: score_attributes score_attributes_score_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.score_attributes
    ADD CONSTRAINT score_attributes_score_id_fkey FOREIGN KEY (score_id) REFERENCES public.scores(id) ON DELETE CASCADE;


--
-- Name: evaluation_rule_links score_rule_links_rule_application_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_rule_links
    ADD CONSTRAINT score_rule_links_rule_application_id_fkey FOREIGN KEY (rule_application_id) REFERENCES public.rule_applications(id) ON DELETE CASCADE;


--
-- Name: scores scores_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scores
    ADD CONSTRAINT scores_evaluation_id_fkey FOREIGN KEY (evaluation_id) REFERENCES public.evaluations(id) ON DELETE CASCADE;


--
-- Name: evaluations scores_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluations
    ADD CONSTRAINT scores_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE CASCADE;


--
-- Name: evaluations scores_hypothesis_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluations
    ADD CONSTRAINT scores_hypothesis_id_fkey FOREIGN KEY (hypothesis_id) REFERENCES public.hypotheses(id) ON DELETE CASCADE;


--
-- Name: scoring_dimensions scoring_dimensions_event_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_dimensions
    ADD CONSTRAINT scoring_dimensions_event_id_fkey FOREIGN KEY (event_id) REFERENCES public.scoring_events(id);


--
-- Name: scoring_history scoring_history_model_version_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_history
    ADD CONSTRAINT scoring_history_model_version_id_fkey FOREIGN KEY (model_version_id) REFERENCES public.model_versions(id);


--
-- Name: scoring_history scoring_history_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_history
    ADD CONSTRAINT scoring_history_pipeline_run_id_fkey FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id);


--
-- Name: search_results search_results_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.search_results
    ADD CONSTRAINT search_results_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id);


--
-- Name: sharpening_predictions sharpening_predictions_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sharpening_predictions
    ADD CONSTRAINT sharpening_predictions_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id);


--
-- Name: symbolic_rules symbolic_rules_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.symbolic_rules
    ADD CONSTRAINT symbolic_rules_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE CASCADE;


--
-- Name: symbolic_rules symbolic_rules_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.symbolic_rules
    ADD CONSTRAINT symbolic_rules_prompt_id_fkey FOREIGN KEY (prompt_id) REFERENCES public.prompts(id) ON DELETE CASCADE;


--
-- Name: theorem_cartridges theorem_cartridges_cartridge_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorem_cartridges
    ADD CONSTRAINT theorem_cartridges_cartridge_id_fkey FOREIGN KEY (cartridge_id) REFERENCES public.cartridges(id);


--
-- Name: theorem_cartridges theorem_cartridges_theorem_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorem_cartridges
    ADD CONSTRAINT theorem_cartridges_theorem_id_fkey FOREIGN KEY (theorem_id) REFERENCES public.theorems(id);


--
-- Name: theorems theorems_embedding_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorems
    ADD CONSTRAINT theorems_embedding_id_fkey FOREIGN KEY (embedding_id) REFERENCES public.embeddings(id);


--
-- Name: training_stats training_stats_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_stats
    ADD CONSTRAINT training_stats_goal_id_fkey FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON DELETE SET NULL;


--
-- Name: training_stats training_stats_model_version_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_stats
    ADD CONSTRAINT training_stats_model_version_id_fkey FOREIGN KEY (model_version_id) REFERENCES public.model_versions(id) ON DELETE SET NULL;


--
-- PostgreSQL database dump complete
--

