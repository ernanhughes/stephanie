--
--
 --
-- Name: pg_trgm; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA PUBLIC;

--
-- Name: EXTENSION pg_trgm; Type: COMMENT; Schema: -; Owner: -
--
 COMMENT ON EXTENSION pg_trgm IS 'text similarity measurement and index searching based on trigrams';

--
-- Name: pgcrypto; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA PUBLIC;

--
-- Name: EXTENSION pgcrypto; Type: COMMENT; Schema: -; Owner: -
--
 COMMENT ON EXTENSION pgcrypto IS 'cryptographic functions';

--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA PUBLIC;

--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: -
--
 COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';

--
-- Name: set_updated_at_timestamp(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.set_updated_at_timestamp() RETURNS TRIGGER LANGUAGE PLPGSQL AS $$
BEGIN
  NEW.updated_at := NOW();
  RETURN NEW;
END;
$$;

--
-- Name: belief_cartridges; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.belief_cartridges (id text NOT NULL,
                                               created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                              updated_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                             source_id text, source_type text NOT NULL,
                                                                                                                                                                              source_url text, markdown_content text NOT NULL,
                                                                                                                                                                                                                     is_active boolean DEFAULT TRUE,
                                                                                                                                                                                                                                               idea_payload JSONB,
                                                                                                                                                                                                                                                            goal_tags text[] DEFAULT ARRAY[]::text[], domain_tags text[] DEFAULT ARRAY[]::text[], derived_from JSONB DEFAULT '[]'::JSONB,
                                                                                                                                                                                                                                                                                                                                                                             applied_in JSONB DEFAULT '[]'::JSONB,
                                                                                                                                                                                                                                                                                                                                                                                                      VERSION integer DEFAULT 1,
                                                                                                                                                                                                                                                                                                                                                                                                                              memcube_id text, goal_id integer, document_id integer, CONSTRAINT belief_cartridges_source_type_check CHECK ((source_type = ANY (ARRAY['paper'::text,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     'blog'::text,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     'experiment'::text,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     'pipeline'::text,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     'manual'::text]))));

--
-- Name: belief_graph_versions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.belief_graph_versions (id integer NOT NULL,
                                                      goal text NOT NULL,
                                                                node_count integer, edge_count integer, avg_strength double precision, avg_relevance double precision, contradictions integer, theorems integer, created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                                                                                                model_path text);

--
-- Name: belief_graph_versions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.belief_graph_versions_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: belief_graph_versions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.belief_graph_versions_id_seq OWNED BY public.belief_graph_versions.id;

--
-- Name: bus_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.bus_events (id integer NOT NULL,
                                           event_id text, subject text NOT NULL,
                                                                       event text, ts real NOT NULL,
                                                                                           run_id text, case_id text, paper_id text, section_name text, agent text, payload_json text NOT NULL,
                                                                                                                                                                                      extras_json text, HASH text);

--
-- Name: bus_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.bus_events_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: bus_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.bus_events_id_seq OWNED BY public.bus_events.id;

--
-- Name: calibration_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.calibration_events (id integer NOT NULL,
                                                   DOMAIN CHARACTER varying NOT NULL,
                                                                            query CHARACTER varying NOT NULL,
                                                                                                    raw_similarity double precision NOT NULL,
                                                                                                                                    scorable_id CHARACTER varying NOT NULL,
                                                                                                                                                                  scorable_type CHARACTER varying NOT NULL,
                                                                                                                                                                                                  entity_type CHARACTER varying, is_relevant boolean NOT NULL,
                                                                                                                                                                                                                                                     context JSON,
                                                                                                                                                                                                                                                             "timestamp" timestamp WITHOUT TIME ZONE DEFAULT (now() AT TIME ZONE 'UTC'::text));

--
-- Name: calibration_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.calibration_events_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: calibration_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.calibration_events_id_seq OWNED BY public.calibration_events.id;

--
-- Name: calibration_models; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.calibration_models (id bigint NOT NULL,
                                                  DOMAIN CHARACTER varying(255) NOT NULL,
                                                                                kind CHARACTER varying(64) NOT NULL,
                                                                                                           threshold double precision DEFAULT 0.5 NOT NULL,
                                                                                                                                                  payload BYTEA NOT NULL,
                                                                                                                                                                updated_at timestamp WITH TIME ZONE DEFAULT now() NOT NULL);

--
-- Name: calibration_models_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.calibration_models
ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY
  (SEQUENCE NAME public.calibration_models_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1);

--
-- Name: cartridge_domains; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cartridge_domains (id integer NOT NULL,
                                                  cartridge_id integer NOT NULL,
                                                                       DOMAIN CHARACTER varying NOT NULL,
                                                                                                score double precision NOT NULL);

--
-- Name: cartridge_domains_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cartridge_domains_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: cartridge_domains_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cartridge_domains_id_seq OWNED BY public.cartridge_domains.id;

--
-- Name: cartridge_triples; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cartridge_triples (id integer NOT NULL,
                                                  cartridge_id integer NOT NULL,
                                                                       subject text NOT NULL,
                                                                                    predicate text NOT NULL,
                                                                                                   OBJECT text NOT NULL,
                                                                                                               confidence double precision DEFAULT 1.0,
                                                                                                                                                   created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP);

--
-- Name: cartridge_triples_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cartridge_triples_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: cartridge_triples_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cartridge_triples_id_seq OWNED BY public.cartridge_triples.id;

--
-- Name: cartridges; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cartridges (id integer NOT NULL,
                                           goal_id integer, source_type text NOT NULL,
                                                                             source_uri text, markdown_content text NOT NULL,
                                                                                                                    embedding_id integer, title text, SUMMARY text, sections JSONB,
                                                                                                                                                                             triples JSONB,
                                                                                                                                                                                     domain_tags JSONB,
                                                                                                                                                                                                 created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                                                                                                                                pipeline_run_id integer);

--
-- Name: cartridges_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cartridges_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: cartridges_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cartridges_id_seq OWNED BY public.cartridges.id;

--
-- Name: case_attributes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.case_attributes (id bigint NOT NULL,
                                               case_id integer NOT NULL,
                                                               KEY text NOT NULL,
                                                                        value_text text, value_num double precision, value_bool boolean, value_json JSONB,
                                                                                                                                                    created_at timestamp WITH TIME ZONE DEFAULT now() NOT NULL,
                                                                                                                                                                                                      CONSTRAINT chk_case_attr_one_value CHECK (((((CASE
                                                                                                                                                                                                                                                        WHEN (value_text IS NOT NULL) THEN 1
                                                                                                                                                                                                                                                        ELSE 0
                                                                                                                                                                                                                                                    END + CASE
                                                                                                                                                                                                                                                              WHEN (value_num IS NOT NULL) THEN 1
                                                                                                                                                                                                                                                              ELSE 0
                                                                                                                                                                                                                                                          END) + CASE
                                                                                                                                                                                                                                                                     WHEN (value_bool IS NOT NULL) THEN 1
                                                                                                                                                                                                                                                                     ELSE 0
                                                                                                                                                                                                                                                                 END) + CASE
                                                                                                                                                                                                                                                                            WHEN ((value_json IS NOT NULL)
                                                                                                                                                                                                                                                                                  AND (jsonb_typeof(value_json) <> 'null'::text)) THEN 1
                                                                                                                                                                                                                                                                            ELSE 0
                                                                                                                                                                                                                                                                        END) = 1)));

--
-- Name: case_attributes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.case_attributes_id_seq
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: case_attributes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.case_attributes_id_seq OWNED BY public.case_attributes.id;

--
-- Name: case_goal_state; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.case_goal_state (id integer NOT NULL,
                                                casebook_id integer NOT NULL,
                                                                    goal_id text NOT NULL,
                                                                                 champion_case_id integer, champion_quality double precision DEFAULT 0.0 NOT NULL,
                                                                                                                                                         run_ix integer DEFAULT 0 NOT NULL,
                                                                                                                                                                                  wins integer DEFAULT 0 NOT NULL,
                                                                                                                                                                                                         losses integer DEFAULT 0 NOT NULL,
                                                                                                                                                                                                                                  avg_delta double precision DEFAULT 0.0 NOT NULL,
                                                                                                                                                                                                                                                                         trust double precision DEFAULT 0.0 NOT NULL,
                                                                                                                                                                                                                                                                                                            created_at timestamp WITH TIME ZONE DEFAULT now() NOT NULL,
                                                                                                                                                                                                                                                                                                                                                              updated_at timestamp WITH TIME ZONE DEFAULT now() NOT NULL);

--
-- Name: case_goal_state_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.case_goal_state_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: case_goal_state_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.case_goal_state_id_seq OWNED BY public.case_goal_state.id;

--
-- Name: case_scorables; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.case_scorables (id integer NOT NULL,
                                               case_id integer NOT NULL,
                                                               scorable_id CHARACTER varying(64) NOT NULL,
                                                                                                 scorable_type CHARACTER varying(64),
                                                                                                                         ROLE CHARACTER varying(64),
                                                                                                                                        created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                       meta JSONB,
                                                                                                                                                                                            rank integer);

--
-- Name: case_scorables_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.case_scorables_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: case_scorables_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.case_scorables_id_seq OWNED BY public.case_scorables.id;

--
-- Name: casebooks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.casebooks (id integer NOT NULL,
                                          name text NOT NULL,
                                                    description text, created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                     pipeline_run_id integer, agent_name text, meta JSONB DEFAULT '{}'::JSONB NOT NULL,
                                                                                                                                                                                              tags JSONB DEFAULT '[]'::JSONB NOT NULL);

--
-- Name: casebooks_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.casebooks_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: casebooks_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.casebooks_id_seq OWNED BY public.casebooks.id;

--
-- Name: cases; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cases (id integer NOT NULL,
                                      casebook_id integer NOT NULL,
                                                          goal_id CHARACTER varying(64) NOT NULL,
                                                                                        agent_name CHARACTER varying(128) NOT NULL,
                                                                                                                          mars_summary JSONB,
                                                                                                                                       scores JSONB,
                                                                                                                                              created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                             meta JSONB,
                                                                                                                                                                                                  rank JSONB,
                                                                                                                                                                                                       prompt_text text);

--
-- Name: cases_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cases_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: cases_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cases_id_seq OWNED BY public.cases.id;

--
-- Name: chat_conversations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.chat_conversations (id integer NOT NULL,
                                                   provider CHARACTER varying(50) DEFAULT 'openai'::CHARACTER varying NOT NULL,
                                                                                                                      external_id CHARACTER varying(255),
                                                                                                                                            title CHARACTER varying(255) NOT NULL,
                                                                                                                                                                         created_at timestamp WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                                                                                                     updated_at timestamp WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                                                                                                                                                 meta JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                                                                                                                                    tsv
                                        TSVECTOR GENERATED ALWAYS AS ((setweight(to_tsvector('english'::regconfig, (COALESCE(title, ''::CHARACTER varying))::text), 'A'::"char") || setweight(to_tsvector('english'::regconfig, COALESCE((meta)::text, ''::text)), 'D'::"char"))) STORED,
                                                           tags JSONB DEFAULT '[]'::JSONB NOT NULL);

--
-- Name: chat_conversations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.chat_conversations_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: chat_conversations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.chat_conversations_id_seq OWNED BY public.chat_conversations.id;

--
-- Name: chat_messages; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.chat_messages (id integer NOT NULL,
                                              conversation_id integer NOT NULL,
                                                                      ROLE CHARACTER varying(50) NOT NULL, text text, parent_id integer, order_index integer NOT NULL,
                                                                                                                                                             created_at timestamp WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                                                                                         meta JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                                                                            tsv
                                   TSVECTOR GENERATED ALWAYS AS ((setweight(to_tsvector('english'::regconfig, (COALESCE(ROLE, ''::CHARACTER varying))::text), 'D'::"char") || setweight(to_tsvector('english'::regconfig, COALESCE(text, ''::text)), 'B'::"char"))) STORED);

--
-- Name: chat_messages_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.chat_messages_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: chat_messages_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.chat_messages_id_seq OWNED BY public.chat_messages.id;

--
-- Name: chat_turns; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.chat_turns (id integer NOT NULL,
                                           conversation_id integer NOT NULL,
                                                                   user_message_id integer NOT NULL,
                                                                                           assistant_message_id integer NOT NULL,
                                                                                                                        order_index integer, star integer DEFAULT 0 NOT NULL,
                                                                                                                                                                    ner JSONB,
                                                                                                                                                                        domains JSONB,
                                                                                                                                                                                order_index_old integer, ai_knowledge_score integer, ai_knowledge_rationale text);

--
-- Name: COLUMN chat_turns.ner; Type: COMMENT; Schema: public; Owner: -
--
 COMMENT ON COLUMN public.chat_turns.ner IS 'NER annotations per turn: array of {text,label,start,end,score?}. NULL = not annotated.';

--
-- Name: COLUMN chat_turns.domains; Type: COMMENT; Schema: public; Owner: -
--
 COMMENT ON COLUMN public.chat_turns.domains IS 'Domain annotations per turn: array of {domain,score,source}. NULL = not annotated.';

--
-- Name: chat_turns_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.chat_turns_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: chat_turns_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.chat_turns_id_seq OWNED BY public.chat_turns.id;

--
-- Name: comparison_preferences; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.comparison_preferences (id UUID DEFAULT gen_random_uuid() NOT NULL,
                                                                              goal_id integer NOT NULL,
                                                                                              preferred_tag text NOT NULL,
                                                                                                                 rejected_tag text NOT NULL,
                                                                                                                                   preferred_run_id UUID NOT NULL,
                                                                                                                                                         rejected_run_id UUID NOT NULL,
                                                                                                                                                                              preferred_score double precision, rejected_score double precision, dimension_scores JSONB,
                                                                                                                                                                                                                                                                  reason text, created_at timestamp WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                                                                                                                                                                                                           SOURCE text);

--
-- Name: component_interfaces; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.component_interfaces (component_id text, protocol text NOT NULL,
                                                                           implemented boolean DEFAULT TRUE,
                                                                                                       last_checked timestamp WITHOUT TIME ZONE DEFAULT now());

--
-- Name: component_versions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.component_versions (id text NOT NULL,
                                                name text NOT NULL,
                                                          protocol text NOT NULL,
                                                                        class_path text NOT NULL,
                                                                                        VERSION text NOT NULL,
                                                                                                     config JSONB,
                                                                                                            performance JSONB,
                                                                                                                        active boolean DEFAULT TRUE,
                                                                                                                                               sensitivity text, created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                                                last_used timestamp WITHOUT TIME ZONE,
                                                                                                                                                                                                                                                 usage_count integer DEFAULT 0,
                                                                                                                                                                                                                                                                             metadata JSONB,
                                                                                                                                                                                                                                                                                      CONSTRAINT component_versions_sensitivity_check CHECK ((sensitivity = ANY (ARRAY['public'::text,
                                                                                                                                                                                                                                                                                                                                                                       'internal'::text,
                                                                                                                                                                                                                                                                                                                                                                       'confidential'::text,
                                                                                                                                                                                                                                                                                                                                                                       'restricted'::text]))));

--
-- Name: context_states; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.context_states (id integer NOT NULL,
                                               run_id text NOT NULL,
                                                           stage_name text NOT NULL,
                                                                           VERSION integer DEFAULT 1,
                                                                                                   context JSONB NOT NULL,
                                                                                                                 preferences JSONB,
                                                                                                                             feedback JSONB,
                                                                                                                                      extra_data JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                               "timestamp" timestamp WITH TIME ZONE DEFAULT now(),
                                                                                                                                                                                                            is_current boolean DEFAULT TRUE,
                                                                                                                                                                                                                                       pipeline_run_id integer, goal_id integer, trace JSONB,
                                                                                                                                                                                                                                                                                       token_count integer);

--
-- Name: context_states_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.context_states_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: context_states_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.context_states_id_seq OWNED BY public.context_states.id;

--
-- Name: cot_pattern_stats; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cot_pattern_stats (id integer NOT NULL,
                                                  goal_id integer, hypothesis_id integer, model_name text NOT NULL,
                                                                                                          agent_name text NOT NULL,
                                                                                                                          dimension text NOT NULL,
                                                                                                                                         label text NOT NULL,
                                                                                                                                                    confidence_score double precision, created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP);

--
-- Name: cot_pattern_stats_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cot_pattern_stats_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: cot_pattern_stats_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cot_pattern_stats_id_seq OWNED BY public.cot_pattern_stats.id;

--
-- Name: cot_patterns; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cot_patterns (id integer NOT NULL,
                                             goal_id integer, hypothesis_id integer, model_name text, agent_name text, dimension text, label text, created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP);

--
-- Name: cot_patterns_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cot_patterns_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: cot_patterns_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cot_patterns_id_seq OWNED BY public.cot_patterns.id;

--
-- Name: scorable_domains; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scorable_domains (id integer NOT NULL,
                                                 DOMAIN text NOT NULL,
                                                             score double precision NOT NULL,
                                                                                    created_at timestamp WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                updated_at timestamp WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                                                            scorable_id text, scorable_type text);

--
-- Name: document_domains_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.document_domains_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: document_domains_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.document_domains_id_seq OWNED BY public.scorable_domains.id;

--
-- Name: scorable_embeddings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scorable_embeddings (id integer NOT NULL,
                                                    scorable_id CHARACTER varying NOT NULL,
                                                                                  scorable_type CHARACTER varying NOT NULL,
                                                                                                                  embedding_id integer NOT NULL,
                                                                                                                                       embedding_type CHARACTER varying NOT NULL,
                                                                                                                                                                        created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP);

--
-- Name: document_embeddings_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.document_embeddings_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: document_embeddings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.document_embeddings_id_seq OWNED BY public.scorable_embeddings.id;

--
-- Name: document_evaluations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_evaluations (id integer NOT NULL,
                                                     document_id integer NOT NULL,
                                                                         agent_name text, model_name text, evaluator_name text, strategy text, scores JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                    extra_data JSONB,
                                                                                                                                                                               created_at timestamp WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);

--
-- Name: document_evaluations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.document_evaluations_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: document_evaluations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.document_evaluations_id_seq OWNED BY public.document_evaluations.id;

--
-- Name: document_scores; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_scores (id integer NOT NULL,
                                                evaluation_id integer NOT NULL,
                                                                      dimension CHARACTER varying NOT NULL,
                                                                                                  score double precision, weight double precision, rationale text);

--
-- Name: document_scores_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.document_scores_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: document_scores_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.document_scores_id_seq OWNED BY public.document_scores.id;

--
-- Name: document_section_domains; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_section_domains (id integer NOT NULL,
                                                         document_section_id integer NOT NULL,
                                                                                     DOMAIN text NOT NULL,
                                                                                                 score double precision NOT NULL);

--
-- Name: document_section_domains_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.document_section_domains_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: document_section_domains_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.document_section_domains_id_seq OWNED BY public.document_section_domains.id;

--
-- Name: document_sections; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.document_sections (id integer NOT NULL,
                                                  document_id integer NOT NULL,
                                                                      section_name text NOT NULL,
                                                                                        section_text text NOT NULL,
                                                                                                          SOURCE text DEFAULT 'unstructured+llm'::text,
                                                                                                                              SUMMARY text, embedding JSON,
                                                                                                                                                      extra_data JSON,
                                                                                                                                                                 domains text[]);

--
-- Name: document_sections_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.document_sections_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: document_sections_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.document_sections_id_seq OWNED BY public.document_sections.id;

--
-- Name: documents; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.documents (id integer NOT NULL,
                                          title text NOT NULL,
                                                     SOURCE text NOT NULL,
                                                                 external_id text, url text, date_added timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                            SUMMARY text, goal_id integer, domain_label text, domains text[], embedding_id integer, text text);

--
-- Name: documents_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.documents_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: documents_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.documents_id_seq OWNED BY public.documents.id;

--
-- Name: dynamic_scorables; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.dynamic_scorables (id integer NOT NULL,
                                                  pipeline_run_id CHARACTER varying NOT NULL,
                                                                                    case_id integer, scorable_type CHARACTER varying NOT NULL,
                                                                                                                                     SOURCE CHARACTER varying, text text, meta JSONB,
                                                                                                                                                                               created_at timestamp WITHOUT TIME ZONE DEFAULT (now() AT TIME ZONE 'UTC'::text), ROLE text, source_scorable_id integer, source_scorable_type text);

--
-- Name: dynamic_scorables_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.dynamic_scorables_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: dynamic_scorables_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.dynamic_scorables_id_seq OWNED BY public.dynamic_scorables.id;

--
-- Name: elo_ranking_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.elo_ranking_log (id integer NOT NULL,
                                                run_id text, hypothesis text, prompt_version integer, prompt_strategy text, score integer, created_at timestamp WITH TIME ZONE DEFAULT now());

--
-- Name: elo_ranking_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.elo_ranking_log_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: elo_ranking_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.elo_ranking_log_id_seq OWNED BY public.elo_ranking_log.id;

--
-- Name: embeddings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.embeddings (id integer NOT NULL, text text, embedding public.vector(1024),
                                                                created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                               text_hash text);

--
-- Name: embeddings_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.embeddings_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: embeddings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.embeddings_id_seq OWNED BY public.embeddings.id;

--
-- Name: entity_cache; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entity_cache (id integer NOT NULL,
                                             embedding_ref integer NOT NULL,
                                                                   results_json JSON,
                                                                                last_updated timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL);

--
-- Name: entity_cache_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.entity_cache_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: entity_cache_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.entity_cache_id_seq OWNED BY public.entity_cache.id;

--
-- Name: evaluation_attributes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.evaluation_attributes (id integer NOT NULL,
                                                      evaluation_id integer NOT NULL,
                                                                            dimension text NOT NULL,
                                                                                           SOURCE text NOT NULL,
                                                                                                       raw_score real, energy real, uncertainty real, advantage real, pi_value real, q_value real, v_value real, extra JSON,
                                                                                                                                                                                                                       entropy double precision, td_error double precision, expected_return double precision, policy_logits JSON,
                                                                                                                                                                                                                                                                                                                            created_at timestamp WITH TIME ZONE DEFAULT now() NOT NULL,
                                                                                                                                                                                                                                                                                                                                                                              start_time double precision, end_time double precision, duration double precision, error JSONB,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                       output_keys JSONB,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   output_size integer, zsa JSONB,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            CONSTRAINT valid_timing CHECK ((((start_time IS NULL)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             AND (end_time IS NULL))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            OR ((start_time IS NOT NULL)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                AND (end_time IS NULL))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            OR ((start_time IS NULL)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                AND (end_time IS NOT NULL))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            OR (start_time <= end_time))));

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

CREATE SEQUENCE public.evaluation_attributes_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: evaluation_attributes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.evaluation_attributes_id_seq OWNED BY public.evaluation_attributes.id;

--
-- Name: evaluation_rule_links; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.evaluation_rule_links (id integer NOT NULL,
                                                      evaluation_id integer NOT NULL,
                                                                            rule_application_id integer NOT NULL,
                                                                                                        created_at timestamp WITHOUT TIME ZONE DEFAULT now());

--
-- Name: evaluations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.evaluations (id integer NOT NULL,
                                            goal_id integer, agent_name text NOT NULL,
                                                                             model_name text NOT NULL,
                                                                                             evaluator_name text NOT NULL,
                                                                                                                 strategy text, reasoning_strategy text, run_id text, extra_data JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                                               created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                                                                                                                              symbolic_rule_id integer, rule_application_id integer, pipeline_run_id integer, scores JSON DEFAULT '{}'::JSON,
                                                                                                                                                                                                                                                                                                                                                  scorable_type text, scorable_id text, belief_cartridge_id text, embedding_type text, SOURCE text, query_id CHARACTER varying, query_type CHARACTER varying, plan_trace_id integer);

--
-- Name: events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.events (id integer NOT NULL,
                                       event_type text NOT NULL,
                                                       icon CHARACTER varying(4) DEFAULT '📦'::CHARACTER varying, DATA text NOT NULL,
                                                                                                                           embedding public.vector(1024),
                                                                                                                           hidden boolean DEFAULT FALSE,
                                                                                                                                                  "timestamp" timestamp WITH TIME ZONE DEFAULT now());

--
-- Name: events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.events_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.events_id_seq OWNED BY public.events.id;

--
-- Name: execution_steps; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.execution_steps (id integer NOT NULL,
                                                plan_trace_id integer NOT NULL,
                                                                      step_order integer NOT NULL,
                                                                                         step_id text NOT NULL,
                                                                                                      description text NOT NULL,
                                                                                                                       output_text text NOT NULL,
                                                                                                                                        output_embedding_id integer, evaluation_id integer, meta JSONB,
                                                                                                                                                                                                 created_at timestamp WITHOUT TIME ZONE DEFAULT (now() AT TIME ZONE 'UTC'::text) NOT NULL,
                                                                                                                                                                                                                                                                                 step_type text DEFAULT 'action'::text,
                                                                                                                                                                                                                                                                                                        input_text text, pipeline_run_id integer, agent_role CHARACTER varying);

--
-- Name: execution_steps_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.execution_steps_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: execution_steps_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.execution_steps_id_seq OWNED BY public.execution_steps.id;

--
-- Name: experiment_model_snapshots; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.experiment_model_snapshots (id integer NOT NULL,
                                                           experiment_id integer NOT NULL,
                                                                                 name CHARACTER varying(128) NOT NULL,
                                                                                                             DOMAIN CHARACTER varying(128),
                                                                                                                              VERSION integer NOT NULL,
                                                                                                                                              payload JSONB DEFAULT '{}'::JSONB NOT NULL,
                                                                                                                                                                                validation JSONB,
                                                                                                                                                                                           committed_at timestamp WITHOUT TIME ZONE DEFAULT now() NOT NULL,
                                                                                                                                                                                                                                                  created_at timestamp WITHOUT TIME ZONE DEFAULT now() NOT NULL);

--
-- Name: experiment_model_snapshots_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.experiment_model_snapshots
ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY
  (SEQUENCE NAME public.experiment_model_snapshots_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1);

--
-- Name: experiment_trial_metrics; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.experiment_trial_metrics (id integer NOT NULL,
                                                         trial_id integer NOT NULL,
                                                                          KEY CHARACTER varying(64) NOT NULL,
                                                                                                    value double precision NOT NULL,
                                                                                                                           created_at timestamp WITHOUT TIME ZONE DEFAULT now() NOT NULL);

--
-- Name: experiment_trial_metrics_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.experiment_trial_metrics
ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY
  (SEQUENCE NAME public.experiment_trial_metrics_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1);

--
-- Name: experiment_trials; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.experiment_trials (id integer NOT NULL,
                                                  variant_id integer NOT NULL,
                                                                     case_id integer, pipeline_run_id CHARACTER varying(64),
                                                                                                                DOMAIN CHARACTER varying(64),
                                                                                                                                 assigned_at timestamp WITHOUT TIME ZONE DEFAULT now() NOT NULL,
                                                                                                                                                                                       completed_at timestamp WITHOUT TIME ZONE,
                                                                                                                                                                                                                           performance double precision, tokens integer, cost double precision, wall_sec double precision, meta JSONB DEFAULT '{}'::JSONB NOT NULL,
                                                                                                                                                                                                                                                                                                                                                          tags_used JSONB DEFAULT '[]'::JSONB NOT NULL,
                                                                                                                                                                                                                                                                                                                                                                                              experiment_group CHARACTER varying);

--
-- Name: experiment_trials_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.experiment_trials
ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY
  (SEQUENCE NAME public.experiment_trials_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1);

--
-- Name: experiment_variants; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.experiment_variants (id integer NOT NULL,
                                                    experiment_id integer NOT NULL,
                                                                          name CHARACTER varying(16) NOT NULL,
                                                                                                     is_control boolean DEFAULT FALSE NOT NULL,
                                                                                                                                      payload JSONB DEFAULT '{}'::JSONB NOT NULL,
                                                                                                                                                                        created_at timestamp WITHOUT TIME ZONE DEFAULT now() NOT NULL);

--
-- Name: experiment_variants_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.experiment_variants
ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY
  (SEQUENCE NAME public.experiment_variants_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1);

--
-- Name: experiments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.experiments (id integer NOT NULL,
                                            name CHARACTER varying(128) NOT NULL,
                                                                        label CHARACTER varying(128),
                                                                                        status CHARACTER varying(32) DEFAULT 'active'::CHARACTER varying NOT NULL,
                                                                                                                                                         DOMAIN CHARACTER varying(64),
                                                                                                                                                                          config JSONB DEFAULT '{}'::JSONB NOT NULL,
                                                                                                                                                                                                           created_at timestamp WITHOUT TIME ZONE DEFAULT now() NOT NULL);

--
-- Name: experiments_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.experiments
ALTER COLUMN id ADD GENERATED BY DEFAULT AS IDENTITY
  (SEQUENCE NAME public.experiments_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1);

--
-- Name: goal_dimensions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.goal_dimensions (id integer NOT NULL,
                                                goal_id integer NOT NULL,
                                                                dimension text NOT NULL,
                                                                               rank integer DEFAULT 0,
                                                                                                    SOURCE text DEFAULT 'llm'::text,
                                                                                                                        similarity_score double precision, created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP);

--
-- Name: goal_dimensions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.goal_dimensions_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: goal_dimensions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.goal_dimensions_id_seq OWNED BY public.goal_dimensions.id;

--
-- Name: goals; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.goals (id integer NOT NULL,
                                      goal_text text NOT NULL,
                                                     goal_type text, focus_area text, strategy text, llm_suggested_strategy text, SOURCE text DEFAULT 'user'::text,
                                                                                                                                                      created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                                     goal_category CHARACTER varying DEFAULT 'analyze'::CHARACTER varying, difficulty CHARACTER varying DEFAULT 'medium'::CHARACTER varying);

--
-- Name: goals_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.goals_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: goals_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.goals_id_seq OWNED BY public.goals.id;

--
-- Name: hf_embeddings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.hf_embeddings (id integer NOT NULL, text text, embedding public.vector(1024),
                                                                   created_at timestamp WITH TIME ZONE DEFAULT now(),
                                                                                                               text_hash text);

--
-- Name: hf_embeddings_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.hf_embeddings_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: hf_embeddings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.hf_embeddings_id_seq OWNED BY public.hf_embeddings.id;

--
-- Name: hnet_embeddings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.hnet_embeddings (id integer NOT NULL, text text, embedding public.vector(1024),
                                                                     created_at timestamp WITH TIME ZONE DEFAULT now(),
                                                                                                                 text_hash text);

--
-- Name: hnet_embeddings_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.hnet_embeddings_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: hnet_embeddings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.hnet_embeddings_id_seq OWNED BY public.hnet_embeddings.id;

--
-- Name: hypotheses; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.hypotheses (id integer NOT NULL, text text NOT NULL,
                                                               confidence double precision DEFAULT 0.0,
                                                                                                   review text, elo_rating double precision DEFAULT 1000.0,
                                                                                                                                                    embedding public.vector(1024),
                                                                                                                                                    features JSONB,
                                                                                                                                                             prompt_id integer, source_hypothesis_id integer, strategy text, VERSION integer DEFAULT 1,
                                                                                                                                                                                                                                                     SOURCE text, enabled boolean DEFAULT TRUE,
                                                                                                                                                                                                                                                                                          created_at timestamp WITH TIME ZONE DEFAULT now(),
                                                                                                                                                                                                                                                                                                                                      updated_at timestamp WITH TIME ZONE DEFAULT now(),
                                                                                                                                                                                                                                                                                                                                                                                  reflection text, goal_id integer, pipeline_signature text, pipeline_run_id integer);

--
-- Name: hypotheses_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.hypotheses_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: hypotheses_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.hypotheses_id_seq OWNED BY public.hypotheses.id;

--
-- Name: ideas; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ideas (id integer NOT NULL,
                                      idea_text text NOT NULL,
                                                     parent_goal text, focus_area text, strategy text, SOURCE text, origin text, extra_data JSON,
                                                                                                                                            goal_id integer, created_at timestamp WITHOUT TIME ZONE DEFAULT now());

--
-- Name: ideas_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ideas_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: ideas_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ideas_id_seq OWNED BY public.ideas.id;

--
-- Name: knowledge_documents; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.knowledge_documents (id integer NOT NULL,
                                                    title text NOT NULL,
                                                               SUMMARY text, text text, url text, external_id text, SOURCE text, goal_id integer, created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                                                                                 domain_label text, content text);

--
-- Name: knowledge_documents_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.knowledge_documents_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: knowledge_documents_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.knowledge_documents_id_seq OWNED BY public.knowledge_documents.id;

--
-- Name: knowledge_sections; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.knowledge_sections (id integer NOT NULL,
                                                   document_id integer, section_title text, section_text text NOT NULL,
                                                                                                              embedding public.vector,
                                                                                                              DOMAIN text, domain_score double precision, created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP);

--
-- Name: knowledge_sections_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.knowledge_sections_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: knowledge_sections_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.knowledge_sections_id_seq OWNED BY public.knowledge_sections.id;

--
-- Name: lookaheads; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.lookaheads (id integer NOT NULL,
                                           goal_id integer, agent_name text NOT NULL,
                                                                            model_name text NOT NULL,
                                                                                            input_pipeline text[], suggested_pipeline text[], rationale text, reflection text, backup_plans text[], extra_data JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                                                                             run_id text, created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP);

--
-- Name: lookaheads_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.lookaheads_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: lookaheads_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.lookaheads_id_seq OWNED BY public.lookaheads.id;

--
-- Name: mars_conflicts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mars_conflicts (id integer NOT NULL,
                                               pipeline_run_id integer, plan_trace_id CHARACTER varying, dimension CHARACTER varying NOT NULL,
                                                                                                                                     primary_conflict JSON NOT NULL,
                                                                                                                                                           delta double precision NOT NULL,
                                                                                                                                                                                  agreement_score double precision, preferred_model CHARACTER varying, explanation text, created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP);

--
-- Name: mars_conflicts_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.mars_conflicts_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: mars_conflicts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.mars_conflicts_id_seq OWNED BY public.mars_conflicts.id;

--
-- Name: mars_results; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mars_results (id integer NOT NULL,
                                             pipeline_run_id integer, plan_trace_id CHARACTER varying, dimension CHARACTER varying NOT NULL,
                                                                                                                                   agreement_score double precision NOT NULL,
                                                                                                                                                                    std_dev double precision NOT NULL,
                                                                                                                                                                                             preferred_model CHARACTER varying, primary_conflict JSONB,
                                                                                                                                                                                                                                                 delta double precision, high_disagreement boolean DEFAULT FALSE NOT NULL,
                                                                                                                                                                                                                                                                                                                 explanation text, scorer_metrics JSONB,
                                                                                                                                                                                                                                                                                                                                                  metric_correlations JSONB,
                                                                                                                                                                                                                                                                                                                                                                      created_at timestamp WITHOUT TIME ZONE DEFAULT (now() AT TIME ZONE 'UTC'::text), SOURCE text, average_score double precision);

--
-- Name: mars_results_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.mars_results_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: mars_results_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.mars_results_id_seq OWNED BY public.mars_results.id;

--
-- Name: measurements; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.measurements (id integer NOT NULL,
                                             entity_type text NOT NULL,
                                                              entity_id integer NOT NULL,
                                                                                metric_name text NOT NULL,
                                                                                                 value JSONB NOT NULL,
                                                                                                             context JSONB,
                                                                                                                     created_at timestamp WITH TIME ZONE DEFAULT now());

--
-- Name: measurements_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.measurements_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: measurements_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.measurements_id_seq OWNED BY public.measurements.id;

--
-- Name: mem_cubes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mem_cubes (id integer NOT NULL,
                                          scorable_type text NOT NULL,
                                                             scorable_id integer NOT NULL,
                                                                                 state text DEFAULT 'raw'::text,
                                                                                                    tags text[], notes text, created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                            updated_at timestamp WITHOUT TIME ZONE DEFAULT now());

--
-- Name: mem_cubes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.mem_cubes_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: mem_cubes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.mem_cubes_id_seq OWNED BY public.mem_cubes.id;

--
-- Name: memcube_transformations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.memcube_transformations (id integer NOT NULL,
                                                        source_cube_id text NOT NULL,
                                                                            target_cube_id text NOT NULL,
                                                                                                transformation_type text, confidence double precision, created_at timestamp WITHOUT TIME ZONE DEFAULT now());

--
-- Name: memcube_transformations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.memcube_transformations_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: memcube_transformations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.memcube_transformations_id_seq OWNED BY public.memcube_transformations.id;

--
-- Name: memcube_versions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.memcube_versions (id text NOT NULL,
                                              cube_id text NOT NULL,
                                                           scorable_type text NOT NULL,
                                                                              content_hash text, VERSION text NOT NULL,
                                                                                                              created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                             last_modified timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                                               sensitivity text, usage_count integer DEFAULT 0,
                                                                                                                                                                                                                                                             extra_data JSONB,
                                                                                                                                                                                                                                                                        CONSTRAINT memcube_versions_sensitivity_check CHECK ((sensitivity = ANY (ARRAY['public'::text,
                                                                                                                                                                                                                                                                                                                                                       'internal'::text,
                                                                                                                                                                                                                                                                                                                                                       'confidential'::text,
                                                                                                                                                                                                                                                                                                                                                       'restricted'::text,
                                                                                                                                                                                                                                                                                                                                                       'archived'::text]))));

--
-- Name: memcubes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.memcubes (id text NOT NULL,
                                      scorable_id bigint NOT NULL,
                                                         scorable_type text NOT NULL,
                                                                            content text NOT NULL,
                                                                                         dimension text, original_score double precision, refined_score double precision, refined_content text, VERSION text NOT NULL,
                                                                                                                                                                                                             SOURCE text, model text, priority integer DEFAULT 5,
                                                                                                                                                                                                                                                               sensitivity text DEFAULT 'public'::text,
                                                                                                                                                                                                                                                                                        ttl integer, usage_count integer DEFAULT 0,
                                                                                                                                                                                                                                                                                                                                 extra_data JSONB,
                                                                                                                                                                                                                                                                                                                                            created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                                                                                                                                                                                                                           last_modified timestamp WITHOUT TIME ZONE DEFAULT now());

--
-- Name: method_plans; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.method_plans (id integer NOT NULL,
                                             idea_text text NOT NULL,
                                                            idea_id integer, goal_id integer NOT NULL,
                                                                                             research_objective text NOT NULL,
                                                                                                                     key_components JSONB,
                                                                                                                                    experimental_plan text, hypothesis_mapping JSONB,
                                                                                                                                                                               search_strategy text, knowledge_gaps text, next_steps text, task_description text, baseline_method text, literature_summary text, code_plan text, focus_area text, strategy text, score_novelty double precision, score_feasibility double precision, score_impact double precision, score_alignment double precision, evolution_level integer DEFAULT 0,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      parent_plan_id integer, is_refinement boolean DEFAULT FALSE,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            created_at timestamp WITHOUT TIME ZONE DEFAULT now());

--
-- Name: method_plans_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.method_plans_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: method_plans_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.method_plans_id_seq OWNED BY public.method_plans.id;

--
-- Name: model_artifacts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_artifacts (id integer NOT NULL,
                                                name CHARACTER varying NOT NULL,
                                                                       VERSION integer DEFAULT 1 NOT NULL,
                                                                                                 PATH CHARACTER varying NOT NULL,
                                                                                                                        tag CHARACTER varying, meta JSONB DEFAULT '{}'::JSONB NOT NULL,
                                                                                                                                                                              created_at timestamp WITH TIME ZONE DEFAULT now() NOT NULL,
                                                                                                                                                                                                                                updated_at timestamp WITH TIME ZONE DEFAULT now() NOT NULL);

--
-- Name: model_artifacts_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

ALTER TABLE public.model_artifacts
ALTER COLUMN id ADD GENERATED ALWAYS AS IDENTITY
  (SEQUENCE NAME public.model_artifacts_id_seq START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1);

--
-- Name: model_performance; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_performance (id integer NOT NULL,
                                                  model_name text NOT NULL,
                                                                  task_type text NOT NULL,
                                                                                 prompt_strategy text NOT NULL,
                                                                                                      preference_used text[], reward double precision NOT NULL,
                                                                                                                                                      confidence_score double precision, metadata JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                                                                created_at timestamp WITH TIME ZONE DEFAULT now());

--
-- Name: model_performance_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.model_performance_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: model_performance_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.model_performance_id_seq OWNED BY public.model_performance.id;

--
-- Name: model_versions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.model_versions (id integer NOT NULL,
                                               model_type text NOT NULL,
                                                               target_type text NOT NULL,
                                                                                dimension text NOT NULL,
                                                                                               VERSION text NOT NULL,
                                                                                                            trained_on JSONB,
                                                                                                                       performance JSONB,
                                                                                                                                   created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                  active boolean DEFAULT TRUE,
                                                                                                                                                                                                         extra_data JSONB,
                                                                                                                                                                                                                    model_path text, encoder_path text, tuner_path text, scaler_path text, meta_path text, description text, SOURCE text DEFAULT 'user'::text,
                                                                                                                                                                                                                                                                                                                                                 score_mode text);

--
-- Name: model_versions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.model_versions_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: model_versions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.model_versions_id_seq OWNED BY public.model_versions.id;

--
-- Name: mrq_evaluations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mrq_evaluations (id integer NOT NULL,
                                                goal text NOT NULL,
                                                          prompt text NOT NULL,
                                                                      output_a text NOT NULL,
                                                                                    output_b text NOT NULL,
                                                                                                  winner text NOT NULL,
                                                                                                              score_a double precision NOT NULL,
                                                                                                                                       score_b double precision NOT NULL,
                                                                                                                                                                metadata JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                                       created_at timestamp WITH TIME ZONE DEFAULT now());

--
-- Name: mrq_evaluations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.mrq_evaluations_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: mrq_evaluations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.mrq_evaluations_id_seq OWNED BY public.mrq_evaluations.id;

--
-- Name: mrq_memory; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mrq_memory (id integer NOT NULL,
                                           goal text NOT NULL,
                                                     strategy text NOT NULL,
                                                                   prompt text NOT NULL,
                                                                               response text NOT NULL,
                                                                                             reward double precision NOT NULL,
                                                                                                                     embedding public.vector(1024),
                                                                                                                     extra_data JSONB DEFAULT '{}'::JSONB,
                                                                                                                                              created_at timestamp WITH TIME ZONE DEFAULT now());

--
-- Name: mrq_memory_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.mrq_memory_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: mrq_memory_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.mrq_memory_id_seq OWNED BY public.mrq_memory.id;

--
-- Name: mrq_preference_pairs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.mrq_preference_pairs (id integer NOT NULL,
                                                     goal text NOT NULL,
                                                               prompt text NOT NULL,
                                                                           output_a text NOT NULL,
                                                                                         output_b text NOT NULL,
                                                                                                       preferred text NOT NULL,
                                                                                                                      fmt_a text, fmt_b text, difficulty text, SOURCE text, run_id text, features JSONB,
                                                                                                                                                                                                  created_at timestamp WITHOUT TIME ZONE DEFAULT now());

--
-- Name: mrq_preference_pairs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.mrq_preference_pairs_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: mrq_preference_pairs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.mrq_preference_pairs_id_seq OWNED BY public.mrq_preference_pairs.id;

--
-- Name: nodes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.nodes (id integer NOT NULL,
                                      goal_id CHARACTER varying NOT NULL,
                                                                pipeline_run_id integer, stage_name CHARACTER varying, config JSON,
                                                                                                                              hypothesis text, metric double precision, VALID boolean DEFAULT TRUE,
                                                                                                                                                                                              created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP);

--
-- Name: nodes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.nodes_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: nodes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.nodes_id_seq OWNED BY public.nodes.id;

--
-- Name: pipeline_references; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pipeline_references (id integer NOT NULL,
                                                    pipeline_run_id integer NOT NULL,
                                                                            scorable_type CHARACTER varying NOT NULL,
                                                                                                            scorable_id CHARACTER varying NOT NULL,
                                                                                                                                          relation_type CHARACTER varying, created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
                                                                                                                                                                                                                                            SOURCE text);

--
-- Name: pipeline_references_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.pipeline_references_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: pipeline_references_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.pipeline_references_id_seq OWNED BY public.pipeline_references.id;

--
-- Name: pipeline_runs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pipeline_runs (id integer NOT NULL,
                                              goal_id integer, run_id text NOT NULL,
                                                                           pipeline JSONB,
                                                                                    strategy text, model_name text, run_config JSONB,
                                                                                                                               lookahead_context JSONB,
                                                                                                                                                 symbolic_suggestion JSONB,
                                                                                                                                                                     extra_data JSONB,
                                                                                                                                                                                created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                                                                                                               pipeline_text_backup text, name text, tag text, description text, embedding_type text, embedding_dimensions integer);

--
-- Name: pipeline_runs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.pipeline_runs_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: pipeline_runs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.pipeline_runs_id_seq OWNED BY public.pipeline_runs.id;

--
-- Name: pipeline_stages; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.pipeline_stages (id integer NOT NULL,
                                                stage_name CHARACTER varying NOT NULL,
                                                                             agent_class CHARACTER varying NOT NULL,
                                                                                                           goal_id CHARACTER varying, run_id CHARACTER varying NOT NULL,
                                                                                                                                                               pipeline_run_id integer, parent_stage_id integer, input_context_id integer, output_context_id integer, "timestamp" timestamp WITHOUT TIME ZONE DEFAULT now() NOT NULL,
                                                                                                                                                                                                                                                                                                                            status CHARACTER varying NOT NULL,
                                                                                                                                                                                                                                                                                                                                                     score numeric, confidence numeric, symbols_applied JSONB,
                                                                                                                                                                                                                                                                                                                                                                                                        extra_data JSONB,
                                                                                                                                                                                                                                                                                                                                                                                                                   exportable boolean, reusable boolean, invalidated boolean);

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

CREATE SEQUENCE public.pipeline_stages_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: pipeline_stages_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.pipeline_stages_id_seq OWNED BY public.pipeline_stages.id;

--
-- Name: plan_trace_reuse_links; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.plan_trace_reuse_links (id integer NOT NULL,
                                                       parent_trace_id CHARACTER varying NOT NULL,
                                                                                         child_trace_id CHARACTER varying NOT NULL,
                                                                                                                          created_at timestamp WITHOUT TIME ZONE DEFAULT (now() AT TIME ZONE 'UTC'::text));

--
-- Name: plan_trace_reuse_links_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.plan_trace_reuse_links_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: plan_trace_reuse_links_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.plan_trace_reuse_links_id_seq OWNED BY public.plan_trace_reuse_links.id;

--
-- Name: plan_traces; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.plan_traces (id integer NOT NULL,
                                            trace_id text NOT NULL,
                                                          goal_id integer, goal_embedding_id integer, plan_signature text NOT NULL,
                                                                                                                          final_output_text text NOT NULL,
                                                                                                                                                 final_output_embedding_id integer, target_epistemic_quality double precision, target_epistemic_quality_source text, meta JSONB,
                                                                                                                                                                                                                                                                          created_at timestamp WITHOUT TIME ZONE DEFAULT (now() AT TIME ZONE 'UTC'::text) NOT NULL,
                                                                                                                                                                                                                                                                                                                                                          updated_at timestamp WITHOUT TIME ZONE DEFAULT (now() AT TIME ZONE 'UTC'::text) NOT NULL,
                                                                                                                                                                                                                                                                                                                                                                                                                                          pipeline_run_id integer, retrieved_cases JSONB DEFAULT '[]'::JSONB,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 strategy_used text, reward_signal JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 skills_used JSONB DEFAULT '[]'::JSONB,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           repair_links JSONB DEFAULT '[]'::JSONB,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      domains text[] DEFAULT '{}'::text[]);

--
-- Name: plan_traces_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.plan_traces_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: plan_traces_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.plan_traces_id_seq OWNED BY public.plan_traces.id;

--
-- Name: prompt_evaluations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_evaluations (id integer NOT NULL,
                                                   prompt_id integer NOT NULL,
                                                                     benchmark_name text NOT NULL,
                                                                                         score double precision, metrics JSONB DEFAULT '{}'::JSONB,
                                                                                                                                       dataset_hash text, evaluator text DEFAULT 'auto'::text,
                                                                                                                                                                                 notes text, created_at timestamp WITH TIME ZONE DEFAULT now());

--
-- Name: prompt_evaluations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompt_evaluations_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: prompt_evaluations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompt_evaluations_id_seq OWNED BY public.prompt_evaluations.id;

--
-- Name: prompt_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_history (id integer NOT NULL,
                                               original_prompt_id integer, prompt_text text NOT NULL,
                                                                                            agent_name text NOT NULL,
                                                                                                            strategy text NOT NULL,
                                                                                                                          prompt_key text NOT NULL,
                                                                                                                                          output_key text, input_key text, extraction_regex text, VERSION integer DEFAULT 1,
                                                                                                                                                                                                                          SOURCE text, is_current boolean DEFAULT FALSE,
                                                                                                                                                                                                                                                                  config JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                                                                                                                                       metadata JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                                                                                                                                                              created_at timestamp WITH TIME ZONE DEFAULT now(),
                                                                                                                                                                                                                                                                                                                                                          updated_at timestamp WITH TIME ZONE DEFAULT now());

--
-- Name: prompt_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompt_history_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: prompt_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompt_history_id_seq OWNED BY public.prompt_history.id;

--
-- Name: prompt_programs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_programs (id text NOT NULL,
                                             goal text NOT NULL,
                                                       TEMPLATE text NOT NULL,
                                                                     inputs JSON DEFAULT '{}'::JSON,
                                                                                         VERSION integer DEFAULT 1,
                                                                                                                 parent_id text, prompt_id integer, pipeline_run_id integer, strategy text DEFAULT 'default'::text,
                                                                                                                                                                                                   prompt_text text, hypothesis text, score double precision, rationale text, mutation_type text, execution_trace text, extra_data JSON DEFAULT '{}'::JSON);

--
-- Name: prompt_versions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_versions (id integer NOT NULL,
                                                agent_name text NOT NULL,
                                                                prompt_key text NOT NULL,
                                                                                prompt_text text NOT NULL,
                                                                                                 previous_prompt_id integer, strategy text, VERSION integer NOT NULL,
                                                                                                                                                            SOURCE text, score_improvement double precision, metadata JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                                                                                    created_at timestamp WITH TIME ZONE DEFAULT now());

--
-- Name: prompt_versions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompt_versions_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: prompt_versions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompt_versions_id_seq OWNED BY public.prompt_versions.id;

--
-- Name: prompts; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompts (id integer NOT NULL,
                                        agent_name text NOT NULL,
                                                        prompt_key text NOT NULL,
                                                                        prompt_text text NOT NULL,
                                                                                         response_text text, SOURCE text, VERSION integer DEFAULT 1,
                                                                                                                                                  is_current boolean DEFAULT FALSE,
                                                                                                                                                                             strategy text, extra_data JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                                                                     "timestamp" timestamp WITH TIME ZONE DEFAULT now(),
                                                                                                                                                                                                                                                                  goal_id integer, pipeline_run_id integer, embedding_id integer);

--
-- Name: prompts_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompts_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: prompts_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompts_id_seq OWNED BY public.prompts.id;

--
-- Name: protocols; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.protocols (name text NOT NULL,
                                         description text, input_format JSONB,
                                                                        output_format JSONB,
                                                                                      failure_modes JSONB,
                                                                                                    depends_on JSONB,
                                                                                                               tags JSONB,
                                                                                                                    capability text, preferred_for JSONB,
                                                                                                                                                   avoid_for JSONB);

--
-- Name: ranking_trace; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.ranking_trace (id integer NOT NULL,
                                              run_id text, prompt_version integer, prompt_strategy text, winner text, loser text, explanation text, created_at timestamp WITH TIME ZONE DEFAULT now());

--
-- Name: ranking_trace_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.ranking_trace_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: ranking_trace_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.ranking_trace_id_seq OWNED BY public.ranking_trace.id;

--
-- Name: refinement_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.refinement_events (id integer NOT NULL,
                                                  context text NOT NULL,
                                                               original text NOT NULL,
                                                                             refined text NOT NULL,
                                                                                          context_hash text NOT NULL,
                                                                                                            original_hash text NOT NULL,
                                                                                                                               refined_hash text NOT NULL,
                                                                                                                                                 original_score double precision, refined_score double precision, dimension text NOT NULL,
                                                                                                                                                                                                                                 improvement double precision, energy_before double precision, energy_after double precision, steps_used integer, SOURCE text DEFAULT 'auto'::text,
                                                                                                                                                                                                                                                                                                                                                                      created_at timestamp WITHOUT TIME ZONE DEFAULT now());

--
-- Name: refinement_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.refinement_events_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: refinement_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.refinement_events_id_seq OWNED BY public.refinement_events.id;

--
-- Name: reflection_deltas; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.reflection_deltas (id integer NOT NULL,
                                                  goal_id integer, run_id_a text NOT NULL,
                                                                                 run_id_b text NOT NULL,
                                                                                               score_a double precision, score_b double precision, score_delta double precision, pipeline_diff JSONB,
                                                                                                                                                                                               strategy_diff boolean, model_diff boolean, rationale_diff JSONB,
                                                                                                                                                                                                                                                         created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                                                                                                                                                                                        pipeline_a JSONB,
                                                                                                                                                                                                                                                                                                                   pipeline_b JSONB);

--
-- Name: reflection_deltas_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.reflection_deltas_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: reflection_deltas_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.reflection_deltas_id_seq OWNED BY public.reflection_deltas.id;

--
-- Name: reports; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.reports (id integer NOT NULL,
                                        goal text, SUMMARY text, PATH text NOT NULL,
                                                                           "timestamp" timestamp WITH TIME ZONE DEFAULT now(),
                                                                                                                        content text, created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                     run_id integer);

--
-- Name: reports_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.reports_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: reports_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.reports_id_seq OWNED BY public.reports.id;

--
-- Name: rule_applications; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.rule_applications (id integer NOT NULL,
                                                  rule_id integer, goal_id integer, pipeline_run_id integer, hypothesis_id integer, applied_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                   agent_name text, change_type text, details JSONB,
                                                                                                                                                                                                                              post_score double precision, pre_score double precision, delta_score double precision, evaluator_name text, rationale text, notes text, context_hash text, stage_details JSON);

--
-- Name: rule_applications_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.rule_applications_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: rule_applications_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.rule_applications_id_seq OWNED BY public.rule_applications.id;

--
-- Name: scorable_entities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scorable_entities
  (id integer NOT NULL,
              scorable_id CHARACTER varying NOT NULL,
                                            scorable_type CHARACTER varying NOT NULL,
                                                                            entity_text text NOT NULL,
                                                                                             entity_type CHARACTER varying, START integer, "end" integer, similarity double precision, source_text text, created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                                                                                        entity_text_norm text, ner_confidence double precision);

--
-- Name: scorable_entities_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scorable_entities_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: scorable_entities_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scorable_entities_id_seq OWNED BY public.scorable_entities.id;

--
-- Name: scores; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scores (id integer NOT NULL,
                                       evaluation_id integer, dimension text NOT NULL,
                                                                             score double precision, weight double precision, rationale text, SOURCE text, prompt_hash text, uncertainty double precision, energy double precision);

--
-- Name: scorable_rankings; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.scorable_rankings AS
SELECT e.id AS evaluation_id,
       e.created_at,
       e.query_type,
       e.query_id,
       e.scorable_type AS candidate_type,
       e.scorable_id AS candidate_id,
       (e.scores ->> 'rank_score'::text) AS rank_score,
       s.dimension,
       s.score AS component_score,
       s.weight
FROM (public.evaluations e
      JOIN public.scores s ON ((s.evaluation_id = e.id)))
WHERE (e.evaluator_name = 'ScorableRanker'::text);

--
-- Name: scorable_ranks; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scorable_ranks (id integer NOT NULL,
                                               query_text text NOT NULL,
                                                               scorable_id text NOT NULL,
                                                                                scorable_type text NOT NULL,
                                                                                                   rank_score double precision NOT NULL,
                                                                                                                               components JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                        embedding_type text, created_at timestamp WITH TIME ZONE DEFAULT now() NOT NULL,
                                                                                                                                                                                                                               evaluation_id integer);

--
-- Name: scorable_ranks_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scorable_ranks_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: scorable_ranks_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scorable_ranks_id_seq OWNED BY public.scorable_ranks.id;

--
-- Name: score_attributes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.score_attributes (id integer NOT NULL,
                                                 score_id integer NOT NULL,
                                                                  KEY text NOT NULL,
                                                                           value text NOT NULL,
                                                                                      data_type CHARACTER varying(32) NOT NULL,
                                                                                                                      created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL);

--
-- Name: score_attributes_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.score_attributes_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: score_attributes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.score_attributes_id_seq OWNED BY public.score_attributes.id;

--
-- Name: score_dimensions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.score_dimensions (id integer NOT NULL,
                                                 name CHARACTER varying NOT NULL,
                                                                        stage CHARACTER varying, prompt_template text NOT NULL,
                                                                                                                      weight double precision, notes text, extra_data JSON);

--
-- Name: score_dimensions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.score_dimensions_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: score_dimensions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.score_dimensions_id_seq OWNED BY public.score_dimensions.id;

--
-- Name: score_rule_links_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.score_rule_links_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: score_rule_links_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.score_rule_links_id_seq OWNED BY public.evaluation_rule_links.id;

--
-- Name: scores_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scores_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: scores_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scores_id_seq OWNED BY public.evaluations.id;

--
-- Name: scores_id_seq1; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scores_id_seq1 AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: scores_id_seq1; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scores_id_seq1 OWNED BY public.scores.id;

--
-- Name: scoring_dimensions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scoring_dimensions (event_id integer NOT NULL,
                                                         dimension text NOT NULL,
                                                                        mrq_score double precision, ebt_energy double precision, uncertainty_score double precision, final_score double precision);

--
-- Name: scoring_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scoring_events (id integer NOT NULL,
                                               document_id integer NOT NULL,
                                                                   goal_text text NOT NULL,
                                                                                  original_text text, refined_text text, final_source text NOT NULL,
                                                                                                                                           used_refinement boolean DEFAULT FALSE,
                                                                                                                                                                           refinement_steps integer, used_llm_fallback boolean DEFAULT FALSE,
                                                                                                                                                                                                                                       created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                                                                                                                      memcube_id text, VERSION text, sensitivity text, SOURCE text, model text);

--
-- Name: scoring_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scoring_events_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: scoring_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scoring_events_id_seq OWNED BY public.scoring_events.id;

--
-- Name: scoring_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.scoring_history (id integer NOT NULL,
                                                model_version_id integer, goal_id integer, target_id integer NOT NULL,
                                                                                                             target_type text NOT NULL,
                                                                                                                              dimension text NOT NULL,
                                                                                                                                             raw_score double precision, transformed_score double precision, uncertainty_score double precision, METHOD text NOT NULL,
                                                                                                                                                                                                                                                             SOURCE text, created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                                                                                                                                                         pipeline_run_id integer, model_type text);

--
-- Name: scoring_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.scoring_history_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: scoring_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.scoring_history_id_seq OWNED BY public.scoring_history.id;

--
-- Name: search_hits; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.search_hits (id integer NOT NULL,
                                            query CHARACTER varying NOT NULL,
                                                                    SOURCE CHARACTER varying NOT NULL,
                                                                                             result_type CHARACTER varying, title CHARACTER varying, SUMMARY text, url CHARACTER varying, goal_id integer, parent_goal text, strategy CHARACTER varying, focus_area CHARACTER varying, extra_data JSON);

--
-- Name: search_hits_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.search_hits_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: search_hits_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.search_hits_id_seq OWNED BY public.search_hits.id;

--
-- Name: search_results; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.search_results (id integer NOT NULL,
                                               query text NOT NULL,
                                                          SOURCE text NOT NULL,
                                                                      result_type text, title text, SUMMARY text, url text, author text, published_at timestamp WITHOUT TIME ZONE,
                                                                                                                                                                             tags text[], goal_id integer, parent_goal text, strategy text, focus_area text, extra_data JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                                                                                                                                      created_at timestamp WITH TIME ZONE DEFAULT now(),
                                                                                                                                                                                                                                                                                                                                  key_concepts text[], technical_insights text[], relevance_score integer, novelty_score integer, related_ideas text[], refined_summary text, extracted_methods text[], domain_knowledge_tags text[], critique_notes text, pid text);

--
-- Name: search_results_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.search_results_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: search_results_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.search_results_id_seq OWNED BY public.search_results.id;

--
-- Name: sharpening_predictions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sharpening_predictions (id integer NOT NULL,
                                                       goal_id integer, prompt_text text NOT NULL,
                                                                                         output_a text NOT NULL,
                                                                                                       output_b text NOT NULL,
                                                                                                                     preferred character(1),
                                                                                                                               predicted character(1),
                                                                                                                                         value_a double precision, value_b double precision, created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                                                                            CONSTRAINT sharpening_predictions_predicted_check CHECK ((predicted = ANY (ARRAY['a'::bpchar,
                                                                                                                                                                                                                                                                                                                             'b'::bpchar]))), CONSTRAINT sharpening_predictions_preferred_check CHECK ((preferred = ANY (ARRAY['a'::bpchar,
                                                                                                                                                                                                                                                                                                                                                                                                                               'b'::bpchar]))));

--
-- Name: sharpening_predictions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.sharpening_predictions_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: sharpening_predictions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.sharpening_predictions_id_seq OWNED BY public.sharpening_predictions.id;

--
-- Name: sharpening_results; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sharpening_results (id integer NOT NULL,
                                                   goal text NOT NULL,
                                                             prompt text NOT NULL,
                                                                         TEMPLATE text NOT NULL,
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
                                                                                                                                                                                                                                                                                                               prompt_template text, created_at timestamp WITH TIME ZONE DEFAULT now());

--
-- Name: sharpening_results_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.sharpening_results_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: sharpening_results_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.sharpening_results_id_seq OWNED BY public.sharpening_results.id;

--
-- Name: sis_cards; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sis_cards (id integer NOT NULL,
                                          ts double precision NOT NULL,
                                                              SCOPE text NOT NULL,
                                                                         KEY text NOT NULL,
                                                                                  title text, cards JSONB NOT NULL,
                                                                                                          meta JSONB,
                                                                                                               HASH text);

--
-- Name: sis_cards_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.sis_cards_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: sis_cards_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.sis_cards_id_seq OWNED BY public.sis_cards.id;

--
-- Name: skill_filters; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.skill_filters (id CHARACTER varying(64) NOT NULL,
                                                            casebook_id integer NOT NULL,
                                                                                DOMAIN CHARACTER varying(32),
                                                                                                 description text, created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                                                  weight_delta_path CHARACTER varying(256),
                                                                                                                                                                                              weight_size_mb double precision, vpm_residual_path CHARACTER varying(256),
                                                                                                                                                                                                                                                           vpm_preview_path CHARACTER varying(256),
                                                                                                                                                                                                                                                                                      alignment_score double precision, improvement_score double precision, stability_score double precision, compatible_domains JSON,
                                                                                                                                                                                                                                                                                                                                                                                                                 negative_interactions JSON);

--
-- Name: summaries; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.summaries (id integer NOT NULL, text text, created_at timestamp WITH TIME ZONE DEFAULT now());

--
-- Name: summaries_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.summaries_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: summaries_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.summaries_id_seq OWNED BY public.summaries.id;

--
-- Name: symbolic_rules; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.symbolic_rules (id integer NOT NULL,
                                               target text NOT NULL,
                                                           rule_text text, SOURCE text, attributes JSONB,
                                                                                                   filter JSONB,
                                                                                                          context_hash text, score double precision, goal_id integer, pipeline_run_id integer, prompt_id integer, agent_name text, goal_type text, goal_category text, difficulty text, focus_area text, created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                                                                                                                                                                                        updated_at timestamp WITHOUT TIME ZONE DEFAULT now());

--
-- Name: symbolic_rules_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.symbolic_rules_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: symbolic_rules_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.symbolic_rules_id_seq OWNED BY public.symbolic_rules.id;

--
-- Name: theorem_applications; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.theorem_applications (id integer NOT NULL,
                                                     theorem_id text NOT NULL,
                                                                     context text, RESULT text, success boolean, energy double precision, uncertainty double precision, applied_at timestamp WITHOUT TIME ZONE DEFAULT now());

--
-- Name: theorem_applications_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.theorem_applications_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: theorem_applications_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.theorem_applications_id_seq OWNED BY public.theorem_applications.id;

--
-- Name: theorem_cartridges; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.theorem_cartridges (theorem_id integer NOT NULL,
                                                           cartridge_id integer NOT NULL);

--
-- Name: theorems; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.theorems (id integer NOT NULL,
                                         STATEMENT text NOT NULL,
                                                        proof text, embedding_id integer, created_at timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                         pipeline_run_id integer);

--
-- Name: theorems_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.theorems_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: theorems_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.theorems_id_seq OWNED BY public.theorems.id;

--
-- Name: training_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.training_events (id bigint NOT NULL,
                                               model_key text NOT NULL,
                                                              dimension text NOT NULL,
                                                                             goal_id text, pipeline_run_id integer, agent_name text, kind text NOT NULL,
                                                                                                                                               query_text text, pos_text text, neg_text text, cand_text text, label smallint, weight double precision DEFAULT 1.0,
                                                                                                                                                                                                                                                              trust double precision DEFAULT 0.0,
                                                                                                                                                                                                                                                                                             SOURCE text DEFAULT 'memento'::text,
                                                                                                                                                                                                                                                                                                                 meta JSONB DEFAULT '{}'::JSONB,
                                                                                                                                                                                                                                                                                                                                    fp character(40),
                                                                                                                                                                                                                                                                                                                                       processed boolean DEFAULT FALSE NOT NULL,
                                                                                                                                                                                                                                                                                                                                                                       created_at timestamp WITH TIME ZONE DEFAULT now() NOT NULL,
                                                                                                                                                                                                                                                                                                                                                                                                                         CONSTRAINT training_events_kind_check CHECK ((kind = ANY (ARRAY['pairwise'::text,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         'pointwise'::text]))));

--
-- Name: training_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.training_events_id_seq
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: training_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.training_events_id_seq OWNED BY public.training_events.id;

--
-- Name: training_stats; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.training_stats (id integer NOT NULL,
                                               model_type CHARACTER varying NOT NULL,
                                                                            target_type CHARACTER varying NOT NULL,
                                                                                                          dimension CHARACTER varying NOT NULL,
                                                                                                                                      VERSION CHARACTER varying NOT NULL,
                                                                                                                                                                embedding_type CHARACTER varying NOT NULL,
                                                                                                                                                                                                 q_loss double precision, v_loss double precision, pi_loss double precision, avg_q_loss double precision, avg_v_loss double precision, avg_pi_loss double precision, policy_entropy double precision, policy_stability double precision, policy_logits JSONB,
                                                                                                                                                                                                                                                                                                                                                                                                                                                       config JSONB,
                                                                                                                                                                                                                                                                                                                                                                                                                                                              sample_count integer DEFAULT 0,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           valid_samples integer DEFAULT 0,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         invalid_samples integer DEFAULT 0,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         start_time timestamp WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        end_time timestamp WITHOUT TIME ZONE,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        goal_id integer, model_version_id integer, pipeline_run_id integer);

--
-- Name: training_stats_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.training_stats_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: training_stats_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.training_stats_id_seq OWNED BY public.training_stats.id;

--
-- Name: turn_knowledge_analysis; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.turn_knowledge_analysis (id integer NOT NULL,
                                                        turn_id integer NOT NULL,
                                                                        conversation_id integer NOT NULL,
                                                                                                pipeline_run_id integer, order_index integer, goal_text_hash text NOT NULL,
                                                                                                                                                                  assistant_text_hash text NOT NULL,
                                                                                                                                                                                           knowledge_score double precision DEFAULT 0 NOT NULL,
                                                                                                                                                                                                                                      verdict text DEFAULT 'not_knowledge'::text NOT NULL,
                                                                                                                                                                                                                                                                                 confidence double precision DEFAULT 0 NOT NULL,
                                                                                                                                                                                                                                                                                                                       dimensions JSONB,
                                                                                                                                                                                                                                                                                                                                  flags JSONB,
                                                                                                                                                                                                                                                                                                                                        rationale text, abstain boolean DEFAULT FALSE NOT NULL,
                                                                                                                                                                                                                                                                                                                                                                                      is_inflection boolean DEFAULT FALSE NOT NULL,
                                                                                                                                                                                                                                                                                                                                                                                                                          judge_model text, judge_prompt_version text, evaluation_id integer, meta JSONB,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   created_at timestamp WITHOUT TIME ZONE DEFAULT (now() AT TIME ZONE 'UTC'::text) NOT NULL);

--
-- Name: turn_knowledge_analysis_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.turn_knowledge_analysis_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: turn_knowledge_analysis_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.turn_knowledge_analysis_id_seq OWNED BY public.turn_knowledge_analysis.id;

--
-- Name: unified_mrq_models; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.unified_mrq_models (id integer NOT NULL,
                                                   dimension text NOT NULL,
                                                                  model_path text NOT NULL,
                                                                                  trained_on timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                 pair_count integer, trainer_version text, notes text, context JSONB);

--
-- Name: unified_mrq_models_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.unified_mrq_models_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: unified_mrq_models_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.unified_mrq_models_id_seq OWNED BY public.unified_mrq_models.id;

--
-- Name: worldviews; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.worldviews (id integer NOT NULL,
                                           name text NOT NULL,
                                                     description text, goal text, created_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                 updated_at timestamp WITHOUT TIME ZONE DEFAULT now(),
                                                                                                                                                                                extra_data JSON,
                                                                                                                                                                                           db_path text, active boolean DEFAULT TRUE);

--
-- Name: worldviews_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.worldviews_id_seq AS integer
START WITH 1 INCREMENT BY 1 NO MINVALUE NO MAXVALUE CACHE 1;

--
-- Name: worldviews_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.worldviews_id_seq OWNED BY public.worldviews.id;

--
-- Name: belief_graph_versions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.belief_graph_versions
ALTER COLUMN id
SET DEFAULT nextval('public.belief_graph_versions_id_seq'::regclass);

--
-- Name: bus_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bus_events
ALTER COLUMN id
SET DEFAULT nextval('public.bus_events_id_seq'::regclass);

--
-- Name: calibration_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.calibration_events
ALTER COLUMN id
SET DEFAULT nextval('public.calibration_events_id_seq'::regclass);

--
-- Name: cartridge_domains id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridge_domains
ALTER COLUMN id
SET DEFAULT nextval('public.cartridge_domains_id_seq'::regclass);

--
-- Name: cartridge_triples id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridge_triples
ALTER COLUMN id
SET DEFAULT nextval('public.cartridge_triples_id_seq'::regclass);

--
-- Name: cartridges id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridges
ALTER COLUMN id
SET DEFAULT nextval('public.cartridges_id_seq'::regclass);

--
-- Name: case_attributes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_attributes
ALTER COLUMN id
SET DEFAULT nextval('public.case_attributes_id_seq'::regclass);

--
-- Name: case_goal_state id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_goal_state
ALTER COLUMN id
SET DEFAULT nextval('public.case_goal_state_id_seq'::regclass);

--
-- Name: case_scorables id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_scorables
ALTER COLUMN id
SET DEFAULT nextval('public.case_scorables_id_seq'::regclass);

--
-- Name: casebooks id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.casebooks
ALTER COLUMN id
SET DEFAULT nextval('public.casebooks_id_seq'::regclass);

--
-- Name: cases id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cases
ALTER COLUMN id
SET DEFAULT nextval('public.cases_id_seq'::regclass);

--
-- Name: chat_conversations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_conversations
ALTER COLUMN id
SET DEFAULT nextval('public.chat_conversations_id_seq'::regclass);

--
-- Name: chat_messages id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_messages
ALTER COLUMN id
SET DEFAULT nextval('public.chat_messages_id_seq'::regclass);

--
-- Name: chat_turns id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_turns
ALTER COLUMN id
SET DEFAULT nextval('public.chat_turns_id_seq'::regclass);

--
-- Name: context_states id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_states
ALTER COLUMN id
SET DEFAULT nextval('public.context_states_id_seq'::regclass);

--
-- Name: cot_pattern_stats id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_pattern_stats
ALTER COLUMN id
SET DEFAULT nextval('public.cot_pattern_stats_id_seq'::regclass);

--
-- Name: cot_patterns id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_patterns
ALTER COLUMN id
SET DEFAULT nextval('public.cot_patterns_id_seq'::regclass);

--
-- Name: document_evaluations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_evaluations
ALTER COLUMN id
SET DEFAULT nextval('public.document_evaluations_id_seq'::regclass);

--
-- Name: document_scores id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_scores
ALTER COLUMN id
SET DEFAULT nextval('public.document_scores_id_seq'::regclass);

--
-- Name: document_section_domains id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_section_domains
ALTER COLUMN id
SET DEFAULT nextval('public.document_section_domains_id_seq'::regclass);

--
-- Name: document_sections id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_sections
ALTER COLUMN id
SET DEFAULT nextval('public.document_sections_id_seq'::regclass);

--
-- Name: documents id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.documents
ALTER COLUMN id
SET DEFAULT nextval('public.documents_id_seq'::regclass);

--
-- Name: dynamic_scorables id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dynamic_scorables
ALTER COLUMN id
SET DEFAULT nextval('public.dynamic_scorables_id_seq'::regclass);

--
-- Name: elo_ranking_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.elo_ranking_log
ALTER COLUMN id
SET DEFAULT nextval('public.elo_ranking_log_id_seq'::regclass);

--
-- Name: embeddings id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embeddings
ALTER COLUMN id
SET DEFAULT nextval('public.embeddings_id_seq'::regclass);

--
-- Name: entity_cache id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_cache
ALTER COLUMN id
SET DEFAULT nextval('public.entity_cache_id_seq'::regclass);

--
-- Name: evaluation_attributes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_attributes
ALTER COLUMN id
SET DEFAULT nextval('public.evaluation_attributes_id_seq'::regclass);

--
-- Name: evaluation_rule_links id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_rule_links
ALTER COLUMN id
SET DEFAULT nextval('public.score_rule_links_id_seq'::regclass);

--
-- Name: evaluations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluations
ALTER COLUMN id
SET DEFAULT nextval('public.scores_id_seq'::regclass);

--
-- Name: events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.events
ALTER COLUMN id
SET DEFAULT nextval('public.events_id_seq'::regclass);

--
-- Name: execution_steps id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.execution_steps
ALTER COLUMN id
SET DEFAULT nextval('public.execution_steps_id_seq'::regclass);

--
-- Name: goal_dimensions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.goal_dimensions
ALTER COLUMN id
SET DEFAULT nextval('public.goal_dimensions_id_seq'::regclass);

--
-- Name: goals id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.goals
ALTER COLUMN id
SET DEFAULT nextval('public.goals_id_seq'::regclass);

--
-- Name: hf_embeddings id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hf_embeddings
ALTER COLUMN id
SET DEFAULT nextval('public.hf_embeddings_id_seq'::regclass);

--
-- Name: hnet_embeddings id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hnet_embeddings
ALTER COLUMN id
SET DEFAULT nextval('public.hnet_embeddings_id_seq'::regclass);

--
-- Name: hypotheses id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses
ALTER COLUMN id
SET DEFAULT nextval('public.hypotheses_id_seq'::regclass);

--
-- Name: ideas id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ideas
ALTER COLUMN id
SET DEFAULT nextval('public.ideas_id_seq'::regclass);

--
-- Name: knowledge_documents id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.knowledge_documents
ALTER COLUMN id
SET DEFAULT nextval('public.knowledge_documents_id_seq'::regclass);

--
-- Name: knowledge_sections id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.knowledge_sections
ALTER COLUMN id
SET DEFAULT nextval('public.knowledge_sections_id_seq'::regclass);

--
-- Name: lookaheads id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lookaheads
ALTER COLUMN id
SET DEFAULT nextval('public.lookaheads_id_seq'::regclass);

--
-- Name: mars_conflicts id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mars_conflicts
ALTER COLUMN id
SET DEFAULT nextval('public.mars_conflicts_id_seq'::regclass);

--
-- Name: mars_results id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mars_results
ALTER COLUMN id
SET DEFAULT nextval('public.mars_results_id_seq'::regclass);

--
-- Name: measurements id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.measurements
ALTER COLUMN id
SET DEFAULT nextval('public.measurements_id_seq'::regclass);

--
-- Name: mem_cubes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_cubes
ALTER COLUMN id
SET DEFAULT nextval('public.mem_cubes_id_seq'::regclass);

--
-- Name: memcube_transformations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memcube_transformations
ALTER COLUMN id
SET DEFAULT nextval('public.memcube_transformations_id_seq'::regclass);

--
-- Name: method_plans id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.method_plans
ALTER COLUMN id
SET DEFAULT nextval('public.method_plans_id_seq'::regclass);

--
-- Name: model_performance id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_performance
ALTER COLUMN id
SET DEFAULT nextval('public.model_performance_id_seq'::regclass);

--
-- Name: model_versions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_versions
ALTER COLUMN id
SET DEFAULT nextval('public.model_versions_id_seq'::regclass);

--
-- Name: mrq_evaluations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mrq_evaluations
ALTER COLUMN id
SET DEFAULT nextval('public.mrq_evaluations_id_seq'::regclass);

--
-- Name: mrq_memory id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mrq_memory
ALTER COLUMN id
SET DEFAULT nextval('public.mrq_memory_id_seq'::regclass);

--
-- Name: mrq_preference_pairs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mrq_preference_pairs
ALTER COLUMN id
SET DEFAULT nextval('public.mrq_preference_pairs_id_seq'::regclass);

--
-- Name: nodes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.nodes
ALTER COLUMN id
SET DEFAULT nextval('public.nodes_id_seq'::regclass);

--
-- Name: pipeline_references id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_references
ALTER COLUMN id
SET DEFAULT nextval('public.pipeline_references_id_seq'::regclass);

--
-- Name: pipeline_runs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_runs
ALTER COLUMN id
SET DEFAULT nextval('public.pipeline_runs_id_seq'::regclass);

--
-- Name: pipeline_stages id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_stages
ALTER COLUMN id
SET DEFAULT nextval('public.pipeline_stages_id_seq'::regclass);

--
-- Name: plan_trace_reuse_links id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_trace_reuse_links
ALTER COLUMN id
SET DEFAULT nextval('public.plan_trace_reuse_links_id_seq'::regclass);

--
-- Name: plan_traces id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_traces
ALTER COLUMN id
SET DEFAULT nextval('public.plan_traces_id_seq'::regclass);

--
-- Name: prompt_evaluations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_evaluations
ALTER COLUMN id
SET DEFAULT nextval('public.prompt_evaluations_id_seq'::regclass);

--
-- Name: prompt_history id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_history
ALTER COLUMN id
SET DEFAULT nextval('public.prompt_history_id_seq'::regclass);

--
-- Name: prompt_versions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_versions
ALTER COLUMN id
SET DEFAULT nextval('public.prompt_versions_id_seq'::regclass);

--
-- Name: prompts id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompts
ALTER COLUMN id
SET DEFAULT nextval('public.prompts_id_seq'::regclass);

--
-- Name: ranking_trace id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ranking_trace
ALTER COLUMN id
SET DEFAULT nextval('public.ranking_trace_id_seq'::regclass);

--
-- Name: refinement_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.refinement_events
ALTER COLUMN id
SET DEFAULT nextval('public.refinement_events_id_seq'::regclass);

--
-- Name: reflection_deltas id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.reflection_deltas
ALTER COLUMN id
SET DEFAULT nextval('public.reflection_deltas_id_seq'::regclass);

--
-- Name: reports id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.reports
ALTER COLUMN id
SET DEFAULT nextval('public.reports_id_seq'::regclass);

--
-- Name: rule_applications id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_applications
ALTER COLUMN id
SET DEFAULT nextval('public.rule_applications_id_seq'::regclass);

--
-- Name: scorable_domains id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scorable_domains
ALTER COLUMN id
SET DEFAULT nextval('public.document_domains_id_seq'::regclass);

--
-- Name: scorable_embeddings id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scorable_embeddings
ALTER COLUMN id
SET DEFAULT nextval('public.document_embeddings_id_seq'::regclass);

--
-- Name: scorable_entities id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scorable_entities
ALTER COLUMN id
SET DEFAULT nextval('public.scorable_entities_id_seq'::regclass);

--
-- Name: scorable_ranks id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scorable_ranks
ALTER COLUMN id
SET DEFAULT nextval('public.scorable_ranks_id_seq'::regclass);

--
-- Name: score_attributes id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.score_attributes
ALTER COLUMN id
SET DEFAULT nextval('public.score_attributes_id_seq'::regclass);

--
-- Name: score_dimensions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.score_dimensions
ALTER COLUMN id
SET DEFAULT nextval('public.score_dimensions_id_seq'::regclass);

--
-- Name: scores id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scores
ALTER COLUMN id
SET DEFAULT nextval('public.scores_id_seq1'::regclass);

--
-- Name: scoring_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_events
ALTER COLUMN id
SET DEFAULT nextval('public.scoring_events_id_seq'::regclass);

--
-- Name: scoring_history id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_history
ALTER COLUMN id
SET DEFAULT nextval('public.scoring_history_id_seq'::regclass);

--
-- Name: search_hits id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.search_hits
ALTER COLUMN id
SET DEFAULT nextval('public.search_hits_id_seq'::regclass);

--
-- Name: search_results id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.search_results
ALTER COLUMN id
SET DEFAULT nextval('public.search_results_id_seq'::regclass);

--
-- Name: sharpening_predictions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sharpening_predictions
ALTER COLUMN id
SET DEFAULT nextval('public.sharpening_predictions_id_seq'::regclass);

--
-- Name: sharpening_results id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sharpening_results
ALTER COLUMN id
SET DEFAULT nextval('public.sharpening_results_id_seq'::regclass);

--
-- Name: sis_cards id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sis_cards
ALTER COLUMN id
SET DEFAULT nextval('public.sis_cards_id_seq'::regclass);

--
-- Name: summaries id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.summaries
ALTER COLUMN id
SET DEFAULT nextval('public.summaries_id_seq'::regclass);

--
-- Name: symbolic_rules id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.symbolic_rules
ALTER COLUMN id
SET DEFAULT nextval('public.symbolic_rules_id_seq'::regclass);

--
-- Name: theorem_applications id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorem_applications
ALTER COLUMN id
SET DEFAULT nextval('public.theorem_applications_id_seq'::regclass);

--
-- Name: theorems id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorems
ALTER COLUMN id
SET DEFAULT nextval('public.theorems_id_seq'::regclass);

--
-- Name: training_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_events
ALTER COLUMN id
SET DEFAULT nextval('public.training_events_id_seq'::regclass);

--
-- Name: training_stats id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_stats
ALTER COLUMN id
SET DEFAULT nextval('public.training_stats_id_seq'::regclass);

--
-- Name: turn_knowledge_analysis id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_knowledge_analysis
ALTER COLUMN id
SET DEFAULT nextval('public.turn_knowledge_analysis_id_seq'::regclass);

--
-- Name: unified_mrq_models id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.unified_mrq_models
ALTER COLUMN id
SET DEFAULT nextval('public.unified_mrq_models_id_seq'::regclass);

--
-- Name: worldviews id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.worldviews
ALTER COLUMN id
SET DEFAULT nextval('public.worldviews_id_seq'::regclass);

--
-- Name: belief_cartridges belief_cartridges_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.belief_cartridges ADD CONSTRAINT belief_cartridges_pkey PRIMARY KEY (id);

--
-- Name: belief_graph_versions belief_graph_versions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.belief_graph_versions ADD CONSTRAINT belief_graph_versions_pkey PRIMARY KEY (id);

--
-- Name: bus_events bus_events_subject_event_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.bus_events ADD CONSTRAINT bus_events_subject_event_id_key UNIQUE (subject,
                                                                                          event_id);

--
-- Name: calibration_events calibration_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.calibration_events ADD CONSTRAINT calibration_events_pkey PRIMARY KEY (id);

--
-- Name: calibration_models calibration_models_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.calibration_models ADD CONSTRAINT calibration_models_pkey PRIMARY KEY (id);

--
-- Name: cartridge_domains cartridge_domains_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridge_domains ADD CONSTRAINT cartridge_domains_pkey PRIMARY KEY (id);

--
-- Name: cartridge_triples cartridge_triples_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridge_triples ADD CONSTRAINT cartridge_triples_pkey PRIMARY KEY (id);

--
-- Name: cartridges cartridges_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridges ADD CONSTRAINT cartridges_pkey PRIMARY KEY (id);

--
-- Name: case_attributes case_attributes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_attributes ADD CONSTRAINT case_attributes_pkey PRIMARY KEY (id);

--
-- Name: case_goal_state case_goal_state_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_goal_state ADD CONSTRAINT case_goal_state_pkey PRIMARY KEY (id);

--
-- Name: case_scorables case_scorables_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_scorables ADD CONSTRAINT case_scorables_pkey PRIMARY KEY (id);

--
-- Name: casebooks casebooks_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.casebooks ADD CONSTRAINT casebooks_pkey PRIMARY KEY (id);

--
-- Name: cases cases_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cases ADD CONSTRAINT cases_pkey PRIMARY KEY (id);

--
-- Name: chat_conversations chat_conversations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_conversations ADD CONSTRAINT chat_conversations_pkey PRIMARY KEY (id);

--
-- Name: chat_messages chat_messages_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_messages ADD CONSTRAINT chat_messages_pkey PRIMARY KEY (id);

--
-- Name: chat_turns chat_turns_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_turns ADD CONSTRAINT chat_turns_pkey PRIMARY KEY (id);

--
-- Name: comparison_preferences comparison_preferences_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.comparison_preferences ADD CONSTRAINT comparison_preferences_pkey PRIMARY KEY (id);

--
-- Name: component_versions component_versions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.component_versions ADD CONSTRAINT component_versions_pkey PRIMARY KEY (id);

--
-- Name: context_states context_states_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_states ADD CONSTRAINT context_states_pkey PRIMARY KEY (id);

--
-- Name: cot_pattern_stats cot_pattern_stats_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_pattern_stats ADD CONSTRAINT cot_pattern_stats_pkey PRIMARY KEY (id);

--
-- Name: cot_patterns cot_patterns_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_patterns ADD CONSTRAINT cot_patterns_pkey PRIMARY KEY (id);

--
-- Name: scorable_domains document_domains_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scorable_domains ADD CONSTRAINT document_domains_pkey PRIMARY KEY (id);

--
-- Name: scorable_embeddings document_embeddings_document_id_document_type_embedding_typ_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scorable_embeddings ADD CONSTRAINT document_embeddings_document_id_document_type_embedding_typ_key UNIQUE (scorable_id,
                                                                                                                                   scorable_type,
                                                                                                                                   embedding_type);

--
-- Name: scorable_embeddings document_embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scorable_embeddings ADD CONSTRAINT document_embeddings_pkey PRIMARY KEY (id);

--
-- Name: document_evaluations document_evaluations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_evaluations ADD CONSTRAINT document_evaluations_pkey PRIMARY KEY (id);

--
-- Name: document_scores document_scores_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_scores ADD CONSTRAINT document_scores_pkey PRIMARY KEY (id);

--
-- Name: document_section_domains document_section_domains_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_section_domains ADD CONSTRAINT document_section_domains_pkey PRIMARY KEY (id);

--
-- Name: document_sections document_sections_document_id_section_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_sections ADD CONSTRAINT document_sections_document_id_section_name_key UNIQUE (document_id,
                                                                                                                section_name);

--
-- Name: document_sections document_sections_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_sections ADD CONSTRAINT document_sections_pkey PRIMARY KEY (id);

--
-- Name: documents documents_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.documents ADD CONSTRAINT documents_pkey PRIMARY KEY (id);

--
-- Name: dynamic_scorables dynamic_scorables_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dynamic_scorables ADD CONSTRAINT dynamic_scorables_pkey PRIMARY KEY (id);

--
-- Name: elo_ranking_log elo_ranking_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.elo_ranking_log ADD CONSTRAINT elo_ranking_log_pkey PRIMARY KEY (id);

--
-- Name: embeddings embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.embeddings ADD CONSTRAINT embeddings_pkey PRIMARY KEY (id);

--
-- Name: entity_cache entity_cache_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_cache ADD CONSTRAINT entity_cache_pkey PRIMARY KEY (id);

--
-- Name: evaluation_attributes evaluation_attributes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_attributes ADD CONSTRAINT evaluation_attributes_pkey PRIMARY KEY (id);

--
-- Name: events events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.events ADD CONSTRAINT events_pkey PRIMARY KEY (id);

--
-- Name: execution_steps execution_steps_evaluation_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.execution_steps ADD CONSTRAINT execution_steps_evaluation_id_key UNIQUE (evaluation_id);

--
-- Name: execution_steps execution_steps_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.execution_steps ADD CONSTRAINT execution_steps_pkey PRIMARY KEY (id);

--
-- Name: experiment_model_snapshots experiment_model_snapshots_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiment_model_snapshots ADD CONSTRAINT experiment_model_snapshots_pkey PRIMARY KEY (id);

--
-- Name: experiment_trial_metrics experiment_trial_metrics_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiment_trial_metrics ADD CONSTRAINT experiment_trial_metrics_pkey PRIMARY KEY (id);

--
-- Name: experiment_trials experiment_trials_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiment_trials ADD CONSTRAINT experiment_trials_pkey PRIMARY KEY (id);

--
-- Name: experiment_variants experiment_variants_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiment_variants ADD CONSTRAINT experiment_variants_pkey PRIMARY KEY (id);

--
-- Name: experiments experiments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiments ADD CONSTRAINT experiments_pkey PRIMARY KEY (id);

--
-- Name: goal_dimensions goal_dimensions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.goal_dimensions ADD CONSTRAINT goal_dimensions_pkey PRIMARY KEY (id);

--
-- Name: goals goals_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.goals ADD CONSTRAINT goals_pkey PRIMARY KEY (id);

--
-- Name: hf_embeddings hf_embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hf_embeddings ADD CONSTRAINT hf_embeddings_pkey PRIMARY KEY (id);

--
-- Name: hnet_embeddings hnet_embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hnet_embeddings ADD CONSTRAINT hnet_embeddings_pkey PRIMARY KEY (id);

--
-- Name: hypotheses hypotheses_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses ADD CONSTRAINT hypotheses_pkey PRIMARY KEY (id);

--
-- Name: ideas ideas_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ideas ADD CONSTRAINT ideas_pkey PRIMARY KEY (id);

--
-- Name: experiment_model_snapshots ix_model_snapshots_unique; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiment_model_snapshots ADD CONSTRAINT ix_model_snapshots_unique UNIQUE (experiment_id,
                                                                                                    name,
                                                                                                    DOMAIN,
                                                                                                    VERSION);

--
-- Name: knowledge_documents knowledge_documents_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.knowledge_documents ADD CONSTRAINT knowledge_documents_pkey PRIMARY KEY (id);

--
-- Name: knowledge_sections knowledge_sections_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.knowledge_sections ADD CONSTRAINT knowledge_sections_pkey PRIMARY KEY (id);

--
-- Name: lookaheads lookaheads_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lookaheads ADD CONSTRAINT lookaheads_pkey PRIMARY KEY (id);

--
-- Name: mars_conflicts mars_conflicts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mars_conflicts ADD CONSTRAINT mars_conflicts_pkey PRIMARY KEY (id);

--
-- Name: mars_results mars_results_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mars_results ADD CONSTRAINT mars_results_pkey PRIMARY KEY (id);

--
-- Name: measurements measurements_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.measurements ADD CONSTRAINT measurements_pkey PRIMARY KEY (id);

--
-- Name: mem_cubes mem_cubes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mem_cubes ADD CONSTRAINT mem_cubes_pkey PRIMARY KEY (id);

--
-- Name: memcube_transformations memcube_transformations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memcube_transformations ADD CONSTRAINT memcube_transformations_pkey PRIMARY KEY (id);

--
-- Name: memcube_versions memcube_versions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memcube_versions ADD CONSTRAINT memcube_versions_pkey PRIMARY KEY (id);

--
-- Name: memcubes memcubes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memcubes ADD CONSTRAINT memcubes_pkey PRIMARY KEY (id);

--
-- Name: method_plans method_plans_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.method_plans ADD CONSTRAINT method_plans_pkey PRIMARY KEY (id);

--
-- Name: model_artifacts model_artifacts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_artifacts ADD CONSTRAINT model_artifacts_pkey PRIMARY KEY (id);

--
-- Name: model_performance model_performance_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_performance ADD CONSTRAINT model_performance_pkey PRIMARY KEY (id);

--
-- Name: model_versions model_versions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_versions ADD CONSTRAINT model_versions_pkey PRIMARY KEY (id);

--
-- Name: mrq_evaluations mrq_evaluations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mrq_evaluations ADD CONSTRAINT mrq_evaluations_pkey PRIMARY KEY (id);

--
-- Name: mrq_memory mrq_memory_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mrq_memory ADD CONSTRAINT mrq_memory_pkey PRIMARY KEY (id);

--
-- Name: mrq_preference_pairs mrq_preference_pairs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mrq_preference_pairs ADD CONSTRAINT mrq_preference_pairs_pkey PRIMARY KEY (id);

--
-- Name: nodes nodes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.nodes ADD CONSTRAINT nodes_pkey PRIMARY KEY (id);

--
-- Name: pipeline_references pipeline_references_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_references ADD CONSTRAINT pipeline_references_pkey PRIMARY KEY (id);

--
-- Name: pipeline_runs pipeline_runs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_runs ADD CONSTRAINT pipeline_runs_pkey PRIMARY KEY (id);

--
-- Name: pipeline_runs pipeline_runs_run_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_runs ADD CONSTRAINT pipeline_runs_run_id_key UNIQUE (run_id);

--
-- Name: pipeline_stages pipeline_stages_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_stages ADD CONSTRAINT pipeline_stages_pkey PRIMARY KEY (id);

--
-- Name: plan_trace_reuse_links plan_trace_reuse_links_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_trace_reuse_links ADD CONSTRAINT plan_trace_reuse_links_pkey PRIMARY KEY (id);

--
-- Name: plan_traces plan_traces_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_traces ADD CONSTRAINT plan_traces_pkey PRIMARY KEY (id);

--
-- Name: plan_traces plan_traces_trace_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_traces ADD CONSTRAINT plan_traces_trace_id_key UNIQUE (trace_id);

--
-- Name: prompt_evaluations prompt_evaluations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_evaluations ADD CONSTRAINT prompt_evaluations_pkey PRIMARY KEY (id);

--
-- Name: prompt_history prompt_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_history ADD CONSTRAINT prompt_history_pkey PRIMARY KEY (id);

--
-- Name: prompt_programs prompt_programs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_programs ADD CONSTRAINT prompt_programs_pkey PRIMARY KEY (id);

--
-- Name: prompt_versions prompt_versions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_versions ADD CONSTRAINT prompt_versions_pkey PRIMARY KEY (id);

--
-- Name: prompts prompts_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompts ADD CONSTRAINT prompts_pkey PRIMARY KEY (id);

--
-- Name: protocols protocols_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.protocols ADD CONSTRAINT protocols_pkey PRIMARY KEY (name);

--
-- Name: ranking_trace ranking_trace_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ranking_trace ADD CONSTRAINT ranking_trace_pkey PRIMARY KEY (id);

--
-- Name: refinement_events refinement_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.refinement_events ADD CONSTRAINT refinement_events_pkey PRIMARY KEY (id);

--
-- Name: reflection_deltas reflection_deltas_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.reflection_deltas ADD CONSTRAINT reflection_deltas_pkey PRIMARY KEY (id);

--
-- Name: reports reports_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.reports ADD CONSTRAINT reports_pkey PRIMARY KEY (id);

--
-- Name: rule_applications rule_applications_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_applications ADD CONSTRAINT rule_applications_pkey PRIMARY KEY (id);

--
-- Name: scorable_entities scorable_entities_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scorable_entities ADD CONSTRAINT scorable_entities_pkey PRIMARY KEY (id);

--
-- Name: scorable_entities scorable_entities_scorable_id_scorable_type_entity_text_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scorable_entities ADD CONSTRAINT scorable_entities_scorable_id_scorable_type_entity_text_key UNIQUE (scorable_id,
                                                                                                                             scorable_type,
                                                                                                                             entity_text);

--
-- Name: scorable_ranks scorable_ranks_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scorable_ranks ADD CONSTRAINT scorable_ranks_pkey PRIMARY KEY (id);

--
-- Name: score_attributes score_attributes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.score_attributes ADD CONSTRAINT score_attributes_pkey PRIMARY KEY (id);

--
-- Name: score_dimensions score_dimensions_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.score_dimensions ADD CONSTRAINT score_dimensions_name_key UNIQUE (name);

--
-- Name: score_dimensions score_dimensions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.score_dimensions ADD CONSTRAINT score_dimensions_pkey PRIMARY KEY (id);

--
-- Name: evaluation_rule_links score_rule_links_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_rule_links ADD CONSTRAINT score_rule_links_pkey PRIMARY KEY (id);

--
-- Name: evaluations scores_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluations ADD CONSTRAINT scores_pkey PRIMARY KEY (id);

--
-- Name: scores scores_pkey1; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scores ADD CONSTRAINT scores_pkey1 PRIMARY KEY (id);

--
-- Name: scoring_dimensions scoring_dimensions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_dimensions ADD CONSTRAINT scoring_dimensions_pkey PRIMARY KEY (event_id,
                                                                                               dimension);

--
-- Name: scoring_events scoring_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_events ADD CONSTRAINT scoring_events_pkey PRIMARY KEY (id);

--
-- Name: scoring_history scoring_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_history ADD CONSTRAINT scoring_history_pkey PRIMARY KEY (id);

--
-- Name: search_hits search_hits_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.search_hits ADD CONSTRAINT search_hits_pkey PRIMARY KEY (id);

--
-- Name: search_results search_results_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.search_results ADD CONSTRAINT search_results_pkey PRIMARY KEY (id);

--
-- Name: sharpening_predictions sharpening_predictions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sharpening_predictions ADD CONSTRAINT sharpening_predictions_pkey PRIMARY KEY (id);

--
-- Name: sharpening_results sharpening_results_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sharpening_results ADD CONSTRAINT sharpening_results_pkey PRIMARY KEY (id);

--
-- Name: sis_cards sis_cards_hash_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sis_cards ADD CONSTRAINT sis_cards_hash_key UNIQUE (HASH);

--
-- Name: sis_cards sis_cards_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sis_cards ADD CONSTRAINT sis_cards_pkey PRIMARY KEY (id);

--
-- Name: skill_filters skill_filters_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.skill_filters ADD CONSTRAINT skill_filters_pkey PRIMARY KEY (id);

--
-- Name: summaries summaries_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.summaries ADD CONSTRAINT summaries_pkey PRIMARY KEY (id);

--
-- Name: symbolic_rules symbolic_rules_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.symbolic_rules ADD CONSTRAINT symbolic_rules_pkey PRIMARY KEY (id);

--
-- Name: theorem_applications theorem_applications_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorem_applications ADD CONSTRAINT theorem_applications_pkey PRIMARY KEY (id);

--
-- Name: theorem_cartridges theorem_cartridges_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorem_cartridges ADD CONSTRAINT theorem_cartridges_pkey PRIMARY KEY (theorem_id,
                                                                                               cartridge_id);

--
-- Name: theorems theorems_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorems ADD CONSTRAINT theorems_pkey PRIMARY KEY (id);

--
-- Name: training_events training_events_fp_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_events ADD CONSTRAINT training_events_fp_key UNIQUE (fp);

--
-- Name: training_events training_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_events ADD CONSTRAINT training_events_pkey PRIMARY KEY (id);

--
-- Name: training_stats training_stats_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_stats ADD CONSTRAINT training_stats_pkey PRIMARY KEY (id);

--
-- Name: turn_knowledge_analysis turn_knowledge_analysis_evaluation_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_knowledge_analysis ADD CONSTRAINT turn_knowledge_analysis_evaluation_id_key UNIQUE (evaluation_id);

--
-- Name: turn_knowledge_analysis turn_knowledge_analysis_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_knowledge_analysis ADD CONSTRAINT turn_knowledge_analysis_pkey PRIMARY KEY (id);

--
-- Name: unified_mrq_models unified_mrq_models_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.unified_mrq_models ADD CONSTRAINT unified_mrq_models_pkey PRIMARY KEY (id);

--
-- Name: document_section_domains unique_document_section_domain; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_section_domains ADD CONSTRAINT unique_document_section_domain UNIQUE (document_section_id,
                                                                                                       DOMAIN);

--
-- Name: cartridges unique_source; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridges ADD CONSTRAINT unique_source UNIQUE (source_type,
                                                                        source_uri);

--
-- Name: hf_embeddings unique_text_hash_hf; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hf_embeddings ADD CONSTRAINT unique_text_hash_hf UNIQUE (text_hash);

--
-- Name: hnet_embeddings unique_text_hash_hnet; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hnet_embeddings ADD CONSTRAINT unique_text_hash_hnet UNIQUE (text_hash);

--
-- Name: calibration_models uq_calibration_models_domain; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.calibration_models ADD CONSTRAINT uq_calibration_models_domain UNIQUE (DOMAIN);

--
-- Name: case_attributes uq_case_attr; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_attributes ADD CONSTRAINT uq_case_attr UNIQUE (case_id,
                                                                            KEY);

--
-- Name: case_goal_state uq_case_goal_state; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_goal_state ADD CONSTRAINT uq_case_goal_state UNIQUE (casebook_id,
                                                                                  goal_id);

--
-- Name: experiments uq_experiment_name_domain; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiments ADD CONSTRAINT uq_experiment_name_domain UNIQUE (name,
                                                                                     DOMAIN);

--
-- Name: model_artifacts uq_model_artifacts_name_version; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.model_artifacts ADD CONSTRAINT uq_model_artifacts_name_version UNIQUE (name,
                                                                                               VERSION);

--
-- Name: scorable_domains uq_scorable_domain; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scorable_domains ADD CONSTRAINT uq_scorable_domain UNIQUE (scorable_id,
                                                                                   scorable_type,
                                                                                   DOMAIN);

--
-- Name: scorable_entities uq_scorable_entity_norm; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scorable_entities ADD CONSTRAINT uq_scorable_entity_norm UNIQUE (scorable_id,
                                                                                         scorable_type,
                                                                                         entity_text_norm);

--
-- Name: sis_cards uq_sis_cards_scope_key_hash; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sis_cards ADD CONSTRAINT uq_sis_cards_scope_key_hash UNIQUE (SCOPE,
                                                                                     KEY,
                                                                                     HASH);

--
-- Name: experiment_trials uq_trial_variant_case; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiment_trials ADD CONSTRAINT uq_trial_variant_case UNIQUE (variant_id,
                                                                                       case_id);

--
-- Name: experiment_variants uq_variant_name_per_experiment; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiment_variants ADD CONSTRAINT uq_variant_name_per_experiment UNIQUE (experiment_id,
                                                                                                  name);

--
-- Name: worldviews worldviews_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.worldviews ADD CONSTRAINT worldviews_name_key UNIQUE (name);

--
-- Name: worldviews worldviews_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.worldviews ADD CONSTRAINT worldviews_pkey PRIMARY KEY (id);

--
-- Name: chat_conversations_tsv_gin; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX chat_conversations_tsv_gin ON public.chat_conversations USING gin (tsv);

--
-- Name: chat_messages_conv_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX chat_messages_conv_idx ON public.chat_messages USING btree (conversation_id);

--
-- Name: chat_messages_role_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX chat_messages_role_idx ON public.chat_messages USING btree (ROLE);

--
-- Name: chat_messages_tsv_gin; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX chat_messages_tsv_gin ON public.chat_messages USING gin (tsv);

--
-- Name: idx_bus_events_case; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_bus_events_case ON public.bus_events USING btree (case_id);

--
-- Name: idx_bus_events_event; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_bus_events_event ON public.bus_events USING btree (event);

--
-- Name: idx_bus_events_paper_sec; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_bus_events_paper_sec ON public.bus_events USING btree (paper_id, section_name);

--
-- Name: idx_bus_events_run; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_bus_events_run ON public.bus_events USING btree (run_id);

--
-- Name: idx_bus_events_subject; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_bus_events_subject ON public.bus_events USING btree (subject);

--
-- Name: idx_bus_events_ts; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_bus_events_ts ON public.bus_events USING btree (ts);

--
-- Name: idx_cal_models_domain; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_cal_models_domain ON public.calibration_models USING btree (DOMAIN);

--
-- Name: idx_case_attr_case_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_case_attr_case_id ON public.case_attributes USING btree (case_id);

--
-- Name: idx_case_attr_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_case_attr_key ON public.case_attributes USING btree (KEY);

--
-- Name: idx_case_attr_key_bool; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_case_attr_key_bool ON public.case_attributes USING btree (KEY, value_bool);

--
-- Name: idx_case_attr_key_json_gin; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_case_attr_key_json_gin ON public.case_attributes USING gin (value_json jsonb_path_ops);

--
-- Name: idx_case_attr_key_num; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_case_attr_key_num ON public.case_attributes USING btree (KEY, value_num);

--
-- Name: idx_case_attr_key_text; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_case_attr_key_text ON public.case_attributes USING btree (KEY, value_text);

--
-- Name: idx_chat_turns_domains_gin; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_chat_turns_domains_gin ON public.chat_turns USING gin (domains)
WHERE (domains IS NOT NULL);

--
-- Name: idx_chat_turns_ner_gin; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_chat_turns_ner_gin ON public.chat_turns USING gin (ner)
WHERE (ner IS NOT NULL);

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
-- Name: idx_evaluations_created_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_evaluations_created_at ON public.evaluations USING btree (created_at DESC);

--
-- Name: idx_evaluations_query; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_evaluations_query ON public.evaluations USING btree (query_type, query_id);

--
-- Name: idx_evaluations_scorable_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_evaluations_scorable_id ON public.evaluations USING btree (scorable_id);

--
-- Name: idx_execution_steps_agent_role; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_execution_steps_agent_role ON public.execution_steps USING btree (agent_role);

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
-- Name: idx_mars_results_dimension; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_mars_results_dimension ON public.mars_results USING btree (dimension);

--
-- Name: idx_mars_results_pipeline_run; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_mars_results_pipeline_run ON public.mars_results USING btree (pipeline_run_id);

--
-- Name: idx_mars_results_plan_trace; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_mars_results_plan_trace ON public.mars_results USING btree (plan_trace_id);

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
-- Name: idx_pipeline_references_pipeline_run_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pipeline_references_pipeline_run_id ON public.pipeline_references USING btree (pipeline_run_id);

--
-- Name: idx_pipeline_references_target; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_pipeline_references_target ON public.pipeline_references USING btree (scorable_type, scorable_id);

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
-- Name: idx_plan_trace_reuse_child; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_plan_trace_reuse_child ON public.plan_trace_reuse_links USING btree (child_trace_id);

--
-- Name: idx_plan_trace_reuse_parent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_plan_trace_reuse_parent ON public.plan_trace_reuse_links USING btree (parent_trace_id);

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

CREATE INDEX idx_prompt_agent ON public.prompts USING btree (SOURCE);

--
-- Name: idx_prompt_strategy; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prompt_strategy ON public.prompts USING btree (strategy);

--
-- Name: idx_prompt_version; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prompt_version ON public.prompts USING btree (VERSION);

--
-- Name: idx_scorable_domains_domain; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_scorable_domains_domain ON public.scorable_domains USING btree (DOMAIN);

--
-- Name: idx_scorable_domains_scorable; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_scorable_domains_scorable ON public.scorable_domains USING btree (scorable_type, scorable_id);

--
-- Name: idx_scorable_ranks_evaluation_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_scorable_ranks_evaluation_id ON public.scorable_ranks USING btree (evaluation_id);

--
-- Name: idx_scorable_ranks_query_text; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_scorable_ranks_query_text ON public.scorable_ranks USING btree (query_text);

--
-- Name: idx_scorable_ranks_scorable; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_scorable_ranks_scorable ON public.scorable_ranks USING btree (scorable_id, scorable_type);

--
-- Name: idx_scores_dimension; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_scores_dimension ON public.scores USING btree (dimension);

--
-- Name: idx_scores_dimension_eval; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_scores_dimension_eval ON public.scores USING btree (dimension, evaluation_id);

--
-- Name: idx_sis_cards_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sis_cards_key ON public.sis_cards USING btree (KEY);

--
-- Name: idx_sis_cards_scope; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sis_cards_scope ON public.sis_cards USING btree (SCOPE);

--
-- Name: idx_sis_cards_scope_key_ts; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sis_cards_scope_key_ts ON public.sis_cards USING btree (SCOPE, KEY, ts);

--
-- Name: idx_sis_cards_ts; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sis_cards_ts ON public.sis_cards USING btree (ts);

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

CREATE INDEX idx_training_stats_version ON public.training_stats USING btree (VERSION);

--
-- Name: idx_trials_experiment_group; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trials_experiment_group ON public.experiment_trials USING btree (experiment_group);

--
-- Name: idx_trials_tags_used; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_trials_tags_used ON public.experiment_trials USING gin (tags_used);

--
-- Name: ix_case_goal; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_case_goal ON public.case_goal_state USING btree (casebook_id, goal_id);

--
-- Name: ix_dynamic_scorables_srcptr; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_dynamic_scorables_srcptr ON public.dynamic_scorables USING btree (SOURCE, source_scorable_type, source_scorable_id);

--
-- Name: ix_entity_cache_embedding_ref; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_entity_cache_embedding_ref ON public.entity_cache USING btree (embedding_ref);

--
-- Name: ix_entity_cache_last_updated; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_entity_cache_last_updated ON public.entity_cache USING btree (last_updated);

--
-- Name: ix_evaluations_plan_trace_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_evaluations_plan_trace_id ON public.evaluations USING btree (plan_trace_id);

--
-- Name: ix_experiments_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_experiments_name ON public.experiments USING btree (name);

--
-- Name: ix_model_artifacts_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_model_artifacts_name ON public.model_artifacts USING btree (name);

--
-- Name: ix_model_artifacts_tag; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_model_artifacts_tag ON public.model_artifacts USING btree (tag);

--
-- Name: ix_model_snapshots_domain; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_model_snapshots_domain ON public.experiment_model_snapshots USING btree (DOMAIN);

--
-- Name: ix_model_snapshots_experiment_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_model_snapshots_experiment_id ON public.experiment_model_snapshots USING btree (experiment_id);

--
-- Name: ix_model_snapshots_name; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_model_snapshots_name ON public.experiment_model_snapshots USING btree (name);

--
-- Name: ix_model_snapshots_version; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_model_snapshots_version ON public.experiment_model_snapshots USING btree (VERSION);

--
-- Name: ix_te_recent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_te_recent ON public.training_events USING btree (created_at DESC);

--
-- Name: ix_te_target; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_te_target ON public.training_events USING btree (model_key, dimension, kind);

--
-- Name: ix_te_unprocessed; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_te_unprocessed ON public.training_events USING btree (processed)
WHERE (processed = FALSE);

--
-- Name: ix_tka_abstain; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tka_abstain ON public.turn_knowledge_analysis USING btree (abstain);

--
-- Name: ix_tka_conversation_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tka_conversation_id ON public.turn_knowledge_analysis USING btree (conversation_id);

--
-- Name: ix_tka_is_inflection; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tka_is_inflection ON public.turn_knowledge_analysis USING btree (is_inflection);

--
-- Name: ix_tka_order_index; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tka_order_index ON public.turn_knowledge_analysis USING btree (order_index);

--
-- Name: ix_tka_score_desc; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tka_score_desc ON public.turn_knowledge_analysis USING btree (knowledge_score DESC);

--
-- Name: ix_tka_verdict; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_tka_verdict ON public.turn_knowledge_analysis USING btree (verdict);

--
-- Name: ix_trial_metric_key; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_trial_metric_key ON public.experiment_trial_metrics USING btree (KEY);

--
-- Name: ix_trial_metric_trial; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_trial_metric_trial ON public.experiment_trial_metrics USING btree (trial_id);

--
-- Name: ix_trials_case_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_trials_case_id ON public.experiment_trials USING btree (case_id);

--
-- Name: ix_trials_completed; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_trials_completed ON public.experiment_trials USING btree (completed_at);

--
-- Name: ix_trials_variant_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_trials_variant_id ON public.experiment_trials USING btree (variant_id);

--
-- Name: ix_variants_experiment_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_variants_experiment_id ON public.experiment_variants USING btree (experiment_id);

--
-- Name: unique_text_hash; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX unique_text_hash ON public.embeddings USING btree (text_hash);

--
-- Name: uq_plan_trace_reuse; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_plan_trace_reuse ON public.plan_trace_reuse_links USING btree (parent_trace_id, child_trace_id);

--
-- Name: uq_tka_turn_id; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_tka_turn_id ON public.turn_knowledge_analysis USING btree (turn_id);

--
-- Name: case_goal_state trg_case_goal_state_set_updated_at; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_case_goal_state_set_updated_at
BEFORE
UPDATE ON public.case_goal_state
FOR EACH ROW EXECUTE FUNCTION public.set_updated_at_timestamp();

--
-- Name: belief_cartridges belief_cartridges_document_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.belief_cartridges ADD CONSTRAINT belief_cartridges_document_id_fkey
FOREIGN KEY (document_id) REFERENCES public.documents(id) ON
DELETE
SET NULL;

--
-- Name: belief_cartridges belief_cartridges_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.belief_cartridges ADD CONSTRAINT belief_cartridges_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE
SET NULL;

--
-- Name: cartridge_domains cartridge_domains_cartridge_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridge_domains ADD CONSTRAINT cartridge_domains_cartridge_id_fkey
FOREIGN KEY (cartridge_id) REFERENCES public.cartridges(id) ON
DELETE CASCADE;

--
-- Name: cartridge_triples cartridge_triples_cartridge_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridge_triples ADD CONSTRAINT cartridge_triples_cartridge_id_fkey
FOREIGN KEY (cartridge_id) REFERENCES public.cartridges(id) ON
DELETE CASCADE;

--
-- Name: cartridges cartridges_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridges ADD CONSTRAINT cartridges_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE
SET NULL;

--
-- Name: cartridges cartridges_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cartridges ADD CONSTRAINT cartridges_pipeline_run_id_fkey
FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id) ON
DELETE
SET NULL;

--
-- Name: case_attributes case_attributes_case_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_attributes ADD CONSTRAINT case_attributes_case_id_fkey
FOREIGN KEY (case_id) REFERENCES public.cases(id) ON
DELETE CASCADE;

--
-- Name: case_goal_state case_goal_state_casebook_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_goal_state ADD CONSTRAINT case_goal_state_casebook_fk
FOREIGN KEY (casebook_id) REFERENCES public.casebooks(id) ON
DELETE CASCADE;

--
-- Name: case_goal_state case_goal_state_champion_case_fk; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_goal_state ADD CONSTRAINT case_goal_state_champion_case_fk
FOREIGN KEY (champion_case_id) REFERENCES public.cases(id) ON
DELETE
SET NULL;

--
-- Name: case_scorables case_scorables_case_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.case_scorables ADD CONSTRAINT case_scorables_case_id_fkey
FOREIGN KEY (case_id) REFERENCES public.cases(id) ON
DELETE CASCADE;

--
-- Name: cases cases_casebook_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cases ADD CONSTRAINT cases_casebook_id_fkey
FOREIGN KEY (casebook_id) REFERENCES public.casebooks(id) ON
DELETE CASCADE;

--
-- Name: chat_messages chat_messages_conversation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_messages ADD CONSTRAINT chat_messages_conversation_id_fkey
FOREIGN KEY (conversation_id) REFERENCES public.chat_conversations(id) ON
DELETE CASCADE;

--
-- Name: chat_messages chat_messages_parent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_messages ADD CONSTRAINT chat_messages_parent_id_fkey
FOREIGN KEY (parent_id) REFERENCES public.chat_messages(id) ON
DELETE CASCADE;

--
-- Name: chat_turns chat_turns_assistant_message_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_turns ADD CONSTRAINT chat_turns_assistant_message_id_fkey
FOREIGN KEY (assistant_message_id) REFERENCES public.chat_messages(id) ON
DELETE CASCADE;

--
-- Name: chat_turns chat_turns_conversation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_turns ADD CONSTRAINT chat_turns_conversation_id_fkey
FOREIGN KEY (conversation_id) REFERENCES public.chat_conversations(id) ON
DELETE CASCADE;

--
-- Name: chat_turns chat_turns_user_message_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_turns ADD CONSTRAINT chat_turns_user_message_id_fkey
FOREIGN KEY (user_message_id) REFERENCES public.chat_messages(id) ON
DELETE CASCADE;

--
-- Name: component_interfaces component_interfaces_component_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.component_interfaces ADD CONSTRAINT component_interfaces_component_id_fkey
FOREIGN KEY (component_id) REFERENCES public.component_versions(id);

--
-- Name: cot_pattern_stats cot_pattern_stats_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_pattern_stats ADD CONSTRAINT cot_pattern_stats_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE CASCADE;

--
-- Name: cot_pattern_stats cot_pattern_stats_hypothesis_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_pattern_stats ADD CONSTRAINT cot_pattern_stats_hypothesis_id_fkey
FOREIGN KEY (hypothesis_id) REFERENCES public.hypotheses(id) ON
DELETE CASCADE;

--
-- Name: cot_patterns cot_patterns_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_patterns ADD CONSTRAINT cot_patterns_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id);

--
-- Name: cot_patterns cot_patterns_hypothesis_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cot_patterns ADD CONSTRAINT cot_patterns_hypothesis_id_fkey
FOREIGN KEY (hypothesis_id) REFERENCES public.hypotheses(id);

--
-- Name: document_evaluations document_evaluations_document_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_evaluations ADD CONSTRAINT document_evaluations_document_id_fkey
FOREIGN KEY (document_id) REFERENCES public.documents(id) ON
DELETE CASCADE;

--
-- Name: document_scores document_scores_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_scores ADD CONSTRAINT document_scores_evaluation_id_fkey
FOREIGN KEY (evaluation_id) REFERENCES public.document_evaluations(id) ON
DELETE CASCADE;

--
-- Name: document_section_domains document_section_domains_document_section_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_section_domains ADD CONSTRAINT document_section_domains_document_section_id_fkey
FOREIGN KEY (document_section_id) REFERENCES public.document_sections(id) ON
DELETE CASCADE;

--
-- Name: document_sections document_sections_document_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_sections ADD CONSTRAINT document_sections_document_id_fkey
FOREIGN KEY (document_id) REFERENCES public.documents(id) ON
DELETE CASCADE;

--
-- Name: dynamic_scorables dynamic_scorables_case_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.dynamic_scorables ADD CONSTRAINT dynamic_scorables_case_id_fkey
FOREIGN KEY (case_id) REFERENCES public.cases(id) ON
DELETE CASCADE;

--
-- Name: entity_cache entity_cache_embedding_ref_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entity_cache ADD CONSTRAINT entity_cache_embedding_ref_fkey
FOREIGN KEY (embedding_ref) REFERENCES public.scorable_embeddings(id);

--
-- Name: evaluation_attributes evaluation_attributes_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_attributes ADD CONSTRAINT evaluation_attributes_evaluation_id_fkey
FOREIGN KEY (evaluation_id) REFERENCES public.evaluations(id) ON
DELETE CASCADE;

--
-- Name: evaluation_rule_links evaluation_rule_links_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_rule_links ADD CONSTRAINT evaluation_rule_links_evaluation_id_fkey
FOREIGN KEY (evaluation_id) REFERENCES public.evaluations(id) ON
DELETE CASCADE;

--
-- Name: execution_steps execution_steps_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.execution_steps ADD CONSTRAINT execution_steps_evaluation_id_fkey
FOREIGN KEY (evaluation_id) REFERENCES public.evaluations(id) ON
DELETE
SET NULL;

--
-- Name: execution_steps execution_steps_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.execution_steps ADD CONSTRAINT execution_steps_pipeline_run_id_fkey
FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id) ON
DELETE CASCADE;

--
-- Name: execution_steps execution_steps_plan_trace_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.execution_steps ADD CONSTRAINT execution_steps_plan_trace_id_fkey
FOREIGN KEY (plan_trace_id) REFERENCES public.plan_traces(id) ON
DELETE CASCADE;

--
-- Name: experiment_model_snapshots experiment_model_snapshots_experiment_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiment_model_snapshots ADD CONSTRAINT experiment_model_snapshots_experiment_id_fkey
FOREIGN KEY (experiment_id) REFERENCES public.experiments(id) ON
DELETE CASCADE;

--
-- Name: experiment_trial_metrics experiment_trial_metrics_trial_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiment_trial_metrics ADD CONSTRAINT experiment_trial_metrics_trial_id_fkey
FOREIGN KEY (trial_id) REFERENCES public.experiment_trials(id) ON
DELETE CASCADE;

--
-- Name: experiment_trials experiment_trials_variant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiment_trials ADD CONSTRAINT experiment_trials_variant_id_fkey
FOREIGN KEY (variant_id) REFERENCES public.experiment_variants(id) ON
DELETE CASCADE;

--
-- Name: experiment_variants experiment_variants_experiment_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.experiment_variants ADD CONSTRAINT experiment_variants_experiment_id_fkey
FOREIGN KEY (experiment_id) REFERENCES public.experiments(id) ON
DELETE CASCADE;

--
-- Name: chat_messages fk_chat_messages_parent; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chat_messages ADD CONSTRAINT fk_chat_messages_parent
FOREIGN KEY (parent_id) REFERENCES public.chat_messages(id) ON
DELETE CASCADE;

--
-- Name: document_section_domains fk_document_section; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.document_section_domains ADD CONSTRAINT fk_document_section
FOREIGN KEY (document_section_id) REFERENCES public.document_sections(id) ON
DELETE CASCADE;

--
-- Name: documents fk_documents_embedding; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.documents ADD CONSTRAINT fk_documents_embedding
FOREIGN KEY (embedding_id) REFERENCES public.embeddings(id) ON
DELETE
SET NULL;

--
-- Name: documents fk_documents_goal_id_goals; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.documents ADD CONSTRAINT fk_documents_goal_id_goals
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE
SET NULL;

--
-- Name: evaluations fk_evaluations_plan_trace; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluations ADD CONSTRAINT fk_evaluations_plan_trace
FOREIGN KEY (plan_trace_id) REFERENCES public.plan_traces(id) ON
DELETE CASCADE;

--
-- Name: hypotheses fk_goal; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses ADD CONSTRAINT fk_goal
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE CASCADE;

--
-- Name: context_states fk_goal; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_states ADD CONSTRAINT fk_goal
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE
SET NULL;

--
-- Name: hypotheses fk_goal_hypothesis; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses ADD CONSTRAINT fk_goal_hypothesis
FOREIGN KEY (goal_id) REFERENCES public.goals(id);

--
-- Name: prompts fk_goal_prompt; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompts ADD CONSTRAINT fk_goal_prompt
FOREIGN KEY (goal_id) REFERENCES public.goals(id);

--
-- Name: context_states fk_pipeline_run; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.context_states ADD CONSTRAINT fk_pipeline_run
FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id) ON
DELETE
SET NULL;

--
-- Name: hypotheses fk_prompt; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses ADD CONSTRAINT fk_prompt
FOREIGN KEY (prompt_id) REFERENCES public.prompts(id);

--
-- Name: evaluations fk_rule_application; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluations ADD CONSTRAINT fk_rule_application
FOREIGN KEY (rule_application_id) REFERENCES public.rule_applications(id) ON
DELETE
SET NULL;

--
-- Name: scorable_ranks fk_scorable_ranks_evaluation; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scorable_ranks ADD CONSTRAINT fk_scorable_ranks_evaluation
FOREIGN KEY (evaluation_id) REFERENCES public.evaluations(id) ON
DELETE
SET NULL;

--
-- Name: evaluations fk_scores_pipeline_run; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluations ADD CONSTRAINT fk_scores_pipeline_run
FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id) ON
DELETE
SET NULL;

--
-- Name: goal_dimensions goal_dimensions_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.goal_dimensions ADD CONSTRAINT goal_dimensions_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE CASCADE;

--
-- Name: hypotheses hypotheses_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses ADD CONSTRAINT hypotheses_prompt_id_fkey
FOREIGN KEY (prompt_id) REFERENCES public.prompts(id);

--
-- Name: hypotheses hypotheses_source_hypothesis_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.hypotheses ADD CONSTRAINT hypotheses_source_hypothesis_fkey
FOREIGN KEY (source_hypothesis_id) REFERENCES public.hypotheses(id);

--
-- Name: ideas ideas_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.ideas ADD CONSTRAINT ideas_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id);

--
-- Name: knowledge_documents knowledge_documents_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.knowledge_documents ADD CONSTRAINT knowledge_documents_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE
SET NULL;

--
-- Name: knowledge_sections knowledge_sections_document_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.knowledge_sections ADD CONSTRAINT knowledge_sections_document_id_fkey
FOREIGN KEY (document_id) REFERENCES public.knowledge_documents(id) ON
DELETE CASCADE;

--
-- Name: lookaheads lookaheads_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.lookaheads ADD CONSTRAINT lookaheads_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE CASCADE;

--
-- Name: mars_conflicts mars_conflicts_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mars_conflicts ADD CONSTRAINT mars_conflicts_pipeline_run_id_fkey
FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id) ON
DELETE CASCADE;

--
-- Name: mars_conflicts mars_conflicts_plan_trace_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mars_conflicts ADD CONSTRAINT mars_conflicts_plan_trace_id_fkey
FOREIGN KEY (plan_trace_id) REFERENCES public.plan_traces(trace_id) ON
DELETE CASCADE;

--
-- Name: mars_results mars_results_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mars_results ADD CONSTRAINT mars_results_pipeline_run_id_fkey
FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id) ON
DELETE CASCADE;

--
-- Name: mars_results mars_results_plan_trace_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.mars_results ADD CONSTRAINT mars_results_plan_trace_id_fkey
FOREIGN KEY (plan_trace_id) REFERENCES public.plan_traces(trace_id) ON
DELETE CASCADE;

--
-- Name: method_plans method_plans_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.method_plans ADD CONSTRAINT method_plans_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE CASCADE;

--
-- Name: method_plans method_plans_idea_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.method_plans ADD CONSTRAINT method_plans_idea_id_fkey
FOREIGN KEY (idea_id) REFERENCES public.ideas(id) ON
DELETE
SET NULL;

--
-- Name: method_plans method_plans_parent_plan_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.method_plans ADD CONSTRAINT method_plans_parent_plan_id_fkey
FOREIGN KEY (parent_plan_id) REFERENCES public.method_plans(id) ON
DELETE
SET NULL;

--
-- Name: pipeline_runs pipeline_runs_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_runs ADD CONSTRAINT pipeline_runs_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE CASCADE;

--
-- Name: pipeline_stages pipeline_stages_input_context_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_stages ADD CONSTRAINT pipeline_stages_input_context_id_fkey
FOREIGN KEY (input_context_id) REFERENCES public.context_states(id);

--
-- Name: pipeline_stages pipeline_stages_output_context_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_stages ADD CONSTRAINT pipeline_stages_output_context_id_fkey
FOREIGN KEY (output_context_id) REFERENCES public.context_states(id);

--
-- Name: pipeline_stages pipeline_stages_parent_stage_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_stages ADD CONSTRAINT pipeline_stages_parent_stage_id_fkey
FOREIGN KEY (parent_stage_id) REFERENCES public.pipeline_stages(id);

--
-- Name: pipeline_stages pipeline_stages_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.pipeline_stages ADD CONSTRAINT pipeline_stages_pipeline_run_id_fkey
FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id);

--
-- Name: plan_trace_reuse_links plan_trace_reuse_links_child_trace_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_trace_reuse_links ADD CONSTRAINT plan_trace_reuse_links_child_trace_id_fkey
FOREIGN KEY (child_trace_id) REFERENCES public.plan_traces(trace_id) ON
DELETE CASCADE;

--
-- Name: plan_trace_reuse_links plan_trace_reuse_links_parent_trace_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_trace_reuse_links ADD CONSTRAINT plan_trace_reuse_links_parent_trace_id_fkey
FOREIGN KEY (parent_trace_id) REFERENCES public.plan_traces(trace_id) ON
DELETE CASCADE;

--
-- Name: plan_traces plan_traces_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_traces ADD CONSTRAINT plan_traces_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE CASCADE;

--
-- Name: plan_traces plan_traces_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.plan_traces ADD CONSTRAINT plan_traces_pipeline_run_id_fkey
FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id) ON
DELETE CASCADE;

--
-- Name: prompt_evaluations prompt_evaluations_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_evaluations ADD CONSTRAINT prompt_evaluations_prompt_id_fkey
FOREIGN KEY (prompt_id) REFERENCES public.prompts(id) ON
DELETE CASCADE;

--
-- Name: prompt_history prompt_history_original_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_history ADD CONSTRAINT prompt_history_original_prompt_id_fkey
FOREIGN KEY (original_prompt_id) REFERENCES public.prompts(id);

--
-- Name: prompt_programs prompt_programs_parent_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_programs ADD CONSTRAINT prompt_programs_parent_id_fkey
FOREIGN KEY (parent_id) REFERENCES public.prompt_programs(id);

--
-- Name: prompt_programs prompt_programs_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_programs ADD CONSTRAINT prompt_programs_pipeline_run_id_fkey
FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id);

--
-- Name: prompt_programs prompt_programs_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_programs ADD CONSTRAINT prompt_programs_prompt_id_fkey
FOREIGN KEY (prompt_id) REFERENCES public.prompts(id);

--
-- Name: prompt_versions prompt_versions_previous_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_versions ADD CONSTRAINT prompt_versions_previous_prompt_id_fkey
FOREIGN KEY (previous_prompt_id) REFERENCES public.prompts(id);

--
-- Name: prompts prompts_embedding_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompts ADD CONSTRAINT prompts_embedding_id_fkey
FOREIGN KEY (embedding_id) REFERENCES public.embeddings(id) ON
DELETE
SET NULL;

--
-- Name: reflection_deltas reflection_deltas_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.reflection_deltas ADD CONSTRAINT reflection_deltas_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE CASCADE;

--
-- Name: rule_applications rule_applications_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_applications ADD CONSTRAINT rule_applications_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE CASCADE;

--
-- Name: rule_applications rule_applications_hypothesis_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_applications ADD CONSTRAINT rule_applications_hypothesis_id_fkey
FOREIGN KEY (hypothesis_id) REFERENCES public.hypotheses(id);

--
-- Name: rule_applications rule_applications_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.rule_applications ADD CONSTRAINT rule_applications_pipeline_run_id_fkey
FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id) ON
DELETE CASCADE;

--
-- Name: score_attributes score_attributes_score_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.score_attributes ADD CONSTRAINT score_attributes_score_id_fkey
FOREIGN KEY (score_id) REFERENCES public.scores(id) ON
DELETE CASCADE;

--
-- Name: evaluation_rule_links score_rule_links_rule_application_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation_rule_links ADD CONSTRAINT score_rule_links_rule_application_id_fkey
FOREIGN KEY (rule_application_id) REFERENCES public.rule_applications(id) ON
DELETE CASCADE;

--
-- Name: scores scores_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scores ADD CONSTRAINT scores_evaluation_id_fkey
FOREIGN KEY (evaluation_id) REFERENCES public.evaluations(id) ON
DELETE CASCADE;

--
-- Name: evaluations scores_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluations ADD CONSTRAINT scores_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE CASCADE;

--
-- Name: scoring_dimensions scoring_dimensions_event_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_dimensions ADD CONSTRAINT scoring_dimensions_event_id_fkey
FOREIGN KEY (event_id) REFERENCES public.scoring_events(id);

--
-- Name: scoring_history scoring_history_model_version_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_history ADD CONSTRAINT scoring_history_model_version_id_fkey
FOREIGN KEY (model_version_id) REFERENCES public.model_versions(id);

--
-- Name: scoring_history scoring_history_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.scoring_history ADD CONSTRAINT scoring_history_pipeline_run_id_fkey
FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id);

--
-- Name: search_hits search_hits_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.search_hits ADD CONSTRAINT search_hits_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id);

--
-- Name: search_results search_results_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.search_results ADD CONSTRAINT search_results_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id);

--
-- Name: sharpening_predictions sharpening_predictions_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sharpening_predictions ADD CONSTRAINT sharpening_predictions_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id);

--
-- Name: skill_filters skill_filters_casebook_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.skill_filters ADD CONSTRAINT skill_filters_casebook_id_fkey
FOREIGN KEY (casebook_id) REFERENCES public.casebooks(id);

--
-- Name: symbolic_rules symbolic_rules_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.symbolic_rules ADD CONSTRAINT symbolic_rules_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE CASCADE;

--
-- Name: symbolic_rules symbolic_rules_prompt_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.symbolic_rules ADD CONSTRAINT symbolic_rules_prompt_id_fkey
FOREIGN KEY (prompt_id) REFERENCES public.prompts(id) ON
DELETE CASCADE;

--
-- Name: theorem_cartridges theorem_cartridges_cartridge_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorem_cartridges ADD CONSTRAINT theorem_cartridges_cartridge_id_fkey
FOREIGN KEY (cartridge_id) REFERENCES public.cartridges(id);

--
-- Name: theorem_cartridges theorem_cartridges_theorem_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorem_cartridges ADD CONSTRAINT theorem_cartridges_theorem_id_fkey
FOREIGN KEY (theorem_id) REFERENCES public.theorems(id);

--
-- Name: theorems theorems_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.theorems ADD CONSTRAINT theorems_pipeline_run_id_fkey
FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id) ON
DELETE
SET NULL;

--
-- Name: training_stats training_stats_goal_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_stats ADD CONSTRAINT training_stats_goal_id_fkey
FOREIGN KEY (goal_id) REFERENCES public.goals(id) ON
DELETE
SET NULL;

--
-- Name: training_stats training_stats_model_version_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.training_stats ADD CONSTRAINT training_stats_model_version_id_fkey
FOREIGN KEY (model_version_id) REFERENCES public.model_versions(id) ON
DELETE
SET NULL;

--
-- Name: turn_knowledge_analysis turn_knowledge_analysis_conversation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_knowledge_analysis ADD CONSTRAINT turn_knowledge_analysis_conversation_id_fkey
FOREIGN KEY (conversation_id) REFERENCES public.chat_conversations(id) ON
DELETE CASCADE;

--
-- Name: turn_knowledge_analysis turn_knowledge_analysis_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_knowledge_analysis ADD CONSTRAINT turn_knowledge_analysis_evaluation_id_fkey
FOREIGN KEY (evaluation_id) REFERENCES public.evaluations(id) ON
DELETE
SET NULL;

--
-- Name: turn_knowledge_analysis turn_knowledge_analysis_pipeline_run_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_knowledge_analysis ADD CONSTRAINT turn_knowledge_analysis_pipeline_run_id_fkey
FOREIGN KEY (pipeline_run_id) REFERENCES public.pipeline_runs(id);

--
-- Name: turn_knowledge_analysis turn_knowledge_analysis_turn_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.turn_knowledge_analysis ADD CONSTRAINT turn_knowledge_analysis_turn_id_fkey
FOREIGN KEY (turn_id) REFERENCES public.chat_turns(id) ON
DELETE CASCADE;

--
--
