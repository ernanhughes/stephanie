CREATE EXTENSION IF NOT EXISTS vector;

CREATE EXTENSION IF NOT EXISTS pgcrypto; -- text hashing


CREATE TABLE IF NOT EXISTS goals (
    id SERIAL PRIMARY KEY,
    goal_text TEXT NOT NULL,
    goal_type TEXT,                   -- e.g., 'research', 'forecast', 'writing'
    focus_area TEXT,                  -- e.g., 'AI', 'stock', 'healthcare'
    strategy TEXT,                    -- e.g., 'generation_reflect_review', 'cot_eval_refine'
    difficulty TEXT,
    goal_category TEXT,         
    llm_suggested_strategy TEXT,
    source TEXT DEFAULT 'user',       -- 'user', 'llm', or 'hybrid'
    created_at TIMESTAMP DEFAULT now()
);



CREATE TABLE IF NOT EXISTS prompts (
    id SERIAL PRIMARY KEY,

    -- Core Prompt Info
    agent_name TEXT NOT NULL,
    prompt_key TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    response_text TEXT,
    source TEXT,
    version INTEGER DEFAULT 1,
    is_current BOOLEAN DEFAULT FALSE,
    strategy TEXT,

    -- Associations
    goal_id INTEGER REFERENCES goals(id) ON DELETE NO ACTION,
    pipeline_run_id INTEGER,

    embedding_id INTEGER REFERENCES embeddings(id) ON DELETE NO ACTION, 

    -- Metadata
    extra_data JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient lookup
CREATE INDEX IF NOT EXISTS idx_prompt_agent
    ON prompts (source ASC NULLS LAST);

CREATE INDEX IF NOT EXISTS idx_prompt_strategy
    ON prompts (strategy ASC NULLS LAST);

CREATE INDEX IF NOT EXISTS idx_prompt_version
    ON prompts (version ASC NULLS LAST);



CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    goal_id INTEGER REFERENCES goals(id) ON DELETE CASCADE,
    run_id TEXT UNIQUE NOT NULL, -- UUID or generated string
    pipeline TEXT NOT NULL, -- list of agent names
    name TEXT,
    tag TEXT,
    embedding_type TEXT,
    embedding_dimensions int, 
    description TEXT,
    strategy TEXT,
    model_name TEXT,
    run_config JSONB,
    lookahead_context JSONB,
    symbolic_suggestion JSONB,
    extra_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Stores all generated hypotheses and their evaluations
CREATE TABLE IF NOT EXISTS hypotheses (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,                 -- Hypothesis statement
    goal_id INT REFERENCES goals(id), -- Prompt used to generate this hypothesis
    prompt_id INT REFERENCES prompts(id), -- Prompt used to generate this hypothesis
    strategy TEXT,                      -- e.g., goal_aligned, out_of_the_box
    confidence FLOAT DEFAULT 0.0 ,      -- Confidence score (0–1 scale)
    review TEXT,                        -- Structured review data
    reflection TEXT,                    -- Structured reflection data
    elo_rating FLOAT DEFAULT 750.0,    -- Tournament ranking score
    embedding VECTOR(1024),             -- Vector representation of hypothesis
    features JSONB,                     -- Mechanism, rationale, experiment plan
    source_hypothesis_id INT REFERENCES hypotheses(id), -- If derived from another
    source TEXT,                        -- e.g., manual, refinement, grafting
    pipeline_signature TEXT,            -- Unique identifier for the pipeline used
    pipeline_run_id INT REFERENCES pipeline_runs(id), -- Pipeline run this hypothesis belongs to
    enabled BOOLEAN DEFAULT TRUE,       -- Soft delete flag
    version INT DEFAULT 1,              -- Evolve count
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
-- CREATE INDEX idx_hypothesis_goal ON hypotheses(goal);
-- CREATE INDEX idx_hypothesis_elo ON hypotheses(elo_rating DESC);
-- CREATE INDEX idx_hypothesis_embedding ON hypotheses USING ivfflat(embedding vector_cosine_ops);
-- CREATE INDEX idx_hypothesis_source ON hypotheses(source);
-- CREATE INDEX idx_hypothesis_strategy ON hypotheses(strategy_used);


CREATE TABLE IF NOT EXISTS evaluations (
    id SERIAL PRIMARY KEY,

    -- Core Evaluation Context
    goal_id INTEGER REFERENCES goals(id) ON DELETE CASCADE,
    hypothesis_id INTEGER REFERENCES hypotheses(id) ON DELETE CASCADE,
    document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,

    -- Evaluator and Execution Info
    agent_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    evaluator_name TEXT NOT NULL,
    strategy TEXT,
    reasoning_strategy TEXT,
    run_id TEXT,

    -- Evaluation Metadata
    symbolic_rule_id INTEGER REFERENCES symbolic_rules(id) ON DELETE SET NULL,
    rule_application_id INTEGER REFERENCES rule_applications(id) ON DELETE SET NULL,
    pipeline_run_id INTEGER REFERENCES pipeline_runs(id) ON DELETE SET NULL,

    -- Scores and Extra Data
    scores JSON DEFAULT '{}'::json,
    extra_data JSONB DEFAULT '{}'::jsonb,

    -- Timestamp
    created_at TIMESTAMP DEFAULT NOW()
);



CREATE TABLE IF NOT EXISTS elo_ranking_log (
    id SERIAL PRIMARY KEY,
    run_id TEXT,
    hypothesis TEXT,
    prompt_version INT,
    prompt_strategy TEXT,
    score INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS  summaries (
    id SERIAL PRIMARY KEY,
    text TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);


CREATE TABLE IF NOT EXISTS ranking_trace (
    id SERIAL PRIMARY KEY,
    run_id TEXT,
    prompt_version INT,
    prompt_strategy TEXT,
    winner TEXT,
    loser TEXT,
    explanation TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    goal TEXT,
    summary TEXT,
    path TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    icon VARCHAR(4) DEFAULT '📦',
    data TEXT NOT NULL,
    embedding VECTOR(1024),
    hidden BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Table to track prompt evolution across agents
CREATE TABLE IF NOT EXISTS prompt_history (
    id SERIAL PRIMARY KEY,
    original_prompt_id INT REFERENCES prompts(id),
    prompt_text TEXT NOT NULL,        -- The actual prompt template
    agent_name TEXT NOT NULL,         -- e.g., "generation", "reflection"
    strategy TEXT NOT NULL,           -- e.g., "goal_aligned", "out_of_the_box"
    prompt_key TEXT NOT NULL,         -- e.g., "generation_goal_aligned.txt"
    output_key TEXT,                  -- Which context key this affects (e.g., "hypotheses")
    input_key TEXT,                 -- Context fields used (e.g., ["goal", "literature"])
    extraction_regex TEXT,            -- Regex used to extract response
    version INT DEFAULT 1,
    source TEXT,                      -- e.g., "manual", "feedback_injection", "dsp_refinement"
    is_current BOOLEAN DEFAULT FALSE,
    config JSONB DEFAULT '{}'::JSONB,
    extra_data JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id SERIAL PRIMARY KEY,
    agent_name TEXT NOT NULL,
    prompt_key TEXT NOT NULL,         -- e.g., "generation_goal_aligned.txt"
    prompt_text TEXT NOT NULL,
    previous_prompt_id INT REFERENCES prompts(id),
    strategy TEXT,
    version INT NOT NULL,
    source TEXT,                     -- manual, feedback_injection, dsp_refinement
    score_improvement FLOAT,         -- How much better is this prompt than last?
    extra_data JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Stores full pipeline context after each stage
CREATE TABLE IF NOT EXISTS context_states (
    id SERIAL PRIMARY KEY,
    goal_id INTEGER REFERENCES goals(id) ON DELETE SET NULL,
    pipeline_run_id INTEGER REFERENCES pipeline_runs(id) ON DELETE SET NULL,

    run_id TEXT NOT NULL,             -- Unique ID per experiment
    stage_name TEXT NOT NULL,         -- Agent name (generation, reflection)
    version INT DEFAULT 1,           -- Iteration number for this stage
    context JSONB NOT NULL,          -- Full context dict after stage
    trace JSONB DEFAULT '{}'::JSONB, -- Execution trace (e.g., {"generation": [...], "reflection": [...]})
    preferences JSONB,              -- Preferences used (novelty, feasibility)
    feedback JSONB,                 -- Feedback from previous stages
    extra_data JSONB DEFAULT '{}'::JSONB, -- Strategy, prompt_version, etc.
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    is_current BOOLEAN DEFAULT TRUE  -- Only one active version per run/stage
);

-- -- Indexes
-- CREATE INDEX idx_context_run ON context_states(run_id);
-- CREATE INDEX idx_context_stage ON context_states(stage_name);
-- CREATE INDEX idx_context_run_stage ON context_states(run_id, stage_name);
-- CREATE INDEX idx_context_preferences ON context_states USING GIN (preferences);

CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    text TEXT,
    text_hash TEXT UNIQUE,
    embedding VECTOR(1024),  -- adjust dimension if needed
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS prompt_evaluations (
    id SERIAL PRIMARY KEY,
    prompt_id INTEGER NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
    benchmark_name TEXT NOT NULL,                      -- e.g. "goal_alignment_test_set_1"
    score FLOAT,                                       -- Aggregated score for this benchmark
    metrics JSONB DEFAULT '{}'::jsonb,                 -- e.g. {"exact_match": 0.8, "precision": 0.75}
    dataset_hash TEXT,                                 -- Optional hash of the dataset used
    evaluator TEXT DEFAULT 'auto',                     -- "manual", "dspy", "llm", etc.
    notes TEXT,                                        -- Freeform notes about the evaluation
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mrq_memory (
    id SERIAL PRIMARY KEY,
    goal TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    strategy TEXT NOT NULL, -- e.g., recap, critic, devil
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    reward FLOAT NOT NULL,
    prompt_embedding VECTOR(1024),
    response_embedding VECTOR(1024),
    review_embedding VECTOR(1024),
    reflection_embedding VECTOR(1024),
    extra_data JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
-- CREATE INDEX idx_mrq_goal ON mrq_memory(goal);
-- CREATE INDEX idx_mrq_strategy ON mrq_memory(strategy);
-- CREATE INDEX idx_mrq_reward ON mrq_memory(reward DESC);


CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    prompt_strategy TEXT NOT NULL,
    preference_used TEXT[],
    reward FLOAT NOT NULL,
    confidence_score FLOAT,
    extra_data JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_model_name ON model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_task_type ON model_performance(task_type);
CREATE INDEX IF NOT EXISTS idx_preference_used ON model_performance USING GIN(preference_used);

CREATE TABLE IF NOT EXISTS mrq_evaluations (
    id SERIAL PRIMARY KEY,
    goal TEXT NOT NULL,
    prompt TEXT NOT NULL,
    output_a TEXT NOT NULL,
    output_b TEXT NOT NULL,
    winner TEXT NOT NULL, -- 'A' or 'B'
    score_a FLOAT NOT NULL,
    score_b FLOAT NOT NULL,
    extra_data JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_mrq_goal ON mrq_evaluations(goal);
CREATE INDEX IF NOT EXISTS idx_mrq_winner ON mrq_evaluations(winner);

CREATE TABLE IF NOT EXISTS sharpening_results (
    id SERIAL PRIMARY KEY,
    goal TEXT NOT NULL,
    prompt TEXT NOT NULL,
    template TEXT NOT NULL,
    original_output TEXT NOT NULL,
    sharpened_output TEXT NOT NULL,
    preferred_output TEXT NOT NULL,
    winner TEXT NOT NULL,  -- 'a' or 'b'
    improved BOOLEAN NOT NULL,
    comparison TEXT NOT NULL,  -- 'sharpened_better' or 'original_better'
    score_a FLOAT NOT NULL,
    score_b FLOAT NOT NULL,
    score_diff FLOAT NOT NULL,
    best_score FLOAT NOT NULL,
    prompt_template TEXT,  -- raw text if you want to log it
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS cot_pattern_stats (
    id SERIAL PRIMARY KEY,
    goal_id INTEGER REFERENCES goals(id) ON DELETE CASCADE,
    hypothesis_id INTEGER REFERENCES hypotheses(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    dimension TEXT NOT NULL,      -- e.g. "Inference Style"
    label TEXT NOT NULL,          -- e.g. "Analogical"
    confidence_score FLOAT,       -- optional if scoring is enabled
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cot_pattern_goal ON cot_pattern_stats (goal_id);
CREATE INDEX IF NOT EXISTS idx_cot_pattern_model ON cot_pattern_stats (model_name);
CREATE INDEX IF NOT EXISTS idx_cot_pattern_dimension ON cot_pattern_stats (dimension);


CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    goal_id INTEGER REFERENCES goals(id) ON DELETE CASCADE,
    hypothesis_id INTEGER REFERENCES hypotheses(id) ON DELETE CASCADE,
    rule_application_id INTEGER REFERENCES rule_applications(id) ON DELETE CASCADE,
    pipeline_run_id INTEGER REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    evaluator_name TEXT NOT NULL,
    score_type TEXT NOT NULL,
    score FLOAT,
    score_text TEXT,
    strategy TEXT,
    reasoning_strategy TEXT,
    rationale TEXT,
    reflection TEXT,            -- NEW
    review TEXT,                -- NEW
    meta_review TEXT,           -- NEW
    run_id TEXT,
    extra_data JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS lookaheads (
    id SERIAL PRIMARY KEY,
    goal_id INTEGER REFERENCES goals(id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    input_pipeline TEXT[],
    suggested_pipeline TEXT[],
    rationale TEXT,
    reflection TEXT,
    backup_plans TEXT[],
    extra_data JSONB DEFAULT '{}'::JSONB,
    run_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS reflection_deltas (
    id SERIAL PRIMARY KEY,
    goal_id INTEGER REFERENCES goals(id) ON DELETE CASCADE,
    run_id_a TEXT NOT NULL,
    run_id_b TEXT NOT NULL,
    score_a FLOAT,
    score_b FLOAT,
    score_delta FLOAT,
    pipeline_a JSONB DEFAULT '{}'::JSONB,
    pipeline_b JSONB DEFAULT '{}'::JSONB,
    pipeline_diff JSONB DEFAULT '{}'::JSONB, -- {"only_in_a": [...], "only_in_b": [...]}
    strategy_diff BOOLEAN DEFAULT FALSE,
    model_diff BOOLEAN DEFAULT FALSE,
    rationale_diff JSONB DEFAULT '["", ""]'::JSONB, -- tuple stored as array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS ideas (
    id SERIAL PRIMARY KEY,

    idea_text VARCHAR NOT NULL,
    parent_goal VARCHAR,
    focus_area VARCHAR,
    strategy VARCHAR,
    source VARCHAR,
    origin VARCHAR,
    extra_data JSON,
    goal_id INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY(goal_id) REFERENCES goals(id)
);

CREATE TABLE IF NOT EXISTS search_results (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    source TEXT NOT NULL,
    result_type TEXT,
    title TEXT,
    summary TEXT,
    url TEXT,
    author TEXT,
    published_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    tags TEXT[],
    goal_id INTEGER REFERENCES goals(id),
    parent_goal TEXT,
    strategy TEXT,
    focus_area TEXT,
    key_concepts TEXT[],
    technical_insights TEXT[],
    relevance_score INTEGER,
    novelty_score INTEGER,
    related_ideas TEXT[],
    refined_summary TEXT,
    extracted_methods TEXT[],
    domain_knowledge_tags TEXT[],
    critique_notes TEXT,
    extra_data JSON,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);


-- Table: method_plans
-- Purpose: Store structured research methodologies generated from hypotheses

CREATE TABLE IF NOT EXISTS method_plans (
    id SERIAL PRIMARY KEY,

    -- Core Research Idea
    idea_text TEXT NOT NULL,
    idea_id INTEGER REFERENCES ideas(id) ON DELETE SET NULL,

    -- Goal Context
    goal_id INTEGER REFERENCES goals(id) ON DELETE CASCADE,

    -- Methodological Components
    research_objective TEXT NOT NULL,
    key_components JSONB,         -- List of method components
    experimental_plan TEXT,
    hypothesis_mapping TEXT,      -- Mapping between hypotheses and plan parts
    search_strategy TEXT,        -- Keywords and sources to use in next round
    knowledge_gaps TEXT,        -- Missing info before testing
    next_steps TEXT,            -- What should be done after this plan

    -- Supporting Metadata
    task_description TEXT,
    baseline_method TEXT,
    literature_summary TEXT,
    code_plan TEXT,             -- Optional starter code / pseudocode
    focus_area TEXT,
    strategy TEXT,

    -- Evaluation Metrics
    score_novelty FLOAT,
    score_feasibility FLOAT,
    score_impact FLOAT,
    score_alignment FLOAT,

    -- Evolution Tracking
    evolution_level INTEGER DEFAULT 0,
    parent_plan_id INTEGER REFERENCES method_plans(id) ON DELETE SET NULL,
    is_refinement BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for faster querying
CREATE INDEX IF NOT EXISTS idx_idea_text ON method_plans USING GIN (to_tsvector('english', idea_text));
CREATE INDEX IF NOT EXISTS idx_research_objective ON method_plans USING GIN (to_tsvector('english', research_objective));
CREATE INDEX IF NOT EXISTS idx_focus_area ON method_plans (focus_area);
CREATE INDEX IF NOT EXISTS idx_evolution_level ON method_plans (evolution_level);
CREATE INDEX IF NOT EXISTS idx_goal_id ON method_plans (goal_id);
CREATE INDEX IF NOT EXISTS idx_parent_plan_id ON method_plans (parent_plan_id);


-- Create table for storing preference pairs used in ARM/MrQ training
CREATE TABLE IF NOT EXISTS mrq_preference_pairs (
    id SERIAL PRIMARY KEY,

    -- Goal or task group key (e.g., "arm_dpo", "math_reasoning")
    goal TEXT NOT NULL,

    -- Prompt/input question that generated the pair
    prompt TEXT NOT NULL,

    -- Output A and B (chosen and rejected responses)
    output_a TEXT NOT NULL,
    output_b TEXT NOT NULL,

    -- Which response was preferred: 'a' or 'b'
    preferred TEXT NOT NULL,

    -- Format used in each output
    fmt_a TEXT,
    fmt_b TEXT,

    -- Difficulty level (easy/medium/hard)
    difficulty TEXT,

    -- Source of this pair (e.g., arm_dataloader, human, agent)
    source TEXT,

    -- Run ID or session ID for tracking training runs
    run_id TEXT,

    -- Optional features (JSON metadata, e.g., reward shaping info)
    features JSONB,

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS symbolic_rules
(
    id SERIAL PRIMARY KEY,
    goal_id integer,
    pipeline_run_id integer,
    prompt_id integer,
    agent_name text,
    target text NOT NULL,
    rule_text text,
    source text,
    attributes jsonb,
    filter jsonb,
    context_hash text,
    score double precision,
    goal_type text,
    goal_category text,
    difficulty text,
    focus_area text,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS rule_applications (
    id SERIAL PRIMARY KEY,                                  -- Unique application ID
    rule_id INTEGER REFERENCES symbolic_rules(id) ON DELETE CASCADE, -- Applied rule
    goal_id INTEGER REFERENCES goals(id) ON DELETE CASCADE, -- Target goal
    pipeline_run_id INTEGER REFERENCES pipeline_runs(id) ON DELETE CASCADE, -- Affected run
    hypothesis_id INTEGER REFERENCES hypotheses(id),        -- Resulting hypothesis (optional)

    -- Metadata
    applied_at TIMESTAMP DEFAULT NOW(),                     -- When the rule was applied
    agent_name TEXT,                                        -- Which agent was affected
    change_type TEXT,                                       -- e.g., "pipeline_override", "hint", "param_tweak"
    details JSONB,                                          -- Any structured data (e.g., {"old":..., "new":...})

    -- Feedback loop: Evaluation
    post_score FLOAT,                                       -- Final score after applying the rule
    pre_score FLOAT,                                        -- Score before the rule (if available)
    delta_score FLOAT,                                      -- Computed delta (post - pre)
    evaluator_name TEXT,                                    -- Evaluator used to compute scores
    rationale TEXT,                                         -- Why the rule was applied (optional)
    notes TEXT                                               -- Extra notes or observations
);
CREATE TABLE IF NOT EXISTS prompt_programs (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    template TEXT NOT NULL,
    inputs JSON DEFAULT '{}',
    version INTEGER DEFAULT 1,
    parent_id TEXT REFERENCES prompt_programs(id),
    prompt_id INTEGER REFERENCES prompts(id),
    pipeline_run_id INTEGER REFERENCES pipeline_runs(id),
    strategy TEXT DEFAULT 'default',
    prompt_text TEXT,
    hypothesis TEXT,
    score FLOAT,
    rationale TEXT,
    mutation_type TEXT,
    execution_trace TEXT,
    extra_data JSON DEFAULT '{}'
);


CREATE TABLE IF NOT EXISTS score_dimensions (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE,
    stage VARCHAR,
    prompt_template TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    notes TEXT,
    extra_data JSON DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS unified_mrq_models (
    id SERIAL PRIMARY KEY,
    dimension TEXT NOT NULL,              -- e.g., 'correctness', 'clarity', etc.
    model_path TEXT NOT NULL,             -- Path to saved model artifact (e.g., .pkl or .pt)
    trained_on TIMESTAMP DEFAULT NOW(),   -- Timestamp of training
    pair_count INTEGER,                   -- Number of contrastive pairs used
    trainer_version TEXT,                 -- Version or hash of the MRQTrainer config
    notes TEXT,                           -- Optional notes (e.g., goal type filter, dataset slice)
    context JSONB                         -- Optional: additional metadata (e.g., embedding model, goal_type, etc.)
);


CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    evaluation_id INTEGER REFERENCES evaluations(id) ON DELETE CASCADE,
    dimension TEXT NOT NULL,
    score FLOAT,
    weight FLOAT,
    rationale TEXT
);

CREATE TABLE IF NOT EXISTS comparison_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    goal_id INTEGER NOT NULL,
    preferred_tag TEXT NOT NULL,
    rejected_tag TEXT NOT NULL,
    preferred_run_id UUID NOT NULL,
    rejected_run_id UUID NOT NULL,
    preferred_score FLOAT,
    rejected_score FLOAT,
    dimension_scores JSONB,
    reason TEXT,
    source TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    embedding_id INTEGER REFERENCES embeddings(id) ON DELETE NO ACTION, 

    -- Metadata
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    external_id TEXT,
    url TEXT,
    content TEXT,
    summary TEXT,
    date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Relationships
    goal_id INTEGER REFERENCES goals(id) ON DELETE SET NULL,

    -- Domain Classification
    domain_label TEXT,
    domains TEXT[]  -- Optional list of additional domain tags
);

CREATE TABLE IF NOT EXISTS document_domains (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    domain TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, domain)
);


CREATE TABLE IF NOT EXISTS document_sections (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_name TEXT NOT NULL,
    section_text TEXT NOT NULL,
    source TEXT DEFAULT 'unstructured+llm',
    summary TEXT,
    embedding json,
    extra_data json,
    domains text[],
    UNIQUE(document_id, section_name)
);

CREATE TABLE IF NOT EXISTS document_section_domains (
    id SERIAL PRIMARY KEY,
    document_section_id INTEGER NOT NULL REFERENCES document_sections(id) ON DELETE CASCADE,
    domain TEXT NOT NULL,
    score FLOAT NOT NULL,
    CONSTRAINT unique_document_section_domain UNIQUE (document_section_id, domain)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id SERIAL PRIMARY KEY,
    goal_id INTEGER REFERENCES goals(id) ON DELETE CASCADE,
    hypothesis_id INTEGER REFERENCES hypotheses(id) ON DELETE CASCADE,
    document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    agent_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    evaluator_name TEXT NOT NULL,
    strategy TEXT,
    reasoning_strategy TEXT,
    run_id TEXT,
    symbolic_rule_id INTEGER REFERENCES symbolic_rules(id) ON DELETE SET NULL,
    rule_application_id INTEGER REFERENCES rule_applications(id) ON DELETE SET NULL,
    pipeline_run_id INTEGER REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    scores JSON DEFAULT '{}'::json,
    extra_data JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()

     -- Add unique constraint here
  ---  CONSTRAINT unique_source_type_uri UNIQUE (source_type, source_uri)
);

CREATE TABLE IF NOT EXISTS evaluation_rule_links (
    id SERIAL PRIMARY KEY,
    evaluation_id INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    rule_application_id INTEGER NOT NULL REFERENCES rule_applications(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS worldviews (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    goal TEXT,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
    metadata TEXT,
    db_path TEXT
);

CREATE TABLE IF NOT EXISTS cartridges (
    id SERIAL PRIMARY KEY,
    goal_id INTEGER REFERENCES goals(id) ON DELETE SET NULL,
    source_type TEXT NOT NULL,
    source_uri TEXT,
    markdown_content TEXT NOT NULL,
    embedding_id INTEGER REFERENCES embeddings(id) ON DELETE SET NULL,
    title TEXT,
    summary TEXT,
    sections JSONB,
    triples JSONB,
    domain_tags JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS cartridge_domains (
    id SERIAL PRIMARY KEY,
    cartridge_id INTEGER NOT NULL,
    domain VARCHAR NOT NULL,
    score FLOAT NOT NULL,
    FOREIGN KEY (cartridge_id) REFERENCES cartridges(id) ON DELETE CASCADE,
    UNIQUE (cartridge_id, domain)
);

CREATE TABLE IF NOT EXISTS cartridge_triples (
    id SERIAL PRIMARY KEY,
    cartridge_id INTEGER NOT NULL REFERENCES cartridges(id) ON DELETE CASCADE,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (cartridge_id, subject, predicate, object)
);


CREATE TABLE IF NOT EXISTS theorems (
	id SERIAL PRIMARY KEY,
    statement TEXT NOT NULL,
    proof TEXT,
    embedding_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (embedding_id) REFERENCES embeddings(id)
);

CREATE TABLE IF NOT EXISTS theorem_cartridges (
    theorem_id INTEGER NOT NULL,
    cartridge_id INTEGER NOT NULL,
    PRIMARY KEY (theorem_id, cartridge_id),
    FOREIGN KEY (theorem_id) REFERENCES theorems(id),
    FOREIGN KEY (cartridge_id) REFERENCES cartridges(id)
);

CREATE INDEX idx_theorem_cartridges_theorem_id ON theorem_cartridges(theorem_id);
CREATE INDEX idx_theorem_cartridges_cartridge_id ON theorem_cartridges(cartridge_id);

-- Create measurements table
CREATE TABLE IF NOT EXISTS  measurements (
    id SERIAL PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    value JSONB NOT NULL,  -- Storing metrics as JSONB for efficient querying
    context JSONB,         -- Optional context metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for common lookup patterns
CREATE INDEX idx_measurements_entity_metric 
ON measurements (entity_type, entity_id, metric_name);

-- Index for time-based queries
CREATE INDEX idx_measurements_created_at 
ON measurements (created_at);

-- Optional: GIN index for searching within JSONB values
CREATE INDEX idx_measurements_value_gin 
ON measurements USING GIN (value);


CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_type TEXT NOT NULL,          -- e.g., "mrq", "svm", "ebt"
    target_type TEXT NOT NULL,        -- e.g., "document", "cartridge"
    dimension TEXT NOT NULL,           -- e.g., "ethics", "clarity"
    version TEXT NOT NULL,             -- e.g., "v1", "v2", "auto_20240315"
    trained_on JSONB,                 -- IDs of training examples used
    performance JSONB,                -- e.g., {"loss": 0.12, "accuracy": 0.89}
    created_at TIMESTAMP DEFAULT NOW(),
    active BOOLEAN DEFAULT TRUE,       -- current active version for inference
    extra_data JSONB,                     -- extra info (e.g., training config)
    model_path TEXT,
    encoder_path TEXT,
    tuner_path TEXT,
    scaler_path TEXT,
    meta_path TEXT,
    description TEXT,
    source TEXT DEFAULT 'user'          -- e.g., 'user', 'auto', 'llm'
);


CREATE TABLE IF NOT EXISTS scoring_history (
    id SERIAL PRIMARY KEY,
    model_version_id INTEGER REFERENCES model_versions(id),
    goal_id INTEGER,                  -- optional: link to goal context
    pipeline_run_id INTEGER REFERENCES pipeline_runs(id) ON DELETE SET NULL,
    model_type TEXT,
    target_id INTEGER NOT NULL,       -- e.g., document_id, cartridge_id
    target_type TEXT NOT NULL,         -- e.g., "document", "cartridge"
    dimension TEXT NOT NULL,            -- e.g., "relevance"
    raw_score FLOAT,                  -- uncalibrated model output
    transformed_score FLOAT,          -- post-processed score (e.g., tuned)
    uncertainty_score FLOAT,          -- confidence measure (e.g., energy)
    method TEXT NOT NULL,             -- e.g., "ebt", "mrq", "llm"
    source TEXT,                      -- e.g., "gpt-4", "human"
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scoring_events (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL,
    goal_text TEXT NOT NULL,
    original_text TEXT,
    refined_text TEXT,
    final_source TEXT NOT NULL, -- "mrq", "ebt", or "llm"
    used_refinement BOOLEAN DEFAULT FALSE,
    refinement_steps INTEGER,
    used_llm_fallback BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scoring_dimensions (
    event_id INTEGER REFERENCES scoring_events(id),
    dimension TEXT,
    mrq_score FLOAT,
    ebt_energy FLOAT,
    uncertainty_score FLOAT,
    final_score FLOAT,
    PRIMARY KEY (event_id, dimension)
);

CREATE TABLE IF NOT EXISTS scores (
    id SERIAL PRIMARY KEY,
    evaluation_id INTEGER REFERENCES evaluations(id),
    target_type TEXT NOT NULL,         -- e.g., "document", "triplet"
    target_id INTEGER NOT NULL,        -- refers to document.id, etc.
    dimension TEXT NOT NULL,

    score FLOAT NOT NULL,
    energy FLOAT,                      -- nullable: only present if EBT was used
    uncertainty FLOAT,                 -- sigmoid(energy), optional

    weight FLOAT DEFAULT 1.0,
    rationale TEXT,
    source TEXT,                       -- "mrq", "llm", "ebt", etc.
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table for storing refinement history
CREATE TABLE IF NOT EXISTS refinement_events (
    id SERIAL PRIMARY KEY,
    context TEXT NOT NULL,
    original TEXT NOT NULL,
    refined TEXT NOT NULL,
    context_hash TEXT NOT NULL,
    original_hash TEXT NOT NULL,
    refined_hash TEXT NOT NULL,
    original_score FLOAT,
    refined_score FLOAT,
    dimension TEXT NOT NULL,
    improvement FLOAT,
    energy_before FLOAT,
    energy_after FLOAT,
    steps_used INTEGER,
    source TEXT DEFAULT 'auto',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS memcubes (
    id TEXT PRIMARY KEY,  -- hash-based ID + version
    scorable_id bigint NOT NULL,  -- Foreign key to Scorable
    scorable_type TEXT NOT NULL,   -- e.g., document, theorem, triple
    content TEXT NOT NULL,        -- Raw text from Scorable
    dimension TEXT,               -- e.g., relevance, clarity, ethics
    original_score FLOAT,         -- Original score before any transformations
    refined_score FLOAT,          -- Post-processed score (e.g., tuned)
    refined_content TEXT,          -- Optional: text after refinement
    version TEXT NOT NULL,        -- v1, v2, etc.
    source TEXT,                -- e.g., user_input, inference_engine
    model TEXT,                 -- e.g., gpt-4, llama3
    priority INT DEFAULT 5,     -- 1–10 scale
    sensitivity TEXT DEFAULT 'public',  -- security tag
    ttl INT,                    -- Time-to-live in days
    usage_count INT DEFAULT 0,
    extra_data JSONB, 
    created_at TIMESTAMP DEFAULT NOW(),
    last_modified TIMESTAMP DEFAULT NOW()
);




-- Track transformations between memory types
CREATE TABLE IF NOT EXISTS  memcube_transformations (
    id SERIAL PRIMARY KEY,
    source_cube_id TEXT NOT NULL,
    target_cube_id TEXT NOT NULL,
    transformation_type TEXT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Track belief graph versions
CREATE TABLE IF NOT EXISTS  belief_graph_versions (
    id SERIAL PRIMARY KEY,
    goal TEXT NOT NULL,
    node_count INT,
    edge_count INT,
    avg_strength FLOAT,
    avg_relevance FLOAT,
    contradictions INT,
    theorems INT,
    created_at TIMESTAMP DEFAULT NOW(),
    model_path TEXT
);

-- Track theorem applications
CREATE TABLE IF NOT EXISTS  theorem_applications (
    id SERIAL PRIMARY KEY,
    theorem_id TEXT NOT NULL,
    context TEXT,
    result TEXT,
    success BOOLEAN,
    energy FLOAT,
    uncertainty FLOAT,
    applied_at TIMESTAMP DEFAULT NOW()
);


-- Track component versions and performance
CREATE TABLE IF NOT EXISTS  component_versions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    protocol TEXT NOT NULL,
    class_path TEXT NOT NULL,
    version TEXT NOT NULL,
    config JSONB,
    performance JSONB,
    active BOOLEAN DEFAULT TRUE,
    sensitivity TEXT CHECK(sensitivity IN ('public', 'internal', 'confidential', 'restricted')),
    created_at TIMESTAMP DEFAULT NOW(),
    last_used TIMESTAMP,
    usage_count INT DEFAULT 0,
    metadata JSONB
);


-- Track interface compliance
CREATE TABLE IF NOT EXISTS component_interfaces (
    component_id TEXT REFERENCES component_versions(id),
    protocol TEXT NOT NULL,
    implemented BOOLEAN DEFAULT TRUE,
    last_checked TIMESTAMP DEFAULT NOW()
);


-- Create table with core fields and relationships
CREATE TABLE belief_cartridges (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Source information
    source_id TEXT,
    source_type TEXT NOT NULL CHECK (source_type IN ('paper', 'blog', 'experiment', 'pipeline', 'manual')),
    source_url TEXT,
    
    -- Core content
    markdown_content TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Structured idea payload
    idea_payload JSONB,  -- e.g., {title: "Q-MAX", description: "...", code_snippet: "..."}
    
    -- Tagging
    goal_tags TEXT[] DEFAULT ARRAY[]::TEXT[],  -- Tags from goal context
    domain_tags TEXT[] DEFAULT ARRAY[]::TEXT[],  -- Tags from domain analysis
    
    -- Provenance
    derived_from JSONB DEFAULT '[]',  -- List of belief_cartridge IDs this was derived from
    applied_in JSONB DEFAULT '[]',    -- List of pipeline_run IDs where this was used
    
    -- Versioning
    version INTEGER DEFAULT 1,
    memcube_id TEXT,
    
    -- Relationships
    goal_id INTEGER REFERENCES goals(id) ON DELETE SET NULL,
    document_id INTEGER REFERENCES documents(id) ON DELETE SET NULL
);

-- Indexes for common queries
CREATE INDEX idx_belief_cartridges_source ON belief_cartridges(source_type, source_id);
CREATE INDEX idx_belief_cartridges_tags ON belief_cartridges USING GIN (goal_tags);
CREATE INDEX idx_belief_cartridges_active ON belief_cartridges(is_active);
CREATE INDEX idx_belief_cartridges_version ON belief_cartridges(version);


CREATE TABLE IF NOT EXISTS pipeline_stages (
    id SERIAL PRIMARY KEY,
    stage_name VARCHAR NOT NULL,
    agent_class VARCHAR NOT NULL,
    protocol_used VARCHAR NOT NULL,
    goal_id VARCHAR,
    run_id VARCHAR NOT NULL,
    pipeline_run_id INTEGER REFERENCES pipeline_runs(id),
    parent_stage_id INTEGER REFERENCES pipeline_stages(id),
    input_context_id INTEGER REFERENCES context_states(id),
    output_context_id INTEGER REFERENCES context_states(id),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    status VARCHAR NOT NULL,
    score NUMERIC,
    confidence NUMERIC,
    symbols_applied JSONB,
    extra_data JSONB,
    exportable BOOLEAN,
    reusable BOOLEAN,
    invalidated BOOLEAN
);

CREATE INDEX idx_pipeline_stages_run_id ON pipeline_stages(run_id);
CREATE INDEX idx_pipeline_stages_status ON pipeline_stages(status);
CREATE INDEX idx_pipeline_stages_goal_id ON pipeline_stages(goal_id);
CREATE INDEX idx_pipeline_stages_parent ON pipeline_stages(parent_stage_id);
CREATE INDEX idx_pipeline_stages_input_context ON pipeline_stages(input_context_id);
CREATE INDEX idx_pipeline_stages_output_context ON pipeline_stages(output_context_id);

-- Comment descriptions (optional but helpful)
COMMENT ON TABLE pipeline_stages IS 'Records each step in Stephanie’s reasoning process with full traceability.';
COMMENT ON COLUMN pipeline_stages.stage_name IS 'Name of this pipeline stage (e.g., "generation", "judge")';
COMMENT ON COLUMN pipeline_stages.agent_class IS 'Fully qualified name of the agent used';
COMMENT ON COLUMN pipeline_stages.protocol_used IS 'Protocol type used (e.g., "g3ps_search", "cot")';
COMMENT ON COLUMN pipeline_stages.goal_id IS 'Optional link to the associated goal ID';
COMMENT ON COLUMN pipeline_stages.run_id IS 'Unique identifier for the current pipeline run';
COMMENT ON COLUMN pipeline_stages.pipeline_run_id IS 'Foreign key to pipeline_runs table';
COMMENT ON COLUMN pipeline_stages.parent_stage_id IS 'Reference to prior stage for tracing reasoning paths';
COMMENT ON COLUMN pipeline_stages.input_context_id IS 'Context before running this stage';
COMMENT ON COLUMN pipeline_stages.output_context_id IS 'Context after running this stage';
COMMENT ON COLUMN pipeline_stages.status IS 'Stage outcome: accepted, rejected, retry, partial, pending';


-- File: versions/XXXX_create_protocols_table.sql

-- Up Migration
CREATE TABLE IF NOT EXISTS protocols (
    name VARCHAR PRIMARY KEY,
    description TEXT,
    input_format JSONB,
    output_format JSONB,
    failure_modes JSONB,
    depends_on JSONB,
    tags JSONB,
    capability VARCHAR,
    preferred_for JSONB,
    avoid_for JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_protocol_name ON protocols(name);
CREATE INDEX idx_protocol_tags ON protocols USING GIN (tags);
CREATE INDEX idx_protocol_capability ON protocols(capability);
CREATE INDEX idx_protocol_depends_on ON protocols USING GIN (depends_on);

-- Comment descriptions (optional but helpful)
COMMENT ON TABLE protocols IS 'Registry of available reasoning protocols used by Stephanie';
COMMENT ON COLUMN protocols.name IS 'Unique name of the protocol (e.g., "g3ps_search", "cot")';
COMMENT ON COLUMN protocols.description IS 'Human-readable description of what the protocol does';
COMMENT ON COLUMN protocols.input_format IS 'JSON schema defining expected input structure';
COMMENT ON COLUMN protocols.output_format IS 'JSON schema defining expected output structure';
COMMENT ON COLUMN protocols.failure_modes IS 'Common failure types (e.g., hallucination, syntax error)';
COMMENT ON COLUMN protocols.depends_on IS 'Other protocols or agents required for this one to work';
COMMENT ON COLUMN protocols.tags IS 'Tags like ["code", "reasoning", "llm"] for filtering';
COMMENT ON COLUMN protocols.capability IS 'High-level capability category (e.g., code_generation, qa)';
COMMENT ON COLUMN protocols.preferred_for IS 'Goal types where this protocol performs well';
COMMENT ON COLUMN protocols.avoid_for IS 'Goal types where this protocol should be avoided';
COMMENT ON COLUMN protocols.created_at IS 'When this protocol was added';
COMMENT ON COLUMN protocols.updated_at IS 'Last time metadata was changed';


CREATE TABLE IF NOT EXISTS goal_dimensions (
    id SERIAL PRIMARY KEY,
    goal_id INTEGER NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
    dimension TEXT NOT NULL,
    rank INTEGER DEFAULT 0,
    source TEXT DEFAULT 'llm',
    similarity_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE protocols (
    name TEXT PRIMARY KEY,
    description TEXT,
    input_format JSONB,
    output_format JSONB,
    failure_modes JSONB,
    depends_on JSONB,
    tags JSONB,
    capability TEXT,
    preferred_for JSONB,
    avoid_for JSONB
);

CREATE TABLE IF NOT EXISTS embeddings (
id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding VECTOR(1024),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    text_hash TEXT UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_embedding_vector
ON embeddings USING ivfflat (embedding vector_cosine_ops);
ALTER TABLE embeddings ADD CONSTRAINT unique_text_hash UNIQUE (text_hash);

CREATE TABLE IF NOT EXISTS hf_embeddings (
    id SERIAL PRIMARY KEY,
    text TEXT,
    embedding VECTOR(1024),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    text_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_hf_embedding_vector
ON hf_embeddings
USING ivfflat (embedding vector_cosine_ops);
ALTER TABLE hf_embeddings ADD CONSTRAINT unique_text_hash_hf UNIQUE (text_hash);


CREATE TABLE IF NOT EXISTS hnet_embeddings (
    id SERIAL PRIMARY KEY,
    text TEXT,
    embedding VECTOR(1024),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    text_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_hnet_embedding_vector
ON hnet_embeddings
USING ivfflat (embedding vector_cosine_ops);
ALTER TABLE hnet_embeddings ADD CONSTRAINT unique_text_hash_hnet UNIQUE (text_hash);

CREATE TABLE IF NOT EXISTS evaluation_attributes (
    id SERIAL PRIMARY KEY,
    evaluation_id INTEGER NOT NULL,
    dimension TEXT NOT NULL,
    source TEXT NOT NULL,
    raw_score FLOAT,
    energy FLOAT,
    uncertainty FLOAT,
    advantage FLOAT,
    pi_value FLOAT,
    q_value FLOAT,
    v_value FLOAT,
    extra JSON,
    entropy FLOAT,
    td_error FLOAT, 
    expected_return FLOAT,

    FOREIGN KEY (evaluation_id) REFERENCES evaluations(id) ON DELETE CASCADE
);


CREATE TABLE IF NOT EXISTS training_stats (
    id SERIAL PRIMARY KEY,
    
    -- Model identification
    model_type VARCHAR NOT NULL,
    target_type VARCHAR NOT NULL,
    dimension VARCHAR NOT NULL,
    version VARCHAR NOT NULL,
    embedding_type VARCHAR NOT NULL,
    
    -- Training metrics
    q_loss DOUBLE PRECISION,
    v_loss DOUBLE PRECISION,
    pi_loss DOUBLE PRECISION,
    avg_q_loss DOUBLE PRECISION,
    avg_v_loss DOUBLE PRECISION,
    avg_pi_loss DOUBLE PRECISION,
    
    -- Policy metrics
    policy_entropy DOUBLE PRECISION,
    policy_stability DOUBLE PRECISION,
    policy_logits JSONB,
    
    -- Configuration
    config JSONB,
    
    -- Dataset stats
    sample_count INTEGER DEFAULT 0,
    valid_samples INTEGER DEFAULT 0,
    invalid_samples INTEGER DEFAULT 0,
    
    -- Timing
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    
    -- Relationships
    goal_id INTEGER REFERENCES goals(id) ON DELETE SET NULL,
    model_version_id INTEGER REFERENCES model_versions(id) ON DELETE SET NULL
);

-- Indexes for common queries
CREATE INDEX idx_training_stats_dimension ON training_stats(dimension);
CREATE INDEX idx_training_stats_model ON training_stats(model_type);
CREATE INDEX idx_training_stats_version ON training_stats(version);
CREATE INDEX idx_training_stats_embedding ON training_stats(embedding_type);



