CREATE EXTENSION IF NOT EXISTS vector;

-- Stores all generated hypotheses and their evaluations
CREATE TABLE IF NOT EXISTS hypotheses (
    id SERIAL PRIMARY KEY,
    goal TEXT NOT NULL,                 -- Research objective
    text TEXT NOT NULL,                 -- Hypothesis statement
    confidence FLOAT DEFAULT 0.0 ,      -- Confidence score (0–1 scale)
    review TEXT,                        -- Structured review data
    reflection TEXT,                    -- Structured reflection data
    elo_rating FLOAT DEFAULT 750.0,    -- Tournament ranking score
    embedding VECTOR(1024),             -- Vector representation of hypothesis
    features JSONB,                     -- Mechanism, rationale, experiment plan
    prompt_id INT REFERENCES prompts(id), -- Prompt used to generate this hypothesis
    source_hypothesis INT REFERENCES hypotheses(id), -- If derived from another
    strategy_used TEXT,                 -- e.g., goal_aligned, out_of_the_box
    version INT DEFAULT 1,              -- Evolve count
    source TEXT,                        -- e.g., manual, refinement, grafting
    enabled BOOLEAN DEFAULT TRUE,       -- Soft delete flag
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
-- CREATE INDEX idx_hypothesis_goal ON hypotheses(goal);
-- CREATE INDEX idx_hypothesis_elo ON hypotheses(elo_rating DESC);
-- CREATE INDEX idx_hypothesis_embedding ON hypotheses USING ivfflat(embedding vector_cosine_ops);
-- CREATE INDEX idx_hypothesis_source ON hypotheses(source);
-- CREATE INDEX idx_hypothesis_strategy ON hypotheses(strategy_used);


CREATE TABLE elo_ranking_log (
    id SERIAL PRIMARY KEY,
    run_id TEXT,
    hypothesis TEXT,
    prompt_version INT,
    prompt_strategy TEXT,
    score INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE summaries (
    id SERIAL PRIMARY KEY,
    text TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);


CREATE TABLE ranking_trace (
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

CREATE TABLE IF NOT EXISTS prompts (
    id SERIAL PRIMARY KEY,
    agent_name TEXT NOT NULL,
    prompt_key TEXT NOT NULL,         -- e.g., generation_goal_aligned.txt
    prompt_text TEXT NOT NULL,
    goal TEXT;
    response_text TEXT,
    source TEXT,                      -- e.g., manual, dsp_refinement, feedback_injection
    version INT DEFAULT 1,
    is_current BOOLEAN DEFAULT FALSE,
    strategy TEXT,                    -- e.g., goal_aligned, out_of_the_box
    metadata JSONB DEFAULT '{}'::JSONB,
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
    metadata JSONB DEFAULT '{}'::JSONB,
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
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Stores full pipeline context after each stage
CREATE TABLE IF NOT EXISTS context_states (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,             -- Unique ID per experiment
    stage_name TEXT NOT NULL,         -- Agent name (generation, reflection)
    version INT DEFAULT 1,           -- Iteration number for this stage
    context JSONB NOT NULL,          -- Full context dict after stage
    preferences JSONB,              -- Preferences used (novelty, feasibility)
    feedback JSONB,                 -- Feedback from previous stages
    metadata JSONB DEFAULT '{}'::JSONB, -- Strategy, prompt_version, etc.
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
    text TEXT UNIQUE,
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
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_mrq_goal ON mrq_memory(goal);
CREATE INDEX idx_mrq_strategy ON mrq_memory(strategy);
CREATE INDEX idx_mrq_reward ON mrq_memory(reward DESC);


CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    prompt_strategy TEXT NOT NULL,
    preference_used TEXT[],
    reward FLOAT NOT NULL,
    confidence_score FLOAT,
    metadata JSONB DEFAULT '{}'::JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_model_name ON model_performance(model_name);
CREATE INDEX idx_task_type ON model_performance(task_type);
CREATE INDEX idx_preference_used ON model_performance USING GIN(preference_used);


CREATE TABLE IF NOT EXISTS goals (
    id SERIAL PRIMARY KEY,
    goal_text TEXT NOT NULL,
    goal_type TEXT,                   -- e.g., 'research', 'forecast', 'writing'
    focus_area TEXT,                  -- e.g., 'AI', 'stock', 'healthcare'
    strategy TEXT,                    -- e.g., 'generation_reflect_review', 'cot_eval_refine'
    llm_suggested_strategy TEXT,
    source TEXT DEFAULT 'user',       -- 'user', 'llm', or 'hybrid'
    created_at TIMESTAMP DEFAULT now()
);

