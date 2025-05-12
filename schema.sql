CREATE EXTENSION IF NOT EXISTS vector;

-- Stores all generated hypotheses and their evaluations
CREATE TABLE IF NOT EXISTS hypotheses (
    id SERIAL PRIMARY KEY,
    goal TEXT NOT NULL,                -- Research objective
    text TEXT NOT NULL,               -- Hypothesis statement
    confidence FLOAT DEFAULT 0.0,     -- Confidence score (0–1 scale)
    review JSONB,                     -- Structured review data
    elo_rating FLOAT DEFAULT 1000.0,  -- Tournament ranking score
    embedding VECTOR(1024),           -- Vector representation of hypothesis
    features JSONB,                   -- Mechanism, rationale, experiment plan
    prompt_id INT REFERENCES prompts(id), -- Prompt used to generate this hypothesis
    source_hypothesis INT REFERENCES hypotheses(id), -- If derived from another
    strategy_used TEXT,               -- e.g., goal_aligned, out_of_the_box
    version INT DEFAULT 1,            -- Evolve count
    source TEXT,                      -- e.g., manual, refinement, grafting
    enabled BOOLEAN DEFAULT TRUE,      -- Soft delete flag
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
    agent_name TEXT NOT NULL,         -- e.g., "generation", "reflection"
    strategy TEXT NOT NULL,          -- e.g., "goal_aligned", "out_of_the_box"
    prompt_key TEXT NOT NULL,        -- e.g., "generation_goal_aligned.txt"
    prompt_text TEXT NOT NULL,       -- The actual prompt template
    output_key TEXT,                -- Which context key this affects (e.g., "hypotheses")
    input_keys JSONB,                -- Context fields used (e.g., ["goal", "literature"])
    extraction_regex TEXT,          -- Regex used to extract response
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    version INT DEFAULT 1,
    source TEXT,                    -- e.g., "manual", "feedback_injection", "dsp_refinement"
    is_current BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}'::JSONB
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