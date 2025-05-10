CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE hypotheses (
    id SERIAL PRIMARY KEY,
    goal TEXT,
    text TEXT,
    source TEXT,
    confidence FLOAT,
    review TEXT,
    embedding VECTOR(1024),
    features JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE elo_ranking_log (
    id SERIAL PRIMARY KEY,
    run_id TEXT,
    hypothesis TEXT,
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

CREATE TABLE IF NOT EXISTS prompts
(
    id SERIAL PRIMARY KEY,
    agent_name text NOT NULL,
    prompt_text text NOT NULL,,
    response_text text,
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