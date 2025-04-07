-- Ensure extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Final schema
DROP TABLE IF EXISTS memory;

CREATE TABLE memory (
    id TEXT PRIMARY KEY,
    title TEXT,
    content TEXT,
    user_text TEXT,
    ai_text TEXT,
    embedding vector(1024),
    tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(user_text, '') || ' ' || coalesce(ai_text, ''))
    ) STORED,
    timestamp TIMESTAMPTZ,
    tags TEXT[],
    summary TEXT,
    source TEXT,
    openai_url TEXT,
    length INTEGER,
    importance FLOAT,
    archived BOOLEAN DEFAULT FALSE
);

-- Indexes
CREATE INDEX idx_memory_tsv ON memory USING GIN(tsv);
CREATE INDEX idx_memory_trgm_title ON memory USING GIN(title gin_trgm_ops);
