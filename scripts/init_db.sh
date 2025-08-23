#!/usr/bin/env bash
set -euo pipefail

DB_NAME="stephanie"
DB_USER="postgres"
DB_PASS="postgres"
DB_HOST="${DB_HOST:-db}"   # default: service name in docker-compose
DB_PORT="${DB_PORT:-5432}"

echo "🔑 Creating role (if not exists)..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -v ON_ERROR_STOP=1 <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$DB_USER') THEN
            CREATE ROLE $DB_USER WITH LOGIN PASSWORD '$DB_PASS';
        END IF;
    END
    \$\$;
EOSQL

echo "📦 Creating database (if not exists)..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -v ON_ERROR_STOP=1 <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_database WHERE datname = '$DB_NAME') THEN
            CREATE DATABASE $DB_NAME OWNER $DB_USER;
        END IF;
    END
    \$\$;
EOSQL

echo "🔌 Enabling pgvector extension..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;
EOSQL

echo "📜 Running schema.sql..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 -f ./schema.sql

echo "✅ Database initialized."
