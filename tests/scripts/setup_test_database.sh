#!/bin/bash
# Week 1: Database Setup Script
# Sets up test databases (PostgreSQL) with required schema

set -e

echo "=========================================="
echo "Setting up Test Database - Week 1"
echo "=========================================="
echo ""

# Configuration
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-password}"
POSTGRES_DB="${POSTGRES_DB:-amodels_test}"

POSTGRES_DSN="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"

echo "PostgreSQL Configuration:"
echo "  Host: ${POSTGRES_HOST}"
echo "  Port: ${POSTGRES_PORT}"
echo "  Database: ${POSTGRES_DB}"
echo "  User: ${POSTGRES_USER}"
echo ""

# Check if psql is available
if ! command -v psql &> /dev/null; then
    echo "⚠️  psql not found. Installing PostgreSQL client..."
    echo "   (This script requires PostgreSQL client tools)"
    exit 1
fi

# Run migration
echo "Running migration: 001_domain_configs.sql"
echo ""

MIGRATION_FILE="services/localai/migrations/001_domain_configs.sql"

if [ ! -f "$MIGRATION_FILE" ]; then
    echo "❌ Migration file not found: $MIGRATION_FILE"
    exit 1
fi

# Export password for psql
export PGPASSWORD="${POSTGRES_PASSWORD}"

# Create database if it doesn't exist
echo "Creating database if it doesn't exist..."
psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d postgres -c "CREATE DATABASE ${POSTGRES_DB};" 2>/dev/null || echo "Database may already exist"

# Run migration
echo "Running migration..."
psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -f "${MIGRATION_FILE}"

if [ $? -eq 0 ]; then
    echo "✅ Migration completed successfully"
else
    echo "❌ Migration failed"
    exit 1
fi

# Verify schema
echo ""
echo "Verifying schema..."
psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "\d domain_configs"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Schema verification successful"
    echo ""
    echo "Database setup complete!"
    echo "DSN: ${POSTGRES_DSN}"
else
    echo "❌ Schema verification failed"
    exit 1
fi

