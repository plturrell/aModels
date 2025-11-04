#!/bin/bash
# Sync domain configs from Postgres to Redis
# This script can be run manually or via cron after training

set -e

POSTGRES_DSN="${POSTGRES_DSN:-postgres://postgres:postgres@postgres:5432/amodels?sslmode=disable}"
REDIS_URL="${REDIS_URL:-redis://redis:6379/0}"
REDIS_KEY="${REDIS_KEY:-localai:domains:config}"

echo "🔄 Syncing domain configs from Postgres to Redis..."
echo "   Postgres: $POSTGRES_DSN"
echo "   Redis: $REDIS_URL"
echo "   Key: $REDIS_KEY"

# Run the config-sync service (one-time sync)
cd "$(dirname "$0")/.."
go run ./cmd/config-sync -postgres "$POSTGRES_DSN" -redis "$REDIS_URL" -redis-key "$REDIS_KEY" -interval 0

echo "✅ Sync complete"

