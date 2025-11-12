#!/bin/bash
# Generate secure secrets for aModels deployment
# Usage: ./scripts/generate-secrets.sh

set -e

echo "Generating secure secrets for aModels..."
echo "=========================================="
echo ""

# Generate JWT Secret Key
echo "JWT_SECRET_KEY=$(openssl rand -base64 32)"
echo ""

# Generate Neo4j Password
echo "NEO4J_PASSWORD=$(openssl rand -base64 24 | tr -d "=+/" | cut -c1-32)"
echo ""

# Generate PostgreSQL Password
echo "POSTGRES_PASSWORD=$(openssl rand -base64 24 | tr -d "=+/" | cut -c1-32)"
echo ""

# Generate DMS Neo4j Password
echo "DMS_NEO4J_PASSWORD=$(openssl rand -base64 24 | tr -d "=+/" | cut -c1-32)"
echo ""

# Generate HANA Password
echo "HANA_PASSWORD=$(openssl rand -base64 24 | tr -d "=+/" | cut -c1-32)"
echo ""

echo "=========================================="
echo "Copy these values to your .env file"
echo "WARNING: Keep these secrets secure and never commit them to version control!"

