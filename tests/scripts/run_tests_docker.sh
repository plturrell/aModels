#!/bin/bash
# Docker Test Runner - Simple wrapper for running tests in Docker environment
# 
# This script ensures tests use Docker network URLs and run with proper configuration
#
# Usage:
#   ./run_tests_docker.sh           # Run all tests with Docker URLs
#   ./run_tests_docker.sh --check   # Run with service health check first
#   ./run_tests_docker.sh --strict  # Exit on first failure

set -e

cd "$(dirname "$0")"

echo "🐳 Running tests in Docker mode..."
echo ""

# Call the main test runner with Docker mode explicitly set
./run_all_tests.sh --docker "$@"
