#!/bin/bash
# Test LocalAI connectivity from within Docker network
# This verifies LocalAI is accessible from other containers

echo "Testing LocalAI from Docker network..."
echo ""

# Test from postgres container (on same network)
echo "Testing from postgres container..."
docker exec postgres sh -c '
if command -v wget >/dev/null 2>&1; then
    wget -q -O- http://localai:8080/readyz 2>&1 | head -3
elif command -v curl >/dev/null 2>&1; then
    curl -s http://localai:8080/readyz 2>&1 | head -3
else
    echo "No HTTP client available"
fi
' 2>&1

echo ""
echo "Testing from redis container..."
docker exec redis sh -c '
if command -v wget >/dev/null 2>&1; then
    wget -q -O- http://localai:8080/readyz 2>&1 | head -3
elif command -v curl >/dev/null 2>&1; then
    curl -s http://localai:8080/readyz 2>&1 | head -3
else
    echo "No HTTP client available"
fi
' 2>&1

echo ""
echo "Testing from inside LocalAI container..."
docker exec localai sh -c '
if command -v wget >/dev/null 2>&1; then
    wget -q -O- http://localhost:8080/readyz 2>&1 | head -3
elif command -v curl >/dev/null 2>&1; then
    curl -s http://localhost:8080/readyz 2>&1 | head -3
else
    echo "No HTTP client available, checking process..."
    ps aux | grep local-ai | head -2
fi
' 2>&1

