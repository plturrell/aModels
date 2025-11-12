#!/bin/bash
# Run Model Benchmarks

set -e
cd "$(dirname "$0")/../benchmarks"

echo "=========================================="
echo "Running Model Benchmarks"
echo "=========================================="
echo ""

echo "Available benchmarks:"
echo "  - arc (Abstraction and Reasoning)"
echo "  - boolq (Boolean Questions)"
echo "  - hellaswag (Commonsense Reasoning)"
echo "  - piqa (Physical Interaction QA)"
echo "  - socialiq (Social Interaction QA)"
echo "  - triviaqa (Trivia Questions)"
echo ""

# Run Go benchmarks
for dir in arc boolq hellaswag piqa socialiq triviaqa; do
    if [ -d "$dir" ]; then
        echo "Running $dir benchmark..."
        cd "$dir"
        go test -v . || echo "⚠️  $dir failed"
        cd ..
    fi
done

echo ""
echo "✅ Benchmarks complete"
