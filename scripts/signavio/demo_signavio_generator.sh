#!/bin/bash
################################################################################
# Signavio Test Generator Demo
# 
# Demonstrates the Signavio test data generator with various use cases
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="/tmp/signavio_demo_$(date +%s)"

echo "========================================================================"
echo "Signavio Test Data Generator - Demo"
echo "========================================================================"
echo ""
echo -e "${BLUE}This demo will showcase different ways to use the generator${NC}"
echo -e "${BLUE}Demo directory: ${DEMO_DIR}${NC}"
echo ""

# Create demo directory
mkdir -p "$DEMO_DIR"

# Demo 1: Quick Telemetry Generation
echo "========================================================================"
echo -e "${GREEN}Demo 1: Quick Telemetry Generation${NC}"
echo "========================================================================"
echo ""
echo "Generating 5 agent telemetry records..."
echo ""

"$SCRIPT_DIR/generate_signavio_testdata.sh" \
  --mode mock \
  --telemetry-only \
  --telemetry 5 \
  --output-dir "$DEMO_DIR/demo1"

echo ""
echo "Files created:"
ls -lh "$DEMO_DIR/demo1/"
echo ""
echo "Sample JSON record:"
head -30 "$DEMO_DIR/demo1/agent_telemetry.json"
echo ""
read -p "Press Enter to continue..."

# Demo 2: Full Test Suite
echo ""
echo "========================================================================"
echo -e "${GREEN}Demo 2: Full Test Suite Generation${NC}"
echo "========================================================================"
echo ""
echo "Generating complete test suite (instances, definitions, activities, telemetry)..."
echo ""

"$SCRIPT_DIR/generate_signavio_testdata.sh" \
  --mode mock \
  --limit 10 \
  --telemetry 10 \
  --output-dir "$DEMO_DIR/demo2"

echo ""
echo "Complete directory structure:"
tree "$DEMO_DIR/demo2/" 2>/dev/null || ls -lh "$DEMO_DIR/demo2/"
echo ""
read -p "Press Enter to continue..."

# Demo 3: Validation
echo ""
echo "========================================================================"
echo -e "${GREEN}Demo 3: Data Validation${NC}"
echo "========================================================================"
echo ""
echo "Validating generated test data..."
echo ""

"$SCRIPT_DIR/validate_test_data.sh" --dir "$DEMO_DIR/demo2"

echo ""
read -p "Press Enter to continue..."

# Demo 4: Large Dataset
echo ""
echo "========================================================================"
echo -e "${GREEN}Demo 4: Large Dataset Generation${NC}"
echo "========================================================================"
echo ""
echo "Generating 100 telemetry records for load testing..."
echo ""

time "$SCRIPT_DIR/generate_signavio_testdata.sh" \
  --mode mock \
  --telemetry-only \
  --telemetry 100 \
  --output-dir "$DEMO_DIR/demo4"

echo ""
echo "Dataset statistics:"
echo "Records: $(jq 'length' "$DEMO_DIR/demo4/agent_telemetry.json")"
echo "JSON size: $(stat -c%s "$DEMO_DIR/demo4/agent_telemetry.json" 2>/dev/null || stat -f%z "$DEMO_DIR/demo4/agent_telemetry.json" 2>/dev/null) bytes"
echo "CSV size: $(stat -c%s "$DEMO_DIR/demo4/agent_telemetry.csv" 2>/dev/null || stat -f%z "$DEMO_DIR/demo4/agent_telemetry.csv" 2>/dev/null) bytes"
echo ""
read -p "Press Enter to continue..."

# Demo 5: Data Inspection
echo ""
echo "========================================================================"
echo -e "${GREEN}Demo 5: Data Inspection${NC}"
echo "========================================================================"
echo ""
echo "Inspecting generated data structure..."
echo ""

if command -v jq &> /dev/null; then
    echo "Agent names distribution:"
    jq -r '.[].agent_name' "$DEMO_DIR/demo4/agent_telemetry.json" | sort | uniq -c
    echo ""
    
    echo "Status distribution:"
    jq -r '.[].status' "$DEMO_DIR/demo4/agent_telemetry.json" | sort | uniq -c
    echo ""
    
    echo "Average latency:"
    jq '[.[].latency_ms] | add / length' "$DEMO_DIR/demo4/agent_telemetry.json"
    echo "ms"
    echo ""
    
    echo "Sample record with all fields:"
    jq '.[0]' "$DEMO_DIR/demo4/agent_telemetry.json"
else
    echo "Install 'jq' for detailed data inspection"
    echo ""
    echo "Sample record:"
    python3 -c "import json; print(json.dumps(json.load(open('$DEMO_DIR/demo4/agent_telemetry.json'))[0], indent=2))"
fi

echo ""
read -p "Press Enter to continue..."

# Demo 6: CSV Preview
echo ""
echo "========================================================================"
echo -e "${GREEN}Demo 6: CSV Data Preview${NC}"
echo "========================================================================"
echo ""
echo "CSV format is ideal for spreadsheet analysis..."
echo ""
echo "Header:"
head -1 "$DEMO_DIR/demo2/agent_telemetry.csv"
echo ""
echo "First 3 records:"
tail -3 "$DEMO_DIR/demo2/agent_telemetry.csv" | cut -c 1-120
echo "..."
echo ""
read -p "Press Enter to continue..."

# Summary
echo ""
echo "========================================================================"
echo -e "${GREEN}Demo Summary${NC}"
echo "========================================================================"
echo ""
echo "You've seen:"
echo "  ✓ Quick telemetry generation (Demo 1)"
echo "  ✓ Full test suite generation (Demo 2)"
echo "  ✓ Data validation (Demo 3)"
echo "  ✓ Large dataset generation (Demo 4)"
echo "  ✓ Data inspection techniques (Demo 5)"
echo "  ✓ CSV format preview (Demo 6)"
echo ""
echo "Demo files are located at:"
echo "  $DEMO_DIR"
echo ""
echo "To clean up demo files:"
echo "  rm -rf $DEMO_DIR"
echo ""
echo "Next steps:"
echo "  1. Read QUICK_START.md for usage guide"
echo "  2. Read SIGNAVIO_GENERATOR_README.md for full documentation"
echo "  3. Try generating data for your own tests"
echo "  4. Integrate with your testing workflow"
echo ""
echo "========================================================================"
echo -e "${GREEN}Demo Complete!${NC}"
echo "========================================================================"
echo ""

# Optional: Keep or clean up
read -p "Would you like to keep the demo files? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleaning up demo files..."
    rm -rf "$DEMO_DIR"
    echo "✓ Demo files removed"
else
    echo "✓ Demo files kept at: $DEMO_DIR"
fi

echo ""
echo "Thank you for trying the Signavio Test Data Generator!"
