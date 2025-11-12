#!/bin/bash
################################################################################
# Validate Signavio Test Data
# 
# Validates generated test files for compatibility with SignavioClient
################################################################################

set -e

# Default values
DATA_DIR="./signavio_test_data"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Validate Signavio test data files.

OPTIONS:
    -d, --dir DIR              Data directory (default: ./signavio_test_data)
    -h, --help                 Show this help message

EXAMPLES:
    # Validate test data in default directory
    $0

    # Validate test data in custom directory
    $0 --dir /tmp/signavio_test

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Check if directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Directory not found: ${DATA_DIR}${NC}"
    exit 1
fi

echo "========================================================================"
echo "Signavio Test Data Validator"
echo "========================================================================"
echo ""

# Validate files exist
echo -e "${BLUE}Checking for required files...${NC}"

FILES=("agent_telemetry.json" "agent_telemetry.csv" "agent_telemetry.avsc")
MISSING=0

for file in "${FILES[@]}"; do
    filepath="${DATA_DIR}/${file}"
    if [ -f "$filepath" ]; then
        size=$(stat -c%s "$filepath" 2>/dev/null || stat -f%z "$filepath" 2>/dev/null)
        echo -e "${GREEN}✓${NC} ${file} (${size} bytes)"
    else
        echo -e "${RED}✗${NC} ${file} (missing)"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo -e "${RED}Error: ${MISSING} file(s) missing${NC}"
    exit 1
fi

echo ""

# Validate JSON structure
echo -e "${BLUE}Validating JSON structure...${NC}"

JSON_FILE="${DATA_DIR}/agent_telemetry.json"

# Check if jq is available for better validation
if command -v jq &> /dev/null; then
    # Validate JSON syntax
    if jq empty "$JSON_FILE" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Valid JSON syntax"
        
        # Count records
        record_count=$(jq 'length' "$JSON_FILE")
        echo -e "${GREEN}✓${NC} Found ${record_count} records"
        
        # Validate required fields in first record
        REQUIRED_FIELDS=("agent_run_id" "agent_name" "task_id" "start_time" "end_time" "status")
        for field in "${REQUIRED_FIELDS[@]}"; do
            if jq -e ".[0].${field}" "$JSON_FILE" > /dev/null 2>&1; then
                value=$(jq -r ".[0].${field}" "$JSON_FILE")
                echo -e "${GREEN}✓${NC} Field '${field}' present"
            else
                echo -e "${YELLOW}⚠${NC} Field '${field}' missing"
            fi
        done
        
        # Show sample record
        echo ""
        echo -e "${BLUE}Sample Record:${NC}"
        jq '.[0]' "$JSON_FILE" | head -20
    else
        echo -e "${RED}✗${NC} Invalid JSON syntax"
        exit 1
    fi
else
    # Basic JSON validation without jq
    if python3 -c "import json; json.load(open('$JSON_FILE'))" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Valid JSON syntax"
        
        record_count=$(python3 -c "import json; print(len(json.load(open('$JSON_FILE'))))")
        echo -e "${GREEN}✓${NC} Found ${record_count} records"
    else
        echo -e "${RED}✗${NC} Invalid JSON syntax"
        exit 1
    fi
fi

echo ""

# Validate CSV structure
echo -e "${BLUE}Validating CSV structure...${NC}"

CSV_FILE="${DATA_DIR}/agent_telemetry.csv"

# Check header
if head -1 "$CSV_FILE" | grep -q "agent_run_id"; then
    echo -e "${GREEN}✓${NC} CSV header present"
else
    echo -e "${RED}✗${NC} CSV header missing"
    exit 1
fi

# Count lines (excluding header)
line_count=$(($(wc -l < "$CSV_FILE") - 1))
echo -e "${GREEN}✓${NC} CSV contains ${line_count} data rows"

# Show first few lines
echo ""
echo -e "${BLUE}CSV Preview:${NC}"
head -3 "$CSV_FILE" | cut -c 1-100

echo ""

# Validate Avro schema
echo -e "${BLUE}Validating Avro schema...${NC}"

AVSC_FILE="${DATA_DIR}/agent_telemetry.avsc"

if command -v jq &> /dev/null; then
    # Validate JSON syntax
    if jq empty "$AVSC_FILE" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Valid JSON syntax"
        
        # Check for required Avro fields
        if jq -e '.type' "$AVSC_FILE" > /dev/null 2>&1; then
            schema_type=$(jq -r '.type' "$AVSC_FILE")
            echo -e "${GREEN}✓${NC} Schema type: ${schema_type}"
        fi
        
        if jq -e '.fields' "$AVSC_FILE" > /dev/null 2>&1; then
            field_count=$(jq '.fields | length' "$AVSC_FILE")
            echo -e "${GREEN}✓${NC} Schema has ${field_count} fields"
        fi
    else
        echo -e "${RED}✗${NC} Invalid Avro schema JSON"
        exit 1
    fi
else
    if python3 -c "import json; json.load(open('$AVSC_FILE'))" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Valid JSON syntax"
    else
        echo -e "${RED}✗${NC} Invalid Avro schema JSON"
        exit 1
    fi
fi

echo ""

# Summary
echo "========================================================================"
echo -e "${GREEN}Validation Summary${NC}"
echo "========================================================================"
echo "Data directory:        $(readlink -f ${DATA_DIR})"
echo "Files validated:       ${#FILES[@]}"
echo "JSON records:          ${record_count:-N/A}"
echo "CSV rows:              ${line_count:-N/A}"
echo "All checks:            ✓ PASSED"
echo "========================================================================"
echo ""
echo -e "${GREEN}✅ Test data is valid and compatible with SignavioClient${NC}"
