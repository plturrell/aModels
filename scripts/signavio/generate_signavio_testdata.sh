#!/bin/bash
################################################################################
# Generate Signavio Test Data
# 
# Wrapper script for easy test data generation from Signavio API
################################################################################

set -e

# Default values
OUTPUT_DIR="./signavio_test_data"
MODE="mock"
LIMIT=50
TELEMETRY_COUNT=20
CONFIG_FILE=""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Generate test files from Signavio API for integration testing.

OPTIONS:
    -o, --output-dir DIR       Output directory (default: ./signavio_test_data)
    -c, --config FILE          Configuration file with API credentials
    -m, --mode MODE            Mode: mock or api (default: mock)
    -l, --limit NUM            Number of process instances (default: 50)
    -t, --telemetry NUM        Number of telemetry records (default: 20)
    --instances-only           Generate only process instances
    --definitions-only         Generate only process definitions
    --activities-only          Generate only activities
    --telemetry-only           Generate only agent telemetry
    -h, --help                 Show this help message

EXAMPLES:
    # Generate all test files with mock data
    $0 --mode mock --output-dir ./test_data

    # Generate 100 telemetry records
    $0 --telemetry-only --telemetry 100

    # Use real API with config file
    $0 --mode api --config signavio_config.json --limit 500

    # Generate only process instances
    $0 --instances-only --limit 200

EOF
}

# Parse arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        -t|--telemetry)
            TELEMETRY_COUNT="$2"
            shift 2
            ;;
        --instances-only|--definitions-only|--activities-only|--telemetry-only)
            EXTRA_ARGS+=("$1")
            shift
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

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    exit 1
fi

# Check if requests module is installed
if ! python3 -c "import requests" &> /dev/null; then
    echo -e "${YELLOW}Warning: 'requests' module not found${NC}"
    echo -e "${YELLOW}Installing requests...${NC}"
    pip install requests || {
        echo -e "${RED}Failed to install requests module${NC}"
        exit 1
    }
fi

# Build command
CMD="python3 ${SCRIPT_DIR}/signavio_test_generator.py"
CMD="${CMD} --output-dir ${OUTPUT_DIR}"
CMD="${CMD} --limit ${LIMIT}"
CMD="${CMD} --telemetry-count ${TELEMETRY_COUNT}"

# Add mode
if [ "$MODE" = "mock" ]; then
    CMD="${CMD} --mock-mode"
elif [ "$MODE" = "api" ]; then
    if [ -n "$CONFIG_FILE" ]; then
        if [ ! -f "$CONFIG_FILE" ]; then
            echo -e "${RED}Error: Config file not found: ${CONFIG_FILE}${NC}"
            exit 1
        fi
        CMD="${CMD} --config ${CONFIG_FILE}"
    else
        echo -e "${RED}Error: API mode requires --config option${NC}"
        exit 1
    fi
else
    echo -e "${RED}Error: Invalid mode: ${MODE}${NC}"
    echo "Mode must be 'mock' or 'api'"
    exit 1
fi

# Add extra arguments
for arg in "${EXTRA_ARGS[@]}"; do
    CMD="${CMD} ${arg}"
done

# Print command
echo -e "${GREEN}Generating Signavio test data...${NC}"
echo -e "${YELLOW}Command: ${CMD}${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Execute command
eval "$CMD"

# Print summary
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Test data generation complete!${NC}"
    echo -e "${GREEN}✓ Files saved to: $(readlink -f ${OUTPUT_DIR})${NC}"
    echo ""
    echo "Generated files:"
    ls -lh "$OUTPUT_DIR" | tail -n +2 | awk '{printf "  - %s (%s)\n", $9, $5}'
else
    echo -e "${RED}✗ Test data generation failed${NC}"
    exit 1
fi
