# Signavio Test Data Generator - Quick Start Guide

## Overview

This toolkit generates test files from Signavio API for integration testing. It supports multiple data sources (process instances, definitions, activities, agent telemetry) and multiple output formats (JSON, CSV, Avro).

## 📦 What You Get

- **Python Generator** (`signavio_test_generator.py`) - Main data generation tool
- **Shell Wrapper** (`generate_signavio_testdata.sh`) - Easy-to-use command-line interface  
- **Validator** (`validate_test_data.sh`) - Validates generated files
- **Config Template** (`signavio_config.template.json`) - API configuration template
- **Documentation** (`SIGNAVIO_GENERATOR_README.md`) - Comprehensive guide

## 🚀 Quick Start (5 minutes)

### Step 1: Generate Test Data

```bash
# Navigate to scripts directory
cd /home/aModels/scripts

# Generate test data (mock mode - no API required)
./generate_signavio_testdata.sh \
  --mode mock \
  --output-dir ./my_test_data \
  --telemetry 50

# Output: 
# - my_test_data/agent_telemetry.json
# - my_test_data/agent_telemetry.csv
# - my_test_data/agent_telemetry.avsc
# - my_test_data/process_instances.json
# - my_test_data/process_definitions.json
# - ... and more
```

### Step 2: Validate the Generated Files

```bash
# Validate the test data
./validate_test_data.sh --dir ./my_test_data

# Output:
# ✓ Valid JSON syntax
# ✓ Found 50 records
# ✓ CSV header present
# ✓ All checks PASSED
```

### Step 3: Use in Your Tests

```bash
# Copy to your test fixtures directory
cp ./my_test_data/agent_telemetry.* /home/aModels/testing/manual/signavio/

# Or use directly in tests
cd /home/aModels/services/testing
go test -v -run TestSignavioClient
```

## 📋 Common Use Cases

### Use Case 1: Generate Large Dataset for Load Testing

```bash
./generate_signavio_testdata.sh \
  --mode mock \
  --telemetry-only \
  --telemetry 1000 \
  --output-dir ./load_test_data
```

**Result:** 1000 agent telemetry records for load testing

### Use Case 2: Generate All Data Types

```bash
./generate_signavio_testdata.sh \
  --mode mock \
  --limit 100 \
  --telemetry 200 \
  --output-dir ./full_suite
```

**Result:** Complete test suite with:
- 100 process instances
- 3 process definitions
- 3 activities
- 200 agent telemetry records

### Use Case 3: Real API Data Collection

```bash
# First, create your config file
cp signavio_config.template.json my_config.json
# Edit my_config.json with your actual credentials

# Generate from real API
./generate_signavio_testdata.sh \
  --mode api \
  --config my_config.json \
  --limit 500 \
  --output-dir ./production_test_data
```

**Result:** Real data from your Signavio instance

### Use Case 4: Quick Prototype Testing

```bash
# Generate minimal dataset
./generate_signavio_testdata.sh \
  --telemetry-only \
  --telemetry 5 \
  --output-dir /tmp/quick_test

# Validate it
./validate_test_data.sh --dir /tmp/quick_test

# Use immediately
cat /tmp/quick_test/agent_telemetry.json
```

## 🔧 Integration Examples

### Integration with Go Tests

```go
// Load generated test data
func TestWithGeneratedData(t *testing.T) {
    data, err := os.ReadFile("./signavio_test_data/agent_telemetry.json")
    require.NoError(t, err)
    
    var records []SignavioTelemetryRecord
    err = json.Unmarshal(data, &records)
    require.NoError(t, err)
    
    // Use records in your test
    assert.Equal(t, 50, len(records))
}
```

### Integration with Shell Scripts

```bash
#!/bin/bash
# automated_test_pipeline.sh

# Generate fresh test data
./scripts/generate_signavio_testdata.sh \
  --telemetry-only \
  --telemetry 100 \
  --output-dir /tmp/test_run

# Run tests with generated data
cd services/testing
TEST_DATA_DIR=/tmp/test_run go test -v

# Clean up
rm -rf /tmp/test_run
```

### Integration with Python Tests

```python
import json

def test_with_generated_data():
    with open('./signavio_test_data/agent_telemetry.json') as f:
        records = json.load(f)
    
    assert len(records) > 0
    assert 'agent_run_id' in records[0]
    assert 'agent_name' in records[0]
```

## 📊 Understanding the Output

### Directory Structure

```
signavio_test_data/
├── agent_telemetry.json       # Agent telemetry records (JSON)
├── agent_telemetry.csv        # Agent telemetry records (CSV)
├── agent_telemetry.avsc       # Avro schema for telemetry
├── process_instances.json     # Process instance data
├── process_instances.csv      # Process instances (CSV)
├── process_instances.avsc     # Avro schema for instances
├── process_definitions.json   # Process definitions
├── process_definitions.csv    # Process definitions (CSV)
├── process_definitions.avsc   # Avro schema for definitions
├── activities.json            # Activity data
├── activities.csv             # Activities (CSV)
└── activities.avsc            # Avro schema for activities
```

### Agent Telemetry Record Format

```json
{
  "agent_run_id": "run-1000",
  "agent_name": "compliance-reasoning-agent",
  "task_id": "task-2000",
  "task_description": "Process regulatory compliance check",
  "start_time": 1762914795762,
  "end_time": 1762914915762,
  "status": "success",
  "latency_ms": 120000,
  "service_name": "regulatory-service",
  "workflow_name": "bcbs239-audit",
  "workflow_version": "v2.1",
  "tools_used": "[{\"tool_name\":\"database_query\",\"call_count\":3}]",
  "llm_calls": "[{\"model\":\"gpt-4\",\"total_tokens\":1200}]",
  "process_steps": "[{\"step_name\":\"Initialize\",\"duration_ms\":200}]"
}
```

## ⚙️ Command Reference

### Generator Options

```bash
./generate_signavio_testdata.sh [OPTIONS]

Options:
  -o, --output-dir DIR      Output directory
  -c, --config FILE         Config file (for API mode)
  -m, --mode MODE           'mock' or 'api'
  -l, --limit NUM           Number of process instances
  -t, --telemetry NUM       Number of telemetry records
  --instances-only          Generate only instances
  --definitions-only        Generate only definitions
  --activities-only         Generate only activities
  --telemetry-only          Generate only telemetry
  -h, --help                Show help
```

### Validator Options

```bash
./validate_test_data.sh [OPTIONS]

Options:
  -d, --dir DIR             Data directory to validate
  -h, --help                Show help
```

### Python Generator Options

```bash
python3 signavio_test_generator.py [OPTIONS]

Options:
  --config FILE             Config file
  --output-dir DIR          Output directory
  --mock-mode               Use mock data (no API)
  --instances-only          Generate instances only
  --definitions-only        Generate definitions only
  --activities-only         Generate activities only
  --telemetry-only          Generate telemetry only
  --limit NUM               Instance limit
  --telemetry-count NUM     Telemetry count
```

## 🐛 Troubleshooting

### Problem: "requests module not found"

```bash
# Solution: Install requests
pip install requests

# Or use the shell wrapper which auto-installs dependencies
./generate_signavio_testdata.sh --mode mock
```

### Problem: "Permission denied"

```bash
# Solution: Make scripts executable
chmod +x ./generate_signavio_testdata.sh
chmod +x ./validate_test_data.sh
```

### Problem: "Output directory not writable"

```bash
# Solution: Create directory with proper permissions
mkdir -p ./signavio_test_data
chmod 755 ./signavio_test_data
```

### Problem: "Invalid JSON" in generated files

```bash
# Solution: Regenerate with fresh output directory
rm -rf ./problematic_data
./generate_signavio_testdata.sh --output-dir ./fresh_data
```

### Problem: API authentication fails

```bash
# Solution 1: Use mock mode instead
./generate_signavio_testdata.sh --mode mock

# Solution 2: Check your config file
cat signavio_config.json  # Verify credentials
```

## 📚 Next Steps

1. **Read Full Documentation**: `SIGNAVIO_GENERATOR_README.md`
2. **Review Existing Tests**: `/home/aModels/testing/manual/signavio/`
3. **Check SignavioClient**: `/home/aModels/services/testing/signavio_client.go`
4. **API Documentation**: `/home/aModels/docs/sap-signavio-process-intelligence-api-guide-en.txt`

## 💡 Tips

- **Start with mock mode** - Test your workflow without API access
- **Validate generated files** - Always run the validator before using in tests
- **Version control test data** - Commit small test datasets to your repo
- **Use appropriate sizes** - Generate only as much data as you need
- **Leverage existing infrastructure** - Files are compatible with your SignavioClient

## 🎯 Success Checklist

- [ ] Generated test data in mock mode
- [ ] Validated generated files
- [ ] Inspected JSON output structure
- [ ] Tested integration with Go tests
- [ ] Read full documentation
- [ ] Configured real API (optional)
- [ ] Generated production-size dataset (optional)

## 📞 Support

For questions or issues:
1. Check the full README: `SIGNAVIO_GENERATOR_README.md`
2. Review examples in `/home/aModels/testing/manual/signavio/`
3. Consult existing implementation: `services/testing/signavio_client.go`

---

**Last Updated:** 2025-01-12  
**Version:** 1.0.0  
**Compatible With:** SignavioClient v2.0+
