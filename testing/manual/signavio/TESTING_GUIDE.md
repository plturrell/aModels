# Signavio Integration Test Package

## Overview
This package provides stubs and test data for integrating with SAP Signavio Process Intelligence APIs. It includes sample telemetry data and process definitions that match the expected formats.

## Test Data

### 1. Agent Telemetry
- **File**: `agent_telemetry.csv`
- **Schema**: `agent_telemetry.avsc`
- **Description**: Sample agent execution data including run IDs, task details, and timestamps.

### 2. Process Library
- **File**: `process_library.json`
- **Description**: Sample process definitions in Signavio-compatible format.

## How to Test

### 1. Uploading Telemetry
```python
from signavio_tool import signavio_stub_upload

# Upload sample telemetry
result = signavio_stub_upload(
    dataset="agent-telemetry",
    file_path="agent_telemetry.csv",
    schema_path="agent_telemetry.avsc",
    primary_keys=["agent_run_id", "task_id"]
)
```

### 2. Fetching Process Data
```python
from signavio_tool import signavio_stub_fetch_view

# Fetch process library
processes = signavio_stub_fetch_view(view_name="ProcessLibrary")
```

## Expected Outputs
- Uploads return a stub URL (e.g., `stub://signavio/uploads/agent-telemetry/agent_telemetry.csv`)
- Process fetches return JSON with process definitions

## Validation
1. Check that the CSV matches the Avro schema
2. Verify that process definitions are correctly formatted
3. Test with different filter conditions

## Next Steps
1. Replace stubs with actual API calls when endpoints are available
2. Update schemas as needed based on Signavio requirements
