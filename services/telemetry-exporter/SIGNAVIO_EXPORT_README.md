# Signavio Export - Test Output

## Generated Files

The following files have been generated and are ready for upload to Signavio Process Intelligence:

### 1. CSV Data File
**File:** `signavio_telemetry_export.csv`
- Format: CSV with header row
- Records: 1 test record
- Size: ~2.9 KB
- Encoding: UTF-8
- Delimiter: Comma

### 2. Avro Schema File
**File:** `signavio_telemetry_schema.json`
- Format: JSON Avro schema
- Size: ~2.0 KB
- Defines all fields and types for Signavio ingestion

### 3. JSON Reference File
**File:** `signavio_telemetry_record.json`
- Format: Pretty-printed JSON
- Size: ~3.1 KB
- Shows the complete structure of a Signavio telemetry record

## Test Data Summary

The generated test record includes:

- **Agent Run ID:** `test-session-1762775872`
- **Service:** `extract-service`
- **Workflow:** `data-extraction-workflow`
- **Agent Type:** `data-extraction-agent`
- **Status:** `partial`
- **Latency:** 540,000 ms (9 minutes)
- **Tools Used:** 3 tool types
  - `database_query` (5 calls, 5 successful)
  - `data_transformation` (7 calls, 7 successful)
  - `file_writer` (3 calls, 2 successful, 1 error)
- **LLM Calls:** 2 calls to `gpt-4`
  - Total tokens: 1,370 (1,250 input, 890 output)
  - Total latency: 2,400 ms
- **Process Steps:** 3 workflow steps
  - user_prompt
  - model_change
  - tool_call_started

## Upload to Signavio

### Prerequisites
1. Signavio Process Intelligence account
2. Ingestion API access configured
3. API key and tenant ID
4. Dataset created in Signavio

### Upload Command

```bash
curl -X POST https://ingestion-eu.signavio.com/ingestion/api/v1/datasets/YOUR_DATASET/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "X-Tenant-ID: YOUR_TENANT_ID" \
  -F "files=@signavio_telemetry_export.csv" \
  -F "schema=@signavio_telemetry_schema.json" \
  -F "primaryKeys=agent_run_id" \
  -F "delimiter=,"
```

### Alternative: Using Signavio UI

1. Log into Signavio Process Intelligence
2. Navigate to Data Ingestion
3. Select your dataset
4. Upload `signavio_telemetry_export.csv`
5. Upload `signavio_telemetry_schema.json` as the schema
6. Set primary key to `agent_run_id`
7. Set delimiter to comma (`,`)

## Field Descriptions

### Core Fields
- `agent_run_id`: Unique identifier for the agent execution session
- `agent_name`: Name of the agent/service
- `task_id`: Task identifier (same as agent_run_id in this case)
- `task_description`: Human-readable description of the task
- `start_time`: Unix timestamp in milliseconds
- `end_time`: Unix timestamp in milliseconds
- `status`: Execution status (success, error, partial)
- `latency_ms`: Total execution time in milliseconds

### Enhanced Fields
- `service_name`: Service that executed the agent
- `workflow_name`: Workflow identifier
- `workflow_version`: Workflow version (if available)
- `agent_type`: Type of agent framework
- `agent_state`: Current state of the agent

### JSON-Encoded Arrays
- `tools_used`: Array of tool usage statistics
- `llm_calls`: Array of LLM call metrics
- `process_steps`: Array of workflow process steps
- `prompt_metrics`: Prompt-related metrics

## Verification

After upload, verify the data in Signavio:

1. Check dataset record count matches expected number
2. Verify all fields are properly mapped
3. Check that JSON-encoded arrays are accessible
4. Validate timestamps are correct
5. Confirm process steps are visible in process mining views

## Next Steps

1. **Generate More Test Data**: Run the generator multiple times to create more records
2. **Real Trace Export**: Use the telemetry-exporter service to export real traces
3. **Process Mining**: Use Signavio's process mining features to analyze agent execution patterns
4. **Dashboards**: Create Signavio dashboards to visualize agent performance metrics

## Troubleshooting

### Upload Fails
- Verify API key and tenant ID are correct
- Check dataset name matches exactly
- Ensure CSV encoding is UTF-8
- Verify schema matches CSV structure

### Data Not Appearing
- Check dataset permissions
- Verify primary key is set correctly
- Check Signavio logs for ingestion errors
- Ensure timestamps are valid Unix milliseconds

### Schema Mismatch
- Compare uploaded schema with `signavio_telemetry_schema.json`
- Verify all required fields are present
- Check field types match schema definition

