#!/bin/bash
# Script to generate Signavio-compatible extract files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create a simple Python script to generate the extract
python3 << 'PYTHON_EOF'
import json
import csv
from datetime import datetime, timedelta
import time

# Create test telemetry data
now = datetime.now()
session_id = f"test-session-{int(time.time())}"

# Create a realistic telemetry record
record = {
    "agent_run_id": session_id,
    "agent_name": "extract-service",
    "task_id": session_id,
    "task_description": f"Agent execution session: {session_id}",
    "start_time": int((now - timedelta(minutes=10)).timestamp() * 1000),
    "end_time": int(now.timestamp() * 1000),
    "status": "partial",
    "outcome_summary": "Agent execution completed; 15 tool calls; 2 model changes; success rate: 93.33% | Service: extract-service | Workflow: data-extraction-workflow | Agent Type: data-extraction-agent | Prompts: 73 chars",
    "latency_ms": 540000,
    "notes": "Total events: 10 | Average tool latency: 245.50 ms | Tool calls: 15 (success: 14, errors: 1) | Model changes: 2 | Tools: 3 types | Models: 1 | Prompt latency: 600000 ms",
    "service_name": "extract-service",
    "workflow_name": "data-extraction-workflow",
    "workflow_version": "",
    "agent_type": "data-extraction-agent",
    "agent_state": "",
    "prompt_metrics": json.dumps({
        "prompt_length": 73,
        "response_length": 0,
        "prompt_type": "task",
        "prompt_category": "data_processing",
        "input_tokens": 450,
        "output_tokens": 0,
        "prompt_latency_ms": 600000
    }),
    "tools_used": json.dumps([
        {
            "tool_name": "database_query",
            "call_count": 5,
            "success_count": 5,
            "total_latency_ms": 902,
            "error_details": "",
            "parameters": {"table": "customers", "limit": 1000},
            "category": "data-access",
            "service_name": "postgres-service"
        },
        {
            "tool_name": "data_transformation",
            "call_count": 7,
            "success_count": 7,
            "total_latency_ms": 2240,
            "error_details": "",
            "parameters": {"operation": "aggregate", "group_by": "region"},
            "category": "data-processing",
            "service_name": "transform-service"
        },
        {
            "tool_name": "file_writer",
            "call_count": 3,
            "success_count": 2,
            "total_latency_ms": 450,
            "error_details": "Permission denied",
            "parameters": {},
            "category": "io",
            "service_name": "storage-service"
        }
    ]),
    "llm_calls": json.dumps([
        {
            "model": "gpt-4",
            "call_count": 2,
            "total_tokens": 1370,
            "input_tokens": 1250,
            "output_tokens": 890,
            "total_latency_ms": 2400,
            "purpose": "report_generation",
            "temperature": 0.7,
            "cost": 0.0,
            "context_length": 8192,
            "service_name": "openai-service"
        }
    ]),
    "process_steps": json.dumps([
        {
            "step_name": "user_prompt",
            "start_time": int((now - timedelta(minutes=10)).timestamp() * 1000),
            "end_time": int((now - timedelta(minutes=9)).timestamp() * 1000),
            "status": "completed",
            "duration_ms": 60000,
            "workflow_name": "data-extraction-workflow",
            "workflow_version": "",
            "dependencies": [],
            "parallel_execution": False
        },
        {
            "step_name": "model_change",
            "start_time": int((now - timedelta(minutes=9)).timestamp() * 1000),
            "end_time": int((now - timedelta(minutes=8)).timestamp() * 1000),
            "status": "completed",
            "duration_ms": 60000,
            "workflow_name": "data-extraction-workflow",
            "workflow_version": "",
            "dependencies": ["user_prompt"],
            "parallel_execution": False
        },
        {
            "step_name": "tool_call_started",
            "start_time": int((now - timedelta(minutes=8)).timestamp() * 1000),
            "end_time": int((now - timedelta(minutes=7)).timestamp() * 1000),
            "status": "completed",
            "duration_ms": 60000,
            "workflow_name": "data-extraction-workflow",
            "workflow_version": "",
            "dependencies": ["model_change"],
            "parallel_execution": False
        }
    ])
}

# Write CSV file
csv_file = "signavio_telemetry_export.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header
    writer.writerow([
        "agent_run_id", "agent_name", "task_id", "task_description",
        "start_time", "end_time", "status", "outcome_summary", "latency_ms", "notes",
        "service_name", "workflow_name", "workflow_version", "agent_type", "agent_state",
        "prompt_metrics", "tools_used", "llm_calls", "process_steps"
    ])
    # Write record
    writer.writerow([
        record["agent_run_id"],
        record["agent_name"],
        record["task_id"],
        record["task_description"],
        record["start_time"],
        record["end_time"],
        record["status"],
        record["outcome_summary"],
        record["latency_ms"],
        record["notes"],
        record["service_name"],
        record["workflow_name"],
        record["workflow_version"],
        record["agent_type"],
        record["agent_state"],
        record["prompt_metrics"],
        record["tools_used"],
        record["llm_calls"],
        record["process_steps"]
    ])

print(f"✓ Generated CSV file: {csv_file}")

# Write Avro schema
schema = {
    "type": "record",
    "name": "AgentTelemetry",
    "namespace": "com.signavio.telemetry",
    "fields": [
        {"name": "agent_run_id", "type": "string"},
        {"name": "agent_name", "type": "string"},
        {"name": "task_id", "type": "string"},
        {"name": "task_description", "type": "string"},
        {"name": "start_time", "type": "long"},
        {"name": "end_time", "type": "long"},
        {"name": "status", "type": "string"},
        {"name": "outcome_summary", "type": ["null", "string"], "default": None},
        {"name": "latency_ms", "type": ["null", "long"], "default": None},
        {"name": "notes", "type": ["null", "string"], "default": None},
        {"name": "service_name", "type": ["null", "string"], "default": None},
        {"name": "workflow_name", "type": ["null", "string"], "default": None},
        {"name": "workflow_version", "type": ["null", "string"], "default": None},
        {"name": "agent_type", "type": ["null", "string"], "default": None},
        {"name": "agent_state", "type": ["null", "string"], "default": None},
        {"name": "prompt_metrics", "type": ["null", "string"], "default": None},
        {"name": "tools_used", "type": ["null", "string"], "default": None},
        {"name": "llm_calls", "type": ["null", "string"], "default": None},
        {"name": "process_steps", "type": ["null", "string"], "default": None}
    ]
}

schema_file = "signavio_telemetry_schema.json"
with open(schema_file, 'w') as f:
    json.dump(schema, f, indent=2)

print(f"✓ Generated Avro schema: {schema_file}")

# Write JSON record for reference
json_file = "signavio_telemetry_record.json"
with open(json_file, 'w') as f:
    json.dump(record, f, indent=2)

print(f"✓ Generated JSON record: {json_file}")

print("\n=== Summary ===")
print(f"Session ID: {record['agent_run_id']}")
print(f"Agent Name: {record['agent_name']}")
print(f"Service: {record['service_name']}")
print(f"Workflow: {record['workflow_name']}")
print(f"Status: {record['status']}")
print(f"\nFiles ready for Signavio upload:")
print(f"  1. CSV file: {csv_file}")
print(f"  2. Avro schema: {schema_file}")
print(f"  3. Primary key: agent_run_id")
print(f"  4. Delimiter: comma (,)")

PYTHON_EOF

chmod +x create_signavio_extract.sh
./create_signavio_extract.sh

