#!/usr/bin/env python3
"""
Convert SGMI Control-M JSON to table row format expected by extract service.

The extract service expects JSON tables as an array of row objects:
[
  {"column1": "value1", "column2": "value2", ...},
  {"column1": "value3", "column2": "value4", ...},
  ...
]

This script converts the nested Control-M structure to this format by extracting
meaningful metadata from job definitions.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def flatten_controlm_job(job_name: str, job_data: Dict[str, Any], parent_path: str = "") -> Dict[str, Any]:
    """Flatten a Control-M job definition into a row object."""
    row = {
        "job_name": job_name,
        "job_type": job_data.get("Type", ""),
        "application": job_data.get("Application", ""),
        "sub_application": job_data.get("SubApplication", ""),
        "description": job_data.get("Description", ""),
        "host": job_data.get("Host", ""),
        "run_as": job_data.get("RunAs", ""),
        "created_by": job_data.get("CreatedBy", ""),
        "priority": job_data.get("Priority", ""),
        "days_keep_active": job_data.get("DaysKeepActive", ""),
        "parent_path": parent_path,
    }
    
    # Extract command if present
    if "Command" in job_data:
        row["command"] = job_data["Command"]
    
    # Extract file name
    if "FileName" in job_data:
        row["file_name"] = job_data["FileName"]
    
    # Extract schedule information
    if "When" in job_data:
        when = job_data["When"]
        row["schedule_from_time"] = when.get("FromTime", "")
        row["schedule_to_time"] = when.get("ToTime", "")
        row["schedule_month_days"] = ",".join(str(d) for d in when.get("MonthDays", []))
        row["schedule_week_days"] = ",".join(when.get("WeekDays", []))
    
    # Extract variables
    if "Variables" in job_data:
        variables = job_data["Variables"]
        if isinstance(variables, list) and len(variables) > 0:
            # Flatten first variable set (most common case)
            if isinstance(variables[0], dict):
                for key, value in variables[0].items():
                    row[f"var_{key}"] = str(value)
    
    # Extract events
    if "eventsToAdd" in job_data:
        events = job_data["eventsToAdd"]
        if isinstance(events, dict) and "Events" in events:
            event_list = events["Events"]
            if isinstance(event_list, list):
                row["events"] = ",".join(str(e.get("Event", "")) for e in event_list if isinstance(e, dict))
    
    # Extract notification
    if "Notify:NotOK_1" in job_data:
        notify = job_data["Notify:NotOK_1"]
        if isinstance(notify, dict):
            row["notify_message"] = notify.get("Message", "")
    
    return row


def extract_jobs_recursive(data: Dict[str, Any], parent_path: str = "", rows: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Recursively extract all jobs from nested Control-M structure."""
    if rows is None:
        rows = []
    
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        
        current_path = f"{parent_path}/{key}" if parent_path else key
        
        # Check if this is a job definition
        job_type = value.get("Type", "")
        if job_type and ("Job:" in job_type or job_type == "Job:Dummy" or job_type == "Job:Command"):
            # This is a job, extract it
            row = flatten_controlm_job(key, value, parent_path)
            rows.append(row)
        elif job_type == "SimpleFolder":
            # This is a folder, recurse into it
            extract_jobs_recursive(value, current_path, rows)
        else:
            # Unknown type, try to recurse anyway in case it contains jobs
            extract_jobs_recursive(value, current_path, rows)
    
    return rows


def convert_sgmi_json(input_path: str, output_path: str) -> int:
    """Convert SGMI Control-M JSON to table format."""
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    
    # Read input JSON
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_path}: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Failed to read {input_path}: {e}", file=sys.stderr)
        return 1
    
    # Extract jobs from nested structure
    if not isinstance(data, dict):
        print(f"Error: Expected JSON object (dict), got {type(data).__name__}", file=sys.stderr)
        return 1
    
    rows = extract_jobs_recursive(data)
    
    if not rows:
        print(f"Warning: No jobs found in {input_path}", file=sys.stderr)
        # Create at least one row with metadata to avoid empty array
        rows = [{
            "source_file": str(input_file.name),
            "total_keys": len(data),
            "note": "No job definitions found, extracted top-level metadata only"
        }]
    
    # Write output JSON
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"Converted {len(rows)} rows from {input_path} to {output_path}")
        return 0
    except Exception as e:
        print(f"Error: Failed to write {output_path}: {e}", file=sys.stderr)
        return 1


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: convert_sgmi_json_to_table_format.py <input_json> <output_json>", file=sys.stderr)
        print("", file=sys.stderr)
        print("Converts nested Control-M JSON structure to array of row objects", file=sys.stderr)
        print("expected by the extract service.", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    exit_code = convert_sgmi_json(input_path, output_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

