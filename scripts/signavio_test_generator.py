#!/usr/bin/env python3
"""
Signavio API Test File Generator

This script connects to Signavio API endpoints and generates test files
in various formats (CSV, JSON, Avro) for integration testing.

Usage:
    python signavio_test_generator.py --config config.json --output-dir ./test_data

Configuration file should contain:
{
    "base_url": "https://ingestion-eu.signavio.com",
    "api_key": "your-api-key",
    "tenant_id": "your-tenant-id",
    "endpoints": ["process-instances", "process-definitions", "activities"]
}
"""

import argparse
import json
import csv
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import requests
from pathlib import Path


class SignavioTestGenerator:
    """Generate test files from Signavio API"""
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config
        self.base_url = config.get("base_url", "https://ingestion-eu.signavio.com")
        self.api_key = config.get("api_key")
        self.tenant_id = config.get("tenant_id")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        })
        if self.tenant_id:
            self.session.headers["X-Tenant-ID"] = self.tenant_id
    
    def fetch_process_instances(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch process instances from Signavio API"""
        print(f"Fetching process instances (limit: {limit})...")
        
        # Signavio Process Intelligence API endpoint
        url = f"{self.base_url}/api/v1/process-instances"
        params = {
            "$top": limit,
            "$orderby": "startTime desc"
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            instances = data.get("value", [])
            print(f"✓ Fetched {len(instances)} process instances")
            return instances
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching process instances: {e}")
            return self._generate_mock_process_instances(limit)
    
    def fetch_process_definitions(self) -> List[Dict[str, Any]]:
        """Fetch process definitions from Signavio API"""
        print("Fetching process definitions...")
        
        url = f"{self.base_url}/api/v1/process-definitions"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            definitions = data.get("value", [])
            print(f"✓ Fetched {len(definitions)} process definitions")
            return definitions
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching process definitions: {e}")
            return self._generate_mock_process_definitions()
    
    def fetch_activities(self, process_instance_id: str) -> List[Dict[str, Any]]:
        """Fetch activities for a specific process instance"""
        print(f"Fetching activities for instance {process_instance_id}...")
        
        url = f"{self.base_url}/api/v1/process-instances/{process_instance_id}/activities"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            activities = data.get("value", [])
            print(f"✓ Fetched {len(activities)} activities")
            return activities
        except requests.exceptions.RequestException as e:
            print(f"✗ Error fetching activities: {e}")
            return self._generate_mock_activities(process_instance_id)
    
    def _generate_mock_process_instances(self, count: int) -> List[Dict[str, Any]]:
        """Generate mock process instances for testing"""
        print(f"Generating {count} mock process instances...")
        instances = []
        base_time = datetime.now()
        
        statuses = ["completed", "running", "failed", "suspended"]
        process_names = [
            "Order-to-Cash", "Procure-to-Pay", "Hire-to-Retire",
            "Lead-to-Opportunity", "Issue-to-Resolution"
        ]
        
        for i in range(count):
            start_time = base_time - timedelta(days=i, hours=i % 24)
            duration = timedelta(hours=2, minutes=i % 60)
            end_time = start_time + duration
            
            instance = {
                "id": f"PI-{1000 + i}",
                "processDefinitionId": f"PD-{i % 5 + 1}",
                "processDefinitionName": process_names[i % len(process_names)],
                "startTime": int(start_time.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000) if i % 4 != 1 else None,
                "status": statuses[i % len(statuses)],
                "durationMs": int(duration.total_seconds() * 1000),
                "initiator": f"user{i % 10 + 1}@example.com",
                "businessKey": f"ORDER-{2000 + i}",
                "variables": {
                    "customer_id": f"CUST-{i % 50 + 1}",
                    "order_value": round(100 + (i * 17.3) % 1000, 2),
                    "region": ["EMEA", "APAC", "AMER"][i % 3],
                    "priority": ["high", "medium", "low"][i % 3]
                }
            }
            instances.append(instance)
        
        return instances
    
    def _generate_mock_process_definitions(self) -> List[Dict[str, Any]]:
        """Generate mock process definitions"""
        print("Generating mock process definitions...")
        
        definitions = [
            {
                "id": "PD-1",
                "name": "Order-to-Cash",
                "version": "v2.1",
                "category": "Finance",
                "description": "End-to-end order processing and payment collection",
                "createdDate": int(datetime(2024, 1, 15).timestamp() * 1000),
                "modifiedDate": int(datetime(2024, 10, 1).timestamp() * 1000),
                "instanceCount": 1245,
                "avgDurationMs": 7200000
            },
            {
                "id": "PD-2",
                "name": "Procure-to-Pay",
                "version": "v1.8",
                "category": "Procurement",
                "description": "Purchase requisition to payment workflow",
                "createdDate": int(datetime(2024, 2, 1).timestamp() * 1000),
                "modifiedDate": int(datetime(2024, 9, 15).timestamp() * 1000),
                "instanceCount": 987,
                "avgDurationMs": 10800000
            },
            {
                "id": "PD-3",
                "name": "Hire-to-Retire",
                "version": "v3.0",
                "category": "HR",
                "description": "Employee lifecycle management process",
                "createdDate": int(datetime(2023, 11, 1).timestamp() * 1000),
                "modifiedDate": int(datetime(2024, 8, 20).timestamp() * 1000),
                "instanceCount": 543,
                "avgDurationMs": 172800000
            }
        ]
        
        return definitions
    
    def _generate_mock_activities(self, instance_id: str) -> List[Dict[str, Any]]:
        """Generate mock activities for a process instance"""
        
        activities = [
            {
                "id": f"{instance_id}-ACT-1",
                "processInstanceId": instance_id,
                "activityName": "Receive Order",
                "activityType": "userTask",
                "startTime": int((datetime.now() - timedelta(hours=3)).timestamp() * 1000),
                "endTime": int((datetime.now() - timedelta(hours=2, minutes=50)).timestamp() * 1000),
                "durationMs": 600000,
                "assignee": "sales.team@example.com",
                "status": "completed"
            },
            {
                "id": f"{instance_id}-ACT-2",
                "processInstanceId": instance_id,
                "activityName": "Validate Credit",
                "activityType": "serviceTask",
                "startTime": int((datetime.now() - timedelta(hours=2, minutes=50)).timestamp() * 1000),
                "endTime": int((datetime.now() - timedelta(hours=2, minutes=45)).timestamp() * 1000),
                "durationMs": 300000,
                "status": "completed"
            },
            {
                "id": f"{instance_id}-ACT-3",
                "processInstanceId": instance_id,
                "activityName": "Approve Order",
                "activityType": "userTask",
                "startTime": int((datetime.now() - timedelta(hours=2, minutes=45)).timestamp() * 1000),
                "endTime": int((datetime.now() - timedelta(hours=1)).timestamp() * 1000),
                "durationMs": 6300000,
                "assignee": "manager@example.com",
                "status": "completed"
            }
        ]
        
        return activities
    
    def save_as_json(self, data: Any, filename: str) -> str:
        """Save data as JSON file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved JSON: {filepath}")
        return str(filepath)
    
    def save_as_csv(self, data: List[Dict[str, Any]], filename: str) -> str:
        """Save data as CSV file"""
        if not data:
            print(f"⚠ No data to save for {filename}")
            return ""
        
        filepath = self.output_dir / filename
        
        # Flatten nested structures
        flattened_data = []
        for record in data:
            flat_record = self._flatten_dict(record)
            flattened_data.append(flat_record)
        
        # Get all unique keys
        fieldnames = set()
        for record in flattened_data:
            fieldnames.update(record.keys())
        fieldnames = sorted(fieldnames)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_data)
        
        print(f"✓ Saved CSV: {filepath}")
        return str(filepath)
    
    def save_avro_schema(self, data: List[Dict[str, Any]], schema_filename: str) -> str:
        """Generate and save Avro schema based on data structure"""
        if not data:
            print(f"⚠ No data to generate schema for {schema_filename}")
            return ""
        
        filepath = self.output_dir / schema_filename
        
        # Infer schema from first record
        sample = self._flatten_dict(data[0])
        fields = []
        
        for key, value in sample.items():
            field_type = self._infer_avro_type(value)
            fields.append({
                "name": key,
                "type": ["null", field_type],
                "default": None
            })
        
        schema = {
            "type": "record",
            "name": schema_filename.replace(".avsc", "").replace("-", "_").title(),
            "namespace": "com.signavio.test",
            "fields": fields
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2)
        
        print(f"✓ Saved Avro schema: {filepath}")
        return str(filepath)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to JSON strings for CSV compatibility
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _infer_avro_type(self, value: Any) -> str:
        """Infer Avro type from Python value"""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "long"
        elif isinstance(value, float):
            return "double"
        else:
            return "string"
    
    def generate_test_suite(self, options: Dict[str, Any]):
        """Generate complete test file suite"""
        print("=" * 60)
        print("Signavio Test File Generator")
        print("=" * 60)
        print()
        
        generated_files = []
        
        # Generate process instances
        if options.get("generate_instances", True):
            print("📊 Generating Process Instances...")
            instances = self.fetch_process_instances(limit=options.get("instance_limit", 50))
            
            if instances:
                # Save in multiple formats
                generated_files.append(self.save_as_json(instances, "process_instances.json"))
                generated_files.append(self.save_as_csv(instances, "process_instances.csv"))
                generated_files.append(self.save_avro_schema(instances, "process_instances.avsc"))
            print()
        
        # Generate process definitions
        if options.get("generate_definitions", True):
            print("📋 Generating Process Definitions...")
            definitions = self.fetch_process_definitions()
            
            if definitions:
                generated_files.append(self.save_as_json(definitions, "process_definitions.json"))
                generated_files.append(self.save_as_csv(definitions, "process_definitions.csv"))
                generated_files.append(self.save_avro_schema(definitions, "process_definitions.avsc"))
            print()
        
        # Generate activities
        if options.get("generate_activities", True):
            print("🔄 Generating Activities...")
            # Use first instance from mock data
            instances = self._generate_mock_process_instances(1)
            if instances:
                activities = self.fetch_activities(instances[0]["id"])
                
                if activities:
                    generated_files.append(self.save_as_json(activities, "activities.json"))
                    generated_files.append(self.save_as_csv(activities, "activities.csv"))
                    generated_files.append(self.save_avro_schema(activities, "activities.avsc"))
            print()
        
        # Generate agent telemetry (compatible with existing format)
        if options.get("generate_telemetry", True):
            print("📡 Generating Agent Telemetry...")
            telemetry = self._generate_agent_telemetry(count=options.get("telemetry_count", 20))
            
            if telemetry:
                generated_files.append(self.save_as_json(telemetry, "agent_telemetry.json"))
                generated_files.append(self.save_as_csv(telemetry, "agent_telemetry.csv"))
                generated_files.append(self.save_avro_schema(telemetry, "agent_telemetry.avsc"))
            print()
        
        # Summary
        print("=" * 60)
        print(f"✓ Generated {len(generated_files)} test files")
        print(f"✓ Output directory: {self.output_dir.absolute()}")
        print("=" * 60)
        
        return generated_files
    
    def _generate_agent_telemetry(self, count: int) -> List[Dict[str, Any]]:
        """Generate agent telemetry records compatible with existing schema"""
        print(f"Generating {count} agent telemetry records...")
        
        records = []
        base_time = datetime.now()
        
        agent_names = [
            "data-extraction-agent", "compliance-reasoning-agent",
            "bcbs-audit-agent", "regulatory-calculation-agent"
        ]
        
        statuses = ["success", "failed", "partial", "timeout"]
        
        for i in range(count):
            start_time = base_time - timedelta(minutes=i * 15)
            duration = timedelta(minutes=2 + (i % 10))
            end_time = start_time + duration
            
            record = {
                "agent_run_id": f"run-{1000 + i}",
                "agent_name": agent_names[i % len(agent_names)],
                "task_id": f"task-{2000 + i}",
                "task_description": f"Process regulatory compliance check #{i}",
                "start_time": int(start_time.timestamp() * 1000),
                "end_time": int(end_time.timestamp() * 1000),
                "status": statuses[i % len(statuses)],
                "outcome_summary": f"Completed with {i % 5} warnings",
                "latency_ms": int(duration.total_seconds() * 1000),
                "notes": f"Test execution #{i}",
                "service_name": "regulatory-service",
                "workflow_name": "bcbs239-audit",
                "workflow_version": "v2.1",
                "agent_type": "compliance",
                "agent_state": "active",
                "tools_used": json.dumps([
                    {
                        "tool_name": "database_query",
                        "call_count": 3 + (i % 5),
                        "success_count": 3 + (i % 5),
                        "total_latency_ms": 450 + (i * 10)
                    }
                ]),
                "llm_calls": json.dumps([
                    {
                        "model": "gpt-4",
                        "call_count": 2,
                        "total_tokens": 1200 + (i * 50),
                        "input_tokens": 800,
                        "output_tokens": 400 + (i * 50),
                        "total_latency_ms": 1800 + (i * 100)
                    }
                ]),
                "process_steps": json.dumps([
                    {
                        "step_name": "Initialize",
                        "duration_ms": 200,
                        "status": "completed"
                    },
                    {
                        "step_name": "Execute",
                        "duration_ms": int(duration.total_seconds() * 1000) - 400,
                        "status": "completed"
                    },
                    {
                        "step_name": "Finalize",
                        "duration_ms": 200,
                        "status": "completed"
                    }
                ])
            }
            records.append(record)
        
        return records


def main():
    parser = argparse.ArgumentParser(
        description="Generate test files from Signavio API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all test files with default settings
  python signavio_test_generator.py --config config.json --output-dir ./test_data

  # Generate only process instances (50 records)
  python signavio_test_generator.py --instances-only --limit 50

  # Generate mock data without API connection
  python signavio_test_generator.py --mock-mode --output-dir ./mock_data

  # Generate agent telemetry only
  python signavio_test_generator.py --telemetry-only --telemetry-count 100
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file with API credentials"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./signavio_test_data",
        help="Output directory for generated files (default: ./signavio_test_data)"
    )
    parser.add_argument(
        "--instances-only",
        action="store_true",
        help="Generate only process instances"
    )
    parser.add_argument(
        "--definitions-only",
        action="store_true",
        help="Generate only process definitions"
    )
    parser.add_argument(
        "--activities-only",
        action="store_true",
        help="Generate only activities"
    )
    parser.add_argument(
        "--telemetry-only",
        action="store_true",
        help="Generate only agent telemetry"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of process instances to fetch (default: 50)"
    )
    parser.add_argument(
        "--telemetry-count",
        type=int,
        default=20,
        help="Number of telemetry records to generate (default: 20)"
    )
    parser.add_argument(
        "--mock-mode",
        action="store_true",
        help="Generate mock data without API connection"
    )
    
    args = parser.parse_args()
    
    # Load config or use defaults
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Use mock configuration
        config = {
            "base_url": "https://ingestion-eu.signavio.com",
            "api_key": "mock-api-key",
            "tenant_id": "mock-tenant"
        }
        if not args.mock_mode and args.config:
            print(f"⚠ Config file not found: {args.config}")
            print("⚠ Running in mock mode")
    
    # Create generator
    generator = SignavioTestGenerator(config, args.output_dir)
    
    # Determine what to generate
    options = {
        "generate_instances": not (args.definitions_only or args.activities_only or args.telemetry_only) or args.instances_only,
        "generate_definitions": not (args.instances_only or args.activities_only or args.telemetry_only) or args.definitions_only,
        "generate_activities": not (args.instances_only or args.definitions_only or args.telemetry_only) or args.activities_only,
        "generate_telemetry": not (args.instances_only or args.definitions_only or args.activities_only) or args.telemetry_only,
        "instance_limit": args.limit,
        "telemetry_count": args.telemetry_count
    }
    
    # Generate test suite
    generated_files = generator.generate_test_suite(options)
    
    print("\n✅ Test file generation complete!")
    print(f"📁 Files saved to: {Path(args.output_dir).absolute()}")


if __name__ == "__main__":
    main()
