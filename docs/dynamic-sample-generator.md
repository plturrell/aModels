# Dynamic Sample Generator Service

## Overview

The Dynamic Sample Generator is an intelligent, database-driven test data generation system that enables comprehensive end-to-end process testing with telemetry, latency metrics, and quality issue detection.

## Key Features

### 1. **Dynamic & Intelligent Generation**
- **Knowledge Graph Driven**: Automatically discovers table schemas from knowledge graph
- **Pattern Learning**: Learns from existing data to generate realistic test data
- **Type-Aware**: Generates appropriate data based on column types and constraints
- **Reference Table Support**: Handles lookup/reference tables with distinct values
- **Transaction Table Support**: Generates realistic transaction data

### 2. **Database-Driven**
- **Schema Discovery**: Queries knowledge graph for table schemas
- **Relationship Awareness**: Understands foreign keys and table relationships
- **Constraint Validation**: Validates generated data against database constraints
- **Dynamic Execution**: Executes test scenarios against actual database

### 3. **End-to-End Process Testing**
- **Process Execution**: Executes Control-M jobs, SQL queries, and workflows
- **Input/Output Validation**: Validates process inputs and outputs
- **Sequence Testing**: Tests table processing sequences from knowledge graph
- **Petri Net Testing**: Generates test scenarios from Petri net workflows

### 4. **Telemetry & Metrics**
- **Latency Metrics**: Tracks process execution durations
- **Query Performance**: Measures SQL query latencies
- **Data Volumes**: Tracks row counts and data volumes
- **Resource Usage**: Monitors memory and CPU usage
- **Error Tracking**: Captures errors with context and severity

### 5. **Quality Issue Detection**
- **Data Quality**: Detects NULL values in non-nullable columns
- **Constraint Violations**: Validates foreign keys and constraints
- **Code Quality**: Detects potential code issues
- **Performance Issues**: Identifies slow queries and processes
- **Relationship Integrity**: Validates table relationships

## Architecture

```
Knowledge Graph (Extract Service)
    ↓
Sample Generator
    ↓
Intelligent Generator (Pattern Learning)
    ↓
Scenario Builder
    ↓
Test Execution
    ↓
Metrics & Quality Analysis
    ↓
Database Storage (test_executions table)
```

## Components

### 1. SampleGenerator
- Loads knowledge graph data
- Generates sample data based on schemas
- Executes test scenarios
- Captures metrics and quality issues

### 2. IntelligentGenerator
- Learns patterns from existing database data
- Generates intelligent values based on learned patterns
- Uses value distributions and enum detection
- Adapts to column naming patterns

### 3. ScenarioBuilder
- Builds test scenarios from Petri nets
- Builds scenarios from process sequences
- Auto-configures quality rules
- Determines appropriate row counts

### 4. TestService
- HTTP API for test operations
- Endpoints for sample generation
- Scenario execution
- Execution history

## Usage

### Initialize Service

```bash
# Set environment variables
export TEST_DB_DSN="postgres://user:pass@localhost/testdb"
export EXTRACT_SERVICE_URL="http://extract-service:8081"

# Run service
./bin/testing-service -port 8082
```

### Load Knowledge Graph

```bash
POST http://localhost:8082/test/load-knowledge-graph
{
  "project_id": "project_1",
  "system_id": "system_1"
}
```

### Generate Sample Data

```bash
POST http://localhost:8082/test/generate-sample
{
  "table_name": "orders",
  "row_count": 1000,
  "seed_data": {
    "status": ["ACTIVE", "PENDING", "COMPLETED"]
  }
}
```

### Execute Test Scenario

```bash
POST http://localhost:8082/test/execute-scenario
{
  "id": "test_001",
  "name": "Order Processing Test",
  "tables": [
    {
      "table_name": "staging_orders",
      "row_count": 1000,
      "quality_rules": [
        {
          "name": "non_null_order_id",
          "type": "constraint",
          "rule": "order_id IS NOT NULL",
          "severity": "error"
        }
      ]
    }
  ],
  "processes": [
    {
      "process_id": "load_orders",
      "process_type": "sql",
      "input_tables": ["staging_orders"],
      "output_tables": ["orders"],
      "expected_rows": {"orders": 1000},
      "validation_sql": [
        "INSERT INTO orders SELECT * FROM staging_orders WHERE status = 'ACTIVE'"
      ]
    }
  ]
}
```

## Metrics Captured

### Execution Metrics
- **Total Duration**: Total test execution time
- **Process Durations**: Duration per process (map: process_id -> duration)
- **Data Volumes**: Row counts per table (map: table_name -> row_count)
- **Query Latencies**: SQL query execution times (map: query -> latency)
- **Memory Usage**: Memory consumption in bytes
- **CPU Usage**: CPU utilization percentage
- **Errors**: Error occurrences with context, timestamp, severity
- **Warnings**: Warning messages

### Quality Issues
- **Data Quality**: NULL violations, constraint violations
- **Code Quality**: Performance issues, error patterns
- **Relationship Integrity**: Foreign key violations
- **Type**: Issue type (data_quality, code_quality, performance)
- **Severity**: Issue severity (critical, high, medium, low)
- **Table/Column**: Specific table and column affected
- **Details**: Additional context and metadata

## Integration Points

### 1. Knowledge Graph Integration
- Queries Extract service for table schemas
- Uses table classifications (transaction, reference, staging)
- Leverages process sequences (PROCESSES_BEFORE edges)
- Uses Petri net workflows for scenario building

### 2. Database Integration
- Connects to Postgres/HANA/other databases
- Inserts generated test data
- Executes SQL queries for validation
- Stores execution results and metrics

### 3. Pattern Learning
- Learns from existing database data
- Detects enum-like columns (low cardinality)
- Identifies common value patterns
- Adapts generation based on learned patterns

## Example Test Execution Flow

```
1. Load Knowledge Graph
   ↓
2. Discover Table Schemas
   ↓
3. Generate Sample Data for Input Tables
   - Reference tables: 50 rows
   - Transaction tables: 1000 rows
   - Staging tables: 500 rows
   ↓
4. Insert Data into Database
   ↓
5. Execute Processes
   - SQL queries
   - Control-M jobs
   - Workflows
   ↓
6. Validate Outputs
   - Check expected row counts
   - Validate data quality
   - Check constraints
   ↓
7. Capture Metrics
   - Process durations
   - Query latencies
   - Data volumes
   ↓
8. Detect Quality Issues
   - NULL violations
   - Constraint violations
   - Performance issues
   ↓
9. Store Execution Results
```

## Benefits

1. **Automated Testing**: End-to-end process testing without manual data setup
2. **Quality Assurance**: Automatic detection of data and code quality issues
3. **Performance Monitoring**: Latency and performance metrics for all processes
4. **Intelligent Generation**: Realistic test data based on knowledge graph and learned patterns
5. **Database-Driven**: Uses actual database schemas and constraints
6. **Dynamic**: Adapts to schema changes automatically
7. **Comprehensive**: Tests entire process flows from input to output

## Future Enhancements

1. **Enhanced Pattern Learning**: Learn from historical test executions
2. **Smart Seeding**: Use anonymized production data as seeds
3. **Test Data Versioning**: Version control for test scenarios
4. **Visualization Dashboard**: Real-time test execution monitoring
5. **Regression Testing**: Compare execution results over time
6. **Performance Baselines**: Establish performance baselines and detect regressions

