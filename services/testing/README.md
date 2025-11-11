# Dynamic Sample Generator Service

## Overview

The Dynamic Sample Generator provides intelligent, database-driven test data generation for all process inputs and reference tables. It enables end-to-end process testing with telemetry, latency metrics, and data/code quality issue detection.

## Features

### 1. **Dynamic & Intelligent Generation**
- **Knowledge Graph Driven**: Uses knowledge graph to understand table schemas, relationships, and classifications
- **AI-Powered Generation**: Uses LocalAI for intelligent value generation based on column semantics
- **Pattern Learning**: Learns from existing data patterns to generate realistic test data (with AI enhancement)
- **Type-Aware**: Generates appropriate data based on column types, constraints, and patterns
- **Reference Table Support**: Handles reference/lookup tables with distinct values
- **Transaction Table Support**: Generates realistic transaction data
- **Foreign Key Resolution**: Automatically ensures foreign key values exist in referenced tables

### 2. **Database-Driven**
- **Schema Discovery**: Automatically discovers table schemas from knowledge graph
- **Relationship Awareness**: Understands foreign keys and table relationships
- **Foreign Key Resolution**: Automatically generates referenced data and ensures FK integrity
- **Constraint Validation**: Validates generated data against database constraints
- **Dynamic Query Execution**: Executes test scenarios against actual database
- **Performance Optimized**: Batch inserts and connection pooling for efficient data generation

### 3. **End-to-End Process Testing**
- **Process Execution**: Executes Control-M jobs, SQL queries, and workflows
- **Input/Output Validation**: Validates process inputs and outputs
- **Sequence Testing**: Tests table processing sequences from knowledge graph
- **Petri Net Testing**: Generates test scenarios from Petri net workflows
- **Search Integration**: Discover similar test scenarios and patterns using semantic search

### 4. **Telemetry & Metrics**
- **Latency Metrics**: Tracks process execution durations
- **Query Performance**: Measures SQL query latencies
- **Data Volumes**: Tracks row counts and data volumes
- **Resource Usage**: Monitors memory and CPU usage
- **Error Tracking**: Captures errors with context and severity

### 5. **Quality Issue Detection**
- **AI-Powered Quality Rules**: LocalAI generates intelligent quality rules based on table semantics
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
Scenario Builder
    ↓
Test Execution
    ↓
Metrics & Quality Analysis
    ↓
Database Storage
```

## Components

### 1. SampleGenerator
- Loads knowledge graph data
- Generates sample data based on schemas
- Executes test scenarios
- Captures metrics and quality issues

### 2. TestService
- HTTP API for test operations
- Endpoints for sample generation
- Scenario execution
- Execution history

### 3. ScenarioBuilder
- Builds test scenarios from Petri nets
- Builds scenarios from process sequences
- Auto-configures quality rules
- Determines appropriate row counts

## Usage

### Initialize Sample Generator

```go
// Connect to database
db, _ := sql.Open("postgres", "postgres://...")

// Create Extract client
extractClient := NewHTTPExtractClient("http://extract-service:8081")

// Create sample generator
generator := NewSampleGenerator(db, extractClient, logger)

// Load knowledge graph
generator.LoadKnowledgeGraph(ctx, "project_id", "system_id")
```

### Generate Sample Data

```go
config := &TableTestConfig{
    TableName: "orders",
    RowCount:  1000,
    SeedData: map[string][]any{
        "status": []any{"ACTIVE", "PENDING", "COMPLETED"},
    },
}

data, err := generator.GenerateSampleData(ctx, config)
```

### Execute Test Scenario

```go
scenario := &TestScenario{
    ID:   "test_001",
    Name: "End-to-End Order Processing",
    Tables: []*TableTestConfig{
        {
            TableName: "staging_orders",
            RowCount:  1000,
        },
    },
    Processes: []*ProcessTestConfig{
        {
            ProcessID:    "load_orders",
            ProcessType:  "sql",
            InputTables:  []string{"staging_orders"},
            OutputTables: []string{"orders"},
            ExpectedRows: map[string]int{"orders": 1000},
            ValidationSQL: []string{
                "INSERT INTO orders SELECT * FROM staging_orders",
            },
        },
    },
}

execution, err := generator.ExecuteTestScenario(ctx, scenario)
```

### Build Scenario from Petri Net

```go
builder := NewScenarioBuilder(generator, logger)
scenario, err := builder.BuildScenarioFromPetriNet(ctx, "controlm_petri_net")
execution, err := generator.ExecuteTestScenario(ctx, scenario)
```

## API Endpoints

### POST /test/generate-sample
Generate sample data for a table.

**Request:**
```json
{
  "table_name": "orders",
  "row_count": 1000,
  "seed_data": {
    "status": ["ACTIVE", "PENDING"]
  }
}
```

**Response:**
```json
{
  "table_name": "orders",
  "row_count": 1000,
  "data": [...]
}
```

### POST /test/execute-scenario
Execute a complete test scenario.

**Request:**
```json
{
  "id": "test_001",
  "name": "Order Processing Test",
  "tables": [...],
  "processes": [...]
}
```

**Response:**
```json
{
  "id": "test_exec_123",
  "status": "completed",
  "metrics": {
    "total_duration": "2.5s",
    "process_durations": {...},
    "data_volumes": {...},
    "query_latencies": {...}
  },
  "quality_issues": [...],
  "results": {...}
}
```

### POST /test/load-knowledge-graph
Load knowledge graph data.

**Request:**
```json
{
  "project_id": "project_1",
  "system_id": "system_1"
}
```

### GET /test/executions
List test executions.

### GET /test/executions/{id}
Get specific test execution details.

## Metrics Captured

### Execution Metrics
- **Total Duration**: Total test execution time
- **Process Durations**: Duration per process
- **Data Volumes**: Row counts per table
- **Query Latencies**: SQL query execution times
- **Memory Usage**: Memory consumption
- **CPU Usage**: CPU utilization
- **Errors**: Error occurrences with context
- **Warnings**: Warning messages

### Quality Issues
- **Data Quality**: NULL violations, constraint violations
- **Code Quality**: Performance issues, error patterns
- **Relationship Integrity**: Foreign key violations
- **Type**: Issue type (data_quality, code_quality, performance)
- **Severity**: Issue severity (critical, high, medium, low)

## Integration with Knowledge Graph

The generator uses the knowledge graph to:

1. **Discover Table Schemas**: Query table and column definitions
2. **Understand Relationships**: Discover foreign keys and relationships
3. **Classify Tables**: Use table classifications (transaction, reference, staging)
4. **Learn Patterns**: Extract learned patterns from advanced extraction
5. **Process Sequences**: Understand table processing order
6. **Petri Net Workflows**: Build test scenarios from Petri nets

## Example Test Scenario

```json
{
  "id": "scenario_orders_001",
  "name": "Order Processing End-to-End",
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

## Benefits

1. **Automated Testing**: End-to-end process testing without manual data setup
2. **Quality Assurance**: Automatic detection of data and code quality issues
3. **Performance Monitoring**: Latency and performance metrics
4. **Intelligent Generation**: Realistic test data based on knowledge graph
5. **Database-Driven**: Uses actual database schemas and constraints
6. **Dynamic**: Adapts to schema changes automatically

## Configuration

The service can be configured using environment variables:

### Server Settings
- `TEST_SERVICE_PORT`: HTTP server port (default: "8082")

### Database Settings
- `TEST_DB_DSN`: Database connection string (required)

### Service URLs
- `EXTRACT_SERVICE_URL`: Extract service URL (default: "http://localhost:8081")
- `LOCALAI_URL`: LocalAI service URL (default: "http://localhost:8080")
- `SEARCH_SERVICE_URL`: Search service URL (defaults to Extract service URL if not set)

### LocalAI Settings
- `LOCALAI_ENABLED`: Enable LocalAI integration (default: true)
- `LOCALAI_MODEL`: Model to use (default: "phi-3.5-mini")
- `LOCALAI_TIMEOUT`: Request timeout (default: "60s")
- `LOCALAI_RETRY_ATTEMPTS`: Number of retry attempts (default: 3)

### Default Row Counts
- `DEFAULT_REFERENCE_ROW_COUNT`: Default rows for reference tables (default: 50)
- `DEFAULT_TRANSACTION_ROW_COUNT`: Default rows for transaction tables (default: 1000)
- `DEFAULT_STAGING_ROW_COUNT`: Default rows for staging tables (default: 500)

### Performance Settings
- `BATCH_INSERT_SIZE`: Batch size for inserts (default: 100)
- `DB_CONNECTION_POOL_SIZE`: Max open connections (default: 10)
- `DB_CONNECTION_MAX_IDLE`: Max idle connections (default: 5)
- `DB_CONNECTION_MAX_LIFETIME`: Connection max lifetime (default: "5m")

### Feature Flags
- `ENABLE_LOCALAI`: Enable LocalAI features (default: true)
- `ENABLE_SEARCH`: Enable search features (default: true)
- `ENABLE_FK_RESOLUTION`: Enable foreign key resolution (default: true)
- `ENABLE_BATCH_INSERTS`: Enable batch inserts (default: true)

## Building and Running

```bash
# Build the service
make build

# Run the service
make run

# Run tests
make test
```

The service expects:
- PostgreSQL database (connection via `TEST_DB_DSN`)
- Extract service running (for knowledge graph queries)
- LocalAI service (optional, for AI-powered features)
- Search service (optional, defaults to Extract service)

## Future Enhancements

1. **Enhanced Pattern Learning**: Improved AI-powered pattern recognition
2. **Smart Seeding**: Use seed data from production (anonymized)
3. **Test Data Management**: Version control for test scenarios
4. **Visualization**: Dashboard for test execution results
5. **Regression Testing**: Compare execution results over time
6. **Parallel Data Generation**: Generate data for multiple tables concurrently

