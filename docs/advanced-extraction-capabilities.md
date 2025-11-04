# Advanced Extraction Capabilities

## Overview

The Extract service now includes advanced extraction capabilities that extract deep insights from code during the parsing process:

1. **Table Process Sequences** - Extract processing order/sequence of tables
2. **Code Parameters** - Extract parameters from SQL, Control-M, and code
3. **Hardcoded Lists** - Extract hardcoded lists, constants, and enum values
4. **Table Classifications** - Classify tables as transaction vs reference vs staging
5. **Testing Endpoints** - Identify testing/test endpoints

## 1. Table Process Sequences

### What It Extracts

Extracts the **processing order/sequence** of tables from:
- SQL queries (INSERT INTO ... SELECT FROM, UPDATE ... FROM, SELECT ... JOIN)
- Control-M job dependencies
- DDL statements with dependencies

### Examples

**SQL INSERT Sequence:**
```sql
INSERT INTO target_table SELECT * FROM source_table
```
**Extracted:** `source_table → target_table` (source processed before target)

**SQL JOIN Sequence:**
```sql
SELECT * FROM table1 JOIN table2 ON ...
```
**Extracted:** `table1 → table2` (left to right processing order)

**Control-M Job Dependencies:**
```xml
<JOB JOBNAME="job2">
  <INCOND NAME="job1-TO-job2"/>
</JOB>
```
**Extracted:** `job1 → job2` (job1 processes before job2)

### Output Format

```json
{
  "sequence_id": "sql_0_insert_0",
  "tables": ["source_table", "target_table"],
  "source_type": "sql",
  "sequence_type": "insert",
  "order": 0
}
```

## 2. Code Parameters

### What It Extracts

Extracts **parameters** from code:
- SQL WHERE clause parameters (`WHERE column = ?` or `WHERE column = :param`)
- SQL function parameters
- Control-M variables
- Stored procedure parameters

### Examples

**SQL Parameters:**
```sql
WHERE status = :status AND amount > :min_amount
```
**Extracted:**
- `status` (string, WHERE clause, required)
- `min_amount` (number, WHERE clause, required)

**Control-M Variables:**
```xml
<VAR NAME="environment" VALUE="production"/>
```
**Extracted:**
- `environment` (string, Control-M variable, optional)

### Output Format

```json
{
  "name": "status",
  "type": "string",
  "source": "sql",
  "is_required": true,
  "context": "WHERE clause"
}
```

## 3. Hardcoded Lists

### What It Extracts

Extracts **hardcoded lists and constants** from code:
- SQL IN clauses (`WHERE id IN (1, 2, 3)`)
- SQL CASE WHEN statements
- JSON enum definitions
- Control-M hardcoded values

### Examples

**SQL IN Clause:**
```sql
WHERE status IN ('active', 'pending', 'completed')
```
**Extracted:**
```json
{
  "name": "in_clause_0",
  "values": ["active", "pending", "completed"],
  "type": "IN clause",
  "context": "SQL IN clause"
}
```

**SQL CASE WHEN:**
```sql
CASE WHEN status = 'A' THEN 'Active' ELSE 'Inactive' END
```
**Extracted:** Hardcoded logic values

**JSON Enum:**
```json
{
  "status": {
    "enum": ["active", "inactive", "pending"]
  }
}
```
**Extracted:** Enum values as hardcoded list

## 4. Table Classifications

### What It Classifies

Classifies tables as:
- **Transaction**: Transaction tables (orders, payments, events, logs)
- **Reference**: Lookup/reference tables (codes, dictionaries, master data)
- **Staging**: Staging/landing tables (staging, temp, raw)
- **Test**: Test/mock tables (test, mock, sample)

### Classification Patterns

**Transaction Tables:**
- Pattern keywords: `trans`, `txn`, `order`, `payment`, `invoice`, `event`, `log`, `fact`
- High write frequency
- Time-series data
- Examples: `orders`, `transactions`, `payment_log`, `event_fact`

**Reference Tables:**
- Pattern keywords: `ref`, `lookup`, `code`, `dict`, `master`, `config`, `dimension`
- Low write frequency
- Static/semi-static data
- Examples: `ref_status`, `lookup_codes`, `master_customer`, `dim_date`

**Staging Tables:**
- Pattern keywords: `staging`, `stage`, `temp`, `tmp`, `intermediate`, `landing`, `raw`
- Temporary data
- ETL processing
- Examples: `staging_orders`, `temp_load`, `raw_source`

**Test Tables:**
- Pattern keywords: `test`, `mock`, `stub`, `fake`, `sample`, `demo`
- Testing data
- Non-production
- Examples: `test_orders`, `mock_data`, `sample_customers`

### Classification Confidence

- **High Confidence (≥0.7)**: Strong pattern match
- **Medium Confidence (0.4-0.7)**: Partial pattern match
- **Low Confidence (<0.4)**: Weak or no pattern match

### Output Format

```json
{
  "table_name": "orders",
  "classification": "transaction",
  "confidence": 0.9,
  "evidence": [
    "Contains transaction pattern: order",
    "Has foreign key relationships"
  ],
  "patterns": ["order", "trans"]
}
```

## 5. Testing Endpoints

### What It Identifies

Identifies **testing/test endpoints** from:
- Control-M job commands (HTTP URLs, API calls)
- Code with test indicators
- Endpoints with test patterns in names

### Test Indicators

- Keywords: `test`, `mock`, `stub`, `fake`, `sample`, `demo`, `trial`
- Patterns: `/test/`, `/mock/`, `/stub/`
- File names: `*test*`, `*mock*`

### Output Format

```json
{
  "endpoint": "https://api.example.com/test/orders",
  "method": "GET",
  "source": "controlm",
  "is_test": true,
  "test_indicators": ["test indicator in content"]
}
```

## Integration with Knowledge Graph

All advanced extraction results are integrated into the knowledge graph:

### Table Classifications → Node Properties
- Added as `table_classification`, `classification_confidence`, `classification_evidence` properties on table nodes

### Table Process Sequences → Edges
- Creates `PROCESSES_BEFORE` edges between tables in sequence
- Edge properties include `sequence_id`, `sequence_type`, `sequence_order`

### Code Parameters → Nodes
- Creates `parameter` nodes with parameter metadata

### Hardcoded Lists → Nodes
- Creates `hardcoded_list` nodes with list values and metadata

### Testing Endpoints → Nodes
- Creates `endpoint` nodes marked as test endpoints

## Usage

Advanced extraction happens automatically during graph processing:

```bash
# POST /knowledge-graph
{
  "json_tables": ["data/tables.json"],
  "hive_ddls": ["data/schema.hql"],
  "sql_queries": ["SELECT * FROM orders WHERE status = ?"],
  "control_m_files": ["data/jobs.xml"]
}

# Response includes enhanced nodes/edges with:
# - table_classification properties
# - PROCESSES_BEFORE edges
# - parameter nodes
# - hardcoded_list nodes
# - endpoint nodes
```

## Querying Advanced Extraction Results

### Neo4j Cypher Queries

**Find all transaction tables:**
```cypher
MATCH (n:Node)
WHERE n.type = 'table' 
  AND n.properties_json CONTAINS '"table_classification":"transaction"'
RETURN n.label, n.properties_json
```

**Find table processing sequences:**
```cypher
MATCH (source:Node)-[r:RELATIONSHIP]->(target:Node)
WHERE r.label = 'PROCESSES_BEFORE'
RETURN source.label, target.label, r.properties_json
ORDER BY r.properties_json.sequence_order
```

**Find all code parameters:**
```cypher
MATCH (n:Node)
WHERE n.type = 'parameter'
RETURN n.label, n.properties_json
```

**Find hardcoded lists:**
```cypher
MATCH (n:Node)
WHERE n.type = 'hardcoded_list'
RETURN n.label, n.properties_json.values
```

**Find testing endpoints:**
```cypher
MATCH (n:Node)
WHERE n.type = 'endpoint' 
  AND n.properties_json.is_test = true
RETURN n.label, n.properties_json
```

## Impact

**Priority 6.1 Enhancement: Advanced Extraction**
- **Impact**: Enhanced data extraction capabilities
- **Status**: ✅ Implemented
- **Files**: `services/extract/advanced_extraction.go`
- **Integration**: Automatic during graph processing

## Benefits

1. **Better Understanding**: Know which tables are transaction vs reference
2. **Process Flow**: Understand table processing sequences
3. **Code Analysis**: Extract parameters and hardcoded logic
4. **Quality Assurance**: Identify test endpoints vs production
5. **Pattern Recognition**: Learn from hardcoded lists and constants

