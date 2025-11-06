# Murex API Terminology Extraction and Training

## Overview

The Murex API provides rich information that can be used to:
1. **Train ML models** - Schema patterns, field patterns, value patterns
2. **Populate catalog terminology** - Domain terms, business roles, naming conventions

## What Information is Available

### From OpenAPI Specification

1. **Schema Definitions** (`components.schemas`)
   - Table/entity definitions
   - Field names and types
   - Field descriptions
   - Required/optional fields
   - Relationships (foreign keys)

2. **Endpoint Descriptions**
   - API endpoint paths
   - Parameter definitions
   - Response schemas
   - Documentation strings

3. **Metadata**
   - API version
   - System information
   - Field descriptions

### From Actual API Data

1. **Value Patterns**
   - Sample values from real data
   - Data type validation patterns
   - Format patterns (dates, currency codes, etc.)

2. **Schema Examples**
   - Real table structures
   - Column examples with actual values
   - Relationship examples

3. **Field Examples**
   - Domain-specific field names
   - Business role patterns
   - Naming convention patterns

## Extracted Information

### Terminology

**Domains:**
- Finance domain terms: `trade`, `cashflow`, `position`, `counterparty`, `instrument`, `pricing`, `market_data`

**Roles:**
- `identifier`: trade_id, id, etc.
- `amount`: notional, amount, price, value
- `date`: trade_date, date, timestamp
- `code`: currency, ccy
- `status`: status, state
- `reference`: counterparty, instrument
- `text`: description, comment
- `name`: name fields

**Naming Patterns:**
- `snake_case`: Most common (trade_id, notional_amount)
- `has_id_suffix`: Fields ending in _id
- `has_id_prefix`: Fields starting with id
- `camelCase`, `PascalCase`, `UPPER_SNAKE`

**Relationships:**
- `trades -> cashflows`
- `trades -> positions`
- `counterparties -> trades`

### Training Data

**Schema Examples:**
- Complete table structures with columns
- Primary keys and foreign keys
- Sample values for each column

**Field Examples:**
- Field name, type, domain, role
- Naming pattern
- Example values

**Value Patterns:**
- Inferred patterns (currency_code, date_string, numeric)
- Frequency counts
- Example values

## API Endpoints

### Extract Terminology

**POST** `/integrations/murex/terminology/extract`

Extracts terminology from OpenAPI spec and/or API data.

**Request:**
```json
{
  "from_openapi": true,
  "from_api_data": true,
  "sample_size": 100
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Terminology extracted successfully",
  "terminology": {
    "domains": 7,
    "roles": 8,
    "patterns": 5,
    "entity_types": 5,
    "relationships": 3
  },
  "training_data": {
    "schema_examples": 5,
    "field_examples": 45,
    "relationship_examples": 3,
    "value_patterns": 45
  }
}
```

### Populate Catalog

**POST** `/integrations/murex/catalog/populate`

Populates the catalog with extracted terminology and training data.

**Response:**
```json
{
  "status": "success",
  "message": "Catalog populated successfully from Murex terminology and training data"
}
```

## Integration with Terminology Learner

The extracted terminology can be fed into the `TerminologyLearner` for model training:

1. **Domain Learning**: Learn finance domain patterns
2. **Role Learning**: Learn business role patterns (identifier, amount, date, etc.)
3. **Pattern Learning**: Learn naming convention patterns
4. **Embedding Enhancement**: Enhance embeddings with domain-specific terminology

## Integration with Catalog

The extracted information populates the ISO 11179 catalog with:

1. **Data Elements**: Registered as ISO 11179 data elements
2. **Terminology**: Domain terms, roles, patterns
3. **Schema Elements**: Tables, columns, relationships
4. **Metadata**: Source system, confidence, timestamps

## Usage Example

```bash
# Extract terminology from Murex OpenAPI and API data
curl -X POST http://localhost:8081/integrations/murex/terminology/extract \
  -H "Content-Type: application/json" \
  -d '{
    "from_openapi": true,
    "from_api_data": true,
    "sample_size": 100
  }'

# Populate catalog with extracted terminology
curl -X POST http://localhost:8081/integrations/murex/catalog/populate
```

## Benefits

1. **Model Training**:
   - Better schema understanding
   - Improved field role inference
   - Enhanced naming pattern recognition
   - Better domain-specific embeddings

2. **Catalog Population**:
   - Automatic registration of Murex data elements
   - Domain-specific terminology
   - Business role definitions
   - Relationship mappings

3. **Knowledge Graph Enrichment**:
   - Richer metadata
   - Better semantic search
   - Improved data discovery
   - Enhanced lineage tracking

## Terminology Learner Integration

The extracted terminology can be automatically fed into the TerminologyLearner for model training:

### Train Terminology Learner

**POST** `/integrations/murex/terminology/train`

Trains the terminology learner from extracted Murex terminology.

**Response:**
```json
{
  "status": "success",
  "message": "Terminology learner trained successfully from Murex data"
}
```

### Export Training Data

**GET** `/integrations/murex/terminology/export`

Exports training data in a format suitable for ML model training.

**Response:**
```json
{
  "terminology": {
    "domains": {...},
    "roles": {...},
    "patterns": {...}
  },
  "training_data": {
    "schema_examples": [...],
    "field_examples": [...],
    "relationship_examples": [...],
    "value_patterns": [...]
  },
  "metadata": {
    "source": "murex",
    "extracted_at": "2024-01-01T00:00:00Z"
  }
}
```

## Complete Workflow

```bash
# 1. Extract terminology from Murex OpenAPI and API data
curl -X POST http://localhost:8081/integrations/murex/terminology/extract \
  -H "Content-Type: application/json" \
  -d '{
    "from_openapi": true,
    "from_api_data": true,
    "sample_size": 100
  }'

# 2. Train terminology learner
curl -X POST http://localhost:8081/integrations/murex/terminology/train

# 3. Populate catalog
curl -X POST http://localhost:8081/integrations/murex/catalog/populate

# 4. Export training data for ML models
curl http://localhost:8081/integrations/murex/terminology/export > murex_training_data.json
```

## Next Steps

1. ✅ Integrate with TerminologyLearner for incremental learning
2. ✅ Export to training data format for ML model training
3. ✅ Sync with catalog service for real-time updates
4. Create scheduled jobs for periodic extraction
5. Add endpoints in extract service for terminology learning via HTTP

