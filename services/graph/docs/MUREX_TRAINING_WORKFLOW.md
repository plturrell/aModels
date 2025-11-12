# Murex API Training and Catalog Population Workflow

## Overview

The Murex API provides rich information that can be used to:
1. **Train ML models** - Schema patterns, field patterns, value patterns
2. **Populate catalog terminology** - Domain terms, business roles, naming conventions

## What Information is Extracted

### From OpenAPI Specification

1. **Schema Definitions**
   - Table/entity definitions with field names and types
   - Field descriptions from OpenAPI `description` fields
   - Required/optional field indicators
   - Relationships (foreign keys)

2. **Endpoint Metadata**
   - API endpoint paths and methods
   - Parameter definitions and types
   - Response schemas
   - Documentation strings

3. **Domain Terminology**
   - Finance domain terms: `trade`, `cashflow`, `position`, `counterparty`, `instrument`
   - Business concepts and relationships

### From Actual API Data

1. **Value Patterns**
   - Sample values from real data (up to configured limit)
   - Data type validation patterns
   - Format patterns (dates, currency codes, numeric patterns)

2. **Schema Examples**
   - Real table structures with actual data
   - Column examples with sample values
   - Relationship examples

3. **Field Examples**
   - Domain-specific field names
   - Business role patterns (identifier, amount, date, etc.)
   - Naming convention patterns (snake_case, etc.)

## Complete Workflow

### Step 1: Extract Terminology

Extract terminology from OpenAPI spec and/or actual API data.

```bash
curl -X POST http://localhost:8081/integrations/murex/terminology/extract \
  -H "Content-Type: application/json" \
  -d '{
    "from_openapi": true,
    "from_api_data": true,
    "sample_size": 100
  }'
```

**What it extracts:**
- Domain terms (finance domain)
- Business roles (identifier, amount, date, code, status, reference, text, name)
- Naming patterns (snake_case, has_id_suffix, etc.)
- Schema examples with sample values
- Field examples with patterns
- Value patterns

### Step 2: Train Terminology Learner

Feed extracted terminology into the TerminologyLearner for automatic model training.

```bash
curl -X POST http://localhost:8081/integrations/murex/terminology/train
```

**What it does:**
- Converts extracted terminology to Node/Edge format
- Feeds into TerminologyLearner.LearnFromExtraction()
- Directly learns domains and roles with high confidence
- Updates LNN (Liquid Neural Network) weights
- Stores terminology in Neo4j for persistence

**Benefits:**
- Improved domain inference (finance domain)
- Better role inference (identifier, amount, date, etc.)
- Enhanced naming pattern recognition
- Better embeddings for finance terminology

### Step 3: Populate Catalog

Populate the ISO 11179 catalog with extracted terminology.

```bash
curl -X POST http://localhost:8081/integrations/murex/catalog/populate
```

**What it populates:**
- Data Elements for all tables/columns
- Domain-specific terminology entries
- Business role definitions
- Relationship mappings
- Metadata with confidence scores

### Step 4: Export Training Data

Export training data for ML model training.

```bash
curl http://localhost:8081/integrations/murex/terminology/export > murex_training_data.json
```

**Exported data includes:**
- Terminology (domains, roles, patterns)
- Schema examples
- Field examples
- Relationship examples
- Value patterns

## Integration Architecture

```
Murex API (OpenAPI + Data)
    ↓
MurexTerminologyExtractor
    ├─→ Extracted Terminology (domains, roles, patterns)
    └─→ Training Data (schemas, fields, values)
         ↓
    ┌─────────────────────────────────────┐
    │                                     │
    ↓                                     ↓
MurexCatalogPopulator      MurexTerminologyLearnerIntegration
    │                                     │
    ↓                                     ↓
ISO 11179 Catalog          TerminologyLearner (LNN)
    │                                     │
    └─────────────────────────────────────┘
                    ↓
            Enhanced Models & Catalog
```

## API Endpoints

### 1. Extract Terminology
**POST** `/integrations/murex/terminology/extract`

Extracts terminology from OpenAPI spec and/or API data.

### 2. Train Terminology Learner
**POST** `/integrations/murex/terminology/train`

Trains the TerminologyLearner from extracted terminology.

### 3. Populate Catalog
**POST** `/integrations/murex/catalog/populate`

Populates catalog with extracted terminology.

### 4. Export Training Data
**GET** `/integrations/murex/terminology/export`

Exports training data for ML model training.

## Example: Complete Training Workflow

```bash
#!/bin/bash

# Configuration
GRAPH_SERVER="http://localhost:8081"
EXTRACT_SERVICE="http://extract-service:8081"

echo "Step 1: Extracting terminology from Murex..."
curl -X POST $GRAPH_SERVER/integrations/murex/terminology/extract \
  -H "Content-Type: application/json" \
  -d '{
    "from_openapi": true,
    "from_api_data": true,
    "sample_size": 100
  }'

echo "Step 2: Training terminology learner..."
curl -X POST $GRAPH_SERVER/integrations/murex/terminology/train

echo "Step 3: Populating catalog..."
curl -X POST $GRAPH_SERVER/integrations/murex/catalog/populate

echo "Step 4: Exporting training data..."
curl $GRAPH_SERVER/integrations/murex/terminology/export > murex_training_data.json

echo "Training workflow completed!"
```

## Benefits

### For Model Training

1. **Better Schema Understanding**
   - Learned patterns from real Murex schemas
   - Improved field type inference
   - Better relationship recognition

2. **Improved Field Role Inference**
   - Finance-specific role patterns
   - Domain-aware role classification
   - Higher confidence for finance domain

3. **Enhanced Naming Pattern Recognition**
   - Murex-specific naming conventions
   - Pattern frequency analysis
   - Better pattern matching

4. **Domain-Specific Embeddings**
   - Finance terminology in embeddings
   - Better semantic search
   - Improved similarity matching

### For Catalog

1. **Automatic Registration**
   - All Murex tables/columns registered
   - Metadata with confidence scores
   - Source system tracking

2. **Rich Terminology**
   - Domain-specific terms
   - Business role definitions
   - Relationship mappings

3. **Better Discovery**
   - Semantic search with finance terms
   - Role-based filtering
   - Pattern-based matching

## Integration with Extract Service

The integration uses HTTP calls to the extract service's TerminologyLearner:

- `/terminology/learn` - Learn from nodes/edges
- `/terminology/learn/domain` - Learn domain term
- `/terminology/learn/role` - Learn role term
- `/terminology/infer/domain` - Infer domain
- `/terminology/infer/role` - Infer role
- `/terminology/enhance/embedding` - Enhance embedding

**Note**: These endpoints need to be implemented in the extract service if they don't exist yet.

## Configuration

Set environment variables:

```bash
# Murex API
MUREX_BASE_URL=https://api.murex.com
MUREX_API_KEY=your-api-key
MUREX_OPENAPI_SPEC_URL=https://raw.githubusercontent.com/mxenabled/openapi/master/openapi/trades.yaml

# Extract Service (for terminology learning)
EXTRACT_SERVICE_URL=http://extract-service:8081

# Neo4j (for catalog)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

## Monitoring

Check terminology extraction results:

```bash
# Check extracted terminology stats
curl http://localhost:8081/integrations/murex/terminology/extract | jq '.terminology'

# Check training data stats
curl http://localhost:8081/integrations/murex/terminology/extract | jq '.training_data'
```

