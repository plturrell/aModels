# SAP HANA Cloud Inbound Integration

This document describes the integration of SAP HANA Cloud relational tables with aModels' extraction, training, LocalAI, and search services, secured with XSUAA authentication.

## Overview

The HANA Cloud inbound integration provides:
- **Secure table extraction** from SAP HANA Cloud using XSUAA authentication
- **Automatic domain detection** based on user's XSUAA scopes
- **Differential privacy** applied based on user permissions and domain sensitivity
- **End-to-end pipeline** from HANA → Extraction → Training → LocalAI → Search
- **Domain intelligence** integration for intelligent routing

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              SAP HANA Cloud Tables                      │
│  (Schema: <schema>, Tables: <table1>, <table2>, ...)     │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ XSUAA Authenticated Request
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Catalog Service (HANA Inbound Integration)     │
│  ┌────────────────────────────────────────────────────┐  │
│  │  1. Extract Data from HANA Tables                  │  │
│  │     - Query tables with user's domain access      │  │
│  │     - Apply privacy filters if needed             │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  2. Domain & Privacy Resolution                    │  │
│  │     - Detect domain from XSUAA scopes              │  │
│  │     - Calculate privacy config (epsilon/delta)    │  │
│  │     - Verify user has domain access                │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  3. Process Through Extraction Service             │  │
│  │     - Create knowledge graph                       │  │
│  │     - Apply differential privacy                   │  │
│  │     - Forward XSUAA token                          │  │
│  └────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────────┐   ┌──────────────────┐
│ Training Service  │   │  Search Service  │
│ - Process graph   │   │ - Index data     │
│ - Train models    │   │ - Enable search  │
│ - Update LocalAI  │   │                  │
└──────────────────┘   └──────────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌──────────────────┐
            │    LocalAI       │
            │ - Fine-tune      │
            │ - Domain models  │
            └──────────────────┘
```

## API Endpoints

### POST /catalog/integration/hana/process

Processes HANA Cloud tables through the full pipeline.

**Authentication:** Required (XSUAA Bearer token)

**Request Body:**
```json
{
  "schema": "MYSCHEMA",
  "tables": ["TABLE1", "TABLE2"],
  "domain_id": "finance",  // Optional: auto-detected from XSUAA scopes
  "project_id": "my-project",
  "system_id": "hana-cloud",  // Optional
  "enable_privacy": true,  // Optional: apply differential privacy
  "output_format": "json"  // Optional: json, jsonl, csv
}
```

**Response:**
```json
{
  "request_id": "hana-inbound-1234567890",
  "status": "completed",
  "extracted_nodes": 150,
  "extracted_edges": 200,
  "domain_id": "finance",
  "privacy_applied": true,
  "training_job_id": "training-job-123",
  "search_index_id": "index-456",
  "processing_time_ms": 5000,
  "metadata": {
    "schema": "MYSCHEMA",
    "tables": ["TABLE1", "TABLE2"],
    "project_id": "my-project",
    "system_id": "hana-cloud",
    "privacy_level": "high"
  }
}
```

### GET /catalog/integration/hana/status/:request_id

Get status of a processing request.

**Authentication:** Optional

**Response:**
```json
{
  "request_id": "hana-inbound-1234567890",
  "status": "processing",
  "progress": 75,
  "message": "Indexing in search service"
}
```

## Configuration

### Environment Variables

```bash
# HANA Cloud Connection
HANA_HOST=hana-instance.hana.provider-us10.hanacloud.ondemand.com
HANA_PORT=39015
HANA_USER=myuser
HANA_PASSWORD=mypassword
HANA_DATABASE=HXE
HANA_SCHEMA=MYSCHEMA
HANA_ENCRYPT=true

# Service URLs
EXTRACT_SERVICE_URL=http://extract-service:19080
TRAINING_SERVICE_URL=http://training-service:8080
LOCALAI_URL=http://localai:8080
SEARCH_SERVICE_URL=http://search-service:8080
```

### XSUAA Configuration

The integration automatically uses XSUAA when deployed on SAP BTP:

- **Domain Detection**: Automatically detects user's accessible domains from XSUAA scopes
- **Privacy Configuration**: Calculates epsilon/delta based on:
  - User's access level (admin/read/write)
  - Domain sensitivity (finance/health/PII = high privacy)
  - Data classification

**Required Scopes:**
- `$XSAPPNAME.Domain.<domain_id>.<access_level>` for domain-specific access
- Or general scopes: `Display`, `Edit`, `Admin`

## Processing Flow

### Step 1: Extract from HANA

```go
// Query HANA tables
SELECT * FROM "MYSCHEMA"."TABLE1" LIMIT 10000

// Add metadata
{
  "_table": "TABLE1",
  "_schema": "MYSCHEMA",
  "_domain": "finance",
  ...table columns...
}
```

### Step 2: Domain & Privacy Resolution

```go
// Auto-detect domain from XSUAA scopes
domains := privacyIntegration.GetUserDomains(ctx)
// Returns: ["finance", "default"]

// Get privacy config
privacyConfig := privacyIntegration.GetPrivacyConfig(ctx, "finance")
// Returns: {epsilon: 0.5, delta: 1e-6, privacy_level: "high"}
```

### Step 3: Process Through Extraction Service

```go
POST /extract
{
  "source": "hana",
  "schema": "MYSCHEMA",
  "tables": ["TABLE1"],
  "data": [...],
  "domain_id": "finance",
  "privacy": {
    "epsilon": 0.5,
    "delta": 1e-6,
    "noise_scale": 0.2,
    "max_queries": 50,
    "privacy_level": "high"
  }
}
```

### Step 4: Send to Training Service

```go
POST /training/process
{
  "source": "hana",
  "graph_data": {...},
  "domain_id": "finance",
  "privacy_config": {
    "epsilon": 0.5,
    "delta": 1e-6,
    "privacy_level": "high"
  }
}
```

### Step 5: Index in Search Service

```go
POST /v1/index
{
  "source": "hana",
  "graph_data": {...},
  "domain_id": "finance"
}
```

## Domain Intelligence Integration

The integration automatically:
1. **Detects user's accessible domains** from XSUAA scopes
2. **Routes to appropriate domain models** in LocalAI
3. **Applies domain-specific filtering** in training
4. **Enables domain-aware search** in search service

**Example:**
- User has scope: `amodels-catalog.Domain.finance.read`
- Integration automatically:
  - Uses finance domain for processing
  - Routes to finance-specific models in LocalAI
  - Applies finance domain filters in training
  - Indexes with finance domain tags in search

## Privacy Protection

### Automatic Privacy Level Detection

| Domain Type | Access Level | Privacy Level | Epsilon | Delta | Max Queries |
|------------|--------------|---------------|---------|-------|-------------|
| finance    | admin        | low           | 2.0     | 1e-4  | 200         |
| finance    | read         | high          | 0.5     | 1e-6  | 50          |
| health     | admin        | medium        | 1.0     | 1e-5  | 100         |
| health     | read         | high          | 0.5     | 1e-6  | 50          |
| default    | admin        | low           | 2.0     | 1e-4  | 200         |
| default    | read         | medium        | 1.0     | 1e-5  | 100         |

### Privacy Budget Management

- **Per-user, per-domain budgets**
- **Daily reset** (or configurable schedule)
- **Automatic enforcement** - requests fail if budget exceeded
- **Audit logging** of all privacy budget consumption

## Error Handling

### Common Errors

**401 Unauthorized**
- Missing or invalid XSUAA token
- Solution: Ensure valid Bearer token in Authorization header

**403 Forbidden**
- User doesn't have access to specified domain
- Solution: Request appropriate XSUAA scope for domain

**500 Internal Server Error**
- HANA connection failed
- Solution: Check HANA credentials and network connectivity

**429 Too Many Requests**
- Privacy budget exceeded
- Solution: Wait for budget reset or request higher quota

## Usage Examples

### Example 1: Process Specific Tables

```bash
curl -X POST https://catalog.cfapps.sap.hana.ondemand.com/catalog/integration/hana/process \
  -H "Authorization: Bearer $XSUAA_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "FINANCE",
    "tables": ["TRANSACTIONS", "ACCOUNTS"],
    "project_id": "finance-project",
    "enable_privacy": true
  }'
```

### Example 2: Process All Tables in Schema

```bash
curl -X POST https://catalog.cfapps.sap.hana.ondemand.com/catalog/integration/hana/process \
  -H "Authorization: Bearer $XSUAA_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "FINANCE",
    "project_id": "finance-project"
  }'
```

### Example 3: Check Processing Status

```bash
curl -X GET https://catalog.cfapps.sap.hana.ondemand.com/catalog/integration/hana/status/hana-inbound-1234567890 \
  -H "Authorization: Bearer $XSUAA_TOKEN"
```

## Security Considerations

1. **XSUAA Authentication**: All requests require valid XSUAA token
2. **Domain Access Control**: Users can only process domains they have access to
3. **Privacy Budget Enforcement**: Strict limits prevent privacy budget exhaustion
4. **Token Forwarding**: XSUAA tokens are forwarded to downstream services
5. **Audit Logging**: All processing events are logged with user context
6. **Sensitive Data Filtering**: High-privacy domains automatically filter sensitive fields

## Integration with Other Services

### Extraction Service
- Receives HANA table data
- Creates knowledge graph
- Applies differential privacy
- Returns graph with nodes/edges

### Training Service
- Receives knowledge graph
- Processes with domain-specific filters
- Trains domain-specific models
- Updates LocalAI with fine-tuned models

### Search Service
- Receives knowledge graph
- Indexes with domain tags
- Enables domain-aware search
- Returns search index ID

### LocalAI
- Receives training data from training service
- Fine-tunes domain-specific models
- Updates model routing based on domain intelligence

## Monitoring

### Metrics

- `hana_inbound_requests_total` - Total processing requests
- `hana_inbound_processing_time_seconds` - Processing duration
- `hana_inbound_nodes_extracted` - Nodes extracted from HANA
- `hana_inbound_privacy_budget_consumed` - Privacy budget usage

### Logs

```
[HANA_INBOUND] Starting extraction: schema=FINANCE tables=[TRANSACTIONS, ACCOUNTS] domain=finance
[HANA_INBOUND] Processing through extraction service: nodes=150 edges=200
[HANA_INBOUND] Sending to training service
[HANA_INBOUND] Indexing in search service
[HANA_INBOUND] Processing completed: request_id=hana-inbound-123 nodes=150 edges=200 time=5s
```

## Troubleshooting

### HANA Connection Issues

**Error:** "Failed to ping HANA"
- Check HANA_HOST, HANA_PORT, HANA_USER, HANA_PASSWORD
- Verify network connectivity
- Check HANA_ENCRYPT setting

### Domain Access Issues

**Error:** "User does not have access to domain: finance"
- Verify XSUAA scope: `amodels-catalog.Domain.finance.read`
- Check role collection assignments in SAP BTP Cockpit

### Privacy Budget Exceeded

**Error:** "Privacy budget exceeded"
- Check privacy budget consumption
- Wait for daily reset
- Request higher quota if needed

## Next Steps

1. **Configure HANA connection** via environment variables
2. **Assign XSUAA scopes** for domain access
3. **Test with sample tables** before processing production data
4. **Monitor privacy budget** consumption
5. **Review audit logs** for security compliance

This integration provides a secure, privacy-preserving, domain-aware pipeline for processing SAP HANA Cloud tables through aModels' extraction, training, LocalAI, and search services.

