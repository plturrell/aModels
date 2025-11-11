# Murex OpenAPI Integration

## Overview

The Murex integration provides seamless connectivity to Murex trading systems using OpenAPI specifications. It automatically discovers endpoints, extracts schemas, and ingests data into the Neo4j knowledge graph.

## Features

- **OpenAPI Specification Support**: Automatically loads and parses Murex OpenAPI specs from GitHub or local files
- **Schema Discovery**: Extracts table/entity schemas from OpenAPI `components.schemas`
- **Endpoint Mapping**: Maps table names to API endpoints (e.g., `trades` → `/api/v1/trades`)
- **Authentication**: Supports Bearer token authentication
- **Knowledge Graph Integration**: Automatically creates nodes and relationships in Neo4j
- **Graceful Fallback**: Returns mock data if API is unavailable (for development/testing)

## Configuration

### Environment Variables

The following environment variables are required for Murex integration:

```bash
# Neo4j Configuration (required)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# Murex API Configuration (required)
MUREX_BASE_URL=https://api.murex.com
MUREX_API_KEY=your-api-key

# OpenAPI Specification (optional, defaults to GitHub)
MUREX_OPENAPI_SPEC_URL=https://raw.githubusercontent.com/mxenabled/openapi/master/openapi/trades.yaml
```

### Optional Configuration

- `MUREX_OPENAPI_SPEC_PATH`: Local path to OpenAPI spec file (alternative to URL)
- If `MUREX_OPENAPI_SPEC_URL` is not provided, defaults to Murex GitHub repository

## API Endpoints

The integration exposes the following REST endpoints:

### 1. Full Synchronization

**POST** `/integrations/murex/sync`

Performs a full synchronization of all Murex data (trades, cashflows, regulatory calculations).

**Response:**
```json
{
  "status": "success",
  "message": "Murex synchronization completed successfully"
}
```

### 2. Ingest Trades

**POST** `/integrations/murex/trades`

Ingests trades from Murex with optional filters.

**Request:**
```json
{
  "filters": {
    "trade_date_from": "2024-01-01",
    "trade_date_to": "2024-12-31",
    "status": "Executed"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Trades ingested successfully"
}
```

### 3. Ingest Cashflows

**POST** `/integrations/murex/cashflows`

Ingests cashflows from Murex.

**Request:**
```json
{
  "filters": {
    "trade_id": "T001"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Cashflows ingested successfully"
}
```

### 4. Discover Schema

**GET** `/integrations/murex/schema`

Discovers the Murex API schema from the OpenAPI specification.

**Response:**
```json
{
  "source_type": "murex",
  "tables": [
    {
      "name": "trades",
      "columns": [
        {
          "name": "trade_id",
          "type": "string",
          "nullable": false
        }
      ],
      "primary_key": ["trade_id"]
    }
  ],
  "metadata": {
    "system": "murex",
    "version": "3.1",
    "openapi_version": "3.0.0"
  }
}
```

## Usage Examples

### Using cURL

```bash
# Full synchronization
curl -X POST http://localhost:8081/integrations/murex/sync

# Ingest trades with filters
curl -X POST http://localhost:8081/integrations/murex/trades \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {
      "trade_date_from": "2024-01-01",
      "status": "Executed"
    }
  }'

# Discover schema
curl http://localhost:8081/integrations/murex/schema
```

### Using Go Client

```go
import (
    "github.com/plturrell/aModels/services/graph"
    "github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// Create Neo4j driver
driver, _ := neo4j.NewDriverWithContext(
    "bolt://localhost:7687",
    neo4j.BasicAuth("neo4j", "password", ""),
)

// Create graph client
graphClient := graph.NewNeo4jGraphClient(driver, logger)

// Create mapper
mapper := graph.NewDefaultModelMapper()

// Configure Murex
config := map[string]interface{}{
    "base_url": "https://api.murex.com",
    "api_key":  "your-api-key",
    "openapi_spec_url": "https://raw.githubusercontent.com/mxenabled/openapi/master/openapi/trades.yaml",
}

// Create integration
integration := graph.NewMurexIntegration(config, mapper, graphClient, logger)

// Ingest trades
ctx := context.Background()
err := integration.IngestTrades(ctx, map[string]interface{}{
    "trade_date_from": "2024-01-01",
})
```

## Knowledge Graph Schema

The integration creates the following node types in Neo4j:

- **Trade**: Financial trade nodes with properties like `trade_id`, `notional_amount`, `currency`, etc.
- **JournalEntry**: Journal entry nodes linked to trades
- **RegulatoryCalculation**: Regulatory calculation nodes from Murex FMRP
- **Counterparty**: Counterparty nodes linked to trades

Relationships:
- `Trade` → `HAS_COUNTERPARTY` → `Counterparty`
- `Trade` → `TRADES_TO` → `Cashflow`
- `Trade` → `REQUIRES_CALCULATION` → `RegulatoryCalculation`

## Agent Integration

The Murex connector is also available through the AgentFactory for autonomous data ingestion:

```go
factory := agents.NewAgentFactory(graphClient, ruleStore, alertManager, logger)

// Create Murex ingestion agent
agent, err := factory.CreateDataIngestionAgent("murex", config)
if err != nil {
    log.Fatal(err)
}

// Run ingestion
err = agentSystem.RunIngestion(ctx, "murex", config)
```

## Troubleshooting

### API Connection Issues

If the Murex API is unavailable, the connector will:
1. Log a warning
2. Return mock data for development/testing
3. Continue processing without errors

### OpenAPI Spec Loading

If the OpenAPI spec fails to load:
- Check the URL is accessible
- Verify the spec format is valid JSON/YAML
- Check network connectivity
- Fallback to default endpoint mappings

### Neo4j Connection

Ensure Neo4j is running and accessible:
```bash
# Test Neo4j connection
neo4j-admin ping
```

## References

- [Murex OpenAPI Repository](https://github.com/mxenabled/openapi)
- [Neo4j Go Driver Documentation](https://github.com/neo4j/neo4j-go-driver)

