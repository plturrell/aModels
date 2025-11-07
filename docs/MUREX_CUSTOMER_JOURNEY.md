# Murex Customer Journey

## Overview

The Murex API customer journey provides a complete experience for connecting to Murex trading systems, processing trades, and performing ETL transformations to SAP GL. This journey mirrors the DMS, Relational, and Perplexity integration experiences, offering the same level of sophistication, intelligence, and visual delight.

## User Journey Flow

```
1. Connect to Murex API
   ↓
2. Discover Schema (OpenAPI)
   ↓
3. Select Trades/Entities to Process
   ↓
4. Processing Triggered (Async)
   ↓
5. Real-time Status Tracking
   ↓
6. Full Pipeline Processing
   ├─→ Trade Data Extraction
   ├─→ Catalog Registration
   ├─→ Training Export
   ├─→ LocalAI Storage
   ├─→ Search Indexing
   └─→ ETL to SAP GL
   ↓
7. Intelligence Collection
   ├─→ Domain Detection (Finance)
   ├─→ Relationship Discovery
   ├─→ Pattern Learning
   └─→ Knowledge Graph Building
   ↓
8. Results Visualization
   ├─→ Processing Dashboard
   ├─→ Results Dashboard
   ├─→ Analytics Dashboard
   └─→ ETL Dashboard
```

## Key Features

### 1. Murex API Connection

Users can connect to Murex trading systems:
- **OpenAPI Specification**: Automatic schema discovery from OpenAPI specs
- **API Authentication**: Bearer token authentication
- **Endpoint Discovery**: Automatic endpoint mapping from OpenAPI
- **Connection String**: Base URL and API key configuration

### 2. Schema Discovery

Automatic schema discovery:
- **OpenAPI Spec**: Load from URL or local file
- **Tables/Entities**: Extract from OpenAPI components.schemas
- **Endpoints**: Map table names to API endpoints
- **Relationships**: Discover relationships from endpoint paths

### 3. Processing Pipeline

Trades are processed through the full pipeline:
- **Data Extraction**: Extract trades from Murex API endpoints
- **Catalog**: Registered in the catalog service with metadata
- **Training**: Exported for ML model training
- **LocalAI**: Stored in domain-aware LocalAI service (finance domain)
- **Search**: Indexed for semantic search
- **ETL to SAP GL**: Transformed and loaded to SAP General Ledger

### 4. ETL to SAP GL

The ETL transformation includes:
- **Field Mapping**: Murex fields mapped to SAP GL fields
- **Data Transformation**: Format conversions and calculations
- **Journal Entry Creation**: SAP GL journal entry format
- **Account Mapping**: Counterparty to account lookup
- **Validation**: Data validation before SAP load

### 5. Intelligence Collection

The system automatically collects intelligence:
- **Domain Detection**: Finance domain classification
- **Relationship Discovery**: Trade relationships and dependencies
- **Pattern Learning**: Learned patterns from trade data
- **Knowledge Graph**: Graph representation of trade relationships

### 6. Real-time Tracking

Users can track processing in real-time:
- **Status API**: Current processing status
- **Progress Updates**: Step-by-step progress tracking
- **Error Reporting**: Detailed error information with recovery steps
- **Webhook Notifications**: Optional webhook callbacks

### 7. Visualization Dashboards

Beautiful dashboards for exploring results:
- **Processing Dashboard**: Real-time status and progress
- **Results Dashboard**: Intelligence visualization
- **Analytics Dashboard**: Trends and patterns
- **ETL Dashboard**: ETL transformation monitoring

## Integration Points

### Orchestration Service

- **Process**: `POST /api/murex/process`
- **Status**: `GET /api/murex/status/{request_id}`
- **Results**: `GET /api/murex/results/{request_id}`
- **Intelligence**: `GET /api/murex/results/{request_id}/intelligence`
- **History**: `GET /api/murex/history`

### Observable Dashboard

- **Landing**: `/murex`
- **Processing**: `/murex-processing`
- **Results**: `/murex-results`
- **Analytics**: `/murex-analytics`
- **ETL**: `/murex-etl`

### Browser Shell

- **Module**: Murex Processing module
- **Views**: Processing, Results, Analytics, ETL
- **Navigation**: Integrated into main navigation

## Design Philosophy

The Murex customer journey follows the **Jobs & Ive lens**:

- **Simplicity**: Clean, focused interfaces
- **Beauty**: Elegant typography, generous whitespace
- **Intuition**: Zero learning curve
- **Delight**: Smooth animations, beautiful interactions

## Getting Started

1. **Connect to Murex**: Provide base URL and API key
2. **Process Trades**: Select trades to process or process with filters
3. **Track Processing**: Monitor real-time status
4. **Explore Results**: View intelligence and ETL status
5. **Monitor ETL**: Track SAP GL transformations

## ETL Transformation Details

### Field Mappings

| Murex Field | SAP GL Field | Transformation |
|------------|--------------|-----------------|
| trade_id | entry_id | JE-{trade_id} |
| trade_date | entry_date | Identity |
| notional_amount | debit_amount | Identity |
| notional_amount | credit_amount | Copy |
| counterparty_id | account | Lookup |

### Transformation Steps

1. **Extract**: Get trades from Murex API
2. **Transform**: Map fields to SAP GL format
3. **Enrich**: Add account mappings from counterparty
4. **Validate**: Ensure data meets SAP requirements
5. **Load**: Post journal entries to SAP GL

## Next Steps

- Enhanced ETL monitoring
- SAP GL reconciliation views
- Advanced filtering and search
- Export and sharing capabilities

