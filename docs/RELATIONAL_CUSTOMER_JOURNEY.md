# Relational Customer Journey

## Overview

The Relational Database customer journey provides a complete experience for connecting to relational databases, processing tables, and exploring results through the full aModels pipeline. This journey mirrors the DMS and Perplexity integration experiences, offering the same level of sophistication, intelligence, and visual delight.

## User Journey Flow

```
1. Connect to Database
   ↓
2. Discover Schema
   ↓
3. Select Tables to Process
   ↓
4. Processing Triggered (Async)
   ↓
5. Real-time Status Tracking
   ↓
6. Full Pipeline Processing
   ├─→ Table Data Extraction
   ├─→ Catalog Registration
   ├─→ Training Export
   ├─→ LocalAI Storage
   └─→ Search Indexing
   ↓
7. Intelligence Collection
   ├─→ Domain Detection
   ├─→ Relationship Discovery
   ├─→ Pattern Learning
   └─→ Knowledge Graph Building
   ↓
8. Results Visualization
   ├─→ Processing Dashboard
   ├─→ Results Dashboard
   ├─→ Analytics Dashboard
   └─→ Tables Library
```

## Key Features

### 1. Database Connection

Users can connect to relational databases:
- **PostgreSQL**: Full support with schema discovery
- **MySQL**: Full support with schema discovery
- **SQLite**: Full support with schema discovery
- **Connection String**: Direct DSN or connection parameters

### 2. Schema Discovery

Automatic schema discovery:
- **Tables**: List all tables in schema
- **Columns**: Column names, types, and constraints
- **Relationships**: Foreign key relationships
- **Primary Keys**: Automatic detection

### 3. Processing Pipeline

Tables are processed through the full pipeline:
- **Data Extraction**: Extract table rows (configurable limit)
- **Catalog**: Registered in the catalog service with metadata
- **Training**: Exported for ML model training
- **LocalAI**: Stored in domain-aware LocalAI service
- **Search**: Indexed for semantic search

### 4. Intelligence Collection

The system automatically collects intelligence:
- **Domain Detection**: Automatic domain classification based on table content
- **Relationship Discovery**: Foreign key relationships and data relationships
- **Pattern Learning**: Learned patterns from table structure and data
- **Knowledge Graph**: Graph representation of table relationships

### 5. Real-time Tracking

Users can track processing in real-time:
- **Status API**: Current processing status
- **Progress Updates**: Step-by-step progress tracking
- **Error Reporting**: Detailed error information with recovery steps
- **Webhook Notifications**: Optional webhook callbacks

### 6. Visualization Dashboards

Beautiful dashboards for exploring results:
- **Processing Dashboard**: Real-time status and progress
- **Results Dashboard**: Intelligence visualization
- **Analytics Dashboard**: Trends and patterns
- **Tables Library**: Browse processed tables

## Integration Points

### Orchestration Service

- **Process**: `POST /api/relational/process`
- **Status**: `GET /api/relational/status/{request_id}`
- **Results**: `GET /api/relational/results/{request_id}`
- **Intelligence**: `GET /api/relational/results/{request_id}/intelligence`
- **History**: `GET /api/relational/history`
- **Search**: `POST /api/relational/search`
- **Export**: `GET /api/relational/results/{request_id}/export`
- **Batch**: `POST /api/relational/batch`
- **Cancel**: `DELETE /api/relational/jobs/{request_id}`

### Observable Dashboard

- **Landing**: `/relational`
- **Processing**: `/relational-processing`
- **Results**: `/relational-results`
- **Analytics**: `/relational-analytics`
- **Tables**: `/relational-tables`

### Browser Shell

- **Module**: Relational Processing module (coming soon)
- **Views**: Processing, Results, Analytics, Tables
- **Navigation**: Integrated into main navigation

## Design Philosophy

The Relational customer journey follows the **Jobs & Ive lens**:

- **Simplicity**: Clean, focused interfaces
- **Beauty**: Elegant typography, generous whitespace
- **Intuition**: Zero learning curve
- **Delight**: Smooth animations, beautiful interactions

## Getting Started

1. **Connect to Database**: Provide connection string or parameters
2. **Process Tables**: Select tables to process or process entire schema
3. **Track Processing**: Monitor real-time status
4. **Explore Results**: View intelligence and relationships

## Supported Databases

- **PostgreSQL**: Full support
- **MySQL**: Full support
- **SQLite**: Full support

## Next Steps

- Enhanced table preview
- Advanced filtering and search
- Schema visualization
- Export and sharing capabilities

