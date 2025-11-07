# DMS Customer Journey

## Overview

The Document Management System (DMS) customer journey provides a complete experience for uploading, processing, and exploring documents through the full aModels pipeline. This journey mirrors the Perplexity integration experience, offering the same level of sophistication, intelligence, and visual delight.

## User Journey Flow

```
1. Upload Document
   ↓
2. Document Stored in DMS
   ↓
3. Processing Triggered (Async)
   ↓
4. Real-time Status Tracking
   ↓
5. Full Pipeline Processing
   ├─→ OCR (DeepSeek)
   ├─→ Catalog Registration
   ├─→ Training Export
   ├─→ LocalAI Storage
   └─→ Search Indexing
   ↓
6. Intelligence Collection
   ├─→ Domain Detection
   ├─→ Relationship Discovery
   ├─→ Pattern Learning
   └─→ Knowledge Graph Building
   ↓
7. Results Visualization
   ├─→ Processing Dashboard
   ├─→ Results Dashboard
   ├─→ Analytics Dashboard
   └─→ Documents Library
```

## Key Features

### 1. Document Upload

Users can upload documents via:
- **DMS FastAPI Service**: Direct upload via `/documents/` endpoint
- **Browser Shell**: Native UI for file upload and processing
- **API Integration**: Programmatic upload via REST API

### 2. Processing Pipeline

Documents are processed through the full pipeline:
- **OCR**: Image documents processed through DeepSeek OCR
- **Catalog**: Registered in the catalog service with metadata
- **Training**: Exported for ML model training
- **LocalAI**: Stored in domain-aware LocalAI service
- **Search**: Indexed for semantic search

### 3. Intelligence Collection

The system automatically collects intelligence:
- **Domain Detection**: Automatic domain classification
- **Relationship Discovery**: Connections between documents
- **Pattern Learning**: Learned patterns from document structure
- **Knowledge Graph**: Graph representation of document relationships

### 4. Real-time Tracking

Users can track processing in real-time:
- **Status API**: Current processing status
- **Progress Updates**: Step-by-step progress tracking
- **Error Reporting**: Detailed error information with recovery steps
- **Webhook Notifications**: Optional webhook callbacks

### 5. Visualization Dashboards

Beautiful dashboards for exploring results:
- **Processing Dashboard**: Real-time status and progress
- **Results Dashboard**: Intelligence visualization
- **Analytics Dashboard**: Trends and patterns
- **Documents Library**: Browse uploaded documents

## Integration Points

### DMS FastAPI Service

- **Upload**: `POST /documents/`
- **List**: `GET /documents/`
- **Get**: `GET /documents/{document_id}`
- **Status**: `GET /documents/{document_id}/status`
- **Results**: `GET /documents/{document_id}/results`
- **Intelligence**: `GET /documents/{document_id}/intelligence`

### Orchestration Service

- **Process**: `POST /api/dms/process`
- **Status**: `GET /api/dms/status/{request_id}`
- **Results**: `GET /api/dms/results/{request_id}`
- **Intelligence**: `GET /api/dms/results/{request_id}/intelligence`
- **History**: `GET /api/dms/history`
- **Search**: `POST /api/dms/search`
- **Export**: `GET /api/dms/results/{request_id}/export`
- **Batch**: `POST /api/dms/batch`
- **Cancel**: `DELETE /api/dms/jobs/{request_id}`

### Observable Dashboard

- **Landing**: `/dms`
- **Processing**: `/dms-processing`
- **Results**: `/dms-results`
- **Analytics**: `/dms-analytics`
- **Documents**: `/dms-documents`

### Browser Shell

- **Module**: DMS Processing module
- **Views**: Processing, Results, Analytics, Documents
- **Navigation**: Integrated into main navigation

## Design Philosophy

The DMS customer journey follows the **Jobs & Ive lens**:

- **Simplicity**: Clean, focused interfaces
- **Beauty**: Elegant typography, generous whitespace
- **Intuition**: Zero learning curve
- **Delight**: Smooth animations, beautiful interactions

## Getting Started

1. **Upload a Document**: Use the DMS API or Browser Shell
2. **Track Processing**: Monitor real-time status
3. **Explore Results**: View intelligence and relationships
4. **Analyze Trends**: Understand processing performance

## Next Steps

- Enhanced document preview
- Advanced filtering and search
- Collaborative features
- Export and sharing capabilities

