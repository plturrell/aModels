# SGMI Complete Data Pipeline

This script processes **all** SGMI data through the complete data pipeline:

1. **Structured Data** → Extract Service → Knowledge Graph
2. **Documents** → DMS Service → Extract/Catalog → Knowledge Graph
3. **Verification** → Knowledge Graph & Catalog Integration

## What Gets Processed

### Structured Data (via Extract Service)
- `json_with_changes.json` - Table metadata and change history
- Hive DDL files (`sgmisit_*.hql`, `sgmisitetl_*.hql`, `sgmisitstg_*.hql`, `sgmisit_view.hql`)
- Control-M XML files (`catalyst migration prod 640.xml`)

### Documents (via DMS Service)
- **Control-M XML files** - Job definitions
- **Excel files** (`.xlsx`) - Migration plans, implementation plans, etc.
- **Word documents** (`.docx`) - Release notes, documentation
- **Hive DDL files** - Uploaded as reference documents
- **JSON metadata** - Uploaded as reference document

## Prerequisites

1. **Services Running:**
   - Extract Service (default: `http://localhost:8083`)
   - DMS Service (default: `http://localhost:8080`)
   - Catalog Service (default: `http://localhost:8084`) - Optional
   - Knowledge Graph (Neo4j) - Accessible via Extract Service

2. **Data Files:**
   - All SGMI data files in `/home/aModels/data/training/sgmi/`

## Usage

```bash
cd /home/aModels/services/extract
./scripts/pipelines/run_sgmi_complete_pipeline.sh
```

### Custom Service URLs

```bash
EXTRACT_SERVICE_URL=http://extract:8083 \
DMS_SERVICE_URL=http://dms:8080 \
CATALOG_SERVICE_URL=http://catalog:8084 \
./scripts/pipelines/run_sgmi_complete_pipeline.sh
```

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    SGMI Complete Pipeline                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │   Step 1: Structured Data           │
        │   (Extract Service)                 │
        │   • JSON tables                      │
        │   • Hive DDLs                        │
        │   • Control-M XML                    │
        └──────────────┬────────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │   Step 2: Documents                 │
        │   (DMS Service)                     │
        │   • Excel files                     │
        │   • Word documents                  │
        │   • XML files                       │
        │   • DDL files (reference)           │
        └──────────────┬────────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │   Step 3: Knowledge Graph           │
        │   Verification                      │
        │   • Query SGMI nodes                │
        │   • Count entities                   │
        └──────────────┬────────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────────┐
        │   Step 4: Catalog                    │
        │   Verification                      │
        │   • Query data elements              │
        │   • Verify integration               │
        └─────────────────────────────────────┘
```

## Integration Points

### Extract Service
- **Endpoint**: `POST /knowledge-graph`
- **Processes**: JSON tables, Hive DDLs, Control-M XML
- **Output**: Knowledge graph nodes and edges in Neo4j

### DMS Service
- **Endpoint**: `POST /documents/`
- **Processes**: All document files (Excel, Word, XML, etc.)
- **Integration**: 
  - Automatically calls Extract service for OCR/extraction
  - Automatically registers in Catalog service
  - Creates relationships in knowledge graph

### Catalog Service
- **Endpoint**: `GET /catalog/data-elements`
- **Purpose**: Metadata registry and semantic search
- **Integration**: Receives data from DMS and Extract services

## Expected Output

```
[2025-11-12 13:30:00] ==========================================
[2025-11-12 13:30:00] SGMI Complete Data Pipeline
[2025-11-12 13:30:00] ==========================================
[2025-11-12 13:30:00] 
[2025-11-12 13:30:00] Checking services...
[2025-11-12 13:30:00]   ✓ Extract Service is healthy
[2025-11-12 13:30:00]   ✓ DMS Service is healthy
[2025-11-12 13:30:00]   ✓ Catalog Service is healthy
[2025-11-12 13:30:00] 
[2025-11-12 13:30:00] ==========================================
[2025-11-12 13:30:00] Step 1: Processing Structured Data
[2025-11-12 13:30:00] ==========================================
[2025-11-12 13:30:05] ✓ Structured data processed successfully
[2025-11-12 13:30:05] 
[2025-11-12 13:30:05] ==========================================
[2025-11-12 13:30:05] Step 2: Processing Documents
[2025-11-12 13:30:05] ==========================================
[2025-11-12 13:30:10]   ✓ Uploaded catalyst migration prod 640.xml
[2025-11-12 13:30:15]   ✓ Uploaded 50896 SGMI Migration RunBook Plan_SAMPLE.xlsx
[2025-11-12 13:30:20]   ✓ Uploaded 50896_SGMI_Release_Notes_updated.docx
[2025-11-12 13:30:25] Document upload summary: 8 uploaded, 0 failed
[2025-11-12 13:30:25] 
[2025-11-12 13:30:25] ==========================================
[2025-11-12 13:30:25] Step 3: Verifying Knowledge Graph
[2025-11-12 13:30:25] ==========================================
[2025-11-12 13:30:25]   ✓ Found 1234 SGMI nodes in knowledge graph
[2025-11-12 13:30:25] 
[2025-11-12 13:30:25] ==========================================
[2025-11-12 13:30:25] Step 4: Verifying Catalog
[2025-11-12 13:30:25] ==========================================
[2025-11-12 13:30:25]   ✓ Found 567 data elements in catalog
[2025-11-12 13:30:25] 
[2025-11-12 13:30:25] ==========================================
[2025-11-12 13:30:25] SGMI Complete Pipeline Finished!
[2025-11-12 13:30:25] ==========================================
```

## Troubleshooting

### Service Not Available
If a service is not available, the script will:
- **Extract Service**: Exit with error (required)
- **DMS Service**: Skip document upload (warning)
- **Catalog Service**: Skip verification (warning)

### Document Upload Failures
- Check DMS service logs
- Verify file permissions
- Check DMS storage configuration

### Knowledge Graph Verification Fails
- Verify Neo4j is accessible via Extract service
- Check Extract service logs
- Verify data was actually processed

## Related Scripts

- `run_sgmi_etl_automated.sh` - Processes only structured data (in same directory)
- Legacy scripts `run_sgmi_full_graph.sh` and `run_sgmi_pipeline_graph.sh` have been merged into `run_sgmi_etl_automated.sh`

