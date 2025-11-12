# Extract Service Scripts

This directory contains scripts organized by function for the Extract service.

## Directory Structure

```
scripts/
├── pipelines/              # Pipeline orchestration scripts
│   ├── run_sgmi_complete_pipeline.sh    # Main SGMI pipeline (complete workflow)
│   ├── run_sgmi_etl_automated.sh        # ETL automation (used by main pipeline)
│   ├── run_sgmi_pipeline_from_docker.sh # Docker wrapper for pipeline
│   ├── sgmi_view_builder.py            # View builder utility
│   └── README_SGMI_COMPLETE_PIPELINE.md # Pipeline documentation
│
├── embeddings/            # Embedding generation scripts
│   ├── embed.py           # Unified embedding (SQL, tables, columns, jobs, etc.)
│   ├── embed_sap_rpt.py   # SAP RPT semantic embeddings
│   ├── embed_sap_rpt_batch.py  # Batch SAP RPT embeddings
│   └── unified_multimodal_extraction.py  # Unified multi-modal extractor
│
├── classification/        # Classification scripts
│   └── sap_rpt_classifier.py  # Unified SAP RPT classifier (basic, full, multi-task)
│
├── services/              # Service management scripts
│   ├── start_extract.sh
│   ├── start_extract_service.sh
│   ├── wait_for_service.sh
│   └── check_extract_env.sh
│
├── utils/                 # Utility scripts
│   ├── deepseek_ocr_cli.py
│   └── parse_hive_ddl.py
│
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Migration from Old Structure

The scripts have been reorganized from a flat structure. Key changes:

### Merged Scripts

1. **Pipeline Scripts**: 
   - `run_sgmi_full_graph.sh` and `run_sgmi_pipeline_graph.sh` functionality merged into `pipelines/run_sgmi_etl_automated.sh`
   - All pipeline scripts now in `pipelines/` directory

2. **Embedding Scripts**:
   - `embed_sql.py` functionality merged into `embeddings/embed.py`
   - Use `--artifact-type sql` with `embeddings/embed.py` instead

3. **SAP RPT Classification**:
   - `classify_table_sap_rpt.py`, `classify_table_sap_rpt_full.py`, and `sap_rpt_advanced.py` consolidated into `classification/sap_rpt_classifier.py`
   - Use `--mode basic|full|multi-task` to select functionality

### Updated Paths

When calling scripts, use the new directory structure:

```bash
# Old way (deprecated)
./scripts/run_sgmi_complete_pipeline.sh

# New way
./scripts/pipelines/run_sgmi_complete_pipeline.sh
```

## Usage Examples

### Running SGMI Pipeline

```bash
# Complete pipeline (structured data + documents)
cd /home/aModels/services/extract
./scripts/pipelines/run_sgmi_complete_pipeline.sh

# ETL only (structured data)
./scripts/pipelines/run_sgmi_etl_automated.sh
```

### Generating Embeddings

```bash
# SQL embedding (merged from embed_sql.py)
./scripts/embeddings/embed.py --artifact-type sql --sql "SELECT * FROM table"

# Table embedding
./scripts/embeddings/embed.py --artifact-type table --table-name "users" --columns '[{"name":"id","type":"int"}]'

# SAP RPT semantic embedding
./scripts/embeddings/embed_sap_rpt.py --table-name "users" --columns '[{"name":"id"}]'
```

### Classifying Tables

```bash
# Basic classification (pattern-based)
./scripts/classification/sap_rpt_classifier.py --mode basic --table-name "users" --columns '[{"name":"id"}]'

# Full ML-based classification
./scripts/classification/sap_rpt_classifier.py --mode full --table-name "users" --columns '[{"name":"id"}]' --training-data /path/to/training.json

# Multi-task (classification + quality score)
./scripts/classification/sap_rpt_classifier.py --mode multi-task --table-name "users" --columns '[{"name":"id"}]' --training-data /path/to/training.json
```

### Service Management

```bash
# Start extract service
./scripts/services/start_extract.sh

# Start extract service with dependencies
./scripts/services/start_extract_service.sh

# Check environment
./scripts/services/check_extract_env.sh .env
```

## Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- All scripts maintain backward compatibility where possible
- Path references have been updated to work with the new structure
- Legacy scripts have been removed after consolidation
- See individual script documentation for detailed usage

