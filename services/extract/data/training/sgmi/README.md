# SGMI Training Data

This directory contains SGMI (SAP Global Master Index) dataset files for lineage graph generation and training.

## Directory Structure

```
data/training/sgmi/
├── json_with_changes.json          # Table metadata and change history
├── hive-ddl/                       # Hive DDL files
│   ├── sgmisit_all_tables_statement.hql
│   ├── sgmisitetl_all_tables_statement.hql
│   ├── sgmisitstg_all_tables_statement.hql
│   └── sgmisit_view.hql
└── sgmi-controlm/                  # Control-M job definitions
    └── catalyst migration prod 640.xml
```

## Required Files

1. **JSON Tables**: `json_with_changes.json` - Table metadata and change history
2. **Hive DDL Files** (in `hive-ddl/`):
   - `sgmisit_all_tables_statement.hql`
   - `sgmisitetl_all_tables_statement.hql`
   - `sgmisitstg_all_tables_statement.hql`
   - `sgmisit_view.hql`
3. **Control-M XML**: `sgmi-controlm/catalyst migration prod 640.xml`

## Usage

Once all files are in place, run:

```bash
cd services/extract
./scripts/run_sgmi_full_graph.sh http://localhost:19080/graph
```

This will:
1. Parse Hive DDLs to extract table/view definitions
2. Build view lineage from SQL joins
3. Submit the graph to the graph service (Neo4j/Graph)
4. Generate view lineage metadata files

## Output

The script generates:
- `sgmi_view_lineage.json` - View registry with columns, joins, metrics
- `sgmi_view_summary.json` - Aggregated lineage metrics

These are stored in the default view store location or where `SGMI_VIEW_REGISTRY_OUT` points.
