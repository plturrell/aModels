# SGMI Training Data

This directory contains SGMI (SAP Global Master Index) dataset files for lineage graph generation and training.

## Required Files

### 1. JSON Tables
- `JSON_with_changes.json` - Table metadata and change history

### 2. Hive DDL Files
Place the following files in the `HIVE DDLS/` subdirectory:
- `sgmisit_all_tables_statement.hql`
- `sgmisitetl_all_tables_statement.hql`
- `sgmisitstg_all_tables_statement.hql`
- `sgmisit_view.hql`

### 3. Control-M XML
- `SGMI-controlm/catalyst migration prod 640.xml` - Control-M job definitions

## Usage

Once all files are in place, run:

```bash
cd extract
./scripts/run_sgmi_full_graph.sh http://localhost:19080/graph
```

This will:
1. Parse Hive DDLs to extract table/view definitions
2. Build view lineage from SQL joins
3. Submit the graph to the graph service
4. Generate view lineage metadata files

## Output

The script generates:
- `sgmi_view_lineage.json` - View registry with columns, joins, metrics
- `sgmi_view_summary.json` - Aggregated lineage metrics

