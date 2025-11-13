# Neo4j Relationship Type Definitions

This directory contains documentation and definitions for all relationship types used in the Neo4j knowledge graph.

## Relationship Structure

All relationships in the graph share a common base structure:

### Base Relationship Type: `RELATIONSHIP`

**Required Properties:**
- `label` (String): Relationship type (see Relationship Types below)
- `properties_json` (String): JSON string containing all relationship-specific properties
- `updated_at` (String): ISO 8601 timestamp of last update

**Optional Properties:**
- `agent_id` (String): Agent identifier (if applicable)
- `domain` (String): Domain identifier (if applicable)

## Relationship Categories

### Data Lineage Relationships (`data_lineage.cypher`)
- **DATA_FLOW** - Represents data flow between columns, capturing ETL transformation logic
- **HAS_COLUMN** - Represents table-to-column containment
- **CONTAINS** - Represents database-to-table/view containment
- **REFERENCES** - Represents foreign key relationships

### Workflow Relationships (`workflow.cypher`)
- **SCHEDULES** - Represents calendar-to-job scheduling relationship
- **BLOCKS** - Represents condition-to-job blocking relationship (input conditions)
- **RELEASES** - Represents job-to-condition release relationship (output conditions)
- **HAS_PETRI_NET** - Represents root-to-Petri net relationship

### Compliance Relationships (`compliance.cypher`)
- **ENSURED_BY** - Links BCBS239 principles to controls
- **APPLIES_TO** - Links controls to targets (processes, data assets)
- **DEPENDS_ON** - Links calculations to data assets
- **SOURCE_FROM** - Links calculations to source data assets
- **DERIVED_FROM** - Links derived calculations to source calculations
- **TRANSFORMS** - Links processes to data assets they transform
- **VALIDATED_BY** - Links calculations to validating controls

## Usage

These relationship definitions are for documentation and reference. The actual relationships are created dynamically when data is loaded into the graph.

For detailed property definitions and examples, see the individual `.cypher` files in this directory.

## Related Documentation

- See `../../docs/neo4j_graph_schema.md` for complete schema documentation
- See `../MIGRATION_MANIFEST.md` for schema execution order

