package migrations

// GetAllMigrations returns all registered migrations in order.
func GetAllMigrations() []Migration {
	return []Migration{
		// Migration 1: Initial schema constraints and indexes
		{
			Version:     1,
			Name:        "initial_schema",
			Description: "Create initial constraints and indexes for Node and RELATIONSHIP types",
			Up: `
				// Node constraints
				CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;
				CREATE CONSTRAINT node_id_not_null IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS NOT NULL;
				
				// Node indexes
				CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Node) ON (n.id);
				CREATE INDEX node_type_index IF NOT EXISTS FOR (n:Node) ON (n.type);
				CREATE INDEX node_label_index IF NOT EXISTS FOR (n:Node) ON (n.label);
				CREATE INDEX node_updated_at_index IF NOT EXISTS FOR (n:Node) ON (n.updated_at);
				
				// Relationship indexes
				CREATE INDEX rel_label_index IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.label);
			`,
			Down: `
				// Drop indexes (constraints will be dropped automatically)
				DROP INDEX node_id_index IF EXISTS;
				DROP INDEX node_type_index IF EXISTS;
				DROP INDEX node_label_index IF EXISTS;
				DROP INDEX node_updated_at_index IF EXISTS;
				DROP INDEX rel_label_index IF EXISTS;
			`,
		},

		// Migration 2: BCBS239 schema
		{
			Version:     2,
			Name:        "bcbs239_schema",
			Description: "Add BCBS239 compliance schema with principles, controls, and calculations",
			Up: `
				// BCBS239 Principle constraints
				CREATE CONSTRAINT bcbs239_principle_id IF NOT EXISTS FOR (p:BCBS239Principle) REQUIRE p.principle_id IS UNIQUE;
				
				// BCBS239 Control constraints
				CREATE CONSTRAINT bcbs239_control_id IF NOT EXISTS FOR (c:BCBS239Control) REQUIRE c.control_id IS UNIQUE;
				
				// Regulatory Calculation constraints
				CREATE CONSTRAINT bcbs239_calculation_id IF NOT EXISTS FOR (c:RegulatoryCalculation) REQUIRE c.calculation_id IS UNIQUE;
				
				// Data Asset constraints
				CREATE CONSTRAINT bcbs239_data_asset_id IF NOT EXISTS FOR (d:DataAsset) REQUIRE d.asset_id IS UNIQUE;
				
				// Process constraints
				CREATE CONSTRAINT bcbs239_process_id IF NOT EXISTS FOR (p:Process) REQUIRE p.process_id IS UNIQUE;
				
				// Indexes
				CREATE INDEX bcbs239_principle_area IF NOT EXISTS FOR (p:BCBS239Principle) ON (p.compliance_area);
				CREATE INDEX bcbs239_control_type IF NOT EXISTS FOR (c:BCBS239Control) ON (c.control_type);
				CREATE INDEX bcbs239_calculation_framework IF NOT EXISTS FOR (c:RegulatoryCalculation) ON (c.regulatory_framework);
				CREATE INDEX bcbs239_calculation_date IF NOT EXISTS FOR (c:RegulatoryCalculation) ON (c.calculation_date);
				CREATE INDEX bcbs239_data_asset_type IF NOT EXISTS FOR (d:DataAsset) ON (d.asset_type);
			`,
			Down: `
				DROP INDEX bcbs239_principle_area IF EXISTS;
				DROP INDEX bcbs239_control_type IF EXISTS;
				DROP INDEX bcbs239_calculation_framework IF EXISTS;
				DROP INDEX bcbs239_calculation_date IF EXISTS;
				DROP INDEX bcbs239_data_asset_type IF EXISTS;
			`,
		},

		// Migration 3: Composite indexes for performance
		{
			Version:     3,
			Name:        "composite_indexes",
			Description: "Add composite indexes for common query patterns",
			Up: `
				// Composite index for table lookups (type + label)
				CREATE INDEX node_table_lookup IF NOT EXISTS FOR (n:Node) ON (n.type, n.label);
				
				// Composite index for temporal queries with type filtering
				CREATE INDEX node_type_temporal IF NOT EXISTS FOR (n:Node) ON (n.type, n.updated_at);
				
				// Composite index for relationship label and temporal queries
				CREATE INDEX relationship_label_temporal IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.label, r.updated_at);
				
				// Composite index for BCBS239 principle area and priority
				CREATE INDEX bcbs_principle_area_priority IF NOT EXISTS FOR (p:BCBS239Principle) ON (p.compliance_area, p.priority);
				
				// Composite index for regulatory calculation date and framework
				CREATE INDEX reg_calc_date_framework IF NOT EXISTS FOR (c:RegulatoryCalculation) ON (c.regulatory_framework, c.calculation_date);
				
				// Composite index for regulatory calculation status
				CREATE INDEX reg_calc_status IF NOT EXISTS FOR (c:RegulatoryCalculation) ON (c.status, c.calculation_date);
				
				// Composite index for data asset type
				CREATE INDEX data_asset_type_id IF NOT EXISTS FOR (d:DataAsset) ON (d.asset_type, d.asset_id);
			`,
			Down: `
				DROP INDEX node_table_lookup IF EXISTS;
				DROP INDEX node_type_temporal IF EXISTS;
				DROP INDEX relationship_label_temporal IF EXISTS;
				DROP INDEX bcbs_principle_area_priority IF EXISTS;
				DROP INDEX reg_calc_date_framework IF EXISTS;
				DROP INDEX reg_calc_status IF EXISTS;
				DROP INDEX data_asset_type_id IF EXISTS;
			`,
		},

		// Migration 4: Full-text indexes
		{
			Version:     4,
			Name:        "fulltext_indexes",
			Description: "Add full-text search indexes for labels and descriptions",
			Up: `
				// Full-text index on node labels for search
				CREATE FULLTEXT INDEX node_label_fulltext IF NOT EXISTS
				FOR (n:Node) ON EACH [n.label, n.id];
				
				// Full-text index on BCBS239 principles
				CREATE FULLTEXT INDEX bcbs_principle_fulltext IF NOT EXISTS
				FOR (p:BCBS239Principle) ON EACH [p.principle_name, p.description];
			`,
			Down: `
				DROP INDEX node_label_fulltext IF EXISTS;
				DROP INDEX bcbs_principle_fulltext IF EXISTS;
			`,
		},

		// Migration 5: Agent and domain tracking
		{
			Version:     5,
			Name:        "agent_domain_tracking",
			Description: "Add indexes for agent_id and domain tracking",
			Up: `
				// Index for agent tracking
				CREATE INDEX node_agent_id IF NOT EXISTS FOR (n:Node) ON (n.agent_id);
				CREATE INDEX rel_agent_id IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.agent_id);
				
				// Index for domain filtering
				CREATE INDEX node_domain IF NOT EXISTS FOR (n:Node) ON (n.domain);
				CREATE INDEX rel_domain IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.domain);
			`,
			Down: `
				DROP INDEX node_agent_id IF EXISTS;
				DROP INDEX rel_agent_id IF EXISTS;
				DROP INDEX node_domain IF EXISTS;
				DROP INDEX rel_domain IF EXISTS;
			`,
		},
	}
}

// GetMigration returns a specific migration by version.
func GetMigration(version int) *Migration {
	migrations := GetAllMigrations()
	for _, m := range migrations {
		if m.Version == version {
			return &m
		}
	}
	return nil
}
