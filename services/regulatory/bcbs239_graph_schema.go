package regulatory

import (
	"context"
	"fmt"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// BCBS239GraphSchema defines the knowledge graph schema for BCBS 239 compliance.
// This schema models the relationships between:
// - BCBS239 Principles (14 principles for risk data aggregation and reporting)
// - Controls (mechanisms ensuring principle compliance)
// - Data Assets (tables, reports, fields)
// - Calculations (regulatory metrics)
// - Processes (business workflows)
type BCBS239GraphSchema struct {
	driver neo4j.DriverWithContext
}

// NewBCBS239GraphSchema creates a new BCBS 239 graph schema manager.
func NewBCBS239GraphSchema(driver neo4j.DriverWithContext) *BCBS239GraphSchema {
	return &BCBS239GraphSchema{
		driver: driver,
	}
}

// InitializeSchema initializes the Neo4j schema with BCBS 239 specific constraints and indexes.
// NOTE: Schema definitions have been centralized in infrastructure/neo4j/schemas/domain/regulatory/001_bcbs239_schema.cypher
// This function maintains backward compatibility but should reference the centralized schema file.
func (s *BCBS239GraphSchema) InitializeSchema(ctx context.Context) error {
	session := s.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	// Define schema initialization queries
	// See: infrastructure/neo4j/schemas/domain/regulatory/001_bcbs239_schema.cypher
	queries := []string{
		// Constraints for unique identifiers
		"CREATE CONSTRAINT bcbs239_principle_id IF NOT EXISTS FOR (p:BCBS239Principle) REQUIRE p.principle_id IS UNIQUE",
		"CREATE CONSTRAINT bcbs239_control_id IF NOT EXISTS FOR (c:BCBS239Control) REQUIRE c.control_id IS UNIQUE",
		"CREATE CONSTRAINT bcbs239_calculation_id IF NOT EXISTS FOR (c:RegulatoryCalculation) REQUIRE c.calculation_id IS UNIQUE",
		"CREATE CONSTRAINT bcbs239_data_asset_id IF NOT EXISTS FOR (d:DataAsset) REQUIRE d.asset_id IS UNIQUE",
		"CREATE CONSTRAINT bcbs239_process_id IF NOT EXISTS FOR (p:Process) REQUIRE p.process_id IS UNIQUE",
		
		// Indexes for efficient queries
		"CREATE INDEX bcbs239_principle_area IF NOT EXISTS FOR (p:BCBS239Principle) ON (p.compliance_area)",
		"CREATE INDEX bcbs239_control_type IF NOT EXISTS FOR (c:BCBS239Control) ON (c.control_type)",
		"CREATE INDEX bcbs239_calculation_framework IF NOT EXISTS FOR (c:RegulatoryCalculation) ON (c.regulatory_framework)",
		"CREATE INDEX bcbs239_calculation_date IF NOT EXISTS FOR (c:RegulatoryCalculation) ON (c.calculation_date)",
		"CREATE INDEX bcbs239_data_asset_type IF NOT EXISTS FOR (d:DataAsset) ON (d.asset_type)",
	}

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		for _, query := range queries {
			if _, err := tx.Run(ctx, query, nil); err != nil {
				// Ignore "already exists" errors, fail on actual errors
				return nil, fmt.Errorf("failed to execute schema query: %w", err)
			}
		}
		return nil, nil
	})

	return err
}

// SeedBCBS239Principles seeds the 14 BCBS 239 principles into Neo4j.
func (s *BCBS239GraphSchema) SeedBCBS239Principles(ctx context.Context) error {
	session := s.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	principles := []map[string]interface{}{
		// Overarching Governance and Infrastructure
		{
			"principle_id":    "P1",
			"principle_name":  "Governance",
			"compliance_area": "Governance and Infrastructure",
			"description":     "A bank's risk data aggregation capabilities and risk reporting practices should be subject to strong governance arrangements.",
			"priority":        "critical",
		},
		{
			"principle_id":    "P2",
			"principle_name":  "Data Architecture and IT Infrastructure",
			"compliance_area": "Governance and Infrastructure",
			"description":     "A bank should design, build and maintain data architecture and IT infrastructure which fully supports risk data aggregation capabilities and risk reporting practices.",
			"priority":        "critical",
		},
		
		// Risk Data Aggregation Capabilities (Principles 3-8)
		{
			"principle_id":    "P3",
			"principle_name":  "Accuracy and Integrity",
			"compliance_area": "Risk Data Aggregation",
			"description":     "A bank should be able to generate accurate and reliable risk data to meet normal and stress/crisis reporting accuracy requirements.",
			"priority":        "critical",
		},
		{
			"principle_id":    "P4",
			"principle_name":  "Completeness",
			"compliance_area": "Risk Data Aggregation",
			"description":     "A bank should be able to capture and aggregate all material risk data across the banking group.",
			"priority":        "critical",
		},
		{
			"principle_id":    "P5",
			"principle_name":  "Timeliness",
			"compliance_area": "Risk Data Aggregation",
			"description":     "A bank should be able to generate aggregate and up-to-date risk data in a timely manner.",
			"priority":        "high",
		},
		{
			"principle_id":    "P6",
			"principle_name":  "Adaptability",
			"compliance_area": "Risk Data Aggregation",
			"description":     "A bank should be able to generate aggregate risk data to meet a broad range of on-demand, ad hoc risk management reporting requests.",
			"priority":        "high",
		},
		
		// Risk Reporting Practices (Principles 7-11)
		{
			"principle_id":    "P7",
			"principle_name":  "Accuracy",
			"compliance_area": "Risk Reporting",
			"description":     "Risk management reports should accurately and precisely convey aggregated risk data and reflect risk in an exact manner.",
			"priority":        "critical",
		},
		{
			"principle_id":    "P8",
			"principle_name":  "Comprehensiveness",
			"compliance_area": "Risk Reporting",
			"description":     "Risk management reports should cover all material risk areas within the organization.",
			"priority":        "high",
		},
		{
			"principle_id":    "P9",
			"principle_name":  "Clarity and Usefulness",
			"compliance_area": "Risk Reporting",
			"description":     "Risk management reports should be clear and concise, easy to understand, yet comprehensive enough.",
			"priority":        "medium",
		},
		{
			"principle_id":    "P10",
			"principle_name":  "Frequency",
			"compliance_area": "Risk Reporting",
			"description":     "Risk management reports should be distributed to relevant parties with the appropriate frequency.",
			"priority":        "high",
		},
		{
			"principle_id":    "P11",
			"principle_name":  "Distribution",
			"compliance_area": "Risk Reporting",
			"description":     "Risk management reports should be distributed to relevant parties while ensuring confidentiality.",
			"priority":        "medium",
		},
		
		// Supervisory Review (Principles 12-14)
		{
			"principle_id":    "P12",
			"principle_name":  "Supervisory Reporting",
			"compliance_area": "Supervisory Review",
			"description":     "Banks should have the capability to produce supervisory reports that are accurate and complete.",
			"priority":        "critical",
		},
		{
			"principle_id":    "P13",
			"principle_name":  "Remediation Plans",
			"compliance_area": "Supervisory Review",
			"description":     "Supervisors should have the ability to require banks to develop remediation plans.",
			"priority":        "high",
		},
		{
			"principle_id":    "P14",
			"principle_name":  "Home-Host Coordination",
			"compliance_area": "Supervisory Review",
			"description":     "Supervisors should have the ability to share information regarding risk data aggregation and reporting.",
			"priority":        "medium",
		},
	}

	cypher := `
		UNWIND $principles AS principle
		MERGE (p:BCBS239Principle {principle_id: principle.principle_id})
		SET p.principle_name = principle.principle_name,
		    p.compliance_area = principle.compliance_area,
		    p.description = principle.description,
		    p.priority = principle.priority,
		    p.framework = 'BCBS 239',
		    p.updated_at = datetime()
	`

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		_, err := tx.Run(ctx, cypher, map[string]interface{}{
			"principles": principles,
		})
		return nil, err
	})

	return err
}

// BCBS239QueryTemplates provides Cypher query templates for BCBS 239 compliance analysis.
type BCBS239QueryTemplates struct{}

// GetLineageQuery returns a Cypher query to trace data lineage for a specific calculation.
func (qt *BCBS239QueryTemplates) GetLineageQuery() string {
	return `
		MATCH path = (calc:RegulatoryCalculation {calculation_id: $calculation_id})
		             -[:DEPENDS_ON|SOURCE_FROM|DERIVED_FROM*1..5]->(asset:DataAsset)
		RETURN path, calc, asset
		ORDER BY length(path) DESC
		LIMIT 100
	`
}

// GetControlMappingQuery returns a Cypher query to find controls ensuring a specific principle.
func (qt *BCBS239QueryTemplates) GetControlMappingQuery() string {
	return `
		MATCH (principle:BCBS239Principle {principle_id: $principle_id})
		      -[:ENSURED_BY]->(control:BCBS239Control)
		      -[:APPLIES_TO]->(target)
		RETURN principle, control, target
	`
}

// GetCompliancePathQuery returns a Cypher query to find full compliance paths.
func (qt *BCBS239QueryTemplates) GetCompliancePathQuery() string {
	return `
		MATCH path = (principle:BCBS239Principle {principle_id: $principle_id})
		             -[:ENSURED_BY]->(control:BCBS239Control)
		             -[:APPLIES_TO]->(process:Process)
		             -[:TRANSFORMS]->(asset:DataAsset)
		             <-[:DEPENDS_ON]-(calc:RegulatoryCalculation)
		WHERE calc.regulatory_framework = 'BCBS 239'
		RETURN path, principle, control, process, asset, calc
		LIMIT 50
	`
}

// GetNonCompliantAreasQuery returns a Cypher query to identify non-compliant areas.
func (qt *BCBS239QueryTemplates) GetNonCompliantAreasQuery() string {
	return `
		MATCH (principle:BCBS239Principle)
		WHERE NOT EXISTS {
			MATCH (principle)-[:ENSURED_BY]->(:BCBS239Control)-[:APPLIES_TO]->()
		}
		RETURN principle.principle_id AS principle_id,
		       principle.principle_name AS principle_name,
		       principle.compliance_area AS compliance_area,
		       'missing_controls' AS issue
	`
}

// GetImpactAnalysisQuery returns a Cypher query to analyze the impact of changes.
func (qt *BCBS239QueryTemplates) GetImpactAnalysisQuery() string {
	return `
		MATCH (asset:DataAsset {asset_id: $asset_id})
		      <-[:DEPENDS_ON|SOURCE_FROM*1..5]-(calc:RegulatoryCalculation)
		      <-[:VALIDATED_BY]-(control:BCBS239Control)
		      <-[:ENSURED_BY]-(principle:BCBS239Principle)
		WHERE calc.regulatory_framework = 'BCBS 239'
		RETURN principle.principle_id AS impacted_principle,
		       principle.principle_name AS principle_name,
		       count(DISTINCT calc) AS affected_calculations,
		       collect(DISTINCT calc.calculation_id) AS calculation_ids
		ORDER BY affected_calculations DESC
	`
}

// GetDataQualityMetricsQuery returns a Cypher query for data quality metrics.
func (qt *BCBS239QueryTemplates) GetDataQualityMetricsQuery() string {
	return `
		MATCH (calc:RegulatoryCalculation)
		WHERE calc.regulatory_framework = 'BCBS 239'
		  AND calc.calculation_date >= datetime($start_date)
		  AND calc.calculation_date <= datetime($end_date)
		WITH calc.status AS status, count(*) AS count
		RETURN status,
		       count,
		       round(100.0 * count / sum(count)) AS percentage
		ORDER BY count DESC
	`
}

// BCBS239GraphNode represents a node in the BCBS 239 compliance graph.
type BCBS239GraphNode struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"` // "Principle", "Control", "DataAsset", "Process", "Calculation"
	Properties map[string]interface{} `json:"properties"`
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at"`
}

// BCBS239GraphEdge represents a relationship in the BCBS 239 compliance graph.
type BCBS239GraphEdge struct {
	ID         string                 `json:"id"`
	SourceID   string                 `json:"source_id"`
	TargetID   string                 `json:"target_id"`
	Type       string                 `json:"type"` // "ENSURED_BY", "APPLIES_TO", "DEPENDS_ON", "TRANSFORMS", "VALIDATED_BY"
	Properties map[string]interface{} `json:"properties"`
	CreatedAt  time.Time              `json:"created_at"`
}
