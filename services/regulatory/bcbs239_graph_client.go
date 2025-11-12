package regulatory

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	graphneo4j "github.com/plturrell/aModels/services/graph/pkg/clients/neo4j"
)

// BCBS239GraphClient provides BCBS 239-specific graph operations.
type BCBS239GraphClient struct {
	driver         neo4j.DriverWithContext
	graphClient    *graphneo4j.Neo4jGraphClient
	queryTemplates *BCBS239QueryTemplates
	logger         *log.Logger
}

// NewBCBS239GraphClient creates a new BCBS 239 graph client.
func NewBCBS239GraphClient(
	driver neo4j.DriverWithContext,
	graphClient *graphneo4j.Neo4jGraphClient,
	logger *log.Logger,
) *BCBS239GraphClient {
	return &BCBS239GraphClient{
		driver:         driver,
		graphClient:    graphClient,
		queryTemplates: &BCBS239QueryTemplates{},
		logger:         logger,
	}
}

// UpsertCalculationWithLineage stores a regulatory calculation and its lineage in Neo4j.
func (c *BCBS239GraphClient) UpsertCalculationWithLineage(
	ctx context.Context,
	calculation RegulatoryCalculation,
	sourceAssets []string,
	controlIDs []string,
) error {
	if c.logger != nil {
		c.logger.Printf("Upserting BCBS 239 calculation %s with lineage to Neo4j", calculation.CalculationID)
	}

	session := c.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		// 1. Create/update the calculation node
		calcCypher := `
			MERGE (calc:RegulatoryCalculation {calculation_id: $calc_id})
			SET calc.calculation_type = $calc_type,
			    calc.calculation_date = datetime($calc_date),
			    calc.report_period = $report_period,
			    calc.result = $result,
			    calc.currency = $currency,
			    calc.regulatory_framework = $framework,
			    calc.source_system = $source_system,
			    calc.status = $status,
			    calc.updated_at = datetime()
		`
		
		calcParams := map[string]interface{}{
			"calc_id":       calculation.CalculationID,
			"calc_type":     calculation.CalculationType,
			"calc_date":     calculation.CalculationDate.Format(time.RFC3339),
			"report_period": calculation.ReportPeriod,
			"result":        calculation.Result,
			"currency":      calculation.Currency,
			"framework":     calculation.RegulatoryFramework,
			"source_system": calculation.SourceSystem,
			"status":        calculation.Status,
		}
		
		if _, err := tx.Run(ctx, calcCypher, calcParams); err != nil {
			return nil, fmt.Errorf("failed to upsert calculation node: %w", err)
		}

		// 2. Create lineage edges to source data assets
		if len(sourceAssets) > 0 {
			lineageCypher := `
				MATCH (calc:RegulatoryCalculation {calculation_id: $calc_id})
				UNWIND $source_assets AS asset_id
				MERGE (asset:DataAsset {asset_id: asset_id})
				MERGE (calc)-[r:DEPENDS_ON]->(asset)
				SET r.created_at = datetime(),
				    r.lineage_type = 'direct'
			`
			
			lineageParams := map[string]interface{}{
				"calc_id":       calculation.CalculationID,
				"source_assets": sourceAssets,
			}
			
			if _, err := tx.Run(ctx, lineageCypher, lineageParams); err != nil {
				return nil, fmt.Errorf("failed to create lineage edges: %w", err)
			}
		}

		// 3. Create validation edges to controls
		if len(controlIDs) > 0 {
			controlCypher := `
				MATCH (calc:RegulatoryCalculation {calculation_id: $calc_id})
				UNWIND $control_ids AS control_id
				MERGE (control:BCBS239Control {control_id: control_id})
				MERGE (calc)-[r:VALIDATED_BY]->(control)
				SET r.validated_at = datetime()
			`
			
			controlParams := map[string]interface{}{
				"calc_id":     calculation.CalculationID,
				"control_ids": controlIDs,
			}
			
			if _, err := tx.Run(ctx, controlCypher, controlParams); err != nil {
				return nil, fmt.Errorf("failed to create control edges: %w", err)
			}
		}

		return nil, nil
	})

	if err != nil {
		return err
	}

	if c.logger != nil {
		c.logger.Printf("Successfully upserted calculation %s with %d source assets and %d controls",
			calculation.CalculationID, len(sourceAssets), len(controlIDs))
	}

	return nil
}

// GetCalculationLineage retrieves the data lineage for a specific calculation.
func (c *BCBS239GraphClient) GetCalculationLineage(
	ctx context.Context,
	calculationID string,
) ([]LineageNode, error) {
	session := c.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	query := c.queryTemplates.GetLineageQuery()
	params := map[string]interface{}{
		"calculation_id": calculationID,
	}

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, params)
		if err != nil {
			return nil, err
		}

		var lineage []LineageNode
		for result.Next(ctx) {
			record := result.Record()
			
			// Extract calculation node
			if calcNode, ok := record.Get("calc"); ok {
				if node, ok := calcNode.(neo4j.Node); ok {
					lineage = append(lineage, LineageNode{
						ID:         node.GetProperties()["calculation_id"].(string),
						Type:       "RegulatoryCalculation",
						Properties: node.GetProperties(),
					})
				}
			}
			
			// Extract asset node
			if assetNode, ok := record.Get("asset"); ok {
				if node, ok := assetNode.(neo4j.Node); ok {
					props := node.GetProperties()
					assetID := ""
					if id, ok := props["asset_id"].(string); ok {
						assetID = id
					}
					lineage = append(lineage, LineageNode{
						ID:         assetID,
						Type:       "DataAsset",
						Properties: props,
					})
				}
			}
		}

		return lineage, result.Err()
	})

	if err != nil {
		return nil, err
	}

	return result.([]LineageNode), nil
}

// GetPrincipleControls retrieves all controls that ensure a specific BCBS 239 principle.
func (c *BCBS239GraphClient) GetPrincipleControls(
	ctx context.Context,
	principleID string,
) ([]ControlMapping, error) {
	session := c.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	query := c.queryTemplates.GetControlMappingQuery()
	params := map[string]interface{}{
		"principle_id": principleID,
	}

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, params)
		if err != nil {
			return nil, err
		}

		var controls []ControlMapping
		for result.Next(ctx) {
			record := result.Record()
			
			var mapping ControlMapping
			
			if principleNode, ok := record.Get("principle"); ok {
				if node, ok := principleNode.(neo4j.Node); ok {
					mapping.PrincipleID = node.GetProperties()["principle_id"].(string)
					if name, ok := node.GetProperties()["principle_name"].(string); ok {
						mapping.PrincipleName = name
					}
				}
			}
			
			if controlNode, ok := record.Get("control"); ok {
				if node, ok := controlNode.(neo4j.Node); ok {
					mapping.ControlID = node.GetProperties()["control_id"].(string)
					if name, ok := node.GetProperties()["control_name"].(string); ok {
						mapping.ControlName = name
					}
					if ctype, ok := node.GetProperties()["control_type"].(string); ok {
						mapping.ControlType = ctype
					}
				}
			}
			
			if targetNode, ok := record.Get("target"); ok {
				if node, ok := targetNode.(neo4j.Node); ok {
					mapping.TargetProperties = node.GetProperties()
				}
			}
			
			controls = append(controls, mapping)
		}

		return controls, result.Err()
	})

	if err != nil {
		return nil, err
	}

	return result.([]ControlMapping), nil
}

// AnalyzeCompliancePath retrieves the full compliance path for a principle.
func (c *BCBS239GraphClient) AnalyzeCompliancePath(
	ctx context.Context,
	principleID string,
) (*CompliancePathAnalysis, error) {
	session := c.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	query := c.queryTemplates.GetCompliancePathQuery()
	params := map[string]interface{}{
		"principle_id": principleID,
	}

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, params)
		if err != nil {
			return nil, err
		}

		analysis := &CompliancePathAnalysis{
			PrincipleID: principleID,
			Paths:       []CompliancePath{},
		}

		for result.Next(ctx) {
			record := result.Record()
			
			path := CompliancePath{
				Nodes: []map[string]interface{}{},
			}
			
			// Extract all nodes from the path
			nodeKeys := []string{"principle", "control", "process", "asset", "calc"}
			for _, key := range nodeKeys {
				if nodeVal, ok := record.Get(key); ok {
					if node, ok := nodeVal.(neo4j.Node); ok {
						path.Nodes = append(path.Nodes, node.GetProperties())
					}
				}
			}
			
			analysis.Paths = append(analysis.Paths, path)
		}

		return analysis, result.Err()
	})

	if err != nil {
		return nil, err
	}

	return result.(*CompliancePathAnalysis), nil
}

// GetNonCompliantAreas identifies areas lacking proper controls or coverage.
func (c *BCBS239GraphClient) GetNonCompliantAreas(ctx context.Context) ([]NonCompliantArea, error) {
	session := c.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	query := c.queryTemplates.GetNonCompliantAreasQuery()

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, nil)
		if err != nil {
			return nil, err
		}

		var areas []NonCompliantArea
		for result.Next(ctx) {
			record := result.Record()
			
			area := NonCompliantArea{}
			if principleID, ok := record.Get("principle_id"); ok {
				area.PrincipleID = principleID.(string)
			}
			if principleName, ok := record.Get("principle_name"); ok {
				area.PrincipleName = principleName.(string)
			}
			if complianceArea, ok := record.Get("compliance_area"); ok {
				area.ComplianceArea = complianceArea.(string)
			}
			if issue, ok := record.Get("issue"); ok {
				area.Issue = issue.(string)
			}
			
			areas = append(areas, area)
		}

		return areas, result.Err()
	})

	if err != nil {
		return nil, err
	}

	return result.([]NonCompliantArea), nil
}

// AnalyzeDataAssetImpact analyzes the impact of changes to a data asset.
func (c *BCBS239GraphClient) AnalyzeDataAssetImpact(
	ctx context.Context,
	assetID string,
) (*ImpactAnalysis, error) {
	session := c.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	query := c.queryTemplates.GetImpactAnalysisQuery()
	params := map[string]interface{}{
		"asset_id": assetID,
	}

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, params)
		if err != nil {
			return nil, err
		}

		analysis := &ImpactAnalysis{
			AssetID:            assetID,
			ImpactedPrinciples: []PrincipleImpact{},
		}

		for result.Next(ctx) {
			record := result.Record()
			
			impact := PrincipleImpact{}
			if principleID, ok := record.Get("impacted_principle"); ok {
				impact.PrincipleID = principleID.(string)
			}
			if principleName, ok := record.Get("principle_name"); ok {
				impact.PrincipleName = principleName.(string)
			}
			if count, ok := record.Get("affected_calculations"); ok {
				impact.AffectedCalculations = int(count.(int64))
			}
			if calcIDs, ok := record.Get("calculation_ids"); ok {
				if ids, ok := calcIDs.([]interface{}); ok {
					for _, id := range ids {
						impact.CalculationIDs = append(impact.CalculationIDs, id.(string))
					}
				}
			}
			
			analysis.ImpactedPrinciples = append(analysis.ImpactedPrinciples, impact)
		}

		return analysis, result.Err()
	})

	if err != nil {
		return nil, err
	}

	return result.(*ImpactAnalysis), nil
}

// LineageNode represents a node in the data lineage graph.
type LineageNode struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties"`
}

// ControlMapping represents a mapping between a principle and its controls.
type ControlMapping struct {
	PrincipleID      string                 `json:"principle_id"`
	PrincipleName    string                 `json:"principle_name"`
	ControlID        string                 `json:"control_id"`
	ControlName      string                 `json:"control_name"`
	ControlType      string                 `json:"control_type"`
	TargetProperties map[string]interface{} `json:"target_properties"`
}

// CompliancePathAnalysis represents the analysis of compliance paths.
type CompliancePathAnalysis struct {
	PrincipleID string            `json:"principle_id"`
	Paths       []CompliancePath  `json:"paths"`
}

// CompliancePath represents a single compliance path through the graph.
type CompliancePath struct {
	Nodes []map[string]interface{} `json:"nodes"`
}

// NonCompliantArea represents an area that lacks proper controls.
type NonCompliantArea struct {
	PrincipleID    string `json:"principle_id"`
	PrincipleName  string `json:"principle_name"`
	ComplianceArea string `json:"compliance_area"`
	Issue          string `json:"issue"`
}

// ImpactAnalysis represents the impact analysis of a data asset change.
type ImpactAnalysis struct {
	AssetID            string             `json:"asset_id"`
	ImpactedPrinciples []PrincipleImpact  `json:"impacted_principles"`
}

// PrincipleImpact represents the impact on a specific principle.
type PrincipleImpact struct {
	PrincipleID          string   `json:"principle_id"`
	PrincipleName        string   `json:"principle_name"`
	AffectedCalculations int      `json:"affected_calculations"`
	CalculationIDs       []string `json:"calculation_ids"`
}
