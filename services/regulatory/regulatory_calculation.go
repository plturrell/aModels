package regulatory

import (
	"context"
	"fmt"
	"log"
	"time"
)

// RegulatoryCalculationEngine performs regulatory calculations.
type RegulatoryCalculationEngine struct {
	logger           *log.Logger
	bcbs239GraphClient *BCBS239GraphClient // Optional: for automatic graph lineage tracking
}

// NewRegulatoryCalculationEngine creates a new regulatory calculation engine.
func NewRegulatoryCalculationEngine(logger *log.Logger) *RegulatoryCalculationEngine {
	return &RegulatoryCalculationEngine{
		logger: logger,
	}
}

// WithBCBS239GraphClient adds Neo4j graph tracking capability to the engine.
func (rce *RegulatoryCalculationEngine) WithBCBS239GraphClient(graphClient *BCBS239GraphClient) *RegulatoryCalculationEngine {
	rce.bcbs239GraphClient = graphClient
	return rce
}

// RegulatoryCalculationRequest represents a request for regulatory calculations.
type RegulatoryCalculationRequest struct {
	Framework    string
	ReportPeriod string
	Metrics      []string
}

// RegulatoryCalculation represents a regulatory calculation result.
type RegulatoryCalculation struct {
	CalculationID     string
	CalculationType   string
	CalculationDate   time.Time
	ReportPeriod      string
	Result            float64
	Currency          string
	RegulatoryFramework string
	SourceSystem      string
	Status            string
}

// CalculateRegulatoryMetrics calculates regulatory metrics for a framework.
func (rce *RegulatoryCalculationEngine) CalculateRegulatoryMetrics(ctx context.Context, req RegulatoryCalculationRequest) ([]RegulatoryCalculation, error) {
	if rce.logger != nil {
		rce.logger.Printf("Calculating regulatory metrics for %s (period: %s)", req.Framework, req.ReportPeriod)
	}

	var calculations []RegulatoryCalculation

	// Calculate metrics based on framework
	switch req.Framework {
	case "MAS 610":
		calculations = rce.calculateMAS610Metrics(ctx, req)
	case "BCBS 239":
		calculations = rce.calculateBCBS239Metrics(ctx, req)
	default:
		calculations = rce.calculateGenericMetrics(ctx, req)
	}

	if rce.logger != nil {
		rce.logger.Printf("Calculated %d regulatory metrics", len(calculations))
	}

	// Emit to Neo4j if BCBS 239 graph client is available
	if req.Framework == "BCBS 239" && rce.bcbs239GraphClient != nil {
		if err := rce.emitBCBS239ToGraph(ctx, calculations); err != nil {
			if rce.logger != nil {
				rce.logger.Printf("Warning: Failed to emit BCBS 239 calculations to Neo4j: %v", err)
			}
			// Don't fail the calculation if graph persistence fails
		}
	}

	return calculations, nil
}

// emitBCBS239ToGraph persists BCBS 239 calculations to the knowledge graph.
func (rce *RegulatoryCalculationEngine) emitBCBS239ToGraph(ctx context.Context, calculations []RegulatoryCalculation) error {
	for _, calc := range calculations {
		// Define source assets and controls based on calculation type
		var sourceAssets []string
		var controlIDs []string
		
		switch calc.CalculationType {
		case "risk_data_aggregation":
			// Example: Link to data quality controls and source systems
			sourceAssets = []string{
				"asset-bcrs-risk-data-warehouse",
				"asset-bcrs-trade-repository",
			}
			controlIDs = []string{
				"control-data-accuracy-p3",
				"control-data-completeness-p4",
				"control-data-timeliness-p5",
			}
		case "accuracy_validation":
			sourceAssets = []string{
				"asset-bcrs-validation-rules",
			}
			controlIDs = []string{
				"control-data-accuracy-p3",
			}
		case "completeness_check":
			sourceAssets = []string{
				"asset-bcrs-completeness-monitor",
			}
			controlIDs = []string{
				"control-data-completeness-p4",
			}
		}
		
		// Upsert to Neo4j with lineage
		if err := rce.bcbs239GraphClient.UpsertCalculationWithLineage(
			ctx,
			calc,
			sourceAssets,
			controlIDs,
		); err != nil {
			return fmt.Errorf("failed to upsert calculation %s to graph: %w", calc.CalculationID, err)
		}
	}
	
	return nil
}

// calculateMAS610Metrics calculates MAS 610 specific metrics.
func (rce *RegulatoryCalculationEngine) calculateMAS610Metrics(ctx context.Context, req RegulatoryCalculationRequest) []RegulatoryCalculation {
	calculations := []RegulatoryCalculation{
		{
			CalculationID:       fmt.Sprintf("MAS610-CAR-%s", req.ReportPeriod),
			CalculationType:     "capital_adequacy_ratio",
			CalculationDate:     time.Now(),
			ReportPeriod:        req.ReportPeriod,
			Result:              15.5, // Example value
			Currency:            "SGD",
			RegulatoryFramework: "MAS 610",
			SourceSystem:        "Murex",
			Status:              "calculated",
		},
		{
			CalculationID:       fmt.Sprintf("MAS610-LCR-%s", req.ReportPeriod),
			CalculationType:     "liquidity_coverage_ratio",
			CalculationDate:     time.Now(),
			ReportPeriod:        req.ReportPeriod,
			Result:              120.0,
			Currency:            "SGD",
			RegulatoryFramework: "MAS 610",
			SourceSystem:        "Murex",
			Status:              "calculated",
		},
	}
	return calculations
}

// calculateBCBS239Metrics calculates BCBS 239 specific metrics.
func (rce *RegulatoryCalculationEngine) calculateBCBS239Metrics(ctx context.Context, req RegulatoryCalculationRequest) []RegulatoryCalculation {
	calculations := []RegulatoryCalculation{
		{
			CalculationID:       fmt.Sprintf("BCBS239-RDA-%s", req.ReportPeriod),
			CalculationType:     "risk_data_aggregation",
			CalculationDate:     time.Now(),
			ReportPeriod:        req.ReportPeriod,
			Result:              0.95, // Compliance score
			Currency:            "USD",
			RegulatoryFramework: "BCBS 239",
			SourceSystem:        "BCRS",
			Status:              "calculated",
		},
	}
	return calculations
}

// calculateGenericMetrics calculates generic regulatory metrics.
func (rce *RegulatoryCalculationEngine) calculateGenericMetrics(ctx context.Context, req RegulatoryCalculationRequest) []RegulatoryCalculation {
	return []RegulatoryCalculation{}
}

