package workflows

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/quality"
	"github.com/plturrell/aModels/services/catalog/research"
	"github.com/plturrell/aModels/services/catalog/security"
)

// UnifiedWorkflowIntegration integrates catalog with unified workflow (graph + orchestration + agentflow).
type UnifiedWorkflowIntegration struct {
	graphServiceURL    string
	orchestrationURL   string
	agentflowURL       string
	localaiURL         string
	deepResearchURL    string
	httpClient         *http.Client
	registry           *iso11179.MetadataRegistry
	qualityMonitor     *quality.QualityMonitor
	deepResearchClient *research.DeepResearchClient
	reportStore        *research.ReportStore
	versionManager     *VersionManager
	logger             *log.Logger
}

// NewUnifiedWorkflowIntegration creates a new unified workflow integration.
func NewUnifiedWorkflowIntegration(
	graphServiceURL string,
	orchestrationURL string,
	agentflowURL string,
	localaiURL string,
	deepResearchURL string,
	registry *iso11179.MetadataRegistry,
	qualityMonitor *quality.QualityMonitor,
	reportStore *research.ReportStore,
	versionManager *VersionManager,
	logger *log.Logger,
) *UnifiedWorkflowIntegration {
	// Create Deep Research client
	deepResearchClient := research.NewDeepResearchClient(deepResearchURL, logger)

	return &UnifiedWorkflowIntegration{
		graphServiceURL:    graphServiceURL,
		orchestrationURL:   orchestrationURL,
		agentflowURL:       agentflowURL,
		localaiURL:         localaiURL,
		deepResearchURL:    deepResearchURL,
		httpClient:         &http.Client{Timeout: 60 * time.Second},
		registry:           registry,
		qualityMonitor:     qualityMonitor,
		deepResearchClient: deepResearchClient,
		reportStore:        reportStore,
		versionManager:     versionManager,
		logger:             logger,
	}
}

// CompleteDataProduct represents a complete, end-to-end data product.
// This is the "thin slice" - one working data product for a real customer.
type CompleteDataProduct struct {
	// ISO 11179 data element
	DataElement *iso11179.DataElement

	// Enhanced capabilities
	EnhancedElement *iso11179.EnhancedDataElement

	// Quality metrics (from Extract service)
	QualityMetrics *quality.QualityMetrics

	// Access control
	AccessControl *security.AccessControl

	// Lineage (from knowledge graph)
	Lineage *DataLineage

	// Usage examples
	UsageExamples []UsageExample

	// Documentation
	DocumentationURL string
	SampleDataURL    string

	// Research report (from Open Deep Research)
	ResearchReport *ResearchReport
}

// DataLineage represents data lineage from knowledge graph.
type DataLineage struct {
	Sources         []string
	Transformations []string
	Destinations    []string
	GraphPath       string
}

// UsageExample represents a usage example.
type UsageExample struct {
	Description string
	Code        string
	Language    string
}

// ResearchReport represents a research report from Open Deep Research.
type ResearchReport struct {
	Topic     string
	Summary   string
	Sections  []ReportSection
	Generated time.Time
}

// ReportSection represents a section of a research report.
type ReportSection struct {
	Title   string
	Content string
	Sources []string
}

// BuildCompleteDataProduct builds a complete, end-to-end data product.
// This implements the "thin slice" approach - one working product for a customer.
func (uwi *UnifiedWorkflowIntegration) BuildCompleteDataProduct(
	ctx context.Context,
	topic string,
	customerNeed string,
) (*CompleteDataProduct, error) {
	if uwi.logger != nil {
		uwi.logger.Printf("Building complete data product for topic: %s, customer need: %s", topic, customerNeed)
	}

	// Step 1: Query knowledge graph for existing data
	graphData, err := uwi.queryKnowledgeGraph(ctx, topic)
	if err != nil {
		return nil, fmt.Errorf("failed to query knowledge graph: %w", err)
	}

	// Step 2: Map to ISO 11179 data element
	dataElement, err := uwi.mapToDataElement(graphData, topic, customerNeed)
	if err != nil {
		return nil, fmt.Errorf("failed to create data element: %w", err)
	}

	// Step 3: Create enhanced data element
	enhanced := iso11179.NewEnhancedDataElement(dataElement)
	enhanced.ProductOwner = "data-product-team" // Would come from config
	enhanced.LifecycleState = "published"

	// Step 4: Fetch quality metrics from Extract service
	qualityMetrics := quality.NewQualityMetrics()
	qualityMetrics.AddSLO("freshness", 0.95, "24h")
	qualityMetrics.AddSLO("completeness", 0.90, "24h")
	qualityMetrics.AddSLO("quality", 0.85, "24h")

	if uwi.qualityMonitor != nil {
		if err := uwi.qualityMonitor.UpdateQualityMetrics(ctx, dataElement.Identifier, qualityMetrics); err != nil {
			if uwi.logger != nil {
				uwi.logger.Printf("Warning: Failed to update quality metrics: %v", err)
			}
		}
	}
	enhanced.QualityMetrics = qualityMetrics

	// Step 5: Set up access control
	accessControl := security.NewAccessControl(enhanced.ProductOwner, "internal")
	accessControl.SetDataClassification("general")
	enhanced.AccessControl = accessControl

	// Step 6: Get lineage from knowledge graph
	lineage, err := uwi.getDataLineage(ctx, dataElement.Identifier)
	if err != nil {
		if uwi.logger != nil {
			uwi.logger.Printf("Warning: Failed to get lineage: %v", err)
		}
		lineage = &DataLineage{}
	}

	// Step 7: Generate research report using Open Deep Research (via unified workflow)
	researchReport, err := uwi.generateResearchReport(ctx, topic, dataElement)
	if err != nil {
		if uwi.logger != nil {
			uwi.logger.Printf("Warning: Failed to generate research report: %v", err)
		}
		researchReport = &ResearchReport{
			Topic:     topic,
			Generated: time.Now(),
		}
	}

	// Step 8: Create usage examples
	usageExamples := uwi.createUsageExamples(dataElement)

	product := &CompleteDataProduct{
		DataElement:      dataElement,
		EnhancedElement:  enhanced,
		QualityMetrics:   qualityMetrics,
		AccessControl:    accessControl,
		Lineage:          lineage,
		UsageExamples:    usageExamples,
		ResearchReport:   researchReport,
		DocumentationURL: fmt.Sprintf("/catalog/data-elements/%s/docs", dataElement.Identifier),
		SampleDataURL:    fmt.Sprintf("/catalog/data-elements/%s/sample", dataElement.Identifier),
	}

	// Register in catalog
	uwi.registry.RegisterDataElement(dataElement)

	// Create initial version if version manager is available
	if uwi.versionManager != nil {
		version := "1.0.0"
		createdBy := enhanced.ProductOwner
		if createdBy == "" {
			createdBy = "system"
		}
		
		_, err := uwi.versionManager.CreateVersion(ctx, dataElement.Identifier, version, product, createdBy)
		if err != nil {
			if uwi.logger != nil {
				uwi.logger.Printf("Warning: Failed to create version for data product %s: %v", dataElement.Identifier, err)
			}
		} else {
			if uwi.logger != nil {
				uwi.logger.Printf("Created version %s for data product %s", version, dataElement.Identifier)
			}
		}
	}

	if uwi.logger != nil {
		uwi.logger.Printf("Complete data product built: %s (quality=%.2f, state=%s)",
			dataElement.Identifier, qualityMetrics.QualityScore, enhanced.LifecycleState)
	}

	return product, nil
}

// queryKnowledgeGraph queries the unified workflow for knowledge graph data.
func (uwi *UnifiedWorkflowIntegration) queryKnowledgeGraph(ctx context.Context, topic string) (map[string]any, error) {
	payload := map[string]any{
		"unified_request": map[string]any{
			"knowledge_graph_request": map[string]any{
				"query":  fmt.Sprintf("MATCH (n) WHERE toLower(n.label) CONTAINS toLower($topic) RETURN n LIMIT 10"),
				"params": map[string]any{"topic": topic},
			},
		},
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("%s/unified/process", uwi.graphServiceURL), bytes.NewReader(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := uwi.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("graph service returned status %d", resp.StatusCode)
	}

	var result map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result, nil
}

// mapToDataElement maps graph data to ISO 11179 data element.
func (uwi *UnifiedWorkflowIntegration) mapToDataElement(graphData map[string]any, topic string, customerNeed string) (*iso11179.DataElement, error) {
	// Extract node information from graph data
	// This is simplified - in production would parse actual graph response
	elementID := fmt.Sprintf("http://amodels.org/catalog/data-element/%s", topic)

	conceptID := fmt.Sprintf("http://amodels.org/catalog/concept/%s", topic)
	concept := iso11179.NewDataElementConcept(
		conceptID,
		topic,
		"Data",
		"Product",
		fmt.Sprintf("Data product for: %s. Customer need: %s", topic, customerNeed),
	)
	uwi.registry.RegisterDataElementConcept(concept)

	representationID := fmt.Sprintf("http://amodels.org/catalog/representation/%s", topic)
	representation := iso11179.NewRepresentation(
		representationID,
		fmt.Sprintf("Representation for %s", topic),
		"Structured",
		"json",
	)
	uwi.registry.RegisterRepresentation(representation)

	element := iso11179.NewDataElement(
		elementID,
		topic,
		conceptID,
		representationID,
		fmt.Sprintf("Complete data product addressing: %s", customerNeed),
	)
	element.SetSource("Unified Workflow")
	element.SetSteward("data-product-team")

	return element, nil
}

// getDataLineage gets data lineage from knowledge graph.
func (uwi *UnifiedWorkflowIntegration) getDataLineage(ctx context.Context, elementID string) (*DataLineage, error) {
	// Query knowledge graph for lineage
	payload := map[string]any{
		"query": `
			MATCH path = (source)-[*]->(target {id: $element_id})
			RETURN path
			LIMIT 10
		`,
		"params": map[string]any{
			"element_id": elementID,
		},
	}

	jsonData, _ := json.Marshal(payload)
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("%s/knowledge-graph/query", uwi.graphServiceURL), bytes.NewReader(jsonData))
	req.Header.Set("Content-Type", "application/json")

	resp, err := uwi.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Parse lineage (simplified)
	return &DataLineage{
		Sources:   []string{"extract-service"},
		GraphPath: "extract-service -> knowledge-graph -> catalog",
	}, nil
}

// generateResearchReport generates a research report using Open Deep Research.
func (uwi *UnifiedWorkflowIntegration) generateResearchReport(ctx context.Context, topic string, element *iso11179.DataElement) (*ResearchReport, error) {
	// Use Deep Research client to generate report
	if uwi.deepResearchClient == nil {
		// Fallback if client not initialized
		if uwi.logger != nil {
			uwi.logger.Printf("Warning: Deep Research client not initialized, using placeholder")
		}
		return &ResearchReport{
			Topic:   topic,
			Summary: fmt.Sprintf("Research report for %s data product", topic),
			Sections: []ReportSection{
				{Title: "Overview", Content: fmt.Sprintf("Data product for %s", topic)},
				{Title: "Quality", Content: "Quality metrics from Extract service"},
			},
			Generated: time.Now(),
		}, nil
	}

	// Call Deep Research service
	rawReport, err := uwi.deepResearchClient.ResearchMetadata(ctx, topic, true, true)
	if err != nil {
		if uwi.logger != nil {
			uwi.logger.Printf("Warning: Deep Research failed: %v, using fallback", err)
		}
		// Return fallback report
		return &ResearchReport{
			Topic:   topic,
			Summary: fmt.Sprintf("Research report for %s data product", topic),
			Sections: []ReportSection{
				{Title: "Overview", Content: fmt.Sprintf("Data product for %s", topic)},
				{Title: "Quality", Content: "Quality metrics from Extract service"},
			},
			Generated: time.Now(),
		}, nil
	}

	if rawReport != nil && uwi.reportStore != nil {
		elementID := ""
		if element != nil {
			elementID = element.Identifier
		}
		if err := uwi.reportStore.SaveReport(ctx, topic, elementID, rawReport); err != nil && uwi.logger != nil {
			uwi.logger.Printf("Warning: Failed to persist research report: %v", err)
		}
	}

	// Convert research report to our format
	if rawReport != nil && rawReport.Report != nil {
		return &ResearchReport{
			Topic:     rawReport.Report.Topic,
			Summary:   rawReport.Report.Summary,
			Sections:  convertSections(rawReport.Report.Sections),
			Generated: rawReport.Report.Generated,
		}, nil
	}

	// Fallback if report structure is unexpected
	return &ResearchReport{
		Topic:   topic,
		Summary: fmt.Sprintf("Research report for %s data product", topic),
		Sections: []ReportSection{
			{Title: "Overview", Content: fmt.Sprintf("Data product for %s", topic)},
		},
		Generated: time.Now(),
	}, nil
}

// convertSections converts research.ReportSection to workflows.ReportSection.
func convertSections(sections []research.ReportSection) []ReportSection {
	result := make([]ReportSection, len(sections))
	for i, s := range sections {
		result[i] = ReportSection{
			Title:   s.Title,
			Content: s.Content,
			Sources: s.Sources,
		}
	}
	return result
}

// createUsageExamples creates usage examples for the data product.
func (uwi *UnifiedWorkflowIntegration) createUsageExamples(element *iso11179.DataElement) []UsageExample {
	return []UsageExample{
		{
			Description: "Query via SPARQL",
			Code:        fmt.Sprintf("SELECT ?element WHERE { ?element rdf:type iso11179:DataElement . ?element rdfs:label \"%s\" }", element.Name),
			Language:    "sparql",
		},
		{
			Description: "Access via REST API",
			Code:        fmt.Sprintf("curl http://localhost:8084/catalog/data-elements/%s", element.Identifier),
			Language:    "bash",
		},
	}
}
