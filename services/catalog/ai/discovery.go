package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/research"
	"github.com/plturrell/aModels/services/catalog/workflows"
)

// MetadataDiscoverer provides intelligent metadata discovery capabilities.
type MetadataDiscoverer struct {
	deepResearchClient *research.DeepResearchClient
	extractServiceURL  string
	httpClient         *http.Client
	logger             *log.Logger
}

// NewMetadataDiscoverer creates a new metadata discoverer.
func NewMetadataDiscoverer(
	deepResearchURL string,
	extractServiceURL string,
	logger *log.Logger,
) *MetadataDiscoverer {
	return &MetadataDiscoverer{
		deepResearchClient: research.NewDeepResearchClient(deepResearchURL, logger),
		extractServiceURL:  extractServiceURL,
		httpClient:         &http.Client{Timeout: 60 * time.Second},
		logger:             logger,
	}
}

// DiscoveredMetadata represents metadata discovered from a source.
type DiscoveredMetadata struct {
	Source          string                        `json:"source"`
	DataElements    []*iso11179.DataElement       `json:"data_elements"`
	Relationships   []Relationship                `json:"relationships"`
	Classifications []Classification             `json:"classifications"`
	Confidence      float64                      `json:"confidence"` // 0.0 - 1.0
	DiscoveredAt    time.Time                    `json:"discovered_at"`
	RawData         map[string]interface{}       `json:"raw_data,omitempty"`
}

// Relationship represents a discovered relationship between data elements.
type Relationship struct {
	SourceID      string  `json:"source_id"`
	TargetID      string  `json:"target_id"`
	RelationshipType string `json:"relationship_type"` // "references", "depends_on", "transforms", "similar"
	Confidence    float64 `json:"confidence"`
	Evidence      string  `json:"evidence,omitempty"`
}

// Classification represents a classification of a data element.
type Classification struct {
	ElementID    string  `json:"element_id"`
	Category     string  `json:"category"`     // "transaction", "reference", "dimension", "fact"
	Sensitivity  string  `json:"sensitivity"` // "public", "internal", "confidential", "restricted"
	Confidence   float64 `json:"confidence"`
	Reasoning    string  `json:"reasoning,omitempty"`
}

// DiscoveryRequest represents a request to discover metadata.
type DiscoveryRequest struct {
	Source      string            `json:"source"`       // Database, API, file path
	SourceType  string            `json:"source_type"` // "database", "api", "file", "code"
	Context     map[string]string `json:"context,omitempty"`
	Options     DiscoveryOptions  `json:"options,omitempty"`
}

// DiscoveryOptions configures discovery behavior.
type DiscoveryOptions struct {
	GenerateDescriptions bool `json:"generate_descriptions"` // Use AI to generate descriptions
	DetectRelationships  bool `json:"detect_relationships"`  // Automatically detect relationships
	ClassifyData         bool `json:"classify_data"`        // Classify data types (transaction vs reference)
	DeepAnalysis         bool `json:"deep_analysis"`        // Deep semantic analysis
}

// DiscoverMetadata discovers metadata from a given source.
func (md *MetadataDiscoverer) DiscoverMetadata(ctx context.Context, req DiscoveryRequest) (*DiscoveredMetadata, error) {
	md.logger.Printf("Discovering metadata from source: %s (type: %s)", req.Source, req.SourceType)

	discovered := &DiscoveredMetadata{
		Source:       req.Source,
		DiscoveredAt: time.Now(),
		Confidence:   0.0,
	}

	// Step 1: Use Extract service to analyze schema (if database)
	if req.SourceType == "database" {
		schemas, err := md.analyzeSchema(ctx, req.Source)
		if err != nil {
			md.logger.Printf("Warning: Failed to analyze schema: %v", err)
		} else {
			discovered.DataElements = md.extractDataElementsFromSchemas(schemas)
			discovered.Confidence = 0.7 // Schema analysis gives good confidence
		}
	}

	// Step 2: Use Deep Research to discover metadata
	if req.Options.DeepAnalysis {
		researchQuery := md.buildResearchQuery(req)
		report, err := md.deepResearchClient.ResearchMetadata(ctx, researchQuery, true, true)
		if err != nil {
			md.logger.Printf("Warning: Deep research failed: %v", err)
		} else {
			// Convert research.ResearchReport to workflows.ResearchReport
			workflowReport := &workflows.ResearchReport{
				Topic:     report.Report.Topic,
				Summary:   report.Report.Summary,
				Generated: report.Report.Generated,
			}
			for _, section := range report.Report.Sections {
				workflowReport.Sections = append(workflowReport.Sections, workflows.ReportSection{
					Title:   section.Title,
					Content: section.Content,
					Sources: section.Sources,
				})
			}
			// Extract metadata from research report
			enrichedElements := md.enrichFromResearch(workflowReport, discovered.DataElements, req.Options)
			discovered.DataElements = enrichedElements
			discovered.Confidence = 0.9 // Deep research gives high confidence
		}
	}

	// Step 3: Detect relationships
	if req.Options.DetectRelationships {
		relationships := md.detectRelationships(ctx, discovered.DataElements)
		discovered.Relationships = relationships
	}

	// Step 4: Classify data
	if req.Options.ClassifyData {
		classifications := md.classifyData(ctx, discovered.DataElements)
		discovered.Classifications = classifications
	}

	return discovered, nil
}

// analyzeSchema analyzes a database schema using the Extract service.
func (md *MetadataDiscoverer) analyzeSchema(ctx context.Context, source string) ([]SchemaInfo, error) {
	url := fmt.Sprintf("%s/schema/analyze", md.extractServiceURL)
	payload := map[string]interface{}{
		"source": source,
	}

	jsonData, _ := json.Marshal(payload)
	httpReq, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := md.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze schema: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Read response body for better error messages
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("extract service returned status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Schemas []SchemaInfo `json:"schemas"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Schemas, nil
}

// SchemaInfo represents schema information from Extract service.
type SchemaInfo struct {
	TableName    string                 `json:"table_name"`
	Columns      []ColumnInfo           `json:"columns"`
	Relationships []SchemaRelationship  `json:"relationships,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// ColumnInfo represents column information.
type ColumnInfo struct {
	Name         string `json:"name"`
	DataType     string `json:"data_type"`
	Nullable     bool   `json:"nullable"`
	PrimaryKey   bool   `json:"primary_key"`
	ForeignKey   bool   `json:"foreign_key"`
	ReferencedTable string `json:"referenced_table,omitempty"`
}

// SchemaRelationship represents a relationship between tables.
type SchemaRelationship struct {
	FromTable string `json:"from_table"`
	ToTable   string `json:"to_table"`
	Type      string `json:"type"` // "one_to_one", "one_to_many", "many_to_many"
}

// extractDataElementsFromSchemas converts schema information to data elements.
func (md *MetadataDiscoverer) extractDataElementsFromSchemas(schemas []SchemaInfo) []*iso11179.DataElement {
	var elements []*iso11179.DataElement

	for _, schema := range schemas {
		// Create a data element for the table
		tableElement := iso11179.NewDataElement(
			fmt.Sprintf("table:%s", schema.TableName),
			schema.TableName,
			fmt.Sprintf("Table: %s", schema.TableName),
			"table",
			fmt.Sprintf("http://amodels.org/catalog/tables/%s", schema.TableName),
		)
		elements = append(elements, tableElement)

		// Create data elements for columns
		for _, column := range schema.Columns {
			columnElement := iso11179.NewDataElement(
				fmt.Sprintf("column:%s.%s", schema.TableName, column.Name),
				column.Name,
				fmt.Sprintf("Column %s in table %s (type: %s)", column.Name, schema.TableName, column.DataType),
				column.DataType,
				fmt.Sprintf("http://amodels.org/catalog/columns/%s/%s", schema.TableName, column.Name),
			)
			// Add metadata
			columnElement.AddMetadata("nullable", column.Nullable)
			columnElement.AddMetadata("primary_key", column.PrimaryKey)
			columnElement.AddMetadata("foreign_key", column.ForeignKey)
			if column.ReferencedTable != "" {
				columnElement.AddMetadata("referenced_table", column.ReferencedTable)
			}
			elements = append(elements, columnElement)
		}
	}

	return elements
}

// buildResearchQuery builds a research query for Deep Research.
func (md *MetadataDiscoverer) buildResearchQuery(req DiscoveryRequest) string {
	return fmt.Sprintf(`
Research and document all metadata related to: %s

Context:
- Source Type: %s
- Source: %s

Please provide:
1. Data elements and their definitions
2. Data lineage (sources, transformations)
3. Quality metrics and patterns
4. Access controls and permissions
5. Usage patterns and examples
6. Related data products
	`, req.Source, req.SourceType, req.Source)
}

// enrichFromResearch enriches data elements with information from research report.
func (md *MetadataDiscoverer) enrichFromResearch(
	report *workflows.ResearchReport,
	elements []*iso11179.DataElement,
	options DiscoveryOptions,
) []*iso11179.DataElement {
	// For each element, try to find matching information in research report
	enriched := make([]*iso11179.DataElement, len(elements))
	copy(enriched, elements)

	if options.GenerateDescriptions {
		// Use research report to generate/enhance descriptions
		for _, element := range enriched {
			// Match element to research report sections
			for _, section := range report.Sections {
				if matchesElement(section.Title, element.Name) {
					// Enhance description
					if element.Definition == "" || len(element.Definition) < 50 {
						element.Definition = section.Content
					}
					break
				}
			}
		}
	}

	return enriched
}

// matchesElement checks if a section title matches an element name.
func matchesElement(sectionTitle, elementName string) bool {
	// Simple matching - in production, would use semantic similarity
	return contains(sectionTitle, elementName) || contains(elementName, sectionTitle)
}

func contains(str, substr string) bool {
	return len(str) >= len(substr) && (str == substr || 
		(len(str) > len(substr) && (str[:len(substr)] == substr || str[len(str)-len(substr):] == substr)))
}

// detectRelationships detects relationships between data elements.
func (md *MetadataDiscoverer) detectRelationships(ctx context.Context, elements []*iso11179.DataElement) []Relationship {
	var relationships []Relationship

	// Analyze element names and metadata to find relationships
	for i, source := range elements {
		for j, target := range elements {
			if i == j {
				continue
			}

			// Check for naming patterns
			if isRelated(source.Name, target.Name) {
				relationships = append(relationships, Relationship{
					SourceID:         source.Identifier,
					TargetID:         target.Identifier,
					RelationshipType: inferRelationshipType(source, target),
					Confidence:       0.7,
					Evidence:         "Naming pattern match",
				})
			}

			// Check for foreign key relationships
			if refTable, ok := source.Metadata["referenced_table"].(string); ok {
				if refTable == target.Name || refTable == target.Identifier {
					relationships = append(relationships, Relationship{
						SourceID:         source.Identifier,
						TargetID:         target.Identifier,
						RelationshipType: "references",
						Confidence:       0.9,
						Evidence:         "Foreign key reference",
					})
				}
			}
		}
	}

	return relationships
}

// isRelated checks if two element names suggest a relationship.
func isRelated(name1, name2 string) bool {
	// Simple heuristics: check for common prefixes/suffixes
	// In production, would use semantic similarity
	commonPrefixes := []string{"customer", "order", "product", "user", "account"}
	for _, prefix := range commonPrefixes {
		if contains(name1, prefix) && contains(name2, prefix) {
			return true
		}
	}
	return false
}

// inferRelationshipType infers the type of relationship.
func inferRelationshipType(source, target *iso11179.DataElement) string {
	// Heuristics to determine relationship type
	if contains(source.Name, "id") && contains(target.Name, source.Name) {
		return "references"
	}
	if contains(source.Name, "depends") || contains(source.Definition, "depends") {
		return "depends_on"
	}
	if contains(source.Name, "transform") || contains(source.Definition, "transform") {
		return "transforms"
	}
	return "related"
}

// classifyData classifies data elements (transaction vs reference, sensitivity, etc.).
func (md *MetadataDiscoverer) classifyData(ctx context.Context, elements []*iso11179.DataElement) []Classification {
	var classifications []Classification

	for _, element := range elements {
		classification := Classification{
			ElementID: element.Identifier,
			Category:   inferCategory(element),
			Sensitivity: inferSensitivity(element),
			Confidence: 0.7,
			Reasoning:  buildReasoning(element),
		}
		classifications = append(classifications, classification)
	}

	return classifications
}

// inferCategory infers the data category.
func inferCategory(element *iso11179.DataElement) string {
	name := element.Name
	definition := element.Definition

	// Transaction tables
	if contains(name, "transaction") || contains(name, "order") || contains(name, "payment") ||
		contains(definition, "transaction") || contains(definition, "order") {
		return "transaction"
	}

	// Reference tables
	if contains(name, "reference") || contains(name, "lookup") || contains(name, "code") ||
		contains(definition, "reference") || contains(definition, "lookup") {
		return "reference"
	}

	// Dimension tables
	if contains(name, "dimension") || contains(name, "dim") ||
		contains(definition, "dimension") {
		return "dimension"
	}

	// Fact tables
	if contains(name, "fact") || contains(definition, "fact") {
		return "fact"
	}

	return "unknown"
}

// inferSensitivity infers data sensitivity level.
func inferSensitivity(element *iso11179.DataElement) string {
	name := element.Name
	definition := element.Definition

	// High sensitivity indicators
	if contains(name, "password") || contains(name, "ssn") || contains(name, "credit_card") ||
		contains(name, "pii") || contains(definition, "password") || contains(definition, "ssn") {
		return "restricted"
	}

	// Confidential indicators
	if contains(name, "email") || contains(name, "phone") || contains(name, "address") ||
		contains(definition, "email") || contains(definition, "phone") {
		return "confidential"
	}

	// Internal indicators
	if contains(name, "internal") || contains(definition, "internal") {
		return "internal"
	}

	return "public"
}

// buildReasoning builds reasoning for classification.
func buildReasoning(element *iso11179.DataElement) string {
	return fmt.Sprintf("Classified based on name '%s' and definition patterns", element.Name)
}

