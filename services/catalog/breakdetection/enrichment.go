package breakdetection

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/plturrell/aModels/services/catalog/research"
)

// EnrichmentService provides semantic enrichment for breaks using Deep Research
type EnrichmentService struct {
	deepResearchClient *research.DeepResearchClient
	logger             *log.Logger
}

// NewEnrichmentService creates a new enrichment service
func NewEnrichmentService(deepResearchClient *research.DeepResearchClient, logger *log.Logger) *EnrichmentService {
	return &EnrichmentService{
		deepResearchClient: deepResearchClient,
		logger:             logger,
	}
}

// EnrichBreakContext enriches a break with semantic context using Deep Research
func (es *EnrichmentService) EnrichBreakContext(ctx context.Context, breakRecord *Break) (map[string]interface{}, error) {
	if es.deepResearchClient == nil {
		return nil, fmt.Errorf("Deep Research client not initialized")
	}

	// Build query for semantic enrichment
	query := es.buildEnrichmentQuery(breakRecord)

	// Prepare context with break details
	context := map[string]interface{}{
		"break_id":          breakRecord.BreakID,
		"system_name":       string(breakRecord.SystemName),
		"detection_type":    string(breakRecord.DetectionType),
		"break_type":        string(breakRecord.BreakType),
		"severity":          string(breakRecord.Severity),
		"current_value":     breakRecord.CurrentValue,
		"baseline_value":    breakRecord.BaselineValue,
		"difference":        breakRecord.Difference,
		"affected_entities": breakRecord.AffectedEntities,
	}

	req := &research.ResearchRequest{
		Query:   query,
		Context: context,
		Tools:   []string{"semantic_analysis", "catalog_search", "knowledge_graph"},
	}

	if es.logger != nil {
		es.logger.Printf("Enriching break context for: %s", breakRecord.BreakID)
	}

	report, err := es.deepResearchClient.Research(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to enrich break context: %w", err)
	}

	if report.Status == "error" {
		return nil, fmt.Errorf("Deep Research returned error: %s", report.Error)
	}

	// Extract enrichment from report
	enrichment := es.extractEnrichmentFromReport(report, breakRecord)

	if es.logger != nil {
		es.logger.Printf("Break context enrichment completed for: %s", breakRecord.BreakID)
	}

	return enrichment, nil
}

// buildEnrichmentQuery builds a query for semantic enrichment
func (es *EnrichmentService) buildEnrichmentQuery(breakRecord *Break) string {
	var queryBuilder strings.Builder

	queryBuilder.WriteString(fmt.Sprintf("Provide semantic context and enrichment for a %s break in %s system.\n\n",
		breakRecord.BreakType, breakRecord.SystemName))

	queryBuilder.WriteString("Break Details:\n")
	queryBuilder.WriteString(fmt.Sprintf("- Break Type: %s\n", breakRecord.BreakType))
	queryBuilder.WriteString(fmt.Sprintf("- System: %s\n", breakRecord.SystemName))
	queryBuilder.WriteString(fmt.Sprintf("- Detection Type: %s\n", breakRecord.DetectionType))
	queryBuilder.WriteString(fmt.Sprintf("- Severity: %s\n", breakRecord.Severity))

	if len(breakRecord.AffectedEntities) > 0 {
		queryBuilder.WriteString(fmt.Sprintf("- Affected Entities: %s\n", strings.Join(breakRecord.AffectedEntities, ", ")))
	}

	queryBuilder.WriteString("\nProvide semantic enrichment:\n")
	queryBuilder.WriteString("1. Business context (what business process is affected)\n")
	queryBuilder.WriteString("2. Data lineage context (where this data comes from and flows to)\n")
	queryBuilder.WriteString("3. Related data products and entities\n")
	queryBuilder.WriteString("4. Regulatory or compliance implications\n")
	queryBuilder.WriteString("5. Impact assessment (what systems/processes are affected downstream)\n")
	queryBuilder.WriteString("6. Historical context (similar breaks, patterns, trends)\n")
	queryBuilder.WriteString("7. Domain-specific terminology and definitions\n")
	queryBuilder.WriteString("8. Related metadata and documentation\n")

	return queryBuilder.String()
}

// extractEnrichmentFromReport extracts semantic enrichment from Deep Research report
func (es *EnrichmentService) extractEnrichmentFromReport(report *research.ResearchReport, breakRecord *Break) map[string]interface{} {
	enrichment := make(map[string]interface{})

	if report.Report == nil {
		return enrichment
	}

	// Extract structured enrichment from sections
	enrichment["summary"] = report.Report.Summary
	enrichment["sections"] = make([]map[string]interface{}, 0, len(report.Report.Sections))

	// Organize sections by topic
	categories := make(map[string][]map[string]interface{})

	for _, section := range report.Report.Sections {
		sectionMap := map[string]interface{}{
			"title":   section.Title,
			"content": section.Content,
		}
		if section.Sources != nil {
			sectionMap["sources"] = section.Sources
		}

		// Categorize sections
		category := es.categorizeSection(section.Title)
		categories[category] = append(categories[category], sectionMap)
		enrichment["sections"] = append(enrichment["sections"].([]map[string]interface{}), sectionMap)
	}

	// Add categorized sections
	enrichment["categories"] = categories

	// Extract specific enrichment fields
	enrichment["business_context"] = es.extractFieldFromSections(report.Report.Sections, "business", "context")
	enrichment["data_lineage"] = es.extractFieldFromSections(report.Report.Sections, "lineage", "flow")
	enrichment["related_entities"] = es.extractFieldFromSections(report.Report.Sections, "related", "entity")
	enrichment["regulatory_implications"] = es.extractFieldFromSections(report.Report.Sections, "regulatory", "compliance")
	enrichment["impact_assessment"] = es.extractFieldFromSections(report.Report.Sections, "impact", "effect")
	enrichment["historical_context"] = es.extractFieldFromSections(report.Report.Sections, "historical", "pattern")

	// Add metadata
	if report.Metadata != nil {
		enrichment["metadata"] = report.Metadata
	}

	// Add sources
	if report.Report.Sources != nil {
		enrichment["sources"] = report.Report.Sources
	}

	return enrichment
}

// categorizeSection categorizes a section based on its title
func (es *EnrichmentService) categorizeSection(title string) string {
	titleLower := strings.ToLower(title)

	switch {
	case strings.Contains(titleLower, "business"):
		return "business_context"
	case strings.Contains(titleLower, "lineage") || strings.Contains(titleLower, "flow"):
		return "data_lineage"
	case strings.Contains(titleLower, "related") || strings.Contains(titleLower, "entity"):
		return "related_entities"
	case strings.Contains(titleLower, "regulatory") || strings.Contains(titleLower, "compliance"):
		return "regulatory"
	case strings.Contains(titleLower, "impact") || strings.Contains(titleLower, "effect"):
		return "impact"
	case strings.Contains(titleLower, "historical") || strings.Contains(titleLower, "pattern"):
		return "historical"
	case strings.Contains(titleLower, "definition") || strings.Contains(titleLower, "terminology"):
		return "terminology"
	default:
		return "general"
	}
}

// extractFieldFromSections extracts a specific field from report sections
func (es *EnrichmentService) extractFieldFromSections(sections []research.ReportSection, keywords ...string) string {
	for _, section := range sections {
		titleLower := strings.ToLower(section.Title)
		contentLower := strings.ToLower(section.Content)

		// Check if section matches keywords
		matches := true
		for _, keyword := range keywords {
			if !strings.Contains(titleLower, strings.ToLower(keyword)) &&
				!strings.Contains(contentLower, strings.ToLower(keyword)) {
				matches = false
				break
			}
		}

		if matches {
			return section.Content
		}
	}

	return ""
}

// EnrichBreakWithDomainKnowledge enriches a break with domain-specific knowledge
func (es *EnrichmentService) EnrichBreakWithDomainKnowledge(ctx context.Context, breakRecord *Break, domain string) (map[string]interface{}, error) {
	if es.deepResearchClient == nil {
		return nil, fmt.Errorf("Deep Research client not initialized")
	}

	query := fmt.Sprintf(`
Provide domain-specific knowledge and context for a %s break in the %s domain.

Break Details:
- System: %s
- Break Type: %s
- Detection Type: %s

Focus on:
1. Domain-specific terminology and definitions
2. Industry standards and best practices
3. Regulatory requirements for this domain
4. Common patterns and anti-patterns
5. Domain-specific validation rules
`, breakRecord.BreakType, domain, breakRecord.SystemName, breakRecord.BreakType, breakRecord.DetectionType)

	context := map[string]interface{}{
		"break_id":       breakRecord.BreakID,
		"domain":         domain,
		"system_name":    string(breakRecord.SystemName),
		"break_type":     string(breakRecord.BreakType),
		"detection_type": string(breakRecord.DetectionType),
	}

	req := &research.ResearchRequest{
		Query:   query,
		Context: context,
		Tools:   []string{"domain_knowledge", "catalog_search", "regulatory_search"},
	}

	report, err := es.deepResearchClient.Research(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to enrich with domain knowledge: %w", err)
	}

	if report.Status == "error" {
		return nil, fmt.Errorf("Deep Research returned error: %s", report.Error)
	}

	// Extract domain knowledge enrichment
	enrichment := es.extractEnrichmentFromReport(report, breakRecord)
	enrichment["domain"] = domain

	return enrichment, nil
}

// EnrichBreakWithLineage enriches a break with data lineage information
func (es *EnrichmentService) EnrichBreakWithLineage(ctx context.Context, breakRecord *Break) (map[string]interface{}, error) {
	if es.deepResearchClient == nil {
		return nil, fmt.Errorf("Deep Research client not initialized")
	}

	query := fmt.Sprintf(`
Research the data lineage for entities affected by this break:
%s

Provide:
1. Source systems (where this data originates)
2. Transformation steps (ETL processes, calculations)
3. Downstream dependencies (what systems consume this data)
4. Data flow diagrams or descriptions
5. Impact analysis (what breaks if this data is incorrect)
`, strings.Join(breakRecord.AffectedEntities, ", "))

	context := map[string]interface{}{
		"break_id":          breakRecord.BreakID,
		"affected_entities": breakRecord.AffectedEntities,
		"system_name":       string(breakRecord.SystemName),
	}

	req := &research.ResearchRequest{
		Query:   query,
		Context: context,
		Tools:   []string{"lineage_query", "catalog_search", "graph_query"},
	}

	report, err := es.deepResearchClient.Research(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to enrich with lineage: %w", err)
	}

	if report.Status == "error" {
		return nil, fmt.Errorf("Deep Research returned error: %s", report.Error)
	}

	// Extract lineage enrichment
	lineageEnrichment := map[string]interface{}{
		"summary":  report.Report.Summary,
		"sections": make([]map[string]interface{}, 0, len(report.Report.Sections)),
	}

	for _, section := range report.Report.Sections {
		sectionMap := map[string]interface{}{
			"title":   section.Title,
			"content": section.Content,
		}
		if section.Sources != nil {
			sectionMap["sources"] = section.Sources
		}
		lineageEnrichment["sections"] = append(lineageEnrichment["sections"].([]map[string]interface{}), sectionMap)
	}

	return lineageEnrichment, nil
}
