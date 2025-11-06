package breakdetection

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/plturrell/aModels/services/catalog/research"
)

// BreakAnalysisService provides root cause analysis for breaks using Deep Research
type BreakAnalysisService struct {
	deepResearchClient *research.DeepResearchClient
	logger             *log.Logger
}

// NewBreakAnalysisService creates a new break analysis service
func NewBreakAnalysisService(deepResearchClient *research.DeepResearchClient, logger *log.Logger) *BreakAnalysisService {
	return &BreakAnalysisService{
		deepResearchClient: deepResearchClient,
		logger:             logger,
	}
}

// AnalyzeRootCause performs root cause analysis on a break using Deep Research
func (bas *BreakAnalysisService) AnalyzeRootCause(ctx context.Context, breakRecord *Break) (string, error) {
	if bas.deepResearchClient == nil {
		return "", fmt.Errorf("Deep Research client not initialized")
	}

	// Build comprehensive query for root cause analysis
	query := bas.buildRootCauseQuery(breakRecord)

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
		"detected_at":        breakRecord.DetectedAt.Format("2006-01-02T15:04:05Z"),
	}

	// Execute Deep Research query
	req := &research.ResearchRequest{
		Query:   query,
		Context: context,
		Tools:   []string{"sparql_query", "catalog_search", "pattern_analysis"},
	}

	if bas.logger != nil {
		bas.logger.Printf("Analyzing root cause for break: %s", breakRecord.BreakID)
	}

	report, err := bas.deepResearchClient.Research(ctx, req)
	if err != nil {
		return "", fmt.Errorf("failed to perform root cause analysis: %w", err)
	}

	if report.Status == "error" {
		return "", fmt.Errorf("Deep Research returned error: %s", report.Error)
	}

	// Extract root cause from report
	rootCause := bas.extractRootCauseFromReport(report)
	if rootCause == "" && report.Report != nil {
		// Fallback to summary if no specific root cause section found
		rootCause = report.Report.Summary
	}

	if bas.logger != nil {
		bas.logger.Printf("Root cause analysis completed for break: %s", breakRecord.BreakID)
	}

	return rootCause, nil
}

// buildRootCauseQuery builds a comprehensive query for root cause analysis
func (bas *BreakAnalysisService) buildRootCauseQuery(breakRecord *Break) string {
	var queryBuilder strings.Builder

	queryBuilder.WriteString(fmt.Sprintf("Analyze the root cause of a %s break in %s system.\n\n", 
		breakRecord.BreakType, breakRecord.SystemName))
	
	queryBuilder.WriteString("Break Details:\n")
	queryBuilder.WriteString(fmt.Sprintf("- Break Type: %s\n", breakRecord.BreakType))
	queryBuilder.WriteString(fmt.Sprintf("- Severity: %s\n", breakRecord.Severity))
	queryBuilder.WriteString(fmt.Sprintf("- Detection Type: %s\n", breakRecord.DetectionType))
	
	if len(breakRecord.AffectedEntities) > 0 {
		queryBuilder.WriteString(fmt.Sprintf("- Affected Entities: %s\n", strings.Join(breakRecord.AffectedEntities, ", ")))
	}

	queryBuilder.WriteString("\nAnalysis Requirements:\n")
	queryBuilder.WriteString("1. Identify the root cause of this break\n")
	queryBuilder.WriteString("2. Analyze the difference between current and baseline values\n")
	queryBuilder.WriteString("3. Determine if this is a systemic issue or isolated incident\n")
	queryBuilder.WriteString("4. Identify potential contributing factors (data quality, system changes, etc.)\n")
	queryBuilder.WriteString("5. Provide a clear explanation of why this break occurred\n")
	queryBuilder.WriteString("6. Suggest potential upstream causes in the data pipeline\n")
	
	queryBuilder.WriteString("\nInvestigate:\n")
	queryBuilder.WriteString("- Recent system changes or migrations\n")
	queryBuilder.WriteString("- Data quality issues in source systems\n")
	queryBuilder.WriteString("- ETL pipeline failures or transformations\n")
	queryBuilder.WriteString("- Configuration changes\n")
	queryBuilder.WriteString("- Similar historical breaks and their resolutions\n")

	return queryBuilder.String()
}

// extractRootCauseFromReport extracts root cause from Deep Research report
func (bas *BreakAnalysisService) extractRootCauseFromReport(report *research.ResearchReport) string {
	if report.Report == nil {
		return ""
	}

	// Look for root cause section
	for _, section := range report.Report.Sections {
		title := strings.ToLower(section.Title)
		if strings.Contains(title, "root cause") || 
		   strings.Contains(title, "caus") ||
		   strings.Contains(title, "reason") ||
		   strings.Contains(title, "explanation") {
			return section.Content
		}
	}

	// If no specific root cause section, look for analysis section
	for _, section := range report.Report.Sections {
		title := strings.ToLower(section.Title)
		if strings.Contains(title, "analysis") || 
		   strings.Contains(title, "finding") ||
		   strings.Contains(title, "conclusion") {
			return section.Content
		}
	}

	// Fallback to first section with substantial content
	for _, section := range report.Report.Sections {
		if len(section.Content) > 100 {
			return section.Content
		}
	}

	return ""
}

// AnalyzeBreakPatterns analyzes patterns across multiple breaks
func (bas *BreakAnalysisService) AnalyzeBreakPatterns(ctx context.Context, breaks []*Break) (map[string]interface{}, error) {
	if bas.deepResearchClient == nil {
		return nil, fmt.Errorf("Deep Research client not initialized")
	}

	// Build query for pattern analysis
	query := bas.buildPatternAnalysisQuery(breaks)

	// Prepare context with break summaries
	breakSummaries := make([]map[string]interface{}, len(breaks))
	for i, b := range breaks {
		breakSummaries[i] = map[string]interface{}{
			"break_id":       b.BreakID,
			"break_type":     string(b.BreakType),
			"severity":       string(b.Severity),
			"detection_type": string(b.DetectionType),
			"detected_at":     b.DetectedAt.Format("2006-01-02T15:04:05Z"),
		}
	}

	context := map[string]interface{}{
		"break_count":     len(breaks),
		"break_summaries": breakSummaries,
	}

	req := &research.ResearchRequest{
		Query:   query,
		Context: context,
		Tools:   []string{"pattern_analysis", "statistical_analysis", "catalog_search"},
	}

	if bas.logger != nil {
		bas.logger.Printf("Analyzing patterns across %d breaks", len(breaks))
	}

	report, err := bas.deepResearchClient.Research(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze break patterns: %w", err)
	}

	if report.Status == "error" {
		return nil, fmt.Errorf("Deep Research returned error: %s", report.Error)
	}

	// Extract pattern analysis results
	patterns := bas.extractPatternsFromReport(report)

	return patterns, nil
}

// buildPatternAnalysisQuery builds a query for pattern analysis
func (bas *BreakAnalysisService) buildPatternAnalysisQuery(breaks []*Break) string {
	var queryBuilder strings.Builder

	queryBuilder.WriteString(fmt.Sprintf("Analyze patterns across %d breaks to identify systemic issues.\n\n", len(breaks)))

	queryBuilder.WriteString("Analysis Requirements:\n")
	queryBuilder.WriteString("1. Identify common patterns across these breaks\n")
	queryBuilder.WriteString("2. Determine if breaks are related or independent\n")
	queryBuilder.WriteString("3. Identify temporal patterns (time-based clustering)\n")
	queryBuilder.WriteString("4. Identify systemic root causes affecting multiple breaks\n")
	queryBuilder.WriteString("5. Suggest preventive measures\n")
	queryBuilder.WriteString("6. Prioritize breaks based on impact and root cause\n")

	return queryBuilder.String()
}

// extractPatternsFromReport extracts pattern analysis from Deep Research report
func (bas *BreakAnalysisService) extractPatternsFromReport(report *research.ResearchReport) map[string]interface{} {
	patterns := make(map[string]interface{})

	if report.Report == nil {
		return patterns
	}

	patterns["summary"] = report.Report.Summary
	patterns["sections"] = make([]map[string]interface{}, 0, len(report.Report.Sections))

	for _, section := range report.Report.Sections {
		sectionMap := map[string]interface{}{
			"title":   section.Title,
			"content": section.Content,
		}
		if section.Sources != nil {
			sectionMap["sources"] = section.Sources
		}
		patterns["sections"] = append(patterns["sections"].([]map[string]interface{}), sectionMap)
	}

	if report.Metadata != nil {
		patterns["metadata"] = report.Metadata
	}

	return patterns
}

// GenerateRecommendations generates recommendations based on root cause analysis
func (bas *BreakAnalysisService) GenerateRecommendations(ctx context.Context, rootCauseAnalysis string, breakRecord *Break) ([]string, error) {
	if bas.deepResearchClient == nil {
		return nil, fmt.Errorf("Deep Research client not initialized")
	}

	query := fmt.Sprintf(`
Based on the following root cause analysis, provide actionable recommendations to resolve and prevent this break:

Root Cause Analysis:
%s

Break Details:
- System: %s
- Break Type: %s
- Severity: %s
- Affected Entities: %s

Provide recommendations for:
1. Immediate resolution steps
2. Short-term preventive measures
3. Long-term system improvements
4. Monitoring and alerting improvements
`, rootCauseAnalysis, breakRecord.SystemName, breakRecord.BreakType, breakRecord.Severity, strings.Join(breakRecord.AffectedEntities, ", "))

	context := map[string]interface{}{
		"break_id":           breakRecord.BreakID,
		"root_cause":         rootCauseAnalysis,
		"system_name":        string(breakRecord.SystemName),
		"break_type":         string(breakRecord.BreakType),
	}

	req := &research.ResearchRequest{
		Query:   query,
		Context: context,
		Tools:   []string{"catalog_search", "best_practices"},
	}

	report, err := bas.deepResearchClient.Research(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to generate recommendations: %w", err)
	}

	if report.Status == "error" {
		return nil, fmt.Errorf("Deep Research returned error: %s", report.Error)
	}

	// Extract recommendations from report
	recommendations := bas.extractRecommendationsFromReport(report)

	return recommendations, nil
}

// extractRecommendationsFromReport extracts recommendations from Deep Research report
func (bas *BreakAnalysisService) extractRecommendationsFromReport(report *research.ResearchReport) []string {
	var recommendations []string

	if report.Report == nil {
		return recommendations
	}

	// Look for recommendations section
	for _, section := range report.Report.Sections {
		title := strings.ToLower(section.Title)
		if strings.Contains(title, "recommendation") || 
		   strings.Contains(title, "action") ||
		   strings.Contains(title, "solution") {
			// Parse recommendations from content (assume bullet points or numbered list)
			lines := strings.Split(section.Content, "\n")
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}
				// Remove bullet points or numbering
				line = strings.TrimPrefix(line, "-")
				line = strings.TrimPrefix(line, "*")
				line = strings.TrimPrefix(line, "•")
				line = strings.TrimSpace(line)
				// Remove numbered prefixes (1., 2., etc.)
				if len(line) > 2 && line[1] == '.' {
					line = strings.TrimSpace(line[2:])
				}
				if line != "" {
					recommendations = append(recommendations, line)
				}
			}
			break
		}
	}

	// If no recommendations section found, extract from summary
	if len(recommendations) == 0 && report.Report.Summary != "" {
		lines := strings.Split(report.Report.Summary, "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "Recommend") || strings.HasPrefix(line, "Action") {
				recommendations = append(recommendations, line)
			}
		}
	}

	return recommendations
}

