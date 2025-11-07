package breakdetection

import (
	"context"
	"fmt"
	"math"
	"strings"
)

// AIAnalysisEnhanced provides enhanced AI analysis with confidence scoring
type AIAnalysisEnhanced struct {
	*AIAnalysisService
	enableConfidenceScoring bool
	confidenceThreshold     float64
}

// NewAIAnalysisEnhanced creates enhanced AI analysis service
func NewAIAnalysisEnhanced(baseService *AIAnalysisService) *AIAnalysisEnhanced {
	return &AIAnalysisEnhanced{
		AIAnalysisService:      baseService,
		enableConfidenceScoring: true,
		confidenceThreshold:     0.7, // 70% confidence threshold
	}
}

// ConfidenceScore represents confidence in AI analysis
type ConfidenceScore struct {
	Overall         float64            `json:"overall"`          // Overall confidence (0.0-1.0)
	Description     float64            `json:"description"`      // Confidence in description
	Category        float64            `json:"category"`         // Confidence in category
	Priority        float64            `json:"priority"`         // Confidence in priority score
	RootCause       float64            `json:"root_cause"`        // Confidence in root cause analysis
	Recommendations float64            `json:"recommendations"`  // Confidence in recommendations
	Factors         map[string]float64 `json:"factors"`          // Individual confidence factors
}

// EnhancedAnalysisResult contains enhanced analysis results
type EnhancedAnalysisResult struct {
	Description     string          `json:"description"`
	Category        string          `json:"category"`
	PriorityScore   float64         `json:"priority_score"`
	Confidence      ConfidenceScore `json:"confidence"`
	Reasoning       string          `json:"reasoning,omitempty"`
	AlternativeViews []string       `json:"alternative_views,omitempty"`
}

// GenerateBreakDescriptionEnhanced generates description with confidence scoring
func (aae *AIAnalysisEnhanced) GenerateBreakDescriptionEnhanced(ctx context.Context, 
	breakRecord *Break) (*EnhancedAnalysisResult, error) {
	
	// Generate description
	description, err := aae.AIAnalysisService.GenerateBreakDescription(ctx, breakRecord)
	if err != nil {
		return nil, err
	}

	// Calculate confidence
	confidence := aae.calculateDescriptionConfidence(breakRecord, description)

	result := &EnhancedAnalysisResult{
		Description: description,
		Confidence:  confidence,
	}

	// If confidence is low, generate alternative views
	if confidence.Overall < aae.confidenceThreshold {
		alternatives, _ := aae.generateAlternativeDescriptions(ctx, breakRecord)
		result.AlternativeViews = alternatives
	}

	return result, nil
}

// CategorizeBreakEnhanced categorizes break with confidence scoring
func (aae *AIAnalysisEnhanced) CategorizeBreakEnhanced(ctx context.Context, 
	breakRecord *Break) (*EnhancedAnalysisResult, error) {
	
	// Categorize
	category, err := aae.AIAnalysisService.CategorizeBreak(ctx, breakRecord)
	if err != nil {
		return nil, err
	}

	// Calculate confidence
	confidence := aae.calculateCategoryConfidence(breakRecord, category)

	result := &EnhancedAnalysisResult{
		Category:  category,
		Confidence: confidence,
	}

	// Generate reasoning if confidence is high
	if confidence.Overall >= aae.confidenceThreshold {
		reasoning, _ := aae.generateCategoryReasoning(ctx, breakRecord, category)
		result.Reasoning = reasoning
	}

	return result, nil
}

// CalculatePriorityScoreEnhanced calculates priority with confidence
func (aae *AIAnalysisEnhanced) CalculatePriorityScoreEnhanced(ctx context.Context, 
	breakRecord *Break) (*EnhancedAnalysisResult, error) {
	
	// Calculate priority
	priorityScore, err := aae.AIAnalysisService.CalculatePriorityScore(ctx, breakRecord)
	if err != nil {
		return nil, err
	}

	// Calculate confidence
	confidence := aae.calculatePriorityConfidence(breakRecord, priorityScore)

	result := &EnhancedAnalysisResult{
		PriorityScore: priorityScore,
		Confidence:    confidence,
	}

	return result, nil
}

// AnalyzeBreakComprehensive performs comprehensive analysis with all features
func (aae *AIAnalysisEnhanced) AnalyzeBreakComprehensive(ctx context.Context, 
	breakRecord *Break) (*EnhancedAnalysisResult, error) {
	
	result := &EnhancedAnalysisResult{}

	// Generate description
	descResult, err := aae.GenerateBreakDescriptionEnhanced(ctx, breakRecord)
	if err == nil {
		result.Description = descResult.Description
		result.Confidence.Description = descResult.Confidence.Description
	}

	// Categorize
	catResult, err := aae.CategorizeBreakEnhanced(ctx, breakRecord)
	if err == nil {
		result.Category = catResult.Category
		result.Confidence.Category = catResult.Confidence.Category
		result.Reasoning = catResult.Reasoning
	}

	// Calculate priority
	priorityResult, err := aae.CalculatePriorityScoreEnhanced(ctx, breakRecord)
	if err == nil {
		result.PriorityScore = priorityResult.PriorityScore
		result.Confidence.Priority = priorityResult.Confidence.Priority
	}

	// Calculate overall confidence
	result.Confidence.Overall = aae.calculateOverallConfidence(result.Confidence)

	return result, nil
}

// Confidence calculation methods
func (aae *AIAnalysisEnhanced) calculateDescriptionConfidence(breakRecord *Break, description string) ConfidenceScore {
	confidence := ConfidenceScore{
		Factors: make(map[string]float64),
	}

	// Factor 1: Description length (longer is generally better)
	lengthScore := math.Min(float64(len(description))/200.0, 1.0)
	confidence.Factors["description_length"] = lengthScore

	// Factor 2: Break has complete data
	dataCompleteness := aae.calculateDataCompleteness(breakRecord)
	confidence.Factors["data_completeness"] = dataCompleteness

	// Factor 3: Break type clarity
	typeClarity := aae.calculateTypeClarity(breakRecord)
	confidence.Factors["type_clarity"] = typeClarity

	// Calculate overall description confidence
	confidence.Description = (lengthScore*0.3 + dataCompleteness*0.4 + typeClarity*0.3)

	return confidence
}

func (aae *AIAnalysisEnhanced) calculateCategoryConfidence(breakRecord *Break, category string) ConfidenceScore {
	confidence := ConfidenceScore{
		Factors: make(map[string]float64),
	}

	// Factor 1: Category matches break type
	typeMatch := aae.calculateCategoryTypeMatch(breakRecord, category)
	confidence.Factors["category_type_match"] = typeMatch

	// Factor 2: Severity alignment
	severityAlignment := aae.calculateSeverityAlignment(breakRecord, category)
	confidence.Factors["severity_alignment"] = severityAlignment

	// Factor 3: Data completeness
	dataCompleteness := aae.calculateDataCompleteness(breakRecord)
	confidence.Factors["data_completeness"] = dataCompleteness

	// Calculate overall category confidence
	confidence.Category = (typeMatch*0.4 + severityAlignment*0.3 + dataCompleteness*0.3)

	return confidence
}

func (aae *AIAnalysisEnhanced) calculatePriorityConfidence(breakRecord *Break, priorityScore float64) ConfidenceScore {
	confidence := ConfidenceScore{
		Factors: make(map[string]float64),
	}

	// Factor 1: Priority aligns with severity
	severityAlignment := aae.calculatePrioritySeverityAlignment(breakRecord, priorityScore)
	confidence.Factors["severity_alignment"] = severityAlignment

	// Factor 2: Break type criticality
	typeCriticality := aae.calculateTypeCriticality(breakRecord)
	confidence.Factors["type_criticality"] = typeCriticality

	// Factor 3: Data completeness
	dataCompleteness := aae.calculateDataCompleteness(breakRecord)
	confidence.Factors["data_completeness"] = dataCompleteness

	// Calculate overall priority confidence
	confidence.Priority = (severityAlignment*0.4 + typeCriticality*0.3 + dataCompleteness*0.3)

	return confidence
}

func (aae *AIAnalysisEnhanced) calculateOverallConfidence(conf ConfidenceScore) float64 {
	// Weighted average of all confidence scores
	weights := map[string]float64{
		"description": 0.3,
		"category":     0.25,
		"priority":     0.25,
		"root_cause":   0.1,
		"recommendations": 0.1,
	}

	overall := 0.0
	totalWeight := 0.0

	if conf.Description > 0 {
		overall += conf.Description * weights["description"]
		totalWeight += weights["description"]
	}
	if conf.Category > 0 {
		overall += conf.Category * weights["category"]
		totalWeight += weights["category"]
	}
	if conf.Priority > 0 {
		overall += conf.Priority * weights["priority"]
		totalWeight += weights["priority"]
	}
	if conf.RootCause > 0 {
		overall += conf.RootCause * weights["root_cause"]
		totalWeight += weights["root_cause"]
	}
	if conf.Recommendations > 0 {
		overall += conf.Recommendations * weights["recommendations"]
		totalWeight += weights["recommendations"]
	}

	if totalWeight > 0 {
		return overall / totalWeight
	}
	return 0.5 // Default confidence if no scores available
}

// Helper methods for confidence factors
func (aae *AIAnalysisEnhanced) calculateDataCompleteness(breakRecord *Break) float64 {
	score := 0.0
	factors := 0

	if breakRecord.CurrentValue != nil && len(breakRecord.CurrentValue) > 0 {
		score += 0.3
		factors++
	}
	if breakRecord.BaselineValue != nil && len(breakRecord.BaselineValue) > 0 {
		score += 0.3
		factors++
	}
	if breakRecord.Difference != nil && len(breakRecord.Difference) > 0 {
		score += 0.2
		factors++
	}
	if len(breakRecord.AffectedEntities) > 0 {
		score += 0.2
		factors++
	}

	if factors == 0 {
		return 0.0
	}
	return score
}

func (aae *AIAnalysisEnhanced) calculateTypeClarity(breakRecord *Break) float64 {
	// Some break types are clearer than others
	clearTypes := map[BreakType]float64{
		BreakTypeReconciliationBreak: 0.9,
		BreakTypeBalanceBreak:        0.9,
		BreakTypeAmountMismatch:      0.8,
		BreakTypeMissingEntry:        0.8,
		BreakTypeAccountMismatch:    0.7,
		BreakTypeCapitalRatioViolation: 0.9,
		BreakTypeLCRViolation:        0.9,
		BreakTypeComplianceViolation: 0.9,
	}

	if score, ok := clearTypes[breakRecord.BreakType]; ok {
		return score
	}
	return 0.6 // Default clarity
}

func (aae *AIAnalysisEnhanced) calculateCategoryTypeMatch(breakRecord *Break, category string) float64 {
	// Map break types to expected categories
	typeCategoryMap := map[BreakType]string{
		BreakTypeMissingEntry:        "data_quality",
		BreakTypeAmountMismatch:      "calculation_error",
		BreakTypeBalanceBreak:        "calculation_error",
		BreakTypeReconciliationBreak: "system_error",
		BreakTypeAccountMismatch:     "data_quality",
		BreakTypeCapitalRatioViolation: "calculation_error",
		BreakTypeRWAError:            "calculation_error",
		BreakTypeLCRViolation:        "calculation_error",
		BreakTypeComplianceViolation: "migration_issue",
	}

	expectedCategory, ok := typeCategoryMap[breakRecord.BreakType]
	if !ok {
		return 0.5 // Unknown mapping
	}

	if strings.EqualFold(category, expectedCategory) {
		return 1.0 // Perfect match
	}

	// Partial match (similar categories)
	if strings.Contains(strings.ToLower(category), strings.ToLower(expectedCategory)) ||
		strings.Contains(strings.ToLower(expectedCategory), strings.ToLower(category)) {
		return 0.7
	}

	return 0.3 // Low match
}

func (aae *AIAnalysisEnhanced) calculateSeverityAlignment(breakRecord *Break, category string) float64 {
	// Critical breaks should be in critical categories
	if breakRecord.Severity == SeverityCritical {
		criticalCategories := []string{"system_error", "migration_issue", "calculation_error"}
		for _, cat := range criticalCategories {
			if strings.EqualFold(category, cat) {
				return 1.0
			}
		}
		return 0.6
	}

	// High severity should align with important categories
	if breakRecord.Severity == SeverityHigh {
		importantCategories := []string{"data_quality", "calculation_error", "configuration_error"}
		for _, cat := range importantCategories {
			if strings.EqualFold(category, cat) {
				return 1.0
			}
		}
		return 0.7
	}

	// Medium/Low can be more flexible
	return 0.8
}

func (aae *AIAnalysisEnhanced) calculatePrioritySeverityAlignment(breakRecord *Break, priorityScore float64) float64 {
	// Map severity to expected priority score
	severityPriorityMap := map[Severity]float64{
		SeverityCritical: 0.9,
		SeverityHigh:     0.7,
		SeverityMedium:  0.5,
		SeverityLow:     0.3,
	}

	expectedPriority, ok := severityPriorityMap[breakRecord.Severity]
	if !ok {
		return 0.5
	}

	// Calculate alignment (closer to expected = higher confidence)
	diff := math.Abs(priorityScore - expectedPriority)
	if diff < 0.1 {
		return 1.0 // Very close
	} else if diff < 0.2 {
		return 0.8 // Close
	} else if diff < 0.3 {
		return 0.6 // Moderate
	}
	return 0.4 // Far
}

func (aae *AIAnalysisEnhanced) calculateTypeCriticality(breakRecord *Break) float64 {
	// Critical break types
	criticalTypes := []BreakType{
		BreakTypeReconciliationBreak,
		BreakTypeCapitalRatioViolation,
		BreakTypeLCRViolation,
		BreakTypeComplianceViolation,
	}

	for _, ct := range criticalTypes {
		if breakRecord.BreakType == ct {
			return 1.0
		}
	}

	// High criticality types
	highTypes := []BreakType{
		BreakTypeBalanceBreak,
		BreakTypeRWAError,
		BreakTypeReportingBreak,
	}

	for _, ht := range highTypes {
		if breakRecord.BreakType == ht {
			return 0.8
		}
	}

	return 0.6 // Default
}

func (aae *AIAnalysisEnhanced) generateAlternativeDescriptions(ctx context.Context, 
	breakRecord *Break) ([]string, error) {
	if aae.AIAnalysisService.localaiURL == "" {
		return []string{}, fmt.Errorf("LocalAI URL not configured")
	}

	// Build prompt for alternative descriptions
	var promptBuilder strings.Builder
	promptBuilder.WriteString("Generate 2-3 alternative descriptions of this break from different perspectives:\n\n")
	promptBuilder.WriteString(fmt.Sprintf("Break Type: %s\n", breakRecord.BreakType))
	promptBuilder.WriteString(fmt.Sprintf("System: %s\n", breakRecord.SystemName))
	promptBuilder.WriteString(fmt.Sprintf("Severity: %s\n", breakRecord.Severity))
	promptBuilder.WriteString(fmt.Sprintf("Detection Type: %s\n", breakRecord.DetectionType))
	
	if len(breakRecord.AffectedEntities) > 0 {
		promptBuilder.WriteString(fmt.Sprintf("Affected Entities: %s\n", strings.Join(breakRecord.AffectedEntities, ", ")))
	}
	
	if breakRecord.Difference != nil {
		promptBuilder.WriteString(fmt.Sprintf("Difference: %v\n", breakRecord.Difference))
	}
	
	if breakRecord.RootCauseAnalysis != "" {
		promptBuilder.WriteString(fmt.Sprintf("Root Cause: %s\n", breakRecord.RootCauseAnalysis))
	}

	prompt := promptBuilder.String()
	systemPrompt := "Generate 2-3 alternative descriptions of this break. Each description should provide a different perspective (technical, business impact, or operational). Return descriptions as a numbered list, one per line."

	// Call LocalAI
	response, err := aae.AIAnalysisService.callLocalAI(ctx, prompt, systemPrompt)
	if err != nil {
		if aae.AIAnalysisService.logger != nil {
			aae.AIAnalysisService.logger.Printf("Failed to generate alternative descriptions: %v", err)
		}
		return []string{}, err
	}

	// Parse response into list of descriptions
	descriptions := aae.parseAlternativeDescriptions(response)

	if aae.AIAnalysisService.logger != nil {
		aae.AIAnalysisService.logger.Printf("Generated %d alternative descriptions for break %s", len(descriptions), breakRecord.BreakID)
	}

	return descriptions, nil
}

// parseAlternativeDescriptions parses alternative descriptions from LocalAI response
func (aae *AIAnalysisEnhanced) parseAlternativeDescriptions(response string) []string {
	var descriptions []string
	lines := strings.Split(response, "\n")
	
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		
		// Remove numbering (1., 2., etc.)
		line = strings.TrimPrefix(line, "1.")
		line = strings.TrimPrefix(line, "2.")
		line = strings.TrimPrefix(line, "3.")
		line = strings.TrimPrefix(line, "-")
		line = strings.TrimSpace(line)
		
		// Remove leading dash or bullet points
		line = strings.TrimPrefix(line, "- ")
		line = strings.TrimPrefix(line, "• ")
		
		if line != "" {
			descriptions = append(descriptions, line)
		}
		
		// Limit to 3 descriptions
		if len(descriptions) >= 3 {
			break
		}
	}
	
	return descriptions
}

func (aae *AIAnalysisEnhanced) generateCategoryReasoning(ctx context.Context, 
	breakRecord *Break, category string) (string, error) {
	if aae.AIAnalysisService.localaiURL == "" {
		return fmt.Sprintf("Categorized as %s based on break type %s and severity %s", 
			category, breakRecord.BreakType, breakRecord.Severity), nil
	}

	// Build prompt for category reasoning
	var promptBuilder strings.Builder
	promptBuilder.WriteString("Explain in detail why this break was categorized as '" + category + "':\n\n")
	promptBuilder.WriteString(fmt.Sprintf("Break Type: %s\n", breakRecord.BreakType))
	promptBuilder.WriteString(fmt.Sprintf("System: %s\n", breakRecord.SystemName))
	promptBuilder.WriteString(fmt.Sprintf("Detection Type: %s\n", breakRecord.DetectionType))
	promptBuilder.WriteString(fmt.Sprintf("Severity: %s\n", breakRecord.Severity))
	
	if len(breakRecord.AffectedEntities) > 0 {
		promptBuilder.WriteString(fmt.Sprintf("Affected Entities: %s\n", strings.Join(breakRecord.AffectedEntities, ", ")))
	}
	
	if breakRecord.Difference != nil {
		promptBuilder.WriteString(fmt.Sprintf("Difference: %v\n", breakRecord.Difference))
	}
	
	if breakRecord.RootCauseAnalysis != "" {
		promptBuilder.WriteString(fmt.Sprintf("Root Cause: %s\n", breakRecord.RootCauseAnalysis))
	}

	prompt := promptBuilder.String()
	systemPrompt := "Provide a detailed explanation of why this break was categorized as the specified category. Include specific factors, patterns, or characteristics that led to this categorization. Be concise but comprehensive."

	// Call LocalAI
	response, err := aae.AIAnalysisService.callLocalAI(ctx, prompt, systemPrompt)
	if err != nil {
		if aae.AIAnalysisService.logger != nil {
			aae.AIAnalysisService.logger.Printf("Failed to generate category reasoning: %v", err)
		}
		// Fallback to simple reasoning
		return fmt.Sprintf("Categorized as %s based on break type %s, severity %s, and detection type %s", 
			category, breakRecord.BreakType, breakRecord.Severity, breakRecord.DetectionType), nil
	}

	// Clean up response
	reasoning := strings.TrimSpace(response)
	// Remove any leading prefixes like "Reasoning:" or "Explanation:"
	reasoning = strings.TrimPrefix(reasoning, "Reasoning:")
	reasoning = strings.TrimPrefix(reasoning, "Explanation:")
	reasoning = strings.TrimSpace(reasoning)

	if aae.AIAnalysisService.logger != nil {
		aae.AIAnalysisService.logger.Printf("Generated category reasoning for break %s: %s", breakRecord.BreakID, category)
	}

	return reasoning, nil
}

