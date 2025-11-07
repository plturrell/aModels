package agents

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"
)

// PerplexityIntelligentProcessor provides intelligent query understanding and optimization.
type PerplexityIntelligentProcessor struct {
	queryAnalyzer    *QueryAnalyzer
	contextBuilder   *ContextBuilder
	intentClassifier *IntentClassifier
	logger           *log.Logger
}

// NewPerplexityIntelligentProcessor creates an intelligent processor.
func NewPerplexityIntelligentProcessor(logger *log.Logger) *PerplexityIntelligentProcessor {
	return &PerplexityIntelligentProcessor{
		queryAnalyzer:    NewQueryAnalyzer(logger),
		contextBuilder:   NewContextBuilder(logger),
		intentClassifier: NewIntentClassifier(logger),
		logger:           logger,
	}
}

// ProcessIntelligently processes queries with intelligent understanding.
func (pip *PerplexityIntelligentProcessor) ProcessIntelligently(
	ctx context.Context,
	rawQuery string,
	pipeline *PerplexityPipeline,
) error {
	// Analyze query
	analysis := pip.queryAnalyzer.Analyze(rawQuery)

	// Classify intent
	intent := pip.intentClassifier.Classify(rawQuery)

	// Build enhanced context
	enhancedQuery := pip.contextBuilder.BuildQuery(rawQuery, analysis, intent)

	if pip.logger != nil {
		pip.logger.Printf("Intelligent processing: query='%s', intent=%s, domain=%s",
			rawQuery, intent.Type, analysis.Domain)
	}

	// Process with enhanced query
	return pipeline.ProcessDocuments(ctx, enhancedQuery)
}

// QueryAnalyzer analyzes queries for better understanding.
type QueryAnalyzer struct {
	logger *log.Logger
}

// NewQueryAnalyzer creates a new query analyzer.
func NewQueryAnalyzer(logger *log.Logger) *QueryAnalyzer {
	return &QueryAnalyzer{logger: logger}
}

// Analyze analyzes a query and extracts insights.
func (qa *QueryAnalyzer) Analyze(query string) *QueryAnalysis {
	analysis := &QueryAnalysis{
		Query:      query,
		Keywords:   qa.extractKeywords(query),
		Domain:      qa.detectDomain(query),
		Complexity:  qa.assessComplexity(query),
		TimeSensitivity: qa.assessTimeSensitivity(query),
		AnalyzedAt: time.Now(),
	}

	return analysis
}

// extractKeywords extracts key terms from query.
func (qa *QueryAnalyzer) extractKeywords(query string) []string {
	// Simple keyword extraction - in production would use NLP
	words := strings.Fields(strings.ToLower(query))
	keywords := make([]string, 0)
	for _, word := range words {
		if len(word) > 3 { // Filter short words
			keywords = append(keywords, word)
		}
	}
	return keywords
}

// detectDomain detects the domain of the query.
func (qa *QueryAnalyzer) detectDomain(query string) string {
	queryLower := strings.ToLower(query)
	
	domains := map[string][]string{
		"ai":        {"ai", "machine learning", "neural", "transformer", "llm"},
		"technology": {"technology", "software", "hardware", "computer"},
		"science":    {"science", "research", "study", "experiment"},
		"business":   {"business", "market", "company", "industry"},
	}

	for domain, keywords := range domains {
		for _, keyword := range keywords {
			if strings.Contains(queryLower, keyword) {
				return domain
			}
		}
	}

	return "general"
}

// assessComplexity assesses query complexity.
func (qa *QueryAnalyzer) assessComplexity(query string) string {
	wordCount := len(strings.Fields(query))
	if wordCount < 5 {
		return "simple"
	} else if wordCount < 15 {
		return "medium"
	}
	return "complex"
}

// assessTimeSensitivity assesses if query is time-sensitive.
func (qa *QueryAnalyzer) assessTimeSensitivity(query string) bool {
	timeKeywords := []string{"latest", "recent", "new", "current", "today", "now", "2024", "2025"}
	queryLower := strings.ToLower(query)
	for _, keyword := range timeKeywords {
		if strings.Contains(queryLower, keyword) {
			return true
		}
	}
	return false
}

// QueryAnalysis represents query analysis results.
type QueryAnalysis struct {
	Query           string
	Keywords         []string
	Domain           string
	Complexity       string
	TimeSensitivity  bool
	AnalyzedAt       time.Time
}

// ContextBuilder builds enhanced query context.
type ContextBuilder struct {
	logger *log.Logger
}

// NewContextBuilder creates a new context builder.
func NewContextBuilder(logger *log.Logger) *ContextBuilder {
	return &ContextBuilder{logger: logger}
}

// BuildQuery builds an enhanced query with context.
func (cb *ContextBuilder) BuildQuery(
	rawQuery string,
	analysis *QueryAnalysis,
	intent *QueryIntent,
) map[string]interface{} {
	query := map[string]interface{}{
		"query": rawQuery,
		"model": "sonar",
		"limit": 10,
	}

	// Add domain-specific optimizations
	if analysis.Domain != "general" {
		query["domain"] = analysis.Domain
	}

	// Add time sensitivity
	if analysis.TimeSensitivity {
		query["search_recency_filter"] = "week"
	}

	// Add intent-based optimizations
	if intent.Type == "research" {
		query["model"] = "sonar-deep-research"
		query["limit"] = 20
	} else if intent.Type == "quick_answer" {
		query["model"] = "sonar"
		query["limit"] = 3
	}

	// Add metadata
	query["metadata"] = map[string]interface{}{
		"complexity":       analysis.Complexity,
		"keywords":         analysis.Keywords,
		"intent":           intent.Type,
		"confidence":       intent.Confidence,
		"analyzed_at":      analysis.AnalyzedAt,
	}

	return query
}

// IntentClassifier classifies query intent.
type IntentClassifier struct {
	logger *log.Logger
}

// NewIntentClassifier creates a new intent classifier.
func NewIntentClassifier(logger *log.Logger) *IntentClassifier {
	return &IntentClassifier{logger: logger}
}

// Classify classifies the intent of a query.
func (ic *IntentClassifier) Classify(query string) *QueryIntent {
	queryLower := strings.ToLower(query)

	// Simple rule-based classification - in production would use ML
	intentType := "general"
	confidence := 0.5

	// Research intent
	if strings.Contains(queryLower, "research") ||
		strings.Contains(queryLower, "study") ||
		strings.Contains(queryLower, "analysis") {
		intentType = "research"
		confidence = 0.8
	}

	// Quick answer intent
	if strings.Contains(queryLower, "what is") ||
		strings.Contains(queryLower, "explain") ||
		strings.Contains(queryLower, "define") {
		intentType = "quick_answer"
		confidence = 0.7
	}

	// Comparison intent
	if strings.Contains(queryLower, "compare") ||
		strings.Contains(queryLower, "difference") ||
		strings.Contains(queryLower, "vs") {
		intentType = "comparison"
		confidence = 0.75
	}

	return &QueryIntent{
		Type:       intentType,
		Confidence: confidence,
		ClassifiedAt: time.Now(),
	}
}

// QueryIntent represents query intent classification.
type QueryIntent struct {
	Type          string
	Confidence    float64
	ClassifiedAt time.Time
}

