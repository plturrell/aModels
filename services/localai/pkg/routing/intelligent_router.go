package routing

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
)

// QueryComplexity represents the complexity analysis of a query
type QueryComplexity struct {
	Score             float64 `json:"score"`              // 0.0 to 1.0, higher = more complex
	TokenCount        int     `json:"token_count"`        // Estimated token count
	DomainSpecific    bool    `json:"domain_specific"`    // Whether query is domain-specific
	TechnicalLevel    string  `json:"technical_level"`    // "beginner", "intermediate", "advanced", "expert"
	RequiresReasoning bool    `json:"requires_reasoning"` // Whether query requires complex reasoning
	ContextLength     int     `json:"context_length"`     // Required context length
}

// ModelCapability represents the capabilities of a model
type ModelCapability struct {
	MaxTokens        int                `json:"max_tokens"`
	ReasoningAbility float64            `json:"reasoning_ability"` // 0.0 to 1.0
	DomainExpertise  map[string]float64 `json:"domain_expertise"`  // domain -> expertise score
	TechnicalLevel   string             `json:"technical_level"`   // "beginner", "intermediate", "advanced", "expert"
	Speed            float64            `json:"speed"`             // tokens per second
	Accuracy         float64            `json:"accuracy"`          // 0.0 to 1.0
	Cost             float64            `json:"cost"`              // relative cost
}

// RoutingDecision represents the final routing decision
type RoutingDecision struct {
	SelectedDomain    string             `json:"selected_domain"`
	SelectedModel     string             `json:"selected_model"`
	Confidence        float64            `json:"confidence"` // 0.0 to 1.0
	Reasoning         string             `json:"reasoning"`
	FallbackDomain    string             `json:"fallback_domain"`
	AlternativeRoutes []AlternativeRoute `json:"alternative_routes"`
	EstimatedLatency  time.Duration      `json:"estimated_latency"`
	EstimatedCost     float64            `json:"estimated_cost"`
}

// AlternativeRoute represents an alternative routing option
type AlternativeRoute struct {
	Domain    string  `json:"domain"`
	Model     string  `json:"model"`
	Score     float64 `json:"score"`
	Reasoning string  `json:"reasoning"`
}

// IntelligentRouter provides intelligent model selection based on query analysis
type IntelligentRouter struct {
	domainManager     *domain.DomainManager
	modelCapabilities map[string]*ModelCapability
	queryHistory      map[string]*QueryComplexity
	performanceStats  map[string]*PerformanceStats
	mu                sync.RWMutex
}

// PerformanceStats tracks performance metrics for routing decisions
type PerformanceStats struct {
	TotalQueries     int           `json:"total_queries"`
	SuccessfulRoutes int           `json:"successful_routes"`
	AverageLatency   time.Duration `json:"average_latency"`
	AverageAccuracy  float64       `json:"average_accuracy"`
	LastUpdated      time.Time     `json:"last_updated"`
}

// NewIntelligentRouter creates a new intelligent router
func NewIntelligentRouter(domainManager *domain.DomainManager) *IntelligentRouter {
	return &IntelligentRouter{
		domainManager:     domainManager,
		modelCapabilities: make(map[string]*ModelCapability),
		queryHistory:      make(map[string]*QueryComplexity),
		performanceStats:  make(map[string]*PerformanceStats),
	}
}

// InitializeCapabilities initializes model capabilities based on domain configurations
func (ir *IntelligentRouter) InitializeCapabilities() error {
	ir.mu.Lock()
	defer ir.mu.Unlock()

	domains := ir.domainManager.ListDomainConfigs()

	for domainName, config := range domains {
		capability := &ModelCapability{
			MaxTokens:        config.MaxTokens,
			ReasoningAbility: ir.calculateReasoningAbility(config),
			DomainExpertise:  ir.calculateDomainExpertise(config),
			TechnicalLevel:   ir.determineTechnicalLevel(config),
			Speed:            ir.estimateModelSpeed(config),
			Accuracy:         ir.estimateModelAccuracy(config),
			Cost:             ir.estimateModelCost(config),
		}

		ir.modelCapabilities[domainName] = capability
	}

	return nil
}

// RouteQuery intelligently routes a query to the most appropriate model
func (ir *IntelligentRouter) RouteQuery(ctx context.Context, query string, userDomains []string, contextData map[string]interface{}) (*RoutingDecision, error) {
	// Analyze query complexity
	complexity, err := ir.analyzeQueryComplexity(query, contextData)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze query complexity: %w", err)
	}

	// Get available domains for user
	availableDomains := ir.getAvailableDomains(userDomains)
	if len(availableDomains) == 0 {
		availableDomains = []string{ir.domainManager.GetDefaultDomain()}
	}

	// Score each available domain
	domainScores := make(map[string]float64)
	for _, domainName := range availableDomains {
		score := ir.calculateDomainScore(domainName, query, complexity, contextData)
		domainScores[domainName] = score
	}

	// Select best domain
	selectedDomain := ir.selectBestDomain(domainScores)

	// Get model capabilities for selected domain
	capability, exists := ir.modelCapabilities[selectedDomain]
	if !exists {
		return nil, fmt.Errorf("no capabilities found for domain: %s", selectedDomain)
	}

	// Check if model can handle the query
	if !ir.canModelHandleQuery(capability, complexity) {
		// Find fallback domain
		fallbackDomain := ir.findFallbackDomain(availableDomains, complexity)
		selectedDomain = fallbackDomain
		capability = ir.modelCapabilities[fallbackDomain]
	}

	// Generate alternative routes
	alternatives := ir.generateAlternativeRoutes(availableDomains, domainScores, complexity)

	// Calculate estimated metrics
	latency := ir.estimateLatency(capability, complexity)
	cost := ir.estimateCost(capability, complexity)

	// Create routing decision
	decision := &RoutingDecision{
		SelectedDomain:    selectedDomain,
		SelectedModel:     ir.getModelName(selectedDomain),
		Confidence:        ir.calculateConfidence(domainScores, selectedDomain),
		Reasoning:         ir.generateReasoning(selectedDomain, complexity, domainScores),
		FallbackDomain:    ir.findFallbackDomain(availableDomains, complexity),
		AlternativeRoutes: alternatives,
		EstimatedLatency:  latency,
		EstimatedCost:     cost,
	}

	// Update performance tracking
	ir.updatePerformanceStats(selectedDomain)

	return decision, nil
}

// analyzeQueryComplexity analyzes the complexity of a query
func (ir *IntelligentRouter) analyzeQueryComplexity(query string, contextData map[string]interface{}) (*QueryComplexity, error) {
	// Check if we have cached analysis
	queryHash := ir.hashQuery(query)
	ir.mu.RLock()
	if ir.queryHistory != nil {
		if cached, exists := ir.queryHistory[queryHash]; exists {
			ir.mu.RUnlock()
			return cached, nil
		}
	}
	ir.mu.RUnlock()

	complexity := &QueryComplexity{}

	// Analyze token count
	complexity.TokenCount = ir.estimateTokenCount(query)

	// Analyze domain specificity
	complexity.DomainSpecific = ir.isDomainSpecific(query)

	// Analyze technical level
	complexity.TechnicalLevel = ir.analyzeTechnicalLevel(query)

	// Analyze reasoning requirements
	complexity.RequiresReasoning = ir.requiresComplexReasoning(query)

	// Long or detailed queries typically require reasoning even without explicit keywords
	if !complexity.RequiresReasoning && complexity.TokenCount > 30 {
		complexity.RequiresReasoning = true
	}

	// Calculate overall complexity score
	complexity.Score = ir.calculateComplexityScore(complexity)

	// Analyze context requirements
	complexity.ContextLength = ir.estimateContextLength(query, contextData)

	// Cache the analysis
	ir.mu.Lock()
	if ir.queryHistory == nil {
		ir.queryHistory = make(map[string]*QueryComplexity)
	}
	ir.queryHistory[queryHash] = complexity
	ir.mu.Unlock()

	return complexity, nil
}

// calculateDomainScore calculates how well a domain matches a query
func (ir *IntelligentRouter) calculateDomainScore(domainName, query string, complexity *QueryComplexity, contextData map[string]interface{}) float64 {
	config, exists := ir.domainManager.GetDomainConfig(domainName)
	if !exists {
		return 0.0
	}

	score := 0.0

	// Keyword matching (40% weight)
	keywordScore := ir.calculateKeywordScore(query, config.Keywords)
	score += keywordScore * 0.4

	// Domain expertise (30% weight)
	expertiseScore := ir.calculateExpertiseScore(domainName, complexity)
	score += expertiseScore * 0.3

	// Technical level matching (20% weight)
	technicalScore := ir.calculateTechnicalScore(domainName, complexity)
	score += technicalScore * 0.2

	// Performance considerations (10% weight)
	performanceScore := ir.calculatePerformanceScore(domainName, complexity)
	score += performanceScore * 0.1

	return math.Min(score, 1.0)
}

// calculateKeywordScore calculates score based on keyword matches
func (ir *IntelligentRouter) calculateKeywordScore(query string, keywords []string) float64 {
	if len(keywords) == 0 {
		return 0.5 // Neutral score if no keywords
	}

	queryLower := strings.ToLower(query)
	matches := 0
	for _, keyword := range keywords {
		if strings.Contains(queryLower, strings.ToLower(keyword)) {
			matches++
		}
	}

	return float64(matches) / float64(len(keywords))
}

// calculateExpertiseScore calculates score based on domain expertise
func (ir *IntelligentRouter) calculateExpertiseScore(domainName string, complexity *QueryComplexity) float64 {
	capability, exists := ir.modelCapabilities[domainName]
	if !exists {
		return 0.5
	}

	// Base expertise on reasoning ability and domain-specific knowledge
	expertise := capability.ReasoningAbility * 0.6

	// Add domain-specific expertise if applicable
	if complexity.DomainSpecific {
		// This would need domain-specific expertise mapping
		expertise += 0.3
	}

	return math.Min(expertise, 1.0)
}

// calculateTechnicalScore calculates score based on technical level matching
func (ir *IntelligentRouter) calculateTechnicalScore(domainName string, complexity *QueryComplexity) float64 {
	capability, exists := ir.modelCapabilities[domainName]
	if !exists {
		return 0.5
	}

	// Map technical levels to scores
	levelScores := map[string]float64{
		"beginner":     0.2,
		"intermediate": 0.5,
		"advanced":     0.8,
		"expert":       1.0,
	}

	modelLevel := levelScores[capability.TechnicalLevel]
	queryLevel := levelScores[complexity.TechnicalLevel]

	// Calculate similarity score
	diff := math.Abs(modelLevel - queryLevel)
	return 1.0 - diff
}

// calculatePerformanceScore calculates score based on performance considerations
func (ir *IntelligentRouter) calculatePerformanceScore(domainName string, complexity *QueryComplexity) float64 {
	capability, exists := ir.modelCapabilities[domainName]
	if !exists {
		return 0.5
	}

	// Check if model can handle the query
	if complexity.TokenCount > capability.MaxTokens {
		return 0.0 // Cannot handle
	}

	// Calculate performance score based on speed, accuracy, and cost
	speedScore := math.Min(capability.Speed/100.0, 1.0) // Normalize speed
	accuracyScore := capability.Accuracy
	costScore := 1.0 - math.Min(capability.Cost/10.0, 1.0) // Lower cost is better

	return (speedScore + accuracyScore + costScore) / 3.0
}

// selectBestDomain selects the domain with the highest score
func (ir *IntelligentRouter) selectBestDomain(domainScores map[string]float64) string {
	bestDomain := ""
	bestScore := 0.0

	for domain, score := range domainScores {
		if score > bestScore {
			bestScore = score
			bestDomain = domain
		}
	}

	if bestDomain == "" {
		return ir.domainManager.GetDefaultDomain()
	}

	return bestDomain
}

// canModelHandleQuery checks if a model can handle the query
func (ir *IntelligentRouter) canModelHandleQuery(capability *ModelCapability, complexity *QueryComplexity) bool {
	// Check token limits
	if complexity.TokenCount > capability.MaxTokens {
		return false
	}

	// Check reasoning requirements
	if complexity.RequiresReasoning && capability.ReasoningAbility < 0.5 {
		return false
	}

	// Check technical level compatibility
	levelScores := map[string]float64{
		"beginner":     0.2,
		"intermediate": 0.5,
		"advanced":     0.8,
		"expert":       1.0,
	}

	modelLevel := levelScores[capability.TechnicalLevel]
	queryLevel := levelScores[complexity.TechnicalLevel]

	// Model should be at least as capable as the query requires
	return modelLevel >= queryLevel
}

// findFallbackDomain finds a suitable fallback domain
func (ir *IntelligentRouter) findFallbackDomain(availableDomains []string, complexity *QueryComplexity) string {
	// Find domains that can handle the query
	suitableDomains := make([]string, 0)

	for _, domain := range availableDomains {
		capability, exists := ir.modelCapabilities[domain]
		if exists && ir.canModelHandleQuery(capability, complexity) {
			suitableDomains = append(suitableDomains, domain)
		}
	}

	if len(suitableDomains) > 0 {
		// Return the first suitable domain
		return suitableDomains[0]
	}

	// Fall back to default domain
	return ir.domainManager.GetDefaultDomain()
}

// generateAlternativeRoutes generates alternative routing options
func (ir *IntelligentRouter) generateAlternativeRoutes(availableDomains []string, domainScores map[string]float64, complexity *QueryComplexity) []AlternativeRoute {
	alternatives := make([]AlternativeRoute, 0)

	// Sort domains by score
	type domainScore struct {
		domain string
		score  float64
	}

	sortedDomains := make([]domainScore, 0, len(domainScores))
	for domain, score := range domainScores {
		sortedDomains = append(sortedDomains, domainScore{domain, score})
	}

	// Sort by score (descending)
	for i := 0; i < len(sortedDomains)-1; i++ {
		for j := i + 1; j < len(sortedDomains); j++ {
			if sortedDomains[i].score < sortedDomains[j].score {
				sortedDomains[i], sortedDomains[j] = sortedDomains[j], sortedDomains[i]
			}
		}
	}

	// Generate alternatives (top 3)
	for i, ds := range sortedDomains {
		if i >= 3 {
			break
		}

		alternative := AlternativeRoute{
			Domain:    ds.domain,
			Model:     ir.getModelName(ds.domain),
			Score:     ds.score,
			Reasoning: ir.generateAlternativeReasoning(ds.domain, complexity),
		}
		alternatives = append(alternatives, alternative)
	}

	return alternatives
}

// Helper methods for analysis

func (ir *IntelligentRouter) estimateTokenCount(query string) int {
	// Simple estimation: ~4 characters per token
	return len(query) / 4
}

func (ir *IntelligentRouter) isDomainSpecific(query string) bool {
	// Check for domain-specific keywords
	domainKeywords := []string{
		"sql", "database", "query", "table", "join",
		"vector", "embedding", "similarity", "cosine",
		"blockchain", "transaction", "smart contract",
		"financial", "accounting", "ledger", "reconcile",
		"ai", "machine learning", "neural network",
	}

	queryLower := strings.ToLower(query)
	for _, keyword := range domainKeywords {
		if strings.Contains(queryLower, keyword) {
			return true
		}
	}

	return false
}

func (ir *IntelligentRouter) analyzeTechnicalLevel(query string) string {
	// Analyze technical complexity indicators
	expertKeywords := []string{"algorithm", "optimization", "architecture", "implementation", "performance"}
	advancedKeywords := []string{"analysis", "design", "development", "integration", "testing"}
	intermediateKeywords := []string{"how to", "explain", "example", "tutorial", "guide"}

	queryLower := strings.ToLower(query)

	expertCount := 0
	advancedCount := 0
	intermediateCount := 0

	for _, keyword := range expertKeywords {
		if strings.Contains(queryLower, keyword) {
			expertCount++
		}
	}

	for _, keyword := range advancedKeywords {
		if strings.Contains(queryLower, keyword) {
			advancedCount++
		}
	}

	for _, keyword := range intermediateKeywords {
		if strings.Contains(queryLower, keyword) {
			intermediateCount++
		}
	}

	if expertCount > advancedCount && expertCount > intermediateCount {
		return "expert"
	} else if advancedCount > intermediateCount {
		return "advanced"
	} else if intermediateCount > 0 {
		return "intermediate"
	}

	return "beginner"
}

func (ir *IntelligentRouter) requiresComplexReasoning(query string) bool {
	reasoningKeywords := []string{
		"analyze", "compare", "evaluate", "explain why", "what if", "how would",
		"pros and cons", "advantages and disadvantages", "trade-offs", "implications",
		"reasoning", "logic", "deduce", "infer", "conclude",
	}

	queryLower := strings.ToLower(query)
	for _, keyword := range reasoningKeywords {
		if strings.Contains(queryLower, keyword) {
			return true
		}
	}

	return false
}

func (ir *IntelligentRouter) calculateComplexityScore(complexity *QueryComplexity) float64 {
	score := 0.0

	// Token count contribution (30%)
	tokenScore := math.Min(float64(complexity.TokenCount)/1000.0, 1.0)
	score += tokenScore * 0.3

	// Domain specificity contribution (25%)
	if complexity.DomainSpecific {
		score += 0.25
	}

	// Technical level contribution (25%)
	levelScores := map[string]float64{
		"beginner":     0.2,
		"intermediate": 0.5,
		"advanced":     0.8,
		"expert":       1.0,
	}
	score += levelScores[complexity.TechnicalLevel] * 0.25

	// Reasoning requirements contribution (20%)
	if complexity.RequiresReasoning {
		score += 0.2
	}

	return math.Min(score, 1.0)
}

func (ir *IntelligentRouter) estimateContextLength(query string, contextData map[string]interface{}) int {
	baseLength := len(query)

	// Add context data length
	if contextData != nil {
		contextJSON, _ := json.Marshal(contextData)
		baseLength += len(contextJSON)
	}

	return baseLength
}

func (ir *IntelligentRouter) hashQuery(query string) string {
	// Simple hash for caching
	return fmt.Sprintf("%d", len(query))
}

func (ir *IntelligentRouter) getAvailableDomains(userDomains []string) []string {
	if len(userDomains) == 0 {
		return ir.domainManager.ListDomains()
	}
	return userDomains
}

func (ir *IntelligentRouter) getModelName(domainName string) string {
	config, exists := ir.domainManager.GetDomainConfig(domainName)
	if !exists {
		return "unknown"
	}

	if config.ModelName != "" {
		return config.ModelName
	}

	// Extract model name from path
	parts := strings.Split(config.ModelPath, "/")
	if len(parts) > 0 {
		return parts[len(parts)-1]
	}

	return "local-model"
}

func (ir *IntelligentRouter) calculateConfidence(domainScores map[string]float64, selectedDomain string) float64 {
	selectedScore := domainScores[selectedDomain]
	if selectedScore == 0 {
		return 0.0
	}

	// Calculate confidence based on score difference with second best
	scores := make([]float64, 0, len(domainScores))
	for _, score := range domainScores {
		scores = append(scores, score)
	}

	// Sort scores
	for i := 0; i < len(scores)-1; i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[i] < scores[j] {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	if len(scores) < 2 {
		return selectedScore
	}

	// Confidence based on gap between first and second
	gap := scores[0] - scores[1]
	return math.Min(selectedScore+gap, 1.0)
}

func (ir *IntelligentRouter) generateReasoning(selectedDomain string, complexity *QueryComplexity, domainScores map[string]float64) string {
	score := domainScores[selectedDomain]

	reasoning := fmt.Sprintf("Selected %s (score: %.2f) based on ", selectedDomain, score)

	// Add reasoning based on complexity
	if complexity.DomainSpecific {
		reasoning += "domain-specific requirements, "
	}

	if complexity.RequiresReasoning {
		reasoning += "complex reasoning needs, "
	}

	reasoning += fmt.Sprintf("technical level: %s", complexity.TechnicalLevel)

	return reasoning
}

func (ir *IntelligentRouter) generateAlternativeReasoning(domain string, complexity *QueryComplexity) string {
	return fmt.Sprintf("Alternative route for %s domain, suitable for %s level queries", domain, complexity.TechnicalLevel)
}

func (ir *IntelligentRouter) estimateLatency(capability *ModelCapability, complexity *QueryComplexity) time.Duration {
	// Estimate based on model speed and query complexity
	baseLatency := time.Duration(complexity.TokenCount) * time.Millisecond / time.Duration(capability.Speed)

	// Add complexity factor
	if complexity.RequiresReasoning {
		baseLatency *= 2
	}

	return baseLatency
}

func (ir *IntelligentRouter) estimateCost(capability *ModelCapability, complexity *QueryComplexity) float64 {
	// Estimate based on model cost and query complexity
	baseCost := capability.Cost * float64(complexity.TokenCount) / 1000.0

	// Add complexity factor
	if complexity.RequiresReasoning {
		baseCost *= 1.5
	}

	return baseCost
}

func (ir *IntelligentRouter) updatePerformanceStats(domain string) {
	ir.mu.Lock()
	defer ir.mu.Unlock()

	stats, exists := ir.performanceStats[domain]
	if !exists {
		stats = &PerformanceStats{
			TotalQueries:     0,
			SuccessfulRoutes: 0,
			AverageLatency:   0,
			AverageAccuracy:  0,
			LastUpdated:      time.Now(),
		}
		ir.performanceStats[domain] = stats
	}

	stats.TotalQueries++
	stats.SuccessfulRoutes++
	stats.LastUpdated = time.Now()
}

// Model capability calculation methods

func (ir *IntelligentRouter) calculateReasoningAbility(config *domain.DomainConfig) float64 {
	// Base reasoning ability on model type and configuration
	baseScore := 0.5

	// Adjust based on model path/name
	if strings.Contains(config.ModelPath, "granite") {
		baseScore = 0.8 // Granite models are good at reasoning
	} else if strings.Contains(config.ModelPath, "phi") {
		baseScore = 0.7 // Phi models have good reasoning
	} else if strings.Contains(config.ModelPath, "vaultgemma") {
		baseScore = 0.6 // VaultGemma is decent at reasoning
	}

	// Adjust based on temperature (lower temperature = more deterministic = better reasoning)
	if config.Temperature < 0.3 {
		baseScore += 0.1
	}

	return math.Min(baseScore, 1.0)
}

func (ir *IntelligentRouter) calculateDomainExpertise(config *domain.DomainConfig) map[string]float64 {
	expertise := make(map[string]float64)

	// Base expertise on attention weights
	for domain, weight := range config.AttentionWeights {
		expertise[domain] = float64(weight)
	}

	// Add domain-specific expertise based on tags
	for _, tag := range config.DomainTags {
		expertise[tag] = 0.8 // High expertise for domain tags
	}

	return expertise
}

func (ir *IntelligentRouter) determineTechnicalLevel(config *domain.DomainConfig) string {
	// Determine technical level based on domain and configuration
	if strings.Contains(config.Name, "Expert") || strings.Contains(config.Name, "Advanced") {
		return "expert"
	}

	if strings.Contains(config.Name, "Intermediate") || strings.Contains(config.Name, "Professional") {
		return "advanced"
	}

	if strings.Contains(config.Name, "Beginner") || strings.Contains(config.Name, "Basic") {
		return "intermediate"
	}

	// Default based on layer
	switch config.Layer {
	case "layer1":
		return "advanced"
	case "layer2":
		return "expert"
	case "layer3":
		return "advanced"
	default:
		return "intermediate"
	}
}

func (ir *IntelligentRouter) estimateModelSpeed(config *domain.DomainConfig) float64 {
	// Estimate speed based on model type and size
	baseSpeed := 50.0 // tokens per second

	// Adjust based on model type
	if strings.Contains(config.ModelPath, "granite") {
		baseSpeed = 40.0 // Granite models are slower but more capable
	} else if strings.Contains(config.ModelPath, "phi") {
		baseSpeed = 60.0 // Phi models are faster
	} else if strings.Contains(config.ModelPath, "vaultgemma") {
		baseSpeed = 70.0 // VaultGemma is fastest
	}

	// Adjust based on max tokens (larger models are slower)
	if config.MaxTokens > 2000 {
		baseSpeed *= 0.8
	}

	return baseSpeed
}

func (ir *IntelligentRouter) estimateModelAccuracy(config *domain.DomainConfig) float64 {
	// Estimate accuracy based on model type and configuration
	baseAccuracy := 0.7

	// Adjust based on model type
	if strings.Contains(config.ModelPath, "granite") {
		baseAccuracy = 0.85 // Granite models are more accurate
	} else if strings.Contains(config.ModelPath, "phi") {
		baseAccuracy = 0.8 // Phi models are accurate
	} else if strings.Contains(config.ModelPath, "vaultgemma") {
		baseAccuracy = 0.75 // VaultGemma is decent
	}

	// Adjust based on temperature (lower temperature = more accurate)
	if config.Temperature < 0.3 {
		baseAccuracy += 0.05
	}

	return math.Min(baseAccuracy, 1.0)
}

func (ir *IntelligentRouter) estimateModelCost(config *domain.DomainConfig) float64 {
	// Estimate relative cost based on model type and size
	baseCost := 1.0

	// Adjust based on model type
	if strings.Contains(config.ModelPath, "granite") {
		baseCost = 2.0 // Granite models are more expensive
	} else if strings.Contains(config.ModelPath, "phi") {
		baseCost = 1.5 // Phi models are moderately expensive
	} else if strings.Contains(config.ModelPath, "vaultgemma") {
		baseCost = 1.0 // VaultGemma is cheapest
	}

	// Adjust based on max tokens (larger models are more expensive)
	if config.MaxTokens > 2000 {
		baseCost *= 1.5
	}

	return baseCost
}
