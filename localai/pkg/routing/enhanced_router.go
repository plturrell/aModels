package routing

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
)

// EnhancedIntelligentRouter provides enhanced intelligent routing with configuration support
type EnhancedIntelligentRouter struct {
	*IntelligentRouter
	configManager *RoutingConfigManager
	ruleEngine    *RuleEngine
	loadBalancer  *LoadBalancer
	performance   *PerformanceMonitor
	mu            sync.RWMutex
}

// LoadBalancer handles load balancing across domains
type LoadBalancer struct {
	strategy       string
	healthCheck    bool
	maxRetries     int
	retryDelay     time.Duration
	circuitBreaker bool
	threshold      float64
	domainStats    map[string]*DomainStats
	mu             sync.RWMutex
}

// DomainStats tracks statistics for a domain
type DomainStats struct {
	TotalRequests    int64         `json:"total_requests"`
	SuccessfulRequests int64       `json:"successful_requests"`
	FailedRequests  int64         `json:"failed_requests"`
	AverageLatency  time.Duration  `json:"average_latency"`
	LastRequest     time.Time      `json:"last_request"`
	HealthStatus    string         `json:"health_status"` // "healthy", "degraded", "unhealthy"
	ErrorRate       float64        `json:"error_rate"`
	Throughput      float64        `json:"throughput"` // requests per minute
}

// PerformanceMonitor tracks performance metrics
type PerformanceMonitor struct {
	metrics      map[string]*PerformanceMetrics
	alerts       []Alert
	thresholds   map[string]float64
	mu           sync.RWMutex
}

// PerformanceMetrics tracks performance for a domain
type PerformanceMetrics struct {
	Domain          string        `json:"domain"`
	TotalRequests   int64         `json:"total_requests"`
	SuccessfulRequests int64      `json:"successful_requests"`
	FailedRequests int64          `json:"failed_requests"`
	AverageLatency time.Duration  `json:"average_latency"`
	MaxLatency     time.Duration  `json:"max_latency"`
	MinLatency     time.Duration  `json:"min_latency"`
	ErrorRate      float64        `json:"error_rate"`
	Throughput     float64        `json:"throughput"`
	LastUpdated    time.Time      `json:"last_updated"`
}

// Alert represents a performance alert
type Alert struct {
	ID          string    `json:"id"`
	Type        string    `json:"type"`
	Severity    string    `json:"severity"`
	Message     string    `json:"message"`
	Domain      string    `json:"domain"`
	Value       float64   `json:"value"`
	Threshold   float64   `json:"threshold"`
	CreatedAt   time.Time `json:"created_at"`
	ResolvedAt  *time.Time `json:"resolved_at,omitempty"`
}

// NewEnhancedIntelligentRouter creates a new enhanced intelligent router
func NewEnhancedIntelligentRouter(domainManager *domain.DomainManager) *EnhancedIntelligentRouter {
	baseRouter := NewIntelligentRouter(domainManager)
	
	return &EnhancedIntelligentRouter{
		IntelligentRouter: baseRouter,
		configManager:     NewRoutingConfigManager(),
		ruleEngine:        &RuleEngine{},
		loadBalancer: &LoadBalancer{
			strategy:       "weighted",
			healthCheck:    true,
			maxRetries:     3,
			retryDelay:     time.Second,
			circuitBreaker: true,
			threshold:      0.5,
			domainStats:    make(map[string]*DomainStats),
		},
		performance: &PerformanceMonitor{
			metrics:    make(map[string]*PerformanceMetrics),
			alerts:     make([]Alert, 0),
			thresholds: make(map[string]float64),
		},
	}
}

// InitializeEnhancedRouter initializes the enhanced router with configuration
func (eir *EnhancedIntelligentRouter) InitializeEnhancedRouter(configPath string) error {
	// Initialize base router
	if err := eir.InitializeCapabilities(); err != nil {
		return fmt.Errorf("failed to initialize base router: %w", err)
	}
	
	// Load routing configuration
	if err := eir.configManager.LoadRoutingConfig(configPath); err != nil {
		return fmt.Errorf("failed to load routing config: %w", err)
	}
	
	// Initialize performance monitoring
	eir.initializePerformanceMonitoring()
	
	// Start background tasks
	go eir.startBackgroundTasks()
	
	return nil
}

// RouteQueryEnhanced provides enhanced routing with configuration support
func (eir *EnhancedIntelligentRouter) RouteQueryEnhanced(ctx context.Context, query string, userDomains []string, contextData map[string]interface{}) (*RoutingDecision, error) {
	// Analyze query complexity
	complexity, err := eir.analyzeQueryComplexity(query, contextData)
	if err != nil {
		return nil, fmt.Errorf("failed to analyze query complexity: %w", err)
	}
	
	// Evaluate routing rules
	matchingRules, err := eir.configManager.EvaluateRules(query, userDomains, complexity, contextData)
	if err != nil {
		return nil, fmt.Errorf("failed to evaluate routing rules: %w", err)
	}
	
	// Apply routing rules
	decision, err := eir.applyRoutingRules(ctx, matchingRules, query, userDomains, complexity, contextData)
	if err != nil {
		return nil, fmt.Errorf("failed to apply routing rules: %w", err)
	}
	
	// Apply load balancing
	decision, err = eir.applyLoadBalancing(decision, userDomains, complexity)
	if err != nil {
		return nil, fmt.Errorf("failed to apply load balancing: %w", err)
	}
	
	// Update performance metrics
	eir.updatePerformanceMetrics(decision.SelectedDomain, complexity)
	
	// Check for alerts
	eir.checkPerformanceAlerts(decision.SelectedDomain)
	
	return decision, nil
}

// applyRoutingRules applies routing rules to determine the best domain
func (eir *EnhancedIntelligentRouter) applyRoutingRules(ctx context.Context, rules []RoutingRule, query string, userDomains []string, complexity *QueryComplexity, contextData map[string]interface{}) (*RoutingDecision, error) {
	if len(rules) == 0 {
		// No rules matched, use base router
		return eir.RouteQuery(ctx, query, userDomains, contextData)
	}
	
	// Apply the highest priority rule
	rule := rules[0]
	
	// Execute rule actions
	selectedDomain := eir.configManager.GetConfig().DefaultDomain
	reasoning := "No specific routing rule applied"
	
	for _, action := range rule.Actions {
		switch action.Type {
		case "route_to_domain":
			selectedDomain = action.Target
			reasoning = action.Parameters["reason"]
		case "set_priority":
			// Handle priority setting
		case "add_metadata":
			// Add metadata to context
			if contextData == nil {
				contextData = make(map[string]interface{})
			}
			contextData[action.Target] = action.Parameters["value"]
		case "log_event":
			// Log the event
			eir.logRoutingEvent(rule.ID, action, query, selectedDomain)
		}
	}
	
	// Create routing decision
	decision := &RoutingDecision{
		SelectedDomain:    selectedDomain,
		SelectedModel:     eir.getModelName(selectedDomain),
		Confidence:        eir.calculateRuleConfidence(rule, complexity),
		Reasoning:         reasoning,
		FallbackDomain:    eir.configManager.GetConfig().FallbackDomain,
		AlternativeRoutes: eir.generateAlternativeRoutesFromRules(rules[1:], userDomains, complexity),
		EstimatedLatency:  eir.estimateLatency(eir.modelCapabilities[selectedDomain], complexity),
		EstimatedCost:     eir.estimateCost(eir.modelCapabilities[selectedDomain], complexity),
	}
	
	return decision, nil
}

// applyLoadBalancing applies load balancing to the routing decision
func (eir *EnhancedIntelligentRouter) applyLoadBalancing(decision *RoutingDecision, userDomains []string, complexity *QueryComplexity) (*RoutingDecision, error) {
	if !eir.loadBalancer.healthCheck {
		return decision, nil
	}
	
	// Check if selected domain is healthy
	if !eir.isDomainHealthy(decision.SelectedDomain) {
		// Find alternative healthy domain
		alternativeDomain := eir.findHealthyAlternative(userDomains, complexity)
		if alternativeDomain != "" {
			decision.SelectedDomain = alternativeDomain
			decision.SelectedModel = eir.getModelName(alternativeDomain)
			decision.Reasoning += " (load balanced due to health check)"
		}
	}
	
	return decision, nil
}

// isDomainHealthy checks if a domain is healthy
func (eir *EnhancedIntelligentRouter) isDomainHealthy(domain string) bool {
	eir.loadBalancer.mu.RLock()
	defer eir.loadBalancer.mu.RUnlock()
	
	stats, exists := eir.loadBalancer.domainStats[domain]
	if !exists {
		return true // Assume healthy if no stats
	}
	
	// Check error rate
	if stats.ErrorRate > eir.loadBalancer.threshold {
		return false
	}
	
	// Check health status
	if stats.HealthStatus == "unhealthy" {
		return false
	}
	
	return true
}

// findHealthyAlternative finds a healthy alternative domain
func (eir *EnhancedIntelligentRouter) findHealthyAlternative(userDomains []string, complexity *QueryComplexity) string {
	for _, domain := range userDomains {
		if eir.isDomainHealthy(domain) && eir.canModelHandleQuery(eir.modelCapabilities[domain], complexity) {
			return domain
		}
	}
	
	// Fall back to default domain
	if eir.isDomainHealthy(eir.configManager.GetConfig().DefaultDomain) {
		return eir.configManager.GetConfig().DefaultDomain
	}
	
	return ""
}

// calculateRuleConfidence calculates confidence based on rule priority and complexity
func (eir *EnhancedIntelligentRouter) calculateRuleConfidence(rule RoutingRule, complexity *QueryComplexity) float64 {
	baseConfidence := float64(rule.Priority) / 100.0
	
	// Adjust based on complexity
	if complexity.Score > 0.7 {
		baseConfidence += 0.1
	}
	
	if complexity.RequiresReasoning {
		baseConfidence += 0.05
	}
	
	return math.Min(baseConfidence, 1.0)
}

// generateAlternativeRoutesFromRules generates alternative routes from remaining rules
func (eir *EnhancedIntelligentRouter) generateAlternativeRoutesFromRules(rules []RoutingRule, userDomains []string, complexity *QueryComplexity) []AlternativeRoute {
	_ = userDomains // Reserved for future domain filtering
	_ = complexity  // Reserved for future complexity-based scoring
	alternatives := make([]AlternativeRoute, 0)
	
	for i, rule := range rules {
		if i >= 3 { // Limit to top 3 alternatives
			break
		}
		
		// Find the route action
		for _, action := range rule.Actions {
			if action.Type == "route_to_domain" {
				alternative := AlternativeRoute{
					Domain:    action.Target,
					Model:     eir.getModelName(action.Target),
					Score:     float64(rule.Priority) / 100.0,
					Reasoning: action.Parameters["reason"],
				}
				alternatives = append(alternatives, alternative)
				break
			}
		}
	}
	
	return alternatives
}

// updatePerformanceMetrics updates performance metrics for a domain
func (eir *EnhancedIntelligentRouter) updatePerformanceMetrics(domain string, complexity *QueryComplexity) {
	_ = complexity // Reserved for future complexity-based metrics
	eir.performance.mu.Lock()
	defer eir.performance.mu.Unlock()
	
	metrics, exists := eir.performance.metrics[domain]
	if !exists {
		metrics = &PerformanceMetrics{
			Domain:       domain,
			LastUpdated:  time.Now(),
		}
		eir.performance.metrics[domain] = metrics
	}
	
	metrics.TotalRequests++
	metrics.LastUpdated = time.Now()
	
	// Update load balancer stats
	eir.loadBalancer.mu.Lock()
	stats, exists := eir.loadBalancer.domainStats[domain]
	if !exists {
		stats = &DomainStats{
			HealthStatus: "healthy",
		}
		eir.loadBalancer.domainStats[domain] = stats
	}
	stats.TotalRequests++
	stats.LastRequest = time.Now()
	eir.loadBalancer.mu.Unlock()
}

// checkPerformanceAlerts checks for performance alerts
func (eir *EnhancedIntelligentRouter) checkPerformanceAlerts(domain string) {
	eir.performance.mu.RLock()
	metrics, exists := eir.performance.metrics[domain]
	if !exists {
		eir.performance.mu.RUnlock()
		return
	}
	eir.performance.mu.RUnlock()
	
	// Check error rate threshold
	if metrics.ErrorRate > eir.performance.thresholds["error_rate"] {
		alert := Alert{
			ID:        fmt.Sprintf("alert_%d", time.Now().UnixNano()),
			Type:      "error_rate",
			Severity:  "warning",
			Message:   fmt.Sprintf("High error rate for domain %s: %.2f%%", domain, metrics.ErrorRate*100),
			Domain:    domain,
			Value:     metrics.ErrorRate,
			Threshold: eir.performance.thresholds["error_rate"],
			CreatedAt: time.Now(),
		}
		eir.addAlert(alert)
	}
	
	// Check latency threshold
	if metrics.AverageLatency > time.Duration(eir.performance.thresholds["latency"])*time.Millisecond {
		alert := Alert{
			ID:        fmt.Sprintf("alert_%d", time.Now().UnixNano()),
			Type:      "latency",
			Severity:  "warning",
			Message:   fmt.Sprintf("High latency for domain %s: %v", domain, metrics.AverageLatency),
			Domain:    domain,
			Value:     float64(metrics.AverageLatency.Milliseconds()),
			Threshold: eir.performance.thresholds["latency"],
			CreatedAt: time.Now(),
		}
		eir.addAlert(alert)
	}
}

// addAlert adds a performance alert
func (eir *EnhancedIntelligentRouter) addAlert(alert Alert) {
	eir.performance.mu.Lock()
	defer eir.performance.mu.Unlock()
	
	eir.performance.alerts = append(eir.performance.alerts, alert)
	
	// Keep only last 100 alerts
	if len(eir.performance.alerts) > 100 {
		eir.performance.alerts = eir.performance.alerts[len(eir.performance.alerts)-100:]
	}
}

// initializePerformanceMonitoring initializes performance monitoring
func (eir *EnhancedIntelligentRouter) initializePerformanceMonitoring() {
	config := eir.configManager.GetConfig()
	
	// Set up thresholds
	eir.performance.thresholds["error_rate"] = 0.05
	eir.performance.thresholds["latency"] = 5000
	eir.performance.thresholds["throughput"] = 1000
	
	// Override with config values if available
	if config.Monitoring.AlertThresholds != nil {
		for key, value := range config.Monitoring.AlertThresholds {
			eir.performance.thresholds[key] = value
		}
	}
}

// startBackgroundTasks starts background monitoring tasks
func (eir *EnhancedIntelligentRouter) startBackgroundTasks() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		eir.updateDomainHealth()
		eir.cleanupOldAlerts()
	}
}

// updateDomainHealth updates domain health status
func (eir *EnhancedIntelligentRouter) updateDomainHealth() {
	eir.loadBalancer.mu.Lock()
	defer eir.loadBalancer.mu.Unlock()
	
	for _, stats := range eir.loadBalancer.domainStats {
		// Calculate error rate
		if stats.TotalRequests > 0 {
			stats.ErrorRate = float64(stats.FailedRequests) / float64(stats.TotalRequests)
		}
		
		// Update health status
		if stats.ErrorRate > 0.5 {
			stats.HealthStatus = "unhealthy"
		} else if stats.ErrorRate > 0.2 {
			stats.HealthStatus = "degraded"
		} else {
			stats.HealthStatus = "healthy"
		}
		
		// Calculate throughput
		if time.Since(stats.LastRequest) < time.Minute {
			stats.Throughput = float64(stats.TotalRequests) / time.Since(stats.LastRequest).Minutes()
		}
	}
}

// cleanupOldAlerts removes old alerts
func (eir *EnhancedIntelligentRouter) cleanupOldAlerts() {
	eir.performance.mu.Lock()
	defer eir.performance.mu.Unlock()
	
	cutoff := time.Now().Add(-24 * time.Hour)
	var activeAlerts []Alert
	
	for _, alert := range eir.performance.alerts {
		if alert.CreatedAt.After(cutoff) {
			activeAlerts = append(activeAlerts, alert)
		}
	}
	
	eir.performance.alerts = activeAlerts
}

// logRoutingEvent logs a routing event
func (eir *EnhancedIntelligentRouter) logRoutingEvent(ruleID string, action RuleAction, query string, domain string) {
	// In a real implementation, this would log to a proper logging system
	fmt.Printf("Routing Event: Rule %s, Action %s, Domain %s, Query: %s\n", 
		ruleID, action.Type, domain, query[:min(50, len(query))])
}

// GetPerformanceMetrics returns performance metrics for all domains
func (eir *EnhancedIntelligentRouter) GetPerformanceMetrics() map[string]*PerformanceMetrics {
	eir.performance.mu.RLock()
	defer eir.performance.mu.RUnlock()
	
	metrics := make(map[string]*PerformanceMetrics)
	for domain, metric := range eir.performance.metrics {
		metrics[domain] = metric
	}
	
	return metrics
}

// GetAlerts returns current performance alerts
func (eir *EnhancedIntelligentRouter) GetAlerts() []Alert {
	eir.performance.mu.RLock()
	defer eir.performance.mu.RUnlock()
	
	alerts := make([]Alert, len(eir.performance.alerts))
	copy(alerts, eir.performance.alerts)
	
	return alerts
}

// GetDomainStats returns domain statistics
func (eir *EnhancedIntelligentRouter) GetDomainStats() map[string]*DomainStats {
	eir.loadBalancer.mu.RLock()
	defer eir.loadBalancer.mu.RUnlock()
	
	stats := make(map[string]*DomainStats)
	for domain, stat := range eir.loadBalancer.domainStats {
		stats[domain] = stat
	}
	
	return stats
}

// UpdateDomainHealth manually updates domain health status
func (eir *EnhancedIntelligentRouter) UpdateDomainHealth(domain string, status string) {
	eir.loadBalancer.mu.Lock()
	defer eir.loadBalancer.mu.Unlock()
	
	stats, exists := eir.loadBalancer.domainStats[domain]
	if !exists {
		stats = &DomainStats{
			HealthStatus: status,
		}
		eir.loadBalancer.domainStats[domain] = stats
	} else {
		stats.HealthStatus = status
	}
}

// RecordRequestSuccess records a successful request
func (eir *EnhancedIntelligentRouter) RecordRequestSuccess(domain string, latency time.Duration) {
	eir.performance.mu.Lock()
	defer eir.performance.mu.Unlock()
	
	metrics, exists := eir.performance.metrics[domain]
	if !exists {
		metrics = &PerformanceMetrics{
			Domain:      domain,
			LastUpdated: time.Now(),
		}
		eir.performance.metrics[domain] = metrics
	}
	
	metrics.SuccessfulRequests++
	metrics.TotalRequests++
	
	// Update average latency
	if metrics.AverageLatency == 0 {
		metrics.AverageLatency = latency
	} else {
		metrics.AverageLatency = (metrics.AverageLatency + latency) / 2
	}
	
	// Update min/max latency
	if metrics.MinLatency == 0 || latency < metrics.MinLatency {
		metrics.MinLatency = latency
	}
	if latency > metrics.MaxLatency {
		metrics.MaxLatency = latency
	}
	
	// Update error rate
	if metrics.TotalRequests > 0 {
		metrics.ErrorRate = float64(metrics.FailedRequests) / float64(metrics.TotalRequests)
	}
	
	metrics.LastUpdated = time.Now()
	
	// Update load balancer stats
	eir.loadBalancer.mu.Lock()
	stats, exists := eir.loadBalancer.domainStats[domain]
	if !exists {
		stats = &DomainStats{
			HealthStatus: "healthy",
		}
		eir.loadBalancer.domainStats[domain] = stats
	}
	stats.SuccessfulRequests++
	stats.AverageLatency = (stats.AverageLatency + latency) / 2
	eir.loadBalancer.mu.Unlock()
}

// RecordRequestFailure records a failed request
func (eir *EnhancedIntelligentRouter) RecordRequestFailure(domain string, error string) {
	eir.performance.mu.Lock()
	defer eir.performance.mu.Unlock()
	
	metrics, exists := eir.performance.metrics[domain]
	if !exists {
		metrics = &PerformanceMetrics{
			Domain:      domain,
			LastUpdated: time.Now(),
		}
		eir.performance.metrics[domain] = metrics
	}
	
	metrics.FailedRequests++
	metrics.TotalRequests++
	
	// Update error rate
	if metrics.TotalRequests > 0 {
		metrics.ErrorRate = float64(metrics.FailedRequests) / float64(metrics.TotalRequests)
	}
	
	metrics.LastUpdated = time.Now()
	
	// Update load balancer stats
	eir.loadBalancer.mu.Lock()
	stats, exists := eir.loadBalancer.domainStats[domain]
	if !exists {
		stats = &DomainStats{
			HealthStatus: "healthy",
		}
		eir.loadBalancer.domainStats[domain] = stats
	}
	stats.FailedRequests++
	eir.loadBalancer.mu.Unlock()
}

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
