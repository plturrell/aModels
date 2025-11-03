package cost

import (
	"fmt"
	"sync"
	"time"
)

// CostOptimizer tracks and optimizes compute costs
type CostOptimizer struct {
	mu              sync.RWMutex
	operationCosts  map[string]float64
	budget          float64
	spent           float64
	recommendations []string
	costHistory     []CostEntry
	alerts          []CostAlert
}

// CostEntry represents a cost entry
type CostEntry struct {
	Operation   string
	Cost        float64
	Timestamp   time.Time
	AgentID     string
	ResourceType string
}

// CostAlert represents a cost alert
type CostAlert struct {
	Type        string
	Message     string
	Severity    string
	Timestamp   time.Time
	Resolved    bool
}

// ResourceType represents different types of compute resources
const (
	ResourceCPU     = "cpu"
	ResourceMemory  = "memory"
	ResourceNetwork = "network"
	ResourceStorage = "storage"
)

// NewCostOptimizer creates a new cost optimizer
func NewCostOptimizer(budget float64) *CostOptimizer {
	return &CostOptimizer{
		operationCosts:  make(map[string]float64),
		budget:          budget,
		spent:           0,
		recommendations: make([]string, 0),
		costHistory:     make([]CostEntry, 0),
		alerts:          make([]CostAlert, 0),
	}
}

// RecordCost records a cost for an operation
func (co *CostOptimizer) RecordCost(operation string, cost float64, agentID string, resourceType string) {
	co.mu.Lock()
	defer co.mu.Unlock()
	
	co.operationCosts[operation] += cost
	co.spent += cost
	
	// Add to cost history
	entry := CostEntry{
		Operation:    operation,
		Cost:         cost,
		Timestamp:    time.Now(),
		AgentID:      agentID,
		ResourceType: resourceType,
	}
	co.costHistory = append(co.costHistory, entry)
	
	// Keep only recent history (last 10000 entries)
	maxHistory := 10000
	if len(co.costHistory) > maxHistory {
		co.costHistory = co.costHistory[len(co.costHistory)-maxHistory:]
	}
	
	// Check for budget alerts
	co.checkBudgetAlerts()
	
	// Generate recommendations if over budget
	if co.spent > co.budget*0.9 {
		co.generateRecommendations()
	}
}

// checkBudgetAlerts checks for budget-related alerts
func (co *CostOptimizer) checkBudgetAlerts() {
	utilization := co.spent / co.budget * 100
	
	// Check for budget thresholds
	if utilization >= 90 && !co.hasActiveAlert("budget_90") {
		co.addAlert("budget_90", "Budget utilization at 90%", "warning")
	} else if utilization >= 95 && !co.hasActiveAlert("budget_95") {
		co.addAlert("budget_95", "Budget utilization at 95%", "critical")
	} else if utilization >= 100 && !co.hasActiveAlert("budget_exceeded") {
		co.addAlert("budget_exceeded", "Budget exceeded", "critical")
	}
	
	// Check for cost spikes
	co.checkCostSpikes()
}

// hasActiveAlert checks if an alert type is already active
func (co *CostOptimizer) hasActiveAlert(alertType string) bool {
	for _, alert := range co.alerts {
		if alert.Type == alertType && !alert.Resolved {
			return true
		}
	}
	return false
}

// addAlert adds a new cost alert
func (co *CostOptimizer) addAlert(alertType, message, severity string) {
	alert := CostAlert{
		Type:      alertType,
		Message:   message,
		Severity:  severity,
		Timestamp: time.Now(),
		Resolved:  false,
	}
	co.alerts = append(co.alerts, alert)
}

// checkCostSpikes detects sudden cost increases
func (co *CostOptimizer) checkCostSpikes() {
	if len(co.costHistory) < 10 {
		return
	}
	
	// Calculate recent cost trend
	recent := co.costHistory[len(co.costHistory)-10:]
	recentCost := 0.0
	for _, entry := range recent {
		recentCost += entry.Cost
	}
	
	// Calculate previous cost trend
	previous := co.costHistory[len(co.costHistory)-20 : len(co.costHistory)-10]
	previousCost := 0.0
	for _, entry := range previous {
		previousCost += entry.Cost
	}
	
	// Check for significant increase
	if previousCost > 0 && recentCost > previousCost*2.0 {
		co.addAlert("cost_spike", 
			"Cost spike detected: recent costs are 2x higher than previous period", 
			"warning")
	}
}

// generateRecommendations generates cost optimization recommendations
func (co *CostOptimizer) generateRecommendations() {
	co.recommendations = make([]string, 0)
	
	// Analyze operation costs
	topCostOperations := co.getTopCostOperations(5)
	for _, op := range topCostOperations {
		co.recommendations = append(co.recommendations, 
			"Consider optimizing "+op.Operation+" (cost: $"+formatFloat(op.Cost)+")")
	}
	
	// Analyze resource usage
	resourceCosts := co.getResourceCosts()
	for resource, cost := range resourceCosts {
		if cost > co.budget*0.3 {
			co.recommendations = append(co.recommendations, 
				"High "+resource+" usage detected (cost: $"+formatFloat(cost)+")")
		}
	}
	
	// Suggest caching for repeated operations
	repeatedOps := co.getRepeatedOperations()
	if len(repeatedOps) > 0 {
		co.recommendations = append(co.recommendations, 
			"Enable caching for repeated operations: "+formatRepeatedOps(repeatedOps))
	}
	
	// Suggest batch processing
	smallOps := co.getSmallOperations()
	if len(smallOps) > 0 {
		co.recommendations = append(co.recommendations, 
			"Consider batch processing for small operations: "+formatSmallOps(smallOps))
	}
}

// getTopCostOperations returns the most expensive operations
func (co *CostOptimizer) getTopCostOperations(limit int) []CostEntry {
	type opCost struct {
		Operation string
		Cost      float64
	}
	
	var operations []opCost
	for op, cost := range co.operationCosts {
		operations = append(operations, opCost{op, cost})
	}
	
	// Sort by cost (descending)
	for i := 0; i < len(operations)-1; i++ {
		for j := i + 1; j < len(operations); j++ {
			if operations[i].Cost < operations[j].Cost {
				operations[i], operations[j] = operations[j], operations[i]
			}
		}
	}
	
	// Return top N
	if limit > len(operations) {
		limit = len(operations)
	}
	
	result := make([]CostEntry, limit)
	for i := 0; i < limit; i++ {
		result[i] = CostEntry{
			Operation: operations[i].Operation,
			Cost:      operations[i].Cost,
		}
	}
	
	return result
}

// getResourceCosts returns costs by resource type
func (co *CostOptimizer) getResourceCosts() map[string]float64 {
	resourceCosts := make(map[string]float64)
	
	for _, entry := range co.costHistory {
		resourceCosts[entry.ResourceType] += entry.Cost
	}
	
	return resourceCosts
}

// getRepeatedOperations finds operations that are called frequently
func (co *CostOptimizer) getRepeatedOperations() []string {
	operationCounts := make(map[string]int)
	
	for _, entry := range co.costHistory {
		operationCounts[entry.Operation]++
	}
	
	var repeated []string
	for op, count := range operationCounts {
		if count > 10 { // Called more than 10 times
			repeated = append(repeated, op)
		}
	}
	
	return repeated
}

// getSmallOperations finds operations with small individual costs
func (co *CostOptimizer) getSmallOperations() []string {
	operationCosts := make(map[string]float64)
	operationCounts := make(map[string]int)
	
	for _, entry := range co.costHistory {
		operationCosts[entry.Operation] += entry.Cost
		operationCounts[entry.Operation]++
	}
	
	var small []string
	for op, totalCost := range operationCosts {
		count := operationCounts[op]
		if count > 0 {
			avgCost := totalCost / float64(count)
			if avgCost < 0.01 { // Less than 1 cent per operation
				small = append(small, op)
			}
		}
	}
	
	return small
}

// GetReport returns a comprehensive cost report
func (co *CostOptimizer) GetReport() map[string]interface{} {
	co.mu.RLock()
	defer co.mu.RUnlock()
	
	utilization := co.spent / co.budget * 100
	
	// Calculate cost trends
	recentCosts := co.getRecentCosts(24) // Last 24 hours
	previousCosts := co.getPreviousCosts(24) // Previous 24 hours
	
	trend := "stable"
	if len(recentCosts) > 0 && len(previousCosts) > 0 {
		recentTotal := 0.0
		previousTotal := 0.0
		
		for _, cost := range recentCosts {
			recentTotal += cost
		}
		for _, cost := range previousCosts {
			previousTotal += cost
		}
		
		if recentTotal > previousTotal*1.1 {
			trend = "increasing"
		} else if recentTotal < previousTotal*0.9 {
			trend = "decreasing"
		}
	}
	
	return map[string]interface{}{
		"budget":              co.budget,
		"spent":               co.spent,
		"remaining":           co.budget - co.spent,
		"utilization_percent": utilization,
		"trend":               trend,
		"operation_costs":     co.operationCosts,
		"resource_costs":      co.getResourceCosts(),
		"recommendations":     co.recommendations,
		"active_alerts":       co.getActiveAlerts(),
		"top_operations":      co.getTopCostOperations(10),
	}
}

// getRecentCosts returns costs from the last N hours
func (co *CostOptimizer) getRecentCosts(hours int) []float64 {
	cutoff := time.Now().Add(-time.Duration(hours) * time.Hour)
	var costs []float64
	
	for _, entry := range co.costHistory {
		if entry.Timestamp.After(cutoff) {
			costs = append(costs, entry.Cost)
		}
	}
	
	return costs
}

// getPreviousCosts returns costs from the previous N hours
func (co *CostOptimizer) getPreviousCosts(hours int) []float64 {
	start := time.Now().Add(-time.Duration(hours*2) * time.Hour)
	end := time.Now().Add(-time.Duration(hours) * time.Hour)
	var costs []float64
	
	for _, entry := range co.costHistory {
		if entry.Timestamp.After(start) && entry.Timestamp.Before(end) {
			costs = append(costs, entry.Cost)
		}
	}
	
	return costs
}

// getActiveAlerts returns unresolved alerts
func (co *CostOptimizer) getActiveAlerts() []CostAlert {
	var active []CostAlert
	for _, alert := range co.alerts {
		if !alert.Resolved {
			active = append(active, alert)
		}
	}
	return active
}

// ResolveAlert resolves an alert
func (co *CostOptimizer) ResolveAlert(alertType string) {
	co.mu.Lock()
	defer co.mu.Unlock()
	
	for i, alert := range co.alerts {
		if alert.Type == alertType && !alert.Resolved {
			co.alerts[i].Resolved = true
		}
	}
}

// SetBudget sets a new budget
func (co *CostOptimizer) SetBudget(budget float64) {
	co.mu.Lock()
	defer co.mu.Unlock()
	
	co.budget = budget
}

// Reset resets all cost data
func (co *CostOptimizer) Reset() {
	co.mu.Lock()
	defer co.mu.Unlock()
	
	co.operationCosts = make(map[string]float64)
	co.spent = 0
	co.recommendations = make([]string, 0)
	co.costHistory = make([]CostEntry, 0)
	co.alerts = make([]CostAlert, 0)
}

// Helper functions

func formatFloat(f float64) string {
	return fmt.Sprintf("%.2f", f)
}

func formatRepeatedOps(ops []string) string {
	if len(ops) == 0 {
		return ""
	}
	
	result := ops[0]
	for i := 1; i < len(ops) && i < 3; i++ {
		result += ", " + ops[i]
	}
	
	if len(ops) > 3 {
		result += "..."
	}
	
	return result
}

func formatSmallOps(ops []string) string {
	return formatRepeatedOps(ops)
}

// Global cost optimizer instance
var globalCostOptimizer *CostOptimizer
var costOptimizerOnce sync.Once

// GetGlobalCostOptimizer returns the global cost optimizer
func GetGlobalCostOptimizer() *CostOptimizer {
	costOptimizerOnce.Do(func() {
		globalCostOptimizer = NewCostOptimizer(1000.0) // Default $1000 budget
	})
	return globalCostOptimizer
}

// SetGlobalBudget sets the global budget
func SetGlobalBudget(budget float64) {
	costOptimizerOnce.Do(func() {
		globalCostOptimizer = NewCostOptimizer(budget)
	})
}
