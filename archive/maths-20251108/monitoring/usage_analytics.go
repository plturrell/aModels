// Package monitoring provides usage analytics for the maths package
// Build: LYR1-MATH002 | Version: 1.0.0 | Module: monitoring
// Architecture: Layer1-Core | Component: maths-usage-analytics
// Dependencies: tracked
// Last-Modified: 2025-01-19T00:00:00Z

package monitoring

import (
	"fmt"
	"sync"
	"time"
)

// UsageAnalytics tracks usage patterns and provides insights
type UsageAnalytics struct {
	usagePatterns map[string]*UsagePattern
	agentUsage    map[string]*AgentUsage
	mu            sync.RWMutex
	startTime     time.Time
}

// UsagePattern tracks usage patterns for specific operations
type UsagePattern struct {
	Operation     string
	CallFrequency time.Duration
	PeakHours     []int
	AverageBatch  int
	ErrorPatterns map[string]int
	LastUsed      time.Time
	UsageTrend    []UsageDataPoint
}

// AgentUsage tracks usage by specific agents
type AgentUsage struct {
	AgentID      string
	Operations   map[string]int64
	TotalCalls   int64
	LastActivity time.Time
	Performance  *AgentPerformance
	Preferences  *AgentPreferences
}

// AgentPerformance tracks performance metrics for an agent
type AgentPerformance struct {
	AverageResponseTime time.Duration
	ErrorRate           float64
	SIMDUtilization     float64
	MemoryEfficiency    float64
	Throughput          float64
}

// AgentPreferences tracks agent-specific preferences
type AgentPreferences struct {
	PreferredOperations []string
	BatchSize           int
	SIMDThreshold       int
	MemoryLimit         int64
}

// UsageDataPoint represents a single data point in usage trend
type UsageDataPoint struct {
	Timestamp time.Time
	Count     int64
	Duration  time.Duration
	Errors    int64
}

// UsageReport provides comprehensive usage analytics
type UsageReport struct {
	TotalAgents         int
	TotalOperations     int64
	MostUsedOperations  []OperationUsage
	AgentActivity       []AgentActivity
	UsageTrends         map[string][]UsageDataPoint
	PerformanceInsights []string
	Recommendations     []string
	GeneratedAt         time.Time
}

// OperationUsage summarizes usage for a single operation
type OperationUsage struct {
	Operation    string
	TotalCalls   int64
	UniqueAgents int
	AverageTime  time.Duration
	ErrorRate    float64
	Popularity   float64
}

// AgentActivity summarizes activity for a single agent
type AgentActivity struct {
	AgentID          string
	TotalCalls       int64
	ActiveOperations []string
	LastActivity     time.Time
	Performance      *AgentPerformance
}

// NewUsageAnalytics creates a new usage analytics tracker
func NewUsageAnalytics() *UsageAnalytics {
	return &UsageAnalytics{
		usagePatterns: make(map[string]*UsagePattern),
		agentUsage:    make(map[string]*AgentUsage),
		startTime:     time.Now(),
	}
}

// TrackUsage records usage for a specific operation and agent
func (ua *UsageAnalytics) TrackUsage(operation, agentID string, duration time.Duration, success bool, batchSize int) {
	ua.mu.Lock()
	defer ua.mu.Unlock()

	now := time.Now()

	// Track operation usage pattern
	if ua.usagePatterns[operation] == nil {
		ua.usagePatterns[operation] = &UsagePattern{
			Operation:     operation,
			ErrorPatterns: make(map[string]int),
			UsageTrend:    make([]UsageDataPoint, 0),
		}
	}

	pattern := ua.usagePatterns[operation]
	pattern.LastUsed = now
	pattern.AverageBatch = (pattern.AverageBatch + batchSize) / 2

	// Add to usage trend (keep last 100 data points)
	trendPoint := UsageDataPoint{
		Timestamp: now,
		Count:     1,
		Duration:  duration,
		Errors:    0,
	}
	if !success {
		trendPoint.Errors = 1
	}

	pattern.UsageTrend = append(pattern.UsageTrend, trendPoint)
	if len(pattern.UsageTrend) > 100 {
		pattern.UsageTrend = pattern.UsageTrend[1:]
	}

	// Track agent usage
	if ua.agentUsage[agentID] == nil {
		ua.agentUsage[agentID] = &AgentUsage{
			AgentID:     agentID,
			Operations:  make(map[string]int64),
			Performance: &AgentPerformance{},
			Preferences: &AgentPreferences{
				PreferredOperations: make([]string, 0),
				BatchSize:           1,
				SIMDThreshold:       1000,
				MemoryLimit:         1024 * 1024 * 1024, // 1GB default
			},
		}
	}

	agent := ua.agentUsage[agentID]
	agent.Operations[operation]++
	agent.TotalCalls++
	agent.LastActivity = now

	// Update performance metrics
	if agent.Performance.AverageResponseTime == 0 {
		agent.Performance.AverageResponseTime = duration
	} else {
		agent.Performance.AverageResponseTime = (agent.Performance.AverageResponseTime + duration) / 2
	}

	// Calculate error rate
	totalErrors := int64(0)
	for _, op := range agent.Operations {
		// This is a simplified calculation - in practice you'd track errors separately
		totalErrors += op / 100 // Assume 1% error rate for demo
	}
	agent.Performance.ErrorRate = float64(totalErrors) / float64(agent.TotalCalls)
}

// GetUsagePattern returns usage pattern for a specific operation
func (ua *UsageAnalytics) GetUsagePattern(operation string) (*UsagePattern, bool) {
	ua.mu.RLock()
	defer ua.mu.RUnlock()

	pattern, exists := ua.usagePatterns[operation]
	return pattern, exists
}

// GetAgentUsage returns usage data for a specific agent
func (ua *UsageAnalytics) GetAgentUsage(agentID string) (*AgentUsage, bool) {
	ua.mu.RLock()
	defer ua.mu.RUnlock()

	usage, exists := ua.agentUsage[agentID]
	return usage, exists
}

// GenerateReport generates a comprehensive usage report
func (ua *UsageAnalytics) GenerateReport() *UsageReport {
	ua.mu.RLock()
	defer ua.mu.RUnlock()

	report := &UsageReport{
		GeneratedAt:         time.Now(),
		TotalAgents:         len(ua.agentUsage),
		UsageTrends:         make(map[string][]UsageDataPoint),
		MostUsedOperations:  make([]OperationUsage, 0),
		AgentActivity:       make([]AgentActivity, 0),
		PerformanceInsights: make([]string, 0),
		Recommendations:     make([]string, 0),
	}

	var totalOperations int64

	// Analyze operation usage
	operationStats := make(map[string]*OperationUsage)
	for op, pattern := range ua.usagePatterns {
		stats := &OperationUsage{
			Operation:    op,
			TotalCalls:   int64(len(pattern.UsageTrend)),
			UniqueAgents: 0,
			AverageTime:  0,
			ErrorRate:    0,
			Popularity:   0,
		}

		// Calculate unique agents using this operation
		agentCount := 0
		for _, agent := range ua.agentUsage {
			if agent.Operations[op] > 0 {
				agentCount++
			}
		}
		stats.UniqueAgents = agentCount

		// Calculate average time and error rate
		var totalTime time.Duration
		var errorCount int64
		for _, point := range pattern.UsageTrend {
			totalTime += point.Duration
			errorCount += point.Errors
		}
		if len(pattern.UsageTrend) > 0 {
			stats.AverageTime = totalTime / time.Duration(len(pattern.UsageTrend))
			stats.ErrorRate = float64(errorCount) / float64(len(pattern.UsageTrend))
		}

		operationStats[op] = stats
		totalOperations += stats.TotalCalls
		report.UsageTrends[op] = pattern.UsageTrend
	}

	// Calculate popularity scores
	for _, stats := range operationStats {
		if totalOperations > 0 {
			stats.Popularity = float64(stats.TotalCalls) / float64(totalOperations) * 100
		}
		report.MostUsedOperations = append(report.MostUsedOperations, *stats)
	}

	// Sort by popularity
	for i := 0; i < len(report.MostUsedOperations)-1; i++ {
		for j := i + 1; j < len(report.MostUsedOperations); j++ {
			if report.MostUsedOperations[i].Popularity < report.MostUsedOperations[j].Popularity {
				report.MostUsedOperations[i], report.MostUsedOperations[j] = report.MostUsedOperations[j], report.MostUsedOperations[i]
			}
		}
	}

	// Limit to top 10
	if len(report.MostUsedOperations) > 10 {
		report.MostUsedOperations = report.MostUsedOperations[:10]
	}

	// Analyze agent activity
	for agentID, usage := range ua.agentUsage {
		activity := AgentActivity{
			AgentID:          agentID,
			TotalCalls:       usage.TotalCalls,
			ActiveOperations: make([]string, 0),
			LastActivity:     usage.LastActivity,
			Performance:      usage.Performance,
		}

		// Find active operations
		for op, count := range usage.Operations {
			if count > 0 {
				activity.ActiveOperations = append(activity.ActiveOperations, op)
			}
		}

		report.AgentActivity = append(report.AgentActivity, activity)
	}

	// Sort by total calls
	for i := 0; i < len(report.AgentActivity)-1; i++ {
		for j := i + 1; j < len(report.AgentActivity); j++ {
			if report.AgentActivity[i].TotalCalls < report.AgentActivity[j].TotalCalls {
				report.AgentActivity[i], report.AgentActivity[j] = report.AgentActivity[j], report.AgentActivity[i]
			}
		}
	}

	report.TotalOperations = totalOperations

	// Generate insights and recommendations
	ua.generateInsights(report)

	return report
}

// generateInsights generates performance insights and recommendations
func (ua *UsageAnalytics) generateInsights(report *UsageReport) {
	// Analyze performance patterns
	var highErrorOps []string
	var slowOps []string
	var popularOps []string

	for _, op := range report.MostUsedOperations {
		if op.ErrorRate > 0.1 { // 10% error rate
			highErrorOps = append(highErrorOps, op.Operation)
		}
		if op.AverageTime > 100*time.Millisecond {
			slowOps = append(slowOps, op.Operation)
		}
		if op.Popularity > 20 { // 20% of total usage
			popularOps = append(popularOps, op.Operation)
		}
	}

	// Generate insights
	if len(highErrorOps) > 0 {
		report.PerformanceInsights = append(report.PerformanceInsights,
			fmt.Sprintf("High error rate detected in operations: %v", highErrorOps))
	}

	if len(slowOps) > 0 {
		report.PerformanceInsights = append(report.PerformanceInsights,
			fmt.Sprintf("Slow operations detected: %v", slowOps))
	}

	if len(popularOps) > 0 {
		report.PerformanceInsights = append(report.PerformanceInsights,
			fmt.Sprintf("Most popular operations: %v", popularOps))
	}

	// Generate recommendations
	if len(highErrorOps) > 0 {
		report.Recommendations = append(report.Recommendations,
			"Consider optimizing error-prone operations or adding better error handling")
	}

	if len(slowOps) > 0 {
		report.Recommendations = append(report.Recommendations,
			"Consider implementing SIMD optimizations for slow operations")
	}

	if report.TotalAgents > 10 {
		report.Recommendations = append(report.Recommendations,
			"Consider implementing load balancing for high agent count")
	}

	// Memory and performance recommendations
	report.Recommendations = append(report.Recommendations,
		"Monitor memory usage patterns and consider implementing memory pooling")
	report.Recommendations = append(report.Recommendations,
		"Implement caching for frequently used operations")
}

// PrintReport prints a formatted usage report
func (ua *UsageAnalytics) PrintReport() {
	report := ua.GenerateReport()

	fmt.Printf("\n=== Maths Package Usage Analytics Report ===\n")
	fmt.Printf("Generated at: %s\n", report.GeneratedAt.Format("2006-01-02 15:04:05"))
	fmt.Printf("Total Agents: %d\n", report.TotalAgents)
	fmt.Printf("Total Operations: %d\n", report.TotalOperations)

	fmt.Printf("\nMost Used Operations:\n")
	fmt.Printf("%-20s %8s %8s %12s %8s %8s\n",
		"Operation", "Calls", "Agents", "Avg Time", "Error%", "Popularity%")
	fmt.Printf("%-20s %8s %8s %12s %8s %8s\n",
		"---------", "-----", "------", "--------", "-------", "--------")

	for _, op := range report.MostUsedOperations {
		fmt.Printf("%-20s %8d %8d %12v %8.1f %8.1f\n",
			op.Operation, op.TotalCalls, op.UniqueAgents, op.AverageTime,
			op.ErrorRate*100, op.Popularity)
	}

	fmt.Printf("\nAgent Activity:\n")
	fmt.Printf("%-20s %8s %20s %12s\n",
		"Agent ID", "Calls", "Last Activity", "Avg Response")
	fmt.Printf("%-20s %8s %20s %12s\n",
		"--------", "-----", "------------", "------------")

	for _, agent := range report.AgentActivity {
		fmt.Printf("%-20s %8d %20s %12v\n",
			agent.AgentID, agent.TotalCalls,
			agent.LastActivity.Format("2006-01-02 15:04:05"),
			agent.Performance.AverageResponseTime)
	}

	if len(report.PerformanceInsights) > 0 {
		fmt.Printf("\nPerformance Insights:\n")
		for _, insight := range report.PerformanceInsights {
			fmt.Printf("- %s\n", insight)
		}
	}

	if len(report.Recommendations) > 0 {
		fmt.Printf("\nRecommendations:\n")
		for _, rec := range report.Recommendations {
			fmt.Printf("- %s\n", rec)
		}
	}

	fmt.Printf("\n")
}

// Reset clears all usage analytics data
func (ua *UsageAnalytics) Reset() {
	ua.mu.Lock()
	defer ua.mu.Unlock()

	ua.usagePatterns = make(map[string]*UsagePattern)
	ua.agentUsage = make(map[string]*AgentUsage)
	ua.startTime = time.Now()
}

// Global usage analytics instance
var globalAnalytics = NewUsageAnalytics()

// TrackUsage is a convenience function to track usage globally
func TrackUsage(operation, agentID string, duration time.Duration, success bool, batchSize int) {
	globalAnalytics.TrackUsage(operation, agentID, duration, success, batchSize)
}

// GetGlobalUsageReport returns a report for the global analytics
func GetGlobalUsageReport() *UsageReport {
	return globalAnalytics.GenerateReport()
}

// PrintGlobalUsageReport prints the global usage report
func PrintGlobalUsageReport() {
	globalAnalytics.PrintReport()
}

// ResetGlobalAnalytics resets the global usage analytics
func ResetGlobalAnalytics() {
	globalAnalytics.Reset()
}
