package models

import "time"

// LibraryStats captures aggregated metrics for a specific library.
type LibraryStats struct {
	LibraryType     string
	TotalOperations int64
	SuccessRate     float64
	AverageLatency  float64
	ErrorCount      int64
}

// PerformanceTrend represents historical metrics for trend analysis.
type PerformanceTrend struct {
	Date           time.Time
	LibraryType    string
	Operations     int64
	SuccessRate    float64
	AverageLatency float64
}

// AnalyticsSummary consolidates system-wide analytics for clients.
type AnalyticsSummary struct {
	TotalOperations  int64
	SuccessRate      float64
	AverageLatency   float64
	ErrorBreakdown   map[string]int64
	LibraryStats     []LibraryStats
	PerformanceTrend []PerformanceTrend
	GeneratedAt      time.Time
}

// AnalyticsFilters supplies optional filters for analytics queries.
type AnalyticsFilters struct {
	StartTime   *time.Time
	EndTime     *time.Time
	LibraryType string
}
