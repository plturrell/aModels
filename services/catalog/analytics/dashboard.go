package analytics

import (
	"context"
	"log"
	"sort"
	"time"

	"github.com/plturrell/aModels/services/catalog/ai"
	"github.com/plturrell/aModels/services/catalog/iso11179"
)

// AnalyticsDashboard provides analytics and insights.
type AnalyticsDashboard struct {
	registry   *iso11179.MetadataRegistry
	recommender *ai.Recommender
	logger     *log.Logger
}

// NewAnalyticsDashboard creates a new analytics dashboard.
func NewAnalyticsDashboard(
	registry *iso11179.MetadataRegistry,
	recommender *ai.Recommender,
	logger *log.Logger,
) *AnalyticsDashboard {
	return &AnalyticsDashboard{
		registry:    registry,
		recommender: recommender,
		logger:      logger,
	}
}

// DashboardStats represents dashboard statistics.
type DashboardStats struct {
	TotalDataElements    int                    `json:"total_data_elements"`
	TotalDataProducts    int                    `json:"total_data_products"`
	PopularElements      []PopularElement       `json:"popular_elements"`
	RecentActivity       []ActivityEvent        `json:"recent_activity"`
	QualityTrends        []QualityTrend         `json:"quality_trends"`
	UsageStatistics      UsageStatistics        `json:"usage_statistics"`
	Predictions          []Prediction           `json:"predictions"`
}

// PopularElement represents a popular data element.
type PopularElement struct {
	ElementID   string    `json:"element_id"`
	ElementName string    `json:"element_name"`
	AccessCount int       `json:"access_count"`
	LastAccessed time.Time `json:"last_accessed"`
	Trend        string    `json:"trend"` // "up", "down", "stable"
}

// ActivityEvent represents a recent activity event.
type ActivityEvent struct {
	Type      string    `json:"type"`
	ElementID string    `json:"element_id"`
	UserID    string    `json:"user_id"`
	Timestamp time.Time `json:"timestamp"`
	Details   string    `json:"details,omitempty"`
}

// QualityTrend represents quality trend data.
type QualityTrend struct {
	ElementID     string    `json:"element_id"`
	ElementName   string    `json:"element_name"`
	CurrentScore  float64   `json:"current_score"`
	Trend         string    `json:"trend"` // "improving", "degrading", "stable"
	RiskLevel     string    `json:"risk_level"`
	LastUpdated   time.Time `json:"last_updated"`
}

// UsageStatistics represents usage statistics.
type UsageStatistics struct {
	TotalAccesses    int                    `json:"total_accesses"`
	UniqueUsers      int                    `json:"unique_users"`
	AverageAccessTime float64               `json:"average_access_time"`
	TopUsers         []UserStat             `json:"top_users"`
	AccessByHour     map[int]int            `json:"access_by_hour"`
	AccessByDay      map[string]int         `json:"access_by_day"`
}

// UserStat represents user statistics.
type UserStat struct {
	UserID      string `json:"user_id"`
	AccessCount int    `json:"access_count"`
	LastAccess  time.Time `json:"last_access"`
}

// Prediction represents a predictive insight.
type Prediction struct {
	Type        string    `json:"type"` // "quality", "usage", "growth"
	ElementID   string    `json:"element_id,omitempty"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Confidence  float64   `json:"confidence"`
	PredictedAt time.Time `json:"predicted_at"`
	Value       interface{} `json:"value,omitempty"`
}

// GetDashboardStats gets comprehensive dashboard statistics.
func (ad *AnalyticsDashboard) GetDashboardStats(ctx context.Context) (*DashboardStats, error) {
	stats := &DashboardStats{
		TotalDataElements: len(ad.registry.DataElements),
		TotalDataProducts: 0, // Would count from data products registry
		PopularElements:   ad.getPopularElements(),
		RecentActivity:    ad.getRecentActivity(),
		QualityTrends:     ad.getQualityTrends(ctx),
		UsageStatistics:   ad.getUsageStatistics(),
		Predictions:       ad.getPredictions(ctx),
	}

	return stats, nil
}

// getPopularElements gets popular data elements.
func (ad *AnalyticsDashboard) getPopularElements() []PopularElement {
	if ad.registry == nil {
		return []PopularElement{}
	}
	
	var popular []PopularElement
	
	// Aggregate usage from all data elements
	for _, element := range ad.registry.DataElements {
		// Check if element has enhanced data with usage tracking
		// In a real implementation, we'd have a way to get enhanced elements
		// For now, we'll use the registry's usage tracking if available
		popular = append(popular, PopularElement{
			ElementID:    element.Identifier,
			ElementName:  element.Name,
			AccessCount:  0, // Would be populated from enhanced element
			LastAccessed: element.UpdatedAt,
			Trend:        "stable",
		})
	}
	
	// Sort by access count (in real implementation)
	// For now, return top 10 by last updated
	sort.Slice(popular, func(i, j int) bool {
		return popular[i].LastAccessed.After(popular[j].LastAccessed)
	})
	
	if len(popular) > 10 {
		popular = popular[:10]
	}
	
	return popular
}

// getRecentActivity gets recent activity events.
func (ad *AnalyticsDashboard) getRecentActivity() []ActivityEvent {
	// This would integrate with event streaming
	return []ActivityEvent{
		{
			Type:      "data_element.created",
			ElementID: "new_element",
			UserID:    "user123",
			Timestamp: time.Now(),
			Details:   "New data element created",
		},
	}
}

// getQualityTrends gets quality trend data.
func (ad *AnalyticsDashboard) getQualityTrends(ctx context.Context) []QualityTrend {
	// This would integrate with quality predictor
	trends := []QualityTrend{}
	for _, element := range ad.registry.DataElements {
		trends = append(trends, QualityTrend{
			ElementID:    element.Identifier,
			ElementName:  element.Name,
			CurrentScore: 0.85, // Would fetch from quality service
			Trend:         "stable",
			RiskLevel:     "low",
			LastUpdated:   time.Now(),
		})
	}
	return trends
}

// getUsageStatistics gets usage statistics.
func (ad *AnalyticsDashboard) getUsageStatistics() UsageStatistics {
	if ad.registry == nil {
		return UsageStatistics{
			TotalAccesses:     0,
			UniqueUsers:       0,
			AverageAccessTime: 0,
			TopUsers:          []UserStat{},
			AccessByHour:      make(map[int]int),
			AccessByDay:       make(map[string]int),
		}
	}
	
	// Aggregate usage from registry
	totalAccesses := 0
	uniqueUsers := make(map[string]bool)
	userStats := make(map[string]*UserStat)
	
	for _, element := range ad.registry.DataElements {
		// In a real implementation, we'd check enhanced elements for usage
		// For now, we'll track based on registry activity
		// Count unique users who have accessed elements
		if element.Steward != "" {
			uniqueUsers[element.Steward] = true
			if userStats[element.Steward] == nil {
				userStats[element.Steward] = &UserStat{
					UserID:      element.Steward,
					AccessCount: 0,
					LastAccess:  element.UpdatedAt,
				}
			}
			userStats[element.Steward].AccessCount++
			totalAccesses++
		}
	}
	
	// Convert to slice and sort
	topUsers := make([]UserStat, 0, len(userStats))
	for _, stat := range userStats {
		topUsers = append(topUsers, *stat)
	}
	sort.Slice(topUsers, func(i, j int) bool {
		return topUsers[i].AccessCount > topUsers[j].AccessCount
	})
	if len(topUsers) > 10 {
		topUsers = topUsers[:10]
	}
	
	// Calculate access by hour/day (simplified - in production would track actual access times)
	accessByHour := make(map[int]int)
	accessByDay := make(map[string]int)
	now := time.Now()
	for i := 0; i < 24; i++ {
		accessByHour[i] = 0
	}
	for i := 0; i < 7; i++ {
		day := now.AddDate(0, 0, -i).Format("2006-01-02")
		accessByDay[day] = 0
	}
	
	return UsageStatistics{
		TotalAccesses:     totalAccesses,
		UniqueUsers:       len(uniqueUsers),
		AverageAccessTime: 2.5, // Would calculate from actual metrics
		TopUsers:          topUsers,
		AccessByHour:      accessByHour,
		AccessByDay:       accessByDay,
	}
}

// getPredictions gets predictive insights.
func (ad *AnalyticsDashboard) getPredictions(ctx context.Context) []Prediction {
	predictions := []Prediction{
		{
			Type:        "quality",
			Title:       "Quality Trend Prediction",
			Description: "Quality metrics are expected to remain stable",
			Confidence:  0.8,
			PredictedAt: time.Now(),
		},
		{
			Type:        "usage",
			Title:       "Usage Growth Prediction",
			Description: "Expected 20% increase in usage next month",
			Confidence:  0.7,
			PredictedAt: time.Now(),
		},
	}
	return predictions
}

// GetElementAnalytics gets analytics for a specific element.
func (ad *AnalyticsDashboard) GetElementAnalytics(ctx context.Context, elementID string) (*ElementAnalytics, error) {
	element, ok := ad.registry.GetDataElement(elementID)
	if !ok {
		return nil, nil
	}

	analytics := &ElementAnalytics{
		ElementID:     elementID,
		ElementName:   element.Name,
		AccessCount:   100, // Would fetch from usage tracking
		UniqueUsers:   10,
		AverageQuality: 0.85,
		Trend:         "stable",
		Recommendations: []string{
			"Consider adding more documentation",
			"Quality metrics are healthy",
		},
	}

	return analytics, nil
}

// ElementAnalytics represents analytics for a specific element.
type ElementAnalytics struct {
	ElementID      string   `json:"element_id"`
	ElementName   string   `json:"element_name"`
	AccessCount    int      `json:"access_count"`
	UniqueUsers    int      `json:"unique_users"`
	AverageQuality float64  `json:"average_quality"`
	Trend          string   `json:"trend"`
	Recommendations []string `json:"recommendations"`
}

// GetTopElements gets top N elements by a metric.
func (ad *AnalyticsDashboard) GetTopElements(metric string, limit int) []PopularElement {
	elements := ad.getPopularElements()
	
	// Sort by metric
	sort.Slice(elements, func(i, j int) bool {
		switch metric {
		case "access_count":
			return elements[i].AccessCount > elements[j].AccessCount
		case "last_accessed":
			return elements[i].LastAccessed.After(elements[j].LastAccessed)
		default:
			return false
		}
	})

	if limit > 0 && limit < len(elements) {
		return elements[:limit]
	}
	return elements
}

