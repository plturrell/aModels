package discoverability

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"math"
)

// RecommendationEngine provides advanced recommendation algorithms.
type RecommendationEngine struct {
	db     *sql.DB
	logger *log.Logger
}

// NewRecommendationEngine creates a new recommendation engine.
func NewRecommendationEngine(db *sql.DB, logger *log.Logger) *RecommendationEngine {
	return &RecommendationEngine{
		db:     db,
		logger: logger,
	}
}

// RecommendationType represents the type of recommendation.
type RecommendationType string

const (
	RecommendationTypeSimilar      RecommendationType = "similar"
	RecommendationTypeTrending     RecommendationType = "trending"
	RecommendationTypePopular      RecommendationType = "popular"
	RecommendationTypePersonalized RecommendationType = "personalized"
)

// Recommendation represents a product recommendation.
type Recommendation struct {
	ProductID     string  `json:"product_id"`
	ProductName   string  `json:"product_name"`
	Score         float64 `json:"score"`
	Reason        string  `json:"reason"`
	Type          RecommendationType `json:"type"`
}

// GetSimilarProducts recommends products similar to a given product.
func (re *RecommendationEngine) GetSimilarProducts(ctx context.Context, productID string, limit int) ([]*Recommendation, error) {
	// Get product tags
	productTags, err := re.getProductTags(ctx, productID)
	if err != nil {
		return nil, fmt.Errorf("failed to get product tags: %w", err)
	}

	// Find products with similar tags
	similarProducts, err := re.findProductsByTags(ctx, productTags, productID, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to find similar products: %w", err)
	}

	// Calculate similarity scores
	recommendations := make([]*Recommendation, 0, len(similarProducts))
	for _, product := range similarProducts {
		score := re.calculateSimilarityScore(productTags, product.Tags)
		recommendations = append(recommendations, &Recommendation{
			ProductID:   product.ID,
			ProductName: product.Name,
			Score:       score,
			Reason:      fmt.Sprintf("Similar tags: %d common tags", len(intersect(productTags, product.Tags))),
			Type:        RecommendationTypeSimilar,
		})
	}

	// Sort by score descending
	sortRecommendations(recommendations)

	return recommendations, nil
}

// GetTrendingProducts recommends trending products.
func (re *RecommendationEngine) GetTrendingProducts(ctx context.Context, limit int) ([]*Recommendation, error) {
	// Calculate trending score based on recent usage
	// Trending = (views_last_24h * 2) + (views_last_7d) + (access_requests_last_7d * 3)

	trendingProducts, err := re.findTrendingProducts(ctx, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to find trending products: %w", err)
	}

	recommendations := make([]*Recommendation, 0, len(trendingProducts))
	for _, product := range trendingProducts {
		recommendations = append(recommendations, &Recommendation{
			ProductID:   product.ID,
			ProductName: product.Name,
			Score:       product.TrendingScore,
			Reason:      fmt.Sprintf("Trending: %d views in last 24h", product.Views24h),
			Type:        RecommendationTypeTrending,
		})
	}

	return recommendations, nil
}

// GetPersonalizedRecommendations provides personalized recommendations for a user.
func (re *RecommendationEngine) GetPersonalizedRecommendations(ctx context.Context, userID string, limit int) ([]*Recommendation, error) {
	// Get user's access history and preferences
	userHistory, err := re.getUserHistory(ctx, userID)
	if err != nil {
		return nil, fmt.Errorf("failed to get user history: %w", err)
	}

	// Calculate preferences from history
	preferences := re.calculateUserPreferences(userHistory)

	// Find products matching preferences
	recommendations := make([]*Recommendation, 0)
	
	// Similar to accessed products
	for _, accessedProduct := range userHistory.AccessedProducts {
		similar, err := re.GetSimilarProducts(ctx, accessedProduct, limit/len(userHistory.AccessedProducts)+1)
		if err != nil {
			continue
		}
		recommendations = append(recommendations, similar...)
	}

	// Add trending products
	trending, err := re.GetTrendingProducts(ctx, limit/2)
	if err == nil {
		recommendations = append(recommendations, trending...)
	}

	// Sort and deduplicate
	sortRecommendations(recommendations)
	recommendations = deduplicateRecommendations(recommendations, userHistory.AccessedProducts)

	// Limit results
	if len(recommendations) > limit {
		recommendations = recommendations[:limit]
	}

	return recommendations, nil
}

// Helper types and functions

type ProductWithTags struct {
	ID    string
	Name  string
	Tags  []string
}

type TrendingProduct struct {
	ID            string
	Name          string
	TrendingScore float64
	Views24h      int
}

type UserHistory struct {
	UserID          string
	AccessedProducts []string
	PreferredTags   []string
	PreferredTeams  []string
}

func (re *RecommendationEngine) getProductTags(ctx context.Context, productID string) ([]string, error) {
	// In production, would query database
	return []string{}, nil
}

func (re *RecommendationEngine) findProductsByTags(ctx context.Context, tags []string, excludeID string, limit int) ([]ProductWithTags, error) {
	// In production, would query database
	return []ProductWithTags{}, nil
}

func (re *RecommendationEngine) findTrendingProducts(ctx context.Context, limit int) ([]TrendingProduct, error) {
	// In production, would query database with trending calculation
	return []TrendingProduct{}, nil
}

func (re *RecommendationEngine) getUserHistory(ctx context.Context, userID string) (*UserHistory, error) {
	// In production, would query database
	return &UserHistory{
		UserID:          userID,
		AccessedProducts: []string{},
		PreferredTags:   []string{},
		PreferredTeams:  []string{},
	}, nil
}

func (re *RecommendationEngine) calculateUserPreferences(history *UserHistory) map[string]float64 {
	preferences := make(map[string]float64)
	
	// Weight tags from accessed products
	for _, tag := range history.PreferredTags {
		preferences[tag] = 1.0
	}

	return preferences
}

func (re *RecommendationEngine) calculateSimilarityScore(tags1, tags2 []string) float64 {
	if len(tags1) == 0 && len(tags2) == 0 {
		return 1.0
	}
	if len(tags1) == 0 || len(tags2) == 0 {
		return 0.0
	}

	common := len(intersect(tags1, tags2))
	total := len(union(tags1, tags2))

	if total == 0 {
		return 0.0
	}

	// Jaccard similarity
	return float64(common) / float64(total)
}

func intersect(a, b []string) []string {
	m := make(map[string]bool)
	for _, v := range a {
		m[v] = true
	}

	result := []string{}
	for _, v := range b {
		if m[v] {
			result = append(result, v)
		}
	}
	return result
}

func union(a, b []string) []string {
	m := make(map[string]bool)
	for _, v := range a {
		m[v] = true
	}
	for _, v := range b {
		m[v] = true
	}

	result := make([]string, 0, len(m))
	for k := range m {
		result = append(result, k)
	}
	return result
}

func sortRecommendations(recommendations []*Recommendation) {
	// Simple bubble sort (in production, use sort.Slice)
	for i := 0; i < len(recommendations)-1; i++ {
		for j := i + 1; j < len(recommendations); j++ {
			if recommendations[i].Score < recommendations[j].Score {
				recommendations[i], recommendations[j] = recommendations[j], recommendations[i]
			}
		}
	}
}

func deduplicateRecommendations(recommendations []*Recommendation, excludeIDs []string) []*Recommendation {
	excludeMap := make(map[string]bool)
	for _, id := range excludeIDs {
		excludeMap[id] = true
	}

	result := make([]*Recommendation, 0)
	seen := make(map[string]bool)

	for _, rec := range recommendations {
		if !excludeMap[rec.ProductID] && !seen[rec.ProductID] {
			result = append(result, rec)
			seen[rec.ProductID] = true
		}
	}

	return result
}

