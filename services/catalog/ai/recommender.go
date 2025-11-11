package ai

import (
	"context"
	"fmt"
	"log"
	"math"
	"sort"
	"time"

	"github.com/plturrell/aModels/services/catalog/iso11179"
)

// Recommender provides intelligent recommendations for data products.
type Recommender struct {
	registry *iso11179.MetadataRegistry
	logger   *log.Logger
	usageHistory map[string][]UsageEvent
}

// NewRecommender creates a new recommender.
func NewRecommender(registry *iso11179.MetadataRegistry, logger *log.Logger) *Recommender {
	return &Recommender{
		registry:     registry,
		logger:       logger,
		usageHistory: make(map[string][]UsageEvent),
	}
}

// UsageEvent represents a usage event.
type UsageEvent struct {
	UserID      string    `json:"user_id"`
	ElementID   string    `json:"element_id"`
	Action      string    `json:"action"` // "view", "query", "download", "subscribe"
	Timestamp   time.Time `json:"timestamp"`
	Context     map[string]interface{} `json:"context,omitempty"`
}

// Recommendation represents a data product recommendation.
type Recommendation struct {
	ElementID      string                 `json:"element_id"`
	Element        *iso11179.DataElement `json:"element"`
	Score          float64               `json:"score"`          // 0.0 - 1.0
	Reason         string                `json:"reason"`
	ReasonType     string                `json:"reason_type"`     // "similar_usage", "related_data", "popular", "trending"
	Confidence     float64               `json:"confidence"`
}

// RecommendationRequest represents a request for recommendations.
type RecommendationRequest struct {
	UserID      string                 `json:"user_id,omitempty"`
	Context     string                 `json:"context,omitempty"`     // "discovery", "analysis", "integration"
	ElementID   string                 `json:"element_id,omitempty"` // For "related" recommendations
	Limit       int                    `json:"limit,omitempty"`
	Filters     RecommendationFilters  `json:"filters,omitempty"`
}

// RecommendationFilters filters recommendations.
type RecommendationFilters struct {
	Categories    []string `json:"categories,omitempty"`
	MinQuality   float64  `json:"min_quality,omitempty"`
	MaxAge        time.Duration `json:"max_age,omitempty"`
}

// GetRecommendations gets recommendations for a user.
func (r *Recommender) GetRecommendations(
	ctx context.Context,
	req RecommendationRequest,
) ([]Recommendation, error) {
	r.logger.Printf("Getting recommendations for user: %s (context: %s)", req.UserID, req.Context)

	var recommendations []Recommendation

	// Strategy 1: User-based recommendations (if user history exists)
	if req.UserID != "" {
		userRecs := r.getUserBasedRecommendations(req.UserID, req.Context, req.Limit)
		recommendations = append(recommendations, userRecs...)
	}

	// Strategy 2: Related data recommendations (if element ID provided)
	if req.ElementID != "" {
		relatedRecs := r.getRelatedRecommendations(req.ElementID, req.Limit)
		recommendations = append(recommendations, relatedRecs...)
	}

	// Strategy 3: Popular/trending recommendations
	popularRecs := r.getPopularRecommendations(req.Limit)
	recommendations = append(recommendations, popularRecs...)

	// Deduplicate and sort by score
	recommendations = r.deduplicateRecommendations(recommendations)
	sort.Slice(recommendations, func(i, j int) bool {
		return recommendations[i].Score > recommendations[j].Score
	})

	// Apply filters
	recommendations = r.applyFilters(recommendations, req.Filters)

	// Limit results
	if req.Limit > 0 && len(recommendations) > req.Limit {
		recommendations = recommendations[:req.Limit]
	}

	return recommendations, nil
}

// getUserBasedRecommendations gets recommendations based on user's usage history.
func (r *Recommender) getUserBasedRecommendations(
	userID string,
	context string,
	limit int,
) []Recommendation {
	events, ok := r.usageHistory[userID]
	if !ok || len(events) == 0 {
		return []Recommendation{}
	}

	// Find elements similar to what user has accessed
	accessedElements := make(map[string]bool)
	for _, event := range events {
		accessedElements[event.ElementID] = true
	}

	var recommendations []Recommendation
	for _, element := range r.registry.DataElements {
		if accessedElements[element.Identifier] {
			continue // Skip already accessed
		}

		// Calculate similarity score
		score := r.calculateSimilarityScore(element, events, context)
		if score > 0.3 { // Threshold
			recommendations = append(recommendations, Recommendation{
				ElementID:  element.Identifier,
				Element:    element,
				Score:      score,
				Reason:     fmt.Sprintf("Similar to elements you've accessed"),
				ReasonType: "similar_usage",
				Confidence: 0.7,
			})
		}
	}

	return recommendations
}

// getRelatedRecommendations gets recommendations related to a specific element.
func (r *Recommender) getRelatedRecommendations(elementID string, limit int) []Recommendation {
	element, ok := r.registry.GetDataElement(elementID)
	if !ok {
		return []Recommendation{}
	}

	var recommendations []Recommendation

	// Find elements with similar names, definitions, or metadata
	for _, candidate := range r.registry.DataElements {
		if candidate.Identifier == elementID {
			continue
		}

		similarity := r.calculateElementSimilarity(element, candidate)
		if similarity > 0.4 {
			recommendations = append(recommendations, Recommendation{
				ElementID:  candidate.Identifier,
				Element:    candidate,
				Score:      similarity,
				Reason:     fmt.Sprintf("Related to %s", element.Name),
				ReasonType: "related_data",
				Confidence: 0.8,
			})
		}
	}

	return recommendations
}

// getPopularRecommendations gets popular/trending recommendations.
func (r *Recommender) getPopularRecommendations(limit int) []Recommendation {
	// Calculate popularity scores based on usage
	popularityScores := make(map[string]float64)

	for _, events := range r.usageHistory {
		for _, event := range events {
			popularityScores[event.ElementID] += 1.0
			// Weight recent events more
			age := time.Since(event.Timestamp).Hours()
			if age < 24 {
				popularityScores[event.ElementID] += 2.0 // Bonus for recent usage
			}
		}
	}

	// Convert to recommendations
	var recommendations []Recommendation
	for elementID, score := range popularityScores {
		element, ok := r.registry.GetDataElement(elementID)
		if !ok {
			continue
		}

		normalizedScore := score / 10.0 // Normalize
		if normalizedScore > 1.0 {
			normalizedScore = 1.0
		}

		recommendations = append(recommendations, Recommendation{
			ElementID:  elementID,
			Element:    element,
			Score:      normalizedScore,
			Reason:     "Popular data product",
			ReasonType: "popular",
			Confidence: 0.6,
		})
	}

	return recommendations
}

// calculateSimilarityScore calculates similarity score for user-based recommendations.
func (r *Recommender) calculateSimilarityScore(
	element *iso11179.DataElement,
	userEvents []UsageEvent,
	context string,
) float64 {
	score := 0.0

	// Find most similar accessed element
	for _, event := range userEvents {
		accessedElement, ok := r.registry.GetDataElement(event.ElementID)
		if !ok {
			continue
		}

		similarity := r.calculateElementSimilarity(element, accessedElement)
		if similarity > score {
			score = similarity
		}
	}

	return score
}

// calculateElementSimilarity calculates similarity between two elements.
func (r *Recommender) calculateElementSimilarity(
	element1, element2 *iso11179.DataElement,
) float64 {
	score := 0.0

	// Name similarity (simple substring matching)
	if contains(element1.Name, element2.Name) || contains(element2.Name, element1.Name) {
		score += 0.3
	}

	// Definition similarity
	if contains(element1.Definition, element2.Definition) || contains(element2.Definition, element1.Definition) {
		score += 0.2
	}

	// Metadata similarity
	commonKeys := 0
	for key := range element1.Metadata {
		if _, ok := element2.Metadata[key]; ok {
			commonKeys++
		}
	}
	if len(element1.Metadata) > 0 {
		score += float64(commonKeys) / float64(len(element1.Metadata)) * 0.3
	}

	// Source similarity
	if element1.Source == element2.Source && element1.Source != "" {
		score += 0.2
	}

	return math.Min(score, 1.0)
}

// RecordUsage records a usage event.
func (r *Recommender) RecordUsage(event UsageEvent) {
	if r.usageHistory[event.UserID] == nil {
		r.usageHistory[event.UserID] = []UsageEvent{}
	}
	r.usageHistory[event.UserID] = append(r.usageHistory[event.UserID], event)

	// Keep only last 1000 events per user
	if len(r.usageHistory[event.UserID]) > 1000 {
		r.usageHistory[event.UserID] = r.usageHistory[event.UserID][len(r.usageHistory[event.UserID])-1000:]
	}
}

// deduplicateRecommendations removes duplicate recommendations.
func (r *Recommender) deduplicateRecommendations(recommendations []Recommendation) []Recommendation {
	seen := make(map[string]bool)
	var unique []Recommendation

	for _, rec := range recommendations {
		if !seen[rec.ElementID] {
			seen[rec.ElementID] = true
			unique = append(unique, rec)
		} else {
			// Merge scores if duplicate
			for i, existing := range unique {
				if existing.ElementID == rec.ElementID {
					// Take higher score
					if rec.Score > existing.Score {
						unique[i] = rec
					}
					break
				}
			}
		}
	}

	return unique
}

// applyFilters applies filters to recommendations.
func (r *Recommender) applyFilters(
	recommendations []Recommendation,
	filters RecommendationFilters,
) []Recommendation {
	var filtered []Recommendation

	for _, rec := range recommendations {
		// Category filter (if we had categories)
		// Min quality filter (if we had quality scores)
		// Max age filter
		if filters.MaxAge > 0 {
			if time.Since(rec.Element.CreatedAt) > filters.MaxAge {
				continue
			}
		}

		filtered = append(filtered, rec)
	}

	return filtered
}

