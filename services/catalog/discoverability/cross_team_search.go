package discoverability

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"strings"
	"time"
)

// CrossTeamSearch provides search across teams.
type CrossTeamSearch struct {
	db     *sql.DB
	logger *log.Logger
}

// SearchRequest represents a search request.
type SearchRequest struct {
	Query       string
	Teams       []string // Empty means all teams
	Categories  []string
	Tags        []string
	Limit       int
	Offset      int
	SortBy      string // "relevance", "popularity", "recent"
}

// SearchResult represents a search result.
type SearchResult struct {
	ProductID      string
	ProductName    string
	Description    string
	Team           string
	Category       string
	Tags           []string
	RelevanceScore float64
	LastUpdated    time.Time
	Metadata       map[string]interface{}
}

// SearchResponse represents a search response.
type SearchResponse struct {
	Results    []SearchResult
	TotalCount int
	Query      string
	Duration   time.Duration
}

// SearchHistory represents search history.
type SearchHistory struct {
	ID        string
	UserID    string
	Query     string
	Results   int
	Clicked   []string // Product IDs that were clicked
	Timestamp time.Time
}

// NewCrossTeamSearch creates a new cross-team search service.
func NewCrossTeamSearch(db *sql.DB, logger *log.Logger) *CrossTeamSearch {
	return &CrossTeamSearch{
		db:     db,
		logger: logger,
	}
}

// Search performs a cross-team search.
func (cts *CrossTeamSearch) Search(ctx context.Context, req SearchRequest) (*SearchResponse, error) {
	startTime := time.Now()

	if req.Limit == 0 {
		req.Limit = 20
	}

	// Build query
	query, args := cts.buildSearchQuery(req)

	// Execute search
	rows, err := cts.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to execute search: %w", err)
	}
	defer rows.Close()

	var results []SearchResult
	for rows.Next() {
		var result SearchResult
		var tagsJSON sql.NullString
		var metadataJSON sql.NullString
		var team sql.NullString

		err := rows.Scan(
			&result.ProductID,
			&result.ProductName,
			&result.Description,
			&team,
			&result.Category,
			&tagsJSON,
			&result.RelevanceScore,
			&result.LastUpdated,
			&metadataJSON,
		)
		if err != nil {
			continue
		}

		if team.Valid {
			result.Team = team.String
		}

		if tagsJSON.Valid {
			// Parse tags JSON array
			tagsStr := strings.Trim(tagsJSON.String, "[]\"")
			if tagsStr != "" {
				result.Tags = strings.Split(tagsStr, ",")
				for i, tag := range result.Tags {
					result.Tags[i] = strings.Trim(tag, "\" ")
				}
			}
		}

		if metadataJSON.Valid {
			// Parse metadata JSON
			// Simplified - would use proper JSON parsing
			result.Metadata = make(map[string]interface{})
		}

		results = append(results, result)
	}

	// Get total count
	totalCount := len(results)

	// Log search
	if req.Query != "" {
		go cts.logSearch(ctx, req.Query, totalCount)
	}

	duration := time.Since(startTime)

	return &SearchResponse{
		Results:    results,
		TotalCount: totalCount,
		Query:      req.Query,
		Duration:   duration,
	}, nil
}

// buildSearchQuery builds the SQL query for search.
func (cts *CrossTeamSearch) buildSearchQuery(req SearchRequest) (string, []interface{}) {
	baseQuery := `
		SELECT 
			dp.id,
			dp.name,
			dp.description,
			dp.team,
			dp.category,
			COALESCE(
				json_agg(DISTINCT t.name) FILTER (WHERE t.name IS NOT NULL),
				'[]'::json
			) as tags,
			CASE 
				WHEN $1 = '' THEN 1.0
				ELSE ts_rank_cd(to_tsvector('english', COALESCE(dp.name, '') || ' ' || COALESCE(dp.description, '')), plainto_tsquery('english', $1))
			END as relevance_score,
			dp.updated_at,
			dp.metadata
		FROM data_products dp
		LEFT JOIN product_tags pt ON dp.id = pt.product_id
		LEFT JOIN tags t ON pt.tag_id = t.id
		WHERE 1=1
	`

	args := []interface{}{req.Query}
	argIndex := 2

	// Team filter
	if len(req.Teams) > 0 {
		placeholders := []string{}
		for _, team := range req.Teams {
			placeholders = append(placeholders, fmt.Sprintf("$%d", argIndex))
			args = append(args, team)
			argIndex++
		}
		baseQuery += fmt.Sprintf(" AND dp.team = ANY(ARRAY[%s])", strings.Join(placeholders, ","))
	}

	// Category filter
	if len(req.Categories) > 0 {
		placeholders := []string{}
		for _, category := range req.Categories {
			placeholders = append(placeholders, fmt.Sprintf("$%d", argIndex))
			args = append(args, category)
			argIndex++
		}
		baseQuery += fmt.Sprintf(" AND dp.category = ANY(ARRAY[%s])", strings.Join(placeholders, ","))
	}

	// Tag filter
	if len(req.Tags) > 0 {
		baseQuery += fmt.Sprintf(" AND EXISTS (SELECT 1 FROM product_tags pt2 INNER JOIN tags t2 ON pt2.tag_id = t2.id WHERE pt2.product_id = dp.id AND t2.id = ANY(ARRAY[")
		placeholders := []string{}
		for _, tag := range req.Tags {
			placeholders = append(placeholders, fmt.Sprintf("$%d", argIndex))
			args = append(args, tag)
			argIndex++
		}
		baseQuery += strings.Join(placeholders, ",") + "]))"
	}

	// Text search
	if req.Query != "" {
		baseQuery += fmt.Sprintf(" AND (to_tsvector('english', COALESCE(dp.name, '') || ' ' || COALESCE(dp.description, '')) @@ plainto_tsquery('english', $1))")
	}

	baseQuery += " GROUP BY dp.id, dp.name, dp.description, dp.team, dp.category, dp.updated_at, dp.metadata"

	// Sorting
	switch req.SortBy {
	case "popularity":
		baseQuery += " ORDER BY dp.usage_count DESC, relevance_score DESC"
	case "recent":
		baseQuery += " ORDER BY dp.updated_at DESC, relevance_score DESC"
	default:
		baseQuery += " ORDER BY relevance_score DESC, dp.updated_at DESC"
	}

	// Limit and offset
	baseQuery += fmt.Sprintf(" LIMIT $%d OFFSET $%d", argIndex, argIndex+1)
	args = append(args, req.Limit, req.Offset)

	return baseQuery, args
}

// logSearch logs a search query.
func (cts *CrossTeamSearch) logSearch(ctx context.Context, query string, resultCount int) {
	historyID := fmt.Sprintf("search-%d", time.Now().UnixNano())
	
	_, err := cts.db.ExecContext(ctx, `
		INSERT INTO search_history (id, query, result_count, timestamp)
		VALUES ($1, $2, $3, $4)
	`, historyID, query, resultCount, time.Now())
	
	if err != nil && cts.logger != nil {
		cts.logger.Printf("Failed to log search: %v", err)
	}
}

// GetSearchSuggestions provides search suggestions based on history and popular queries.
func (cts *CrossTeamSearch) GetSearchSuggestions(ctx context.Context, prefix string, limit int) ([]string, error) {
	if limit == 0 {
		limit = 10
	}

	query := `
		SELECT DISTINCT query
		FROM search_history
		WHERE LOWER(query) LIKE LOWER($1)
		GROUP BY query
		ORDER BY COUNT(*) DESC, MAX(timestamp) DESC
		LIMIT $2
	`

	rows, err := cts.db.QueryContext(ctx, query, prefix+"%", limit)
	if err != nil {
		return nil, fmt.Errorf("failed to get suggestions: %w", err)
	}
	defer rows.Close()

	var suggestions []string
	for rows.Next() {
		var suggestion string
		if err := rows.Scan(&suggestion); err == nil {
			suggestions = append(suggestions, suggestion)
		}
	}

	return suggestions, nil
}

// GetPopularSearches returns popular search queries.
func (cts *CrossTeamSearch) GetPopularSearches(ctx context.Context, limit int) ([]string, error) {
	if limit == 0 {
		limit = 10
	}

	query := `
		SELECT query, COUNT(*) as count
		FROM search_history
		WHERE timestamp > NOW() - INTERVAL '30 days'
		GROUP BY query
		ORDER BY count DESC
		LIMIT $1
	`

	rows, err := cts.db.QueryContext(ctx, query, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to get popular searches: %w", err)
	}
	defer rows.Close()

	var searches []string
	for rows.Next() {
		var searchQuery string
		var count int64
		if err := rows.Scan(&searchQuery, &count); err == nil {
			searches = append(searches, searchQuery)
		}
	}

	return searches, nil
}

