package discoverability

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"time"
)

// Marketplace provides data product marketplace functionality.
type Marketplace struct {
	db     *sql.DB
	logger *log.Logger
}

// ProductListing represents a product listing in the marketplace.
type ProductListing struct {
	ProductID      string
	ProductName    string
	Description    string
	Team           string
	Category       string
	Tags           []string
	Version        string
	Status         string // "published", "draft", "archived"
	PublishedAt    time.Time
	Views          int64
	AccessRequests int64
	Rating         float64
	ReviewCount    int64
	Metadata       map[string]interface{}
}

// AccessRequest represents a request to access a product.
type AccessRequest struct {
	ID          string
	ProductID   string
	RequesterID string
	RequesterTeam string
	Status      string // "pending", "approved", "rejected"
	Reason      string
	RequestedAt time.Time
	ProcessedAt *time.Time
	ProcessedBy string
	Comments    string
}

// UsageStatistics represents usage statistics for a product.
type UsageStatistics struct {
	ProductID      string
	TotalViews     int64
	UniqueViewers  int64
	AccessRequests int64
	ApprovedAccess int64
	RejectedAccess int64
	AverageRating  float64
	ReviewCount    int64
	LastAccessed   time.Time
}

// ProductRecommendation represents a product recommendation.
type ProductRecommendation struct {
	ProductID   string
	ProductName string
	Score       float64
	Reason      string
	Tags        []string
}

// NewMarketplace creates a new marketplace.
func NewMarketplace(db *sql.DB, logger *log.Logger) *Marketplace {
	return &Marketplace{
		db:     db,
		logger: logger,
	}
}

// ListProducts lists products in the marketplace.
func (m *Marketplace) ListProducts(ctx context.Context, filters MarketplaceFilters) ([]ProductListing, error) {
	query, args := m.buildListQuery(filters)

	rows, err := m.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list products: %w", err)
	}
	defer rows.Close()

	var listings []ProductListing
	for rows.Next() {
		var listing ProductListing
		var tagsJSON sql.NullString
		var metadataJSON sql.NullString
		var publishedAt sql.NullTime

		err := rows.Scan(
			&listing.ProductID,
			&listing.ProductName,
			&listing.Description,
			&listing.Team,
			&listing.Category,
			&tagsJSON,
			&listing.Version,
			&listing.Status,
			&publishedAt,
			&listing.Views,
			&listing.AccessRequests,
			&listing.Rating,
			&listing.ReviewCount,
			&metadataJSON,
		)
		if err != nil {
			continue
		}

		if publishedAt.Valid {
			listing.PublishedAt = publishedAt.Time
		}

		// Parse tags and metadata (simplified)
		if tagsJSON.Valid {
			listing.Tags = []string{} // Would parse JSON
		}

		if metadataJSON.Valid {
			listing.Metadata = make(map[string]interface{}) // Would parse JSON
		}

		listings = append(listings, listing)
	}

	return listings, nil
}

// buildListQuery builds the query for listing products.
func (m *Marketplace) buildListQuery(filters MarketplaceFilters) (string, []interface{}) {
	query := `
		SELECT 
			dp.id,
			dp.name,
			dp.description,
			dp.team,
			dp.category,
			COALESCE(json_agg(DISTINCT t.name), '[]'::json) as tags,
			dp.version,
			dp.status,
			dp.published_at,
			COALESCE(us.total_views, 0) as views,
			COALESCE(us.access_requests, 0) as access_requests,
			COALESCE(us.average_rating, 0.0) as rating,
			COALESCE(us.review_count, 0) as review_count,
			dp.metadata
		FROM data_products dp
		LEFT JOIN product_tags pt ON dp.id = pt.product_id
		LEFT JOIN tags t ON pt.tag_id = t.id
		LEFT JOIN product_usage_stats us ON dp.id = us.product_id
		WHERE dp.status = 'published'
	`

	args := []interface{}{}
	argIndex := 1

	if filters.Category != "" {
		query += fmt.Sprintf(" AND dp.category = $%d", argIndex)
		args = append(args, filters.Category)
		argIndex++
	}

	if filters.Team != "" {
		query += fmt.Sprintf(" AND dp.team = $%d", argIndex)
		args = append(args, filters.Team)
		argIndex++
	}

	query += " GROUP BY dp.id, dp.name, dp.description, dp.team, dp.category, dp.version, dp.status, dp.published_at, dp.metadata, us.total_views, us.access_requests, us.average_rating, us.review_count"

	// Sorting
	switch filters.SortBy {
	case "popularity":
		query += " ORDER BY views DESC, rating DESC"
	case "rating":
		query += " ORDER BY rating DESC, review_count DESC"
	case "recent":
		query += " ORDER BY published_at DESC"
	default:
		query += " ORDER BY published_at DESC"
	}

	// Limit and offset
	query += fmt.Sprintf(" LIMIT $%d OFFSET $%d", argIndex, argIndex+1)
	args = append(args, filters.Limit, filters.Offset)

	return query, args
}

// MarketplaceFilters filters for marketplace listings.
type MarketplaceFilters struct {
	Category string
	Team     string
	Tags     []string
	SortBy   string // "popularity", "rating", "recent"
	Limit    int
	Offset   int
}

// RequestAccess requests access to a product.
func (m *Marketplace) RequestAccess(ctx context.Context, req AccessRequest) error {
	req.ID = fmt.Sprintf("access-request-%d", time.Now().UnixNano())
	req.Status = "pending"
	req.RequestedAt = time.Now()

	query := `
		INSERT INTO access_requests (id, product_id, requester_id, requester_team, status, reason, requested_at, comments)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`

	_, err := m.db.ExecContext(ctx, query,
		req.ID,
		req.ProductID,
		req.RequesterID,
		req.RequesterTeam,
		req.Status,
		req.Reason,
		req.RequestedAt,
		req.Comments,
	)

	if err != nil {
		return fmt.Errorf("failed to create access request: %w", err)
	}

	// Update usage statistics
	_, err = m.db.ExecContext(ctx, `
		INSERT INTO product_usage_stats (product_id, access_requests, updated_at)
		VALUES ($1, 1, NOW())
		ON CONFLICT (product_id) DO UPDATE
		SET access_requests = product_usage_stats.access_requests + 1,
		    updated_at = NOW()
	`, req.ProductID)

	return err
}

// ApproveAccessRequest approves an access request.
func (m *Marketplace) ApproveAccessRequest(ctx context.Context, requestID string, approverID string, comments string) error {
	query := `
		UPDATE access_requests
		SET status = 'approved',
		    processed_at = NOW(),
		    processed_by = $1,
		    comments = $2
		WHERE id = $3 AND status = 'pending'
	`

	result, err := m.db.ExecContext(ctx, query, approverID, comments, requestID)
	if err != nil {
		return fmt.Errorf("failed to approve access request: %w", err)
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected == 0 {
		return fmt.Errorf("access request not found or already processed")
	}

	return nil
}

// RejectAccessRequest rejects an access request.
func (m *Marketplace) RejectAccessRequest(ctx context.Context, requestID string, approverID string, comments string) error {
	query := `
		UPDATE access_requests
		SET status = 'rejected',
		    processed_at = NOW(),
		    processed_by = $1,
		    comments = $2
		WHERE id = $3 AND status = 'pending'
	`

	result, err := m.db.ExecContext(ctx, query, approverID, comments, requestID)
	if err != nil {
		return fmt.Errorf("failed to reject access request: %w", err)
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected == 0 {
		return fmt.Errorf("access request not found or already processed")
	}

	return nil
}

// GetUsageStatistics retrieves usage statistics for a product.
func (m *Marketplace) GetUsageStatistics(ctx context.Context, productID string) (*UsageStatistics, error) {
	query := `
		SELECT 
			product_id,
			COALESCE(total_views, 0),
			COALESCE(unique_viewers, 0),
			COALESCE(access_requests, 0),
			COALESCE(approved_access, 0),
			COALESCE(rejected_access, 0),
			COALESCE(average_rating, 0.0),
			COALESCE(review_count, 0),
			last_accessed
		FROM product_usage_stats
		WHERE product_id = $1
	`

	var stats UsageStatistics
	var lastAccessed sql.NullTime

	err := m.db.QueryRowContext(ctx, query, productID).Scan(
		&stats.ProductID,
		&stats.TotalViews,
		&stats.UniqueViewers,
		&stats.AccessRequests,
		&stats.ApprovedAccess,
		&stats.RejectedAccess,
		&stats.AverageRating,
		&stats.ReviewCount,
		&lastAccessed,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			// Return empty stats
			return &UsageStatistics{
				ProductID: productID,
			}, nil
		}
		return nil, fmt.Errorf("failed to get usage statistics: %w", err)
	}

	if lastAccessed.Valid {
		stats.LastAccessed = lastAccessed.Time
	}

	return &stats, nil
}

// RecordView records a product view.
func (m *Marketplace) RecordView(ctx context.Context, productID string, viewerID string) error {
	query := `
		INSERT INTO product_usage_stats (product_id, total_views, unique_viewers, updated_at)
		VALUES ($1, 1, 1, NOW())
		ON CONFLICT (product_id) DO UPDATE
		SET total_views = product_usage_stats.total_views + 1,
		    updated_at = NOW()
	`

	_, err := m.db.ExecContext(ctx, query, productID)
	return err
}

// GetRecommendations gets product recommendations based on user behavior.
func (m *Marketplace) GetRecommendations(ctx context.Context, userID string, limit int) ([]ProductRecommendation, error) {
	if limit == 0 {
		limit = 10
	}

	// Simplified recommendation algorithm
	// In production, would use collaborative filtering, content-based filtering, etc.
	query := `
		SELECT 
			dp.id,
			dp.name,
			COALESCE(us.average_rating, 0.0) * 0.5 + (COALESCE(us.total_views, 0)::float / 1000.0) * 0.3 + (COALESCE(us.review_count, 0)::float / 100.0) * 0.2 as score,
			COALESCE(json_agg(DISTINCT t.name), '[]'::json) as tags
		FROM data_products dp
		LEFT JOIN product_usage_stats us ON dp.id = us.product_id
		LEFT JOIN product_tags pt ON dp.id = pt.product_id
		LEFT JOIN tags t ON pt.tag_id = t.id
		WHERE dp.status = 'published'
		GROUP BY dp.id, dp.name, us.average_rating, us.total_views, us.review_count
		ORDER BY score DESC
		LIMIT $1
	`

	rows, err := m.db.QueryContext(ctx, query, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to get recommendations: %w", err)
	}
	defer rows.Close()

	var recommendations []ProductRecommendation
	for rows.Next() {
		var rec ProductRecommendation
		var tagsJSON sql.NullString

		err := rows.Scan(&rec.ProductID, &rec.ProductName, &rec.Score, &tagsJSON)
		if err != nil {
			continue
		}

		rec.Reason = "Based on popularity and ratings"
		
		if tagsJSON.Valid {
			rec.Tags = []string{} // Would parse JSON
		}

		recommendations = append(recommendations, rec)
	}

	return recommendations, nil
}

