package discoverability

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"
)

// TagManager manages tags for data products.
type TagManager struct {
	db     *sql.DB
	logger *log.Logger
}

// Tag represents a tag for categorization.
type Tag struct {
	ID          string
	Name        string
	Category    string // "domain", "technology", "business", "regulatory", "custom"
	ParentTagID string // For hierarchical tags
	Description string
	UsageCount  int64
	CreatedAt   time.Time
	UpdatedAt   time.Time
	Metadata    map[string]interface{}
}

// TagHierarchy represents the hierarchy of tags.
type TagHierarchy struct {
	Tag      *Tag
	Children []*TagHierarchy
	Depth    int
}

// ProductTag represents a tag association with a product.
type ProductTag struct {
	ProductID string
	TagID     string
	Confidence float64 // 0.0 to 1.0 - auto-tagged tags have lower confidence
	Source    string // "manual", "auto", "suggested"
	CreatedAt time.Time
	CreatedBy string
}

// AutoTaggingSuggestion represents a suggested tag.
type AutoTaggingSuggestion struct {
	TagID     string
	TagName   string
	Confidence float64
	Reason    string
}

// TagAnalytics represents analytics for tags.
type TagAnalytics struct {
	TagID          string
	UsageCount     int64
	ProductCount   int64
	Popularity     float64
	Trend          string // "increasing", "decreasing", "stable"
	RelatedTags    []string
	MostUsedWith   []TagUsage
}

// TagUsage represents tag usage statistics.
type TagUsage struct {
	TagID    string
	TagName  string
	Count    int64
	Products []string
}

// NewTagManager creates a new tag manager.
func NewTagManager(db *sql.DB, logger *log.Logger) *TagManager {
	return &TagManager{
		db:     db,
		logger: logger,
	}
}

// CreateTag creates a new tag.
func (tm *TagManager) CreateTag(ctx context.Context, tag *Tag) error {
	metadataJSON, _ := json.Marshal(tag.Metadata)

	query := `
		INSERT INTO tags (id, name, category, parent_tag_id, description, usage_count, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`

	_, err := tm.db.ExecContext(ctx, query,
		tag.ID,
		tag.Name,
		tag.Category,
		tag.ParentTagID,
		tag.Description,
		tag.UsageCount,
		metadataJSON,
		tag.CreatedAt,
		tag.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to create tag: %w", err)
	}

	return nil
}

// GetTag retrieves a tag by ID.
func (tm *TagManager) GetTag(ctx context.Context, id string) (*Tag, error) {
	query := `
		SELECT id, name, category, parent_tag_id, description, usage_count, metadata, created_at, updated_at
		FROM tags
		WHERE id = $1
	`

	var tag Tag
	var parentTagID sql.NullString
	var metadataJSON []byte

	err := tm.db.QueryRowContext(ctx, query, id).Scan(
		&tag.ID,
		&tag.Name,
		&tag.Category,
		&parentTagID,
		&tag.Description,
		&tag.UsageCount,
		&metadataJSON,
		&tag.CreatedAt,
		&tag.UpdatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("tag not found: %s", id)
		}
		return nil, fmt.Errorf("failed to get tag: %w", err)
	}

	if parentTagID.Valid {
		tag.ParentTagID = parentTagID.String
	}

	if len(metadataJSON) > 0 {
		json.Unmarshal(metadataJSON, &tag.Metadata)
	}

	return &tag, nil
}

// AddTagToProduct adds a tag to a product.
func (tm *TagManager) AddTagToProduct(ctx context.Context, productID, tagID string, source string, confidence float64, createdBy string) error {
	query := `
		INSERT INTO product_tags (product_id, tag_id, confidence, source, created_at, created_by)
		VALUES ($1, $2, $3, $4, $5, $6)
		ON CONFLICT (product_id, tag_id) DO UPDATE
		SET confidence = EXCLUDED.confidence,
		    source = EXCLUDED.source,
		    updated_at = NOW()
	`

	_, err := tm.db.ExecContext(ctx, query, productID, tagID, confidence, source, time.Now(), createdBy)
	if err != nil {
		return fmt.Errorf("failed to add tag to product: %w", err)
	}

	// Update tag usage count
	_, err = tm.db.ExecContext(ctx, `
		UPDATE tags SET usage_count = usage_count + 1, updated_at = NOW()
		WHERE id = $1
	`, tagID)

	return err
}

// RemoveTagFromProduct removes a tag from a product.
func (tm *TagManager) RemoveTagFromProduct(ctx context.Context, productID, tagID string) error {
	query := `DELETE FROM product_tags WHERE product_id = $1 AND tag_id = $2`
	_, err := tm.db.ExecContext(ctx, query, productID, tagID)
	if err != nil {
		return fmt.Errorf("failed to remove tag from product: %w", err)
	}

	// Update tag usage count
	_, err = tm.db.ExecContext(ctx, `
		UPDATE tags SET usage_count = GREATEST(usage_count - 1, 0), updated_at = NOW()
		WHERE id = $1
	`, tagID)

	return err
}

// GetProductTags retrieves all tags for a product.
func (tm *TagManager) GetProductTags(ctx context.Context, productID string) ([]*Tag, error) {
	query := `
		SELECT t.id, t.name, t.category, t.parent_tag_id, t.description, t.usage_count, t.metadata, t.created_at, t.updated_at
		FROM tags t
		INNER JOIN product_tags pt ON t.id = pt.tag_id
		WHERE pt.product_id = $1
		ORDER BY pt.confidence DESC, t.usage_count DESC
	`

	rows, err := tm.db.QueryContext(ctx, query, productID)
	if err != nil {
		return nil, fmt.Errorf("failed to query product tags: %w", err)
	}
	defer rows.Close()

	var tags []*Tag
	for rows.Next() {
		var tag Tag
		var parentTagID sql.NullString
		var metadataJSON []byte

		err := rows.Scan(
			&tag.ID,
			&tag.Name,
			&tag.Category,
			&parentTagID,
			&tag.Description,
			&tag.UsageCount,
			&metadataJSON,
			&tag.CreatedAt,
			&tag.UpdatedAt,
		)
		if err != nil {
			continue
		}

		if parentTagID.Valid {
			tag.ParentTagID = parentTagID.String
		}

		if len(metadataJSON) > 0 {
			json.Unmarshal(metadataJSON, &tag.Metadata)
		}

		tags = append(tags, &tag)
	}

	return tags, nil
}

// SuggestTags suggests tags for a product based on content and existing tags.
func (tm *TagManager) SuggestTags(ctx context.Context, productID string, productDescription string, existingTags []string) ([]AutoTaggingSuggestion, error) {
	var suggestions []AutoTaggingSuggestion

	// Search for tags matching product description
	keywords := strings.Fields(strings.ToLower(productDescription))
	
	query := `
		SELECT id, name, usage_count
		FROM tags
		WHERE LOWER(name) LIKE ANY($1) OR LOWER(description) LIKE ANY($1)
		ORDER BY usage_count DESC
		LIMIT 10
	`

	// Build LIKE patterns
	patterns := []string{}
	for _, keyword := range keywords {
		if len(keyword) > 2 {
			patterns = append(patterns, "%"+keyword+"%")
		}
	}

	rows, err := tm.db.QueryContext(ctx, query, patterns)
	if err != nil {
		return suggestions, nil // Return empty suggestions on error
	}
	defer rows.Close()

	for rows.Next() {
		var tagID, tagName string
		var usageCount int64

		if err := rows.Scan(&tagID, &tagName, &usageCount); err != nil {
			continue
		}

		// Skip if already tagged
		alreadyTagged := false
		for _, existing := range existingTags {
			if existing == tagID {
				alreadyTagged = true
				break
			}
		}

		if !alreadyTagged {
			// Calculate confidence based on usage and keyword match
			confidence := 0.5
			if usageCount > 10 {
				confidence += 0.2
			}
			if usageCount > 50 {
				confidence += 0.2
			}

			suggestions = append(suggestions, AutoTaggingSuggestion{
				TagID:     tagID,
				TagName:   tagName,
				Confidence: confidence,
				Reason:    fmt.Sprintf("Matches product description keywords"),
			})
		}
	}

	return suggestions, nil
}

// GetTagHierarchy retrieves the tag hierarchy.
func (tm *TagManager) GetTagHierarchy(ctx context.Context, category string) ([]*TagHierarchy, error) {
	query := `
		SELECT id, name, category, parent_tag_id, description, usage_count, metadata, created_at, updated_at
		FROM tags
		WHERE category = $1 OR ($1 = '' AND category IS NOT NULL)
		ORDER BY parent_tag_id NULLS FIRST, usage_count DESC
	`

	rows, err := tm.db.QueryContext(ctx, query, category)
	if err != nil {
		return nil, fmt.Errorf("failed to query tags: %w", err)
	}
	defer rows.Close()

	// Build tag map
	tagMap := make(map[string]*Tag)
	var rootTags []*Tag

	for rows.Next() {
		var tag Tag
		var parentTagID sql.NullString
		var metadataJSON []byte

		err := rows.Scan(
			&tag.ID,
			&tag.Name,
			&tag.Category,
			&parentTagID,
			&tag.Description,
			&tag.UsageCount,
			&metadataJSON,
			&tag.CreatedAt,
			&tag.UpdatedAt,
		)
		if err != nil {
			continue
		}

		if parentTagID.Valid {
			tag.ParentTagID = parentTagID.String
		}

		if len(metadataJSON) > 0 {
			json.Unmarshal(metadataJSON, &tag.Metadata)
		}

		tagMap[tag.ID] = &tag

		if tag.ParentTagID == "" {
			rootTags = append(rootTags, &tag)
		}
	}

	// Build hierarchy
	var hierarchy []*TagHierarchy
	for _, rootTag := range rootTags {
		h := tm.buildHierarchy(rootTag, tagMap, 0)
		hierarchy = append(hierarchy, h)
	}

	return hierarchy, nil
}

// buildHierarchy builds a hierarchy tree recursively.
func (tm *TagManager) buildHierarchy(tag *Tag, tagMap map[string]*Tag, depth int) *TagHierarchy {
	h := &TagHierarchy{
		Tag:   tag,
		Depth: depth,
	}

	// Find children
	for _, t := range tagMap {
		if t.ParentTagID == tag.ID {
			child := tm.buildHierarchy(t, tagMap, depth+1)
			h.Children = append(h.Children, child)
		}
	}

	return h
}

// GetTagAnalytics retrieves analytics for a tag.
func (tm *TagManager) GetTagAnalytics(ctx context.Context, tagID string) (*TagAnalytics, error) {
	// Get usage count
	var usageCount, productCount int64
	err := tm.db.QueryRowContext(ctx, `
		SELECT usage_count, COUNT(DISTINCT product_id)
		FROM tags t
		LEFT JOIN product_tags pt ON t.id = pt.tag_id
		WHERE t.id = $1
		GROUP BY t.usage_count
	`, tagID).Scan(&usageCount, &productCount)
	if err != nil {
		return nil, fmt.Errorf("failed to get tag analytics: %w", err)
	}

	// Get related tags (tags used together)
	relatedTagsQuery := `
		SELECT DISTINCT pt2.tag_id, t.name, COUNT(*) as count
		FROM product_tags pt1
		INNER JOIN product_tags pt2 ON pt1.product_id = pt2.product_id
		INNER JOIN tags t ON pt2.tag_id = t.id
		WHERE pt1.tag_id = $1 AND pt2.tag_id != $1
		GROUP BY pt2.tag_id, t.name
		ORDER BY count DESC
		LIMIT 10
	`

	rows, err := tm.db.QueryContext(ctx, relatedTagsQuery, tagID)
	if err == nil {
		defer rows.Close()
		var relatedTags []string
		for rows.Next() {
			var relatedTagID, relatedTagName string
			var count int64
			if err := rows.Scan(&relatedTagID, &relatedTagName, &count); err == nil {
				relatedTags = append(relatedTags, relatedTagID)
			}
		}
	}

	analytics := &TagAnalytics{
		TagID:        tagID,
		UsageCount:   usageCount,
		ProductCount: productCount,
		Popularity:   float64(usageCount) / 100.0, // Normalized
		Trend:        "stable", // Simplified
		RelatedTags:  []string{},
	}

	return analytics, nil
}

// SearchTags searches for tags by name or description.
func (tm *TagManager) SearchTags(ctx context.Context, query string, category string, limit int) ([]*Tag, error) {
	sqlQuery := `
		SELECT id, name, category, parent_tag_id, description, usage_count, metadata, created_at, updated_at
		FROM tags
		WHERE (LOWER(name) LIKE $1 OR LOWER(description) LIKE $1)
	`

	args := []interface{}{"%" + strings.ToLower(query) + "%"}

	if category != "" {
		sqlQuery += " AND category = $2"
		args = append(args, category)
	}

	sqlQuery += " ORDER BY usage_count DESC LIMIT $"
	if category != "" {
		sqlQuery += "3"
		args = append(args, limit)
	} else {
		sqlQuery += "2"
		args = append(args, limit)
	}

	rows, err := tm.db.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to search tags: %w", err)
	}
	defer rows.Close()

	var tags []*Tag
	for rows.Next() {
		var tag Tag
		var parentTagID sql.NullString
		var metadataJSON []byte

		err := rows.Scan(
			&tag.ID,
			&tag.Name,
			&tag.Category,
			&parentTagID,
			&tag.Description,
			&tag.UsageCount,
			&metadataJSON,
			&tag.CreatedAt,
			&tag.UpdatedAt,
		)
		if err != nil {
			continue
		}

		if parentTagID.Valid {
			tag.ParentTagID = parentTagID.String
		}

		if len(metadataJSON) > 0 {
			json.Unmarshal(metadataJSON, &tag.Metadata)
		}

		tags = append(tags, &tag)
	}

	return tags, nil
}

