package iso11179

import (
	"fmt"
	"time"

	"github.com/plturrell/aModels/services/catalog/quality"
	"github.com/plturrell/aModels/services/catalog/security"
)

// EnhancedDataElement extends DataElement with Data Product Thinking capabilities.
type EnhancedDataElement struct {
	*DataElement

	// Quality metrics (Trustworthy principle)
	QualityMetrics *quality.QualityMetrics

	// Access control (Secure principle)
	AccessControl *security.AccessControl

	// Product ownership
	ProductOwner string
	ProductTeam  []string

	// Lifecycle state
	LifecycleState string // "draft", "published", "deprecated", "archived"

	// Consumer feedback
	ConsumerRating    float64
	ConsumerFeedback  []Feedback

	// Usage analytics
	UsageCount    int64
	LastAccessed  time.Time
	PopularUsers  []string

	// Documentation
	DocumentationURL string
	SampleDataURL    string
	SchemaDocumentationURL string
}

// Feedback represents consumer feedback on a data product.
type Feedback struct {
	UserID    string
	Rating    float64
	Comment   string
	Timestamp time.Time
}

// NewEnhancedDataElement creates a new enhanced data element.
func NewEnhancedDataElement(element *DataElement) *EnhancedDataElement {
	return &EnhancedDataElement{
		DataElement:    element,
		QualityMetrics: quality.NewQualityMetrics(),
		AccessControl:  security.NewAccessControl(element.Steward, "internal"),
		LifecycleState: "draft",
		ConsumerFeedback: []Feedback{},
		PopularUsers:    []string{},
	}
}

// SetLifecycleState sets the lifecycle state.
func (ede *EnhancedDataElement) SetLifecycleState(state string) {
	validStates := []string{"draft", "published", "deprecated", "archived"}
	for _, validState := range validStates {
		if state == validState {
			ede.LifecycleState = state
			ede.UpdatedAt = time.Now()
			return
		}
	}
}

// Publish publishes the data product (moves from draft to published).
func (ede *EnhancedDataElement) Publish(registry *MetadataRegistry) error {
	if ede.LifecycleState != "draft" {
		return fmt.Errorf("can only publish from draft state, current state: %s", ede.LifecycleState)
	}

	// Validate before publishing
	if err := ede.Validate(registry); err != nil {
		return fmt.Errorf("validation failed: %w", err)
	}

	ede.SetLifecycleState("published")
	return nil
}

// Deprecate deprecates the data product.
func (ede *EnhancedDataElement) Deprecate(reason string) {
	ede.SetLifecycleState("deprecated")
	ede.AddMetadata("deprecation_reason", reason)
	ede.AddMetadata("deprecated_at", time.Now().UTC().Format(time.RFC3339))
}

// AddFeedback adds consumer feedback.
func (ede *EnhancedDataElement) AddFeedback(userID string, rating float64, comment string) {
	feedback := Feedback{
		UserID:    userID,
		Rating:    rating,
		Comment:   comment,
		Timestamp: time.Now(),
	}
	ede.ConsumerFeedback = append(ede.ConsumerFeedback, feedback)

	// Recalculate average rating
	totalRating := 0.0
	for _, fb := range ede.ConsumerFeedback {
		totalRating += fb.Rating
	}
	ede.ConsumerRating = totalRating / float64(len(ede.ConsumerFeedback))
}

// RecordUsage records usage of the data product.
func (ede *EnhancedDataElement) RecordUsage(userID string) {
	ede.UsageCount++
	ede.LastAccessed = time.Now()

	// Track popular users
	found := false
	for _, user := range ede.PopularUsers {
		if user == userID {
			found = true
			break
		}
	}
	if !found && len(ede.PopularUsers) < 10 {
		ede.PopularUsers = append(ede.PopularUsers, userID)
	}
}

// Validate validates the enhanced data element before publishing.
// Note: This requires a registry context for full validation.
func (ede *EnhancedDataElement) Validate(registry *MetadataRegistry) error {
	// Validate base data element using registry
	if errors := registry.ValidateDataElement(ede.DataElement); len(errors) > 0 {
		return fmt.Errorf("data element validation failed: %v", errors)
	}

	// Validate access control
	if ede.AccessControl != nil {
		if err := ede.AccessControl.ValidateAccessControl(); err != nil {
			return fmt.Errorf("access control validation failed: %w", err)
		}
	}

	// Validate quality metrics
	if ede.QualityMetrics != nil {
		if ede.QualityMetrics.ValidationStatus == "failed" {
			return fmt.Errorf("quality validation failed")
		}
	}

	// Validate ownership
	if ede.ProductOwner == "" && ede.Steward == "" {
		return fmt.Errorf("product owner or steward is required")
	}

	return nil
}

