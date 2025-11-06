package vectorstore

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/plturrell/aModels/services/catalog/breakdetection"
)

// HANABreakPatternStore stores break patterns in HANA Cloud
type HANABreakPatternStore struct {
	store *HANACloudVectorStore
}

// NewHANABreakPatternStore creates a new break pattern store
func NewHANABreakPatternStore(store *HANACloudVectorStore) *HANABreakPatternStore {
	return &HANABreakPatternStore{store: store}
}

// StoreBreakPattern stores a break pattern in HANA Cloud
func (hbps *HANABreakPatternStore) StoreBreakPattern(ctx context.Context,
	systemName breakdetection.SystemName,
	detectionType breakdetection.DetectionType,
	breakType breakdetection.BreakType,
	pattern *BreakPattern,
	vector []float32) error {

	info := &PublicInformation{
		ID:       fmt.Sprintf("pattern-%s-%s-%s-%d", systemName, detectionType, breakType, time.Now().Unix()),
		Type:     "break_pattern",
		System:   string(systemName),
		Category: string(detectionType),
		Title:    fmt.Sprintf("Break Pattern: %s in %s", breakType, systemName),
		Content:  pattern.Description,
		Vector:   vector,
		Metadata: map[string]interface{}{
			"system_name":    string(systemName),
			"detection_type": string(detectionType),
			"break_type":     string(breakType),
			"frequency":      pattern.Frequency,
			"resolution":     pattern.Resolution,
			"prevention":     pattern.Prevention,
		},
		Tags:      pattern.Tags,
		IsPublic:  true,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	return hbps.store.StorePublicInformation(ctx, info)
}

// BreakPattern represents a break pattern
type BreakPattern struct {
	Description string   `json:"description"`
	Frequency   int      `json:"frequency"`   // How often this pattern occurs
	Resolution  string   `json:"resolution"`  // How to resolve
	Prevention  string   `json:"prevention"`  // How to prevent
	Tags        []string `json:"tags"`
}

// HANARegulatoryRuleStore stores regulatory rules in HANA Cloud
type HANARegulatoryRuleStore struct {
	store *HANACloudVectorStore
}

// NewHANARegulatoryRuleStore creates a new regulatory rule store
func NewHANARegulatoryRuleStore(store *HANACloudVectorStore) *HANARegulatoryRuleStore {
	return &HANARegulatoryRuleStore{store: store}
}

// StoreRegulatoryRule stores a regulatory rule in HANA Cloud
func (hrrs *HANARegulatoryRuleStore) StoreRegulatoryRule(ctx context.Context,
	rule *RegulatoryRule,
	vector []float32) error {

	info := &PublicInformation{
		ID:       fmt.Sprintf("rule-%s-%d", rule.Regulation, time.Now().Unix()),
		Type:     "regulatory_rule",
		System:   "general", // Public regulatory rules
		Category: rule.Regulation,
		Title:    rule.Title,
		Content:  rule.Description,
		Vector:   vector,
		Metadata: map[string]interface{}{
			"regulation":    rule.Regulation,
			"requirement":   rule.Requirement,
			"compliance":    rule.Compliance,
			"effective_date": rule.EffectiveDate.Format(time.RFC3339),
		},
		Tags:      rule.Tags,
		IsPublic:  true,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	return hrrs.store.StorePublicInformation(ctx, info)
}

// RegulatoryRule represents a regulatory rule
type RegulatoryRule struct {
	Regulation   string    `json:"regulation"`    // e.g., "Basel III", "IFRS 9"
	Title        string    `json:"title"`
	Description  string    `json:"description"`
	Requirement  string    `json:"requirement"`
	Compliance   string    `json:"compliance"`
	EffectiveDate time.Time `json:"effective_date"`
	Tags         []string  `json:"tags"`
}

// HANABestPracticeStore stores best practices in HANA Cloud
type HANABestPracticeStore struct {
	store *HANACloudVectorStore
}

// NewHANABestPracticeStore creates a new best practice store
func NewHANABestPracticeStore(store *HANACloudVectorStore) *HANABestPracticeStore {
	return &HANABestPracticeStore{store: store}
}

// StoreBestPractice stores a best practice in HANA Cloud
func (hbps *HANABestPracticeStore) StoreBestPractice(ctx context.Context,
	practice *BestPractice,
	vector []float32) error {

	info := &PublicInformation{
		ID:       fmt.Sprintf("practice-%s-%d", practice.Category, time.Now().Unix()),
		Type:     "best_practice",
		System:   practice.System, // Can be "general" or specific system
		Category: practice.Category,
		Title:    practice.Title,
		Content:  practice.Description,
		Vector:   vector,
		Metadata: map[string]interface{}{
			"category":    practice.Category,
			"application": practice.Application,
			"benefits":    practice.Benefits,
		},
		Tags:      practice.Tags,
		IsPublic:  true,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	return hbps.store.StorePublicInformation(ctx, info)
}

// BestPractice represents a best practice
type BestPractice struct {
	System      string   `json:"system"`      // "general" or specific system
	Category    string   `json:"category"`    // e.g., "data_quality", "break_detection"
	Title       string   `json:"title"`
	Description string   `json:"description"`
	Application string   `json:"application"` // How to apply
	Benefits    []string `json:"benefits"`
	Tags        []string `json:"tags"`
}

// HANAKnowledgeBaseStore stores general knowledge base entries in HANA Cloud
type HANAKnowledgeBaseStore struct {
	store *HANACloudVectorStore
}

// NewHANAKnowledgeBaseStore creates a new knowledge base store
func NewHANAKnowledgeBaseStore(store *HANACloudVectorStore) *HANAKnowledgeBaseStore {
	return &HANAKnowledgeBaseStore{store: store}
}

// StoreKnowledgeEntry stores a knowledge base entry in HANA Cloud
func (hkbs *HANAKnowledgeBaseStore) StoreKnowledgeEntry(ctx context.Context,
	entry *KnowledgeEntry,
	vector []float32) error {

	info := &PublicInformation{
		ID:       entry.ID,
		Type:     "knowledge_base",
		System:   entry.System,
		Category: entry.Category,
		Title:    entry.Title,
		Content:  entry.Content,
		Vector:   vector,
		Metadata: entry.Metadata,
		Tags:     entry.Tags,
		IsPublic: entry.IsPublic,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		CreatedBy: entry.CreatedBy,
	}

	return hkbs.store.StorePublicInformation(ctx, info)
}

// KnowledgeEntry represents a knowledge base entry
type KnowledgeEntry struct {
	ID        string                 `json:"id"`
	System   string                 `json:"system"`
	Category  string                 `json:"category"`
	Title     string                 `json:"title"`
	Content   string                 `json:"content"`
	Metadata  map[string]interface{} `json:"metadata"`
	Tags      []string               `json:"tags"`
	IsPublic  bool                   `json:"is_public"`
	CreatedBy string                 `json:"created_by,omitempty"`
}

// SearchPublicKnowledge searches public knowledge across all types
func SearchPublicKnowledge(ctx context.Context,
	store *HANACloudVectorStore,
	queryVector []float32,
	query string,
	options *SearchOptions) ([]*PublicInformation, error) {

	if options == nil {
		options = &SearchOptions{
			IsPublic:  &[]bool{true}[0], // Only public
			Limit:     10,
			Threshold: 0.7,
		}
	}

	// Ensure only public information
	if options.IsPublic == nil {
		public := true
		options.IsPublic = &public
	}

	return store.SearchPublicInformation(ctx, queryVector, options)
}

// Integration with break detection service
func StoreBreakForPublicKnowledge(ctx context.Context,
	store *HANACloudVectorStore,
	breakRecord *breakdetection.Break,
	vector []float32,
	logger *log.Logger) error {

	// Store as public break pattern (anonymized)
	info := &PublicInformation{
		ID:       fmt.Sprintf("break-pattern-%s-%d", breakRecord.BreakType, time.Now().Unix()),
		Type:     "break_pattern",
		System:   string(breakRecord.SystemName),
		Category: string(breakRecord.DetectionType),
		Title:    fmt.Sprintf("Break Pattern: %s in %s", breakRecord.BreakType, breakRecord.SystemName),
		Content:  buildBreakContent(breakRecord),
		Vector:   vector,
		Metadata: map[string]interface{}{
			"break_type":     string(breakRecord.BreakType),
			"detection_type": string(breakRecord.DetectionType),
			"severity":       string(breakRecord.Severity),
			"resolution":     breakRecord.ResolutionNotes,
		},
		Tags: []string{
			string(breakRecord.BreakType),
			string(breakRecord.DetectionType),
			string(breakRecord.SystemName),
		},
		IsPublic:  true,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := store.StorePublicInformation(ctx, info); err != nil {
		if logger != nil {
			logger.Printf("Failed to store break pattern: %v", err)
		}
		return err
	}

	if logger != nil {
		logger.Printf("Stored break pattern in public knowledge base: %s", info.ID)
	}

	return nil
}

// BuildBreakContent builds content string from break record for embedding
func BuildBreakContent(breakRecord *breakdetection.Break) string {
	var content string
	content += fmt.Sprintf("Break Type: %s\n", breakRecord.BreakType)
	content += fmt.Sprintf("Detection Type: %s\n", breakRecord.DetectionType)
	content += fmt.Sprintf("Severity: %s\n", breakRecord.Severity)
	
	if breakRecord.AIDescription != "" {
		content += fmt.Sprintf("Description: %s\n", breakRecord.AIDescription)
	}
	
	if breakRecord.RootCauseAnalysis != "" {
		content += fmt.Sprintf("Root Cause: %s\n", breakRecord.RootCauseAnalysis)
	}
	
	if len(breakRecord.Recommendations) > 0 {
		content += "Recommendations:\n"
		for _, rec := range breakRecord.Recommendations {
			content += fmt.Sprintf("- %s\n", rec)
		}
	}
	
	if breakRecord.ResolutionNotes != "" {
		content += fmt.Sprintf("Resolution: %s\n", breakRecord.ResolutionNotes)
	}

	return content
}

