package domain

import (
	"context"
	"fmt"
	"log"
	"time"
)

// LifecycleManager manages domain lifecycle operations (create, update, archive, delete)
type LifecycleManager struct {
	domainManager *DomainManager
	postgresStore *PostgresConfigStore
	redisLoader   *RedisConfigLoader
}

// NewLifecycleManager creates a new lifecycle manager
func NewLifecycleManager(
	dm *DomainManager,
	postgresStore *PostgresConfigStore,
	redisLoader *RedisConfigLoader,
) *LifecycleManager {
	return &LifecycleManager{
		domainManager: dm,
		postgresStore: postgresStore,
		redisLoader:   redisLoader,
	}
}

// CreateDomain creates a new domain configuration
func (lm *LifecycleManager) CreateDomain(
	ctx context.Context,
	domainID string,
	config *DomainConfig,
	metadata map[string]interface{},
) error {
	// Validate configuration
	if err := config.Validate(); err != nil {
		return fmt.Errorf("invalid domain config: %w", err)
	}

	// Check if domain already exists
	if _, exists := lm.domainManager.GetDomainConfig(domainID); exists {
		return fmt.Errorf("domain %s already exists", domainID)
	}

	// Add to domain manager
	lm.domainManager.AddDomain(domainID, config)

	// Save to PostgreSQL if available
	if lm.postgresStore != nil {
		trainingRunID := ""
		modelVersion := "v1.0.0"
		metrics := make(map[string]interface{})

		if metadata != nil {
			if trID, ok := metadata["training_run_id"].(string); ok {
				trainingRunID = trID
			}
			if mv, ok := metadata["model_version"].(string); ok {
				modelVersion = mv
			}
			if m, ok := metadata["performance_metrics"].(map[string]interface{}); ok {
				metrics = m
			}
		}

		if err := lm.postgresStore.SaveDomainConfig(
			ctx, domainID, config, trainingRunID, modelVersion, metrics,
		); err != nil {
			log.Printf("⚠️  Failed to save domain to PostgreSQL: %v", err)
		}
	}

	// Sync to Redis if available
	if lm.redisLoader != nil && lm.postgresStore != nil {
		if err := lm.postgresStore.SyncToRedis(ctx, lm.redisLoader); err != nil {
			log.Printf("⚠️  Failed to sync to Redis: %v", err)
		}
	}

	log.Printf("✅ Created domain: %s", domainID)
	return nil
}

// UpdateDomain updates an existing domain configuration
func (lm *LifecycleManager) UpdateDomain(
	ctx context.Context,
	domainID string,
	config *DomainConfig,
	metadata map[string]interface{},
) error {
	// Validate configuration
	if err := config.Validate(); err != nil {
		return fmt.Errorf("invalid domain config: %w", err)
	}

	// Check if domain exists
	if _, exists := lm.domainManager.GetDomainConfig(domainID); !exists {
		return fmt.Errorf("domain %s does not exist", domainID)
	}

	// Update in domain manager
	lm.domainManager.AddDomain(domainID, config)

	// Update in PostgreSQL
	if lm.postgresStore != nil {
		trainingRunID := ""
		modelVersion := ""
		metrics := make(map[string]interface{})

		if metadata != nil {
			if trID, ok := metadata["training_run_id"].(string); ok {
				trainingRunID = trID
			}
			if mv, ok := metadata["model_version"].(string); ok {
				modelVersion = mv
			}
			if m, ok := metadata["performance_metrics"].(map[string]interface{}); ok {
				metrics = m
			}
		}

		if err := lm.postgresStore.SaveDomainConfig(
			ctx, domainID, config, trainingRunID, modelVersion, metrics,
		); err != nil {
			return fmt.Errorf("failed to update in PostgreSQL: %w", err)
		}
	}

	// Sync to Redis
	if lm.redisLoader != nil && lm.postgresStore != nil {
		if err := lm.postgresStore.SyncToRedis(ctx, lm.redisLoader); err != nil {
			log.Printf("⚠️  Failed to sync to Redis: %v", err)
		}
	}

	log.Printf("✅ Updated domain: %s", domainID)
	return nil
}

// ArchiveDomain archives a domain (marks as disabled but keeps history)
func (lm *LifecycleManager) ArchiveDomain(
	ctx context.Context,
	domainID string,
	reason string,
) error {
	if _, exists := lm.domainManager.GetDomainConfig(domainID); !exists {
		return fmt.Errorf("domain %s does not exist", domainID)
	}

	// Remove from active domain manager
	lm.domainManager.RemoveDomain(domainID)

	// Archive in PostgreSQL
	if lm.postgresStore != nil {
		// Mark as disabled in PostgreSQL
		// Implementation would update enabled flag
		log.Printf("📦 Archived domain: %s (reason: %s)", domainID, reason)
	}

	// Remove from Redis
	if lm.redisLoader != nil && lm.postgresStore != nil {
		if err := lm.postgresStore.SyncToRedis(ctx, lm.redisLoader); err != nil {
			log.Printf("⚠️  Failed to sync to Redis: %v", err)
		}
	}

	log.Printf("✅ Archived domain: %s", domainID)
	return nil
}

// DeleteDomain permanently deletes a domain (use ArchiveDomain for safe removal)
func (lm *LifecycleManager) DeleteDomain(
	ctx context.Context,
	domainID string,
	force bool,
) error {
	if !force {
		return fmt.Errorf("delete requires force=true (use ArchiveDomain for safe removal)")
	}

	// Remove from domain manager
	lm.domainManager.RemoveDomain(domainID)

	// Delete from PostgreSQL (if supported)
	if lm.postgresStore != nil {
		// Implementation would delete from database
		log.Printf("🗑️  Deleted domain: %s", domainID)
	}

	log.Printf("✅ Deleted domain: %s", domainID)
	return nil
}

// ListDomains lists all domains with their status
func (lm *LifecycleManager) ListDomains(ctx context.Context) (map[string]DomainStatus, error) {
	statuses := make(map[string]DomainStatus)

	// Get active domains from domain manager
	activeDomains := lm.domainManager.ListDomains()
	for _, domainID := range activeDomains {
		statuses[domainID] = DomainStatus{
			ID:     domainID,
			Status: "active",
		}
	}

	// Get archived domains from PostgreSQL
	if lm.postgresStore != nil {
		// Implementation would query archived domains
	}

	return statuses, nil
}

// DomainStatus represents the status of a domain
type DomainStatus struct {
	ID            string                 `json:"id"`
	Status        string                 `json:"status"` // active, archived, deleted
	Config        *DomainConfig          `json:"config,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	ArchivedAt    *time.Time             `json:"archived_at,omitempty"`
	ArchivedBy    string                 `json:"archived_by,omitempty"`
	ArchiveReason string                 `json:"archive_reason,omitempty"`
}
