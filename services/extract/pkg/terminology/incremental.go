package terminology

import (
	"context"
	"log"
	"sync"
	"time"
)

// IncrementalTerminologyUpdater updates terminology on every extraction run.
type IncrementalTerminologyUpdater struct {
	terminologyLearner *TerminologyLearner
	mu                 sync.RWMutex
	logger             *log.Logger
	updateCount        int64
	lastUpdate         time.Time
}

// NewIncrementalTerminologyUpdater creates a new incremental terminology updater.
func NewIncrementalTerminologyUpdater(terminologyLearner *TerminologyLearner, logger *log.Logger) *IncrementalTerminologyUpdater {
	return &IncrementalTerminologyUpdater{
		terminologyLearner: terminologyLearner,
		logger:             logger,
		lastUpdate:         time.Now(),
	}
}

// UpdateFromExtraction updates terminology from an extraction run.
func (itu *IncrementalTerminologyUpdater) UpdateFromExtraction(ctx context.Context, nodes []Node, edges []Edge) error {
	itu.mu.Lock()
	defer itu.mu.Unlock()

	// Learn from extraction
	if err := itu.terminologyLearner.LearnFromExtraction(ctx, nodes, edges); err != nil {
		itu.logger.Printf("Failed to learn from extraction: %v", err)
		return err
	}

	itu.updateCount++
	itu.lastUpdate = time.Now()

	itu.logger.Printf("Incremental terminology update #%d completed", itu.updateCount)
	return nil
}

// GetUpdateStats returns update statistics.
func (itu *IncrementalTerminologyUpdater) GetUpdateStats() (int64, time.Time) {
	itu.mu.RLock()
	defer itu.mu.RUnlock()

	return itu.updateCount, itu.lastUpdate
}

