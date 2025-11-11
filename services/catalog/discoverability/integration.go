package discoverability

import (
	"database/sql"
	"log"
)

// DiscoverabilitySystem integrates all discoverability components.
type DiscoverabilitySystem struct {
	tagManager      *TagManager
	crossTeamSearch *CrossTeamSearch
	marketplace     *Marketplace
	logger          *log.Logger
}

// NewDiscoverabilitySystem creates a new discoverability system.
func NewDiscoverabilitySystem(db *sql.DB, logger *log.Logger) *DiscoverabilitySystem {
	return &DiscoverabilitySystem{
		tagManager:      NewTagManager(db, logger),
		crossTeamSearch: NewCrossTeamSearch(db, logger),
		marketplace:     NewMarketplace(db, logger),
		logger:          logger,
	}
}

// GetTagManager returns the tag manager.
func (ds *DiscoverabilitySystem) GetTagManager() *TagManager {
	return ds.tagManager
}

// GetCrossTeamSearch returns the cross-team search service.
func (ds *DiscoverabilitySystem) GetCrossTeamSearch() *CrossTeamSearch {
	return ds.crossTeamSearch
}

// GetMarketplace returns the marketplace.
func (ds *DiscoverabilitySystem) GetMarketplace() *Marketplace {
	return ds.marketplace
}

