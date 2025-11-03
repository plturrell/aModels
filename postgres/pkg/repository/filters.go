package repository

import (
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres/pkg/models"
)

// ListFilters captures optional filters for listing operations.
type ListFilters struct {
	LibraryType   string
	SessionID     string
	Status        models.OperationStatus
	CreatedAfter  *time.Time
	CreatedBefore *time.Time
	PageSize      int
	PageToken     string
}

// AnalyticsFilters simply aliases the models representation for convenience.
type AnalyticsFilters = models.AnalyticsFilters
