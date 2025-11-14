package workflows

import (
	"context"
	"fmt"
	"log"

	"github.com/plturrell/aModels/services/orchestration/digitaltwin"
)

// DigitalTwinIntegration integrates digital twin with data products.
type DigitalTwinIntegration struct {
	twinSystem *digitaltwin.DigitalTwinSystem
	logger     *log.Logger
}

// NewDigitalTwinIntegration creates a new digital twin integration.
func NewDigitalTwinIntegration(twinSystem *digitaltwin.DigitalTwinSystem, logger *log.Logger) *DigitalTwinIntegration {
	return &DigitalTwinIntegration{
		twinSystem: twinSystem,
		logger:     logger,
	}
}

// CreateTwinForDataProduct creates a digital twin for a data product.
func (dti *DigitalTwinIntegration) CreateTwinForDataProduct(ctx context.Context, productID string, productName string) (*digitaltwin.Twin, error) {
	if dti.twinSystem == nil {
		return nil, fmt.Errorf("digital twin system not initialized")
	}

	twin, err := dti.twinSystem.CreateTwinFromDataProduct(ctx, productID, productName)
	if err != nil {
		return nil, fmt.Errorf("failed to create twin: %w", err)
	}

	if dti.logger != nil {
		dti.logger.Printf("Created digital twin %s for data product %s", twin.ID, productID)
	}

	return twin, nil
}

// SimulateDataProductChange simulates a change to a data product using its twin.
func (dti *DigitalTwinIntegration) SimulateDataProductChange(ctx context.Context, productID string, change digitaltwin.Change) (*digitaltwin.Rehearsal, error) {
	if dti.twinSystem == nil {
		return nil, fmt.Errorf("digital twin system not initialized")
	}

	// Find twin for product
	filters := digitaltwin.TwinFilters{SourceID: productID, Limit: 1}
	twins, err := dti.twinSystem.GetTwinManager().ListTwins(ctx, filters)
	if err != nil {
		return nil, fmt.Errorf("failed to locate twin for product %s: %w", productID, err)
	}
	if len(twins) == 0 {
		return nil, fmt.Errorf("twin not found for product %s", productID)
	}
	twin := twins[0]

	// Rehearse change
	result, err := dti.twinSystem.RehearseChange(ctx, twin.ID, change, true, false)
	if err != nil {
		return nil, fmt.Errorf("failed to rehearse change: %w", err)
	}

	return result, nil
}
