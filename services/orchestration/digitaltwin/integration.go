package digitaltwin

import (
	"context"
	"database/sql"
	"log"
)

// DigitalTwinSystem integrates all digital twin components.
type DigitalTwinSystem struct {
	twinManager      *TwinManager
	simulationEngine *SimulationEngine
	stressTester     *StressTester
	rehearsalMode    *RehearsalMode
	logger           *log.Logger
}

// NewDigitalTwinSystem creates a new digital twin system.
func NewDigitalTwinSystem(db *sql.DB, logger *log.Logger) *DigitalTwinSystem {
	// Create twin store and manager
	twinStore := NewPostgresTwinStore(db, logger)
	twinManager := NewTwinManager(twinStore, logger)

	// Create simulation engine
	simulationEngine := NewSimulationEngine(twinManager, logger)

	// Create stress tester
	stressTester := NewStressTester(twinManager, logger)

	// Create rehearsal mode
	rehearsalMode := NewRehearsalMode(twinManager, simulationEngine, stressTester, logger)

	return &DigitalTwinSystem{
		twinManager:      twinManager,
		simulationEngine: simulationEngine,
		stressTester:     stressTester,
		rehearsalMode:    rehearsalMode,
		logger:           logger,
	}
}

// GetTwinManager returns the twin manager.
func (dts *DigitalTwinSystem) GetTwinManager() *TwinManager {
	return dts.twinManager
}

// GetSimulationEngine returns the simulation engine.
func (dts *DigitalTwinSystem) GetSimulationEngine() *SimulationEngine {
	return dts.simulationEngine
}

// GetStressTester returns the stress tester.
func (dts *DigitalTwinSystem) GetStressTester() *StressTester {
	return dts.stressTester
}

// GetRehearsalMode returns the rehearsal mode.
func (dts *DigitalTwinSystem) GetRehearsalMode() *RehearsalMode {
	return dts.rehearsalMode
}

// CreateTwinFromDataProduct creates a twin from a data product.
func (dts *DigitalTwinSystem) CreateTwinFromDataProduct(ctx context.Context, dataProductID string, name string) (*Twin, error) {
	req := CreateTwinRequest{
		Name:     name,
		Type:     "data_product",
		SourceID: dataProductID,
		Configuration: TwinConfiguration{
			ReplicationLevel: 0.8,
			SimulationMode:   "full",
			DataGeneration: DataGenerationConfig{
				Strategy:    "synthetic",
				Volume:      10000,
				Distribution: "normal",
			},
		},
		Metadata: map[string]interface{}{
			"source_type": "data_product",
			"source_id":   dataProductID,
		},
	}

	return dts.twinManager.CreateTwin(ctx, req)
}

// RehearseChange rehearses a change on a twin.
func (dts *DigitalTwinSystem) RehearseChange(ctx context.Context, twinID string, change Change, runSimulation bool, runStressTest bool) (*Rehearsal, error) {
	req := StartRehearsalRequest{
		TwinID:        twinID,
		Change:        change,
		RunSimulation: runSimulation,
		RunStressTest: runStressTest,
	}

	return dts.rehearsalMode.StartRehearsal(ctx, req)
}

