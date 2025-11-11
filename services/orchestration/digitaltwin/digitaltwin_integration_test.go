package digitaltwin

import (
	"context"
	"database/sql"
	"log"
	"os"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

func setupTestDB(t *testing.T) *sql.DB {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("Failed to open test database: %v", err)
	}

	schema := `
		CREATE TABLE IF NOT EXISTS digital_twins (
			id TEXT PRIMARY KEY,
			name TEXT NOT NULL,
			type TEXT NOT NULL,
			source_id TEXT,
			version TEXT,
			state TEXT,
			configuration TEXT,
			metadata TEXT,
			created_at TIMESTAMP,
			updated_at TIMESTAMP
		);
	`
	if _, err := db.Exec(schema); err != nil {
		t.Fatalf("Failed to create schema: %v", err)
	}

	return db
}

func TestTwinManager_CreateTwin(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	store := NewPostgresTwinStore(db, logger)
	manager := NewTwinManager(store, logger)

	ctx := context.Background()
	req := CreateTwinRequest{
		Name:     "Test Twin",
		Type:     "data_product",
		SourceID: "test-product-1",
		Configuration: TwinConfiguration{
			ReplicationLevel: 0.8,
			SimulationMode:   "full",
		},
	}

	twin, err := manager.CreateTwin(ctx, req)
	if err != nil {
		t.Fatalf("CreateTwin() error = %v", err)
	}

	if twin.ID == "" {
		t.Error("Expected twin ID to be set")
	}

	if twin.Name != req.Name {
		t.Errorf("Name = %v, want %v", twin.Name, req.Name)
	}
}

func TestTwinManager_CreateSnapshot(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	store := NewPostgresTwinStore(db, logger)
	manager := NewTwinManager(store, logger)

	ctx := context.Background()
	req := CreateTwinRequest{
		Name:     "Test Twin",
		Type:     "data_product",
		SourceID: "test-product-1",
	}

	twin, err := manager.CreateTwin(ctx, req)
	if err != nil {
		t.Fatalf("CreateTwin() error = %v", err)
	}

	snapshot, err := manager.CreateSnapshot(ctx, twin.ID, "Test snapshot")
	if err != nil {
		t.Fatalf("CreateSnapshot() error = %v", err)
	}

	if snapshot.ID == "" {
		t.Error("Expected snapshot ID to be set")
	}
}

func TestSimulationEngine_StartSimulation(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	store := NewPostgresTwinStore(db, logger)
	manager := NewTwinManager(store, logger)
	engine := NewSimulationEngine(manager, logger)

	ctx := context.Background()
	
	// Create twin first
	req := CreateTwinRequest{
		Name:     "Test Twin",
		Type:     "data_product",
		SourceID: "test-product-1",
	}
	twin, err := manager.CreateTwin(ctx, req)
	if err != nil {
		t.Fatalf("CreateTwin() error = %v", err)
	}

	simReq := StartSimulationRequest{
		TwinID: twin.ID,
		Type:   "pipeline",
		Config: SimulationConfig{
			Duration:   1 * time.Second,
			TimeStep:   100 * time.Millisecond,
			DataVolume: 100,
		},
	}

	simulation, err := engine.StartSimulation(ctx, simReq)
	if err != nil {
		t.Fatalf("StartSimulation() error = %v", err)
	}

	if simulation.ID == "" {
		t.Error("Expected simulation ID to be set")
	}

	// Wait a bit for simulation to complete
	time.Sleep(2 * time.Second)

	// Get simulation status
	sim, err := engine.GetSimulation(simulation.ID)
	if err != nil {
		t.Fatalf("GetSimulation() error = %v", err)
	}

	if sim.Status != "completed" && sim.Status != "running" {
		t.Errorf("Simulation status = %v, want completed or running", sim.Status)
	}
}

func TestStressTester_RunStressTest(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	store := NewPostgresTwinStore(db, logger)
	manager := NewTwinManager(store, logger)
	tester := NewStressTester(manager, logger)

	ctx := context.Background()
	
	// Create twin first
	req := CreateTwinRequest{
		Name:     "Test Twin",
		Type:     "data_product",
		SourceID: "test-product-1",
	}
	twin, err := manager.CreateTwin(ctx, req)
	if err != nil {
		t.Fatalf("CreateTwin() error = %v", err)
	}

	stressReq := StressTestRequest{
		TwinID: twin.ID,
		Config: StressTestConfig{
			Duration:       1 * time.Second,
			TargetRPS:      10,
			MaxConcurrency: 5,
			LoadProfile: LoadProfile{
				Type: "constant",
				Stages: []LoadStage{
					{Duration: 1 * time.Second, TargetRPS: 10, Concurrency: 5},
				},
			},
		},
	}

	test, err := tester.RunStressTest(ctx, stressReq)
	if err != nil {
		t.Fatalf("RunStressTest() error = %v", err)
	}

	if test.ID == "" {
		t.Error("Expected test ID to be set")
	}
}

func TestRehearsalMode_StartRehearsal(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	store := NewPostgresTwinStore(db, logger)
	manager := NewTwinManager(store, logger)
	engine := NewSimulationEngine(manager, logger)
	tester := NewStressTester(manager, logger)
	rehearsalMode := NewRehearsalMode(manager, engine, tester, logger)

	ctx := context.Background()
	
	// Create twin first
	req := CreateTwinRequest{
		Name:     "Test Twin",
		Type:     "data_product",
		SourceID: "test-product-1",
	}
	twin, err := manager.CreateTwin(ctx, req)
	if err != nil {
		t.Fatalf("CreateTwin() error = %v", err)
	}

	rehearsalReq := StartRehearsalRequest{
		TwinID:        twin.ID,
		Change: Change{
			ID:          "change-1",
			Type:        "schema",
			Description: "Test change",
			Before:      map[string]interface{}{"field": "old"},
			After:       map[string]interface{}{"field": "new"},
			Priority:    "medium",
			Risk:        "low",
		},
		RunSimulation: false,
		RunStressTest: false,
	}

	rehearsal, err := rehearsalMode.StartRehearsal(ctx, rehearsalReq)
	if err != nil {
		t.Fatalf("StartRehearsal() error = %v", err)
	}

	if rehearsal.ID == "" {
		t.Error("Expected rehearsal ID to be set")
	}

	if !rehearsal.Validation.Valid {
		t.Error("Expected validation to pass for basic change")
	}
}

