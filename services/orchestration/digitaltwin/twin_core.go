package digitaltwin

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"
)

// Twin represents a digital twin of a data product or pipeline.
type Twin struct {
	ID            string
	Name          string
	Type          string // "data_product", "pipeline", "system"
	SourceID      string // ID of the source entity being twinned
	Version       string
	State         TwinState
	Configuration TwinConfiguration
	Metadata      map[string]interface{}
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

// TwinState represents the current state of a digital twin.
type TwinState struct {
	Status        string                 // "active", "inactive", "simulating", "error"
	LastSnapshot  time.Time
	Snapshots     []StateSnapshot
	Metrics       map[string]interface{}
	CurrentData   map[string]interface{}
}

// StateSnapshot captures a point-in-time state of the twin.
type StateSnapshot struct {
	ID          string
	Timestamp   time.Time
	State       map[string]interface{}
	Metrics     map[string]interface{}
	Description string
}

// TwinConfiguration defines the configuration of a digital twin.
type TwinConfiguration struct {
	ReplicationLevel float64                // 0.0 to 1.0 - how closely to replicate source
	SimulationMode   string                 // "full", "partial", "lightweight"
	DataGeneration   DataGenerationConfig
	Constraints      []Constraint
	Monitoring       MonitoringConfig
}

// DataGenerationConfig configures how data is generated for the twin.
type DataGenerationConfig struct {
	Strategy    string // "synthetic", "sampled", "historical"
	Volume      int64  // Number of records to generate
	Distribution string // "uniform", "normal", "exponential"
	Seed        int64  // Random seed for reproducibility
}

// Constraint defines a constraint on the twin.
type Constraint struct {
	Type        string // "performance", "data_quality", "resource"
	Condition   string
	Threshold   interface{}
	Action      string // "warn", "fail", "log"
}

// MonitoringConfig configures monitoring for the twin.
type MonitoringConfig struct {
	Enabled     bool
	Metrics     []string
	Alerts      []AlertConfig
	Frequency   time.Duration
}

// AlertConfig configures alerts for the twin.
type AlertConfig struct {
	Type        string
	Condition   string
	Threshold   interface{}
	Severity    string // "low", "medium", "high", "critical"
}

// TwinManager manages digital twins.
type TwinManager struct {
	twins  map[string]*Twin
	store  TwinStore
	logger *log.Logger
}

// TwinStore stores and retrieves twins.
type TwinStore interface {
	SaveTwin(ctx context.Context, twin *Twin) error
	GetTwin(ctx context.Context, id string) (*Twin, error)
	ListTwins(ctx context.Context, filters TwinFilters) ([]*Twin, error)
	DeleteTwin(ctx context.Context, id string) error
}

// TwinFilters filters for listing twins.
type TwinFilters struct {
	Type     string
	Status   string
	SourceID string
	Limit    int
	Offset   int
}

// NewTwinManager creates a new twin manager.
func NewTwinManager(store TwinStore, logger *log.Logger) *TwinManager {
	return &TwinManager{
		twins:  make(map[string]*Twin),
		store:  store,
		logger: logger,
	}
}

// CreateTwin creates a new digital twin.
func (tm *TwinManager) CreateTwin(ctx context.Context, req CreateTwinRequest) (*Twin, error) {
	twin := &Twin{
		ID:       fmt.Sprintf("twin-%s-%d", req.Type, time.Now().UnixNano()),
		Name:     req.Name,
		Type:     req.Type,
		SourceID: req.SourceID,
		Version:  "1.0.0",
		State: TwinState{
			Status:      "inactive",
			Snapshots:   []StateSnapshot{},
			Metrics:     make(map[string]interface{}),
			CurrentData: make(map[string]interface{}),
		},
		Configuration: req.Configuration,
		Metadata:      req.Metadata,
		CreatedAt:     time.Now(),
		UpdatedAt:     time.Now(),
	}

	if err := tm.store.SaveTwin(ctx, twin); err != nil {
		return nil, fmt.Errorf("failed to save twin: %w", err)
	}

	tm.twins[twin.ID] = twin

	if tm.logger != nil {
		tm.logger.Printf("Created digital twin: %s (%s)", twin.ID, twin.Name)
	}

	return twin, nil
}

// GetTwin retrieves a twin by ID.
func (tm *TwinManager) GetTwin(ctx context.Context, id string) (*Twin, error) {
	// Check cache first
	if twin, exists := tm.twins[id]; exists {
		return twin, nil
	}

	// Load from store
	twin, err := tm.store.GetTwin(ctx, id)
	if err != nil {
		return nil, err
	}

	tm.twins[id] = twin
	return twin, nil
}

// UpdateTwinState updates the state of a twin.
func (tm *TwinManager) UpdateTwinState(ctx context.Context, id string, state TwinState) error {
	twin, err := tm.GetTwin(ctx, id)
	if err != nil {
		return err
	}

	twin.State = state
	twin.UpdatedAt = time.Now()

	if err := tm.store.SaveTwin(ctx, twin); err != nil {
		return fmt.Errorf("failed to update twin state: %w", err)
	}

	return nil
}

// CreateSnapshot creates a state snapshot of the twin.
func (tm *TwinManager) CreateSnapshot(ctx context.Context, id string, description string) (*StateSnapshot, error) {
	twin, err := tm.GetTwin(ctx, id)
	if err != nil {
		return nil, err
	}

	snapshot := StateSnapshot{
		ID:          fmt.Sprintf("snapshot-%s-%d", id, time.Now().UnixNano()),
		Timestamp:   time.Now(),
		State:       make(map[string]interface{}),
		Metrics:     make(map[string]interface{}),
		Description: description,
	}

	// Copy current state
	for k, v := range twin.State.CurrentData {
		snapshot.State[k] = v
	}
	for k, v := range twin.State.Metrics {
		snapshot.Metrics[k] = v
	}

	twin.State.Snapshots = append(twin.State.Snapshots, snapshot)
	twin.State.LastSnapshot = snapshot.Timestamp

	if err := tm.store.SaveTwin(ctx, twin); err != nil {
		return nil, fmt.Errorf("failed to save snapshot: %w", err)
	}

	if tm.logger != nil {
		tm.logger.Printf("Created snapshot for twin %s: %s", id, snapshot.ID)
	}

	return &snapshot, nil
}

// RestoreSnapshot restores a twin to a previous snapshot.
func (tm *TwinManager) RestoreSnapshot(ctx context.Context, id string, snapshotID string) error {
	twin, err := tm.GetTwin(ctx, id)
	if err != nil {
		return err
	}

	var snapshot *StateSnapshot
	for _, snap := range twin.State.Snapshots {
		if snap.ID == snapshotID {
			snapshot = &snap
			break
		}
	}

	if snapshot == nil {
		return fmt.Errorf("snapshot not found: %s", snapshotID)
	}

	// Restore state
	twin.State.CurrentData = make(map[string]interface{})
	for k, v := range snapshot.State {
		twin.State.CurrentData[k] = v
	}

	twin.State.Metrics = make(map[string]interface{})
	for k, v := range snapshot.Metrics {
		twin.State.Metrics[k] = v
	}

	twin.UpdatedAt = time.Now()

	if err := tm.store.SaveTwin(ctx, twin); err != nil {
		return fmt.Errorf("failed to restore snapshot: %w", err)
	}

	if tm.logger != nil {
		tm.logger.Printf("Restored twin %s to snapshot %s", id, snapshotID)
	}

	return nil
}

// ListTwins lists all twins matching filters.
func (tm *TwinManager) ListTwins(ctx context.Context, filters TwinFilters) ([]*Twin, error) {
	return tm.store.ListTwins(ctx, filters)
}

// DeleteTwin deletes a twin.
func (tm *TwinManager) DeleteTwin(ctx context.Context, id string) error {
	if err := tm.store.DeleteTwin(ctx, id); err != nil {
		return err
	}

	delete(tm.twins, id)

	if tm.logger != nil {
		tm.logger.Printf("Deleted twin: %s", id)
	}

	return nil
}

// CreateTwinRequest represents a request to create a twin.
type CreateTwinRequest struct {
	Name          string
	Type          string
	SourceID      string
	Configuration TwinConfiguration
	Metadata      map[string]interface{}
}

// ToJSON converts a twin to JSON.
func (t *Twin) ToJSON() ([]byte, error) {
	return json.MarshalIndent(t, "", "  ")
}

// FromJSON creates a twin from JSON.
func TwinFromJSON(data []byte) (*Twin, error) {
	var twin Twin
	if err := json.Unmarshal(data, &twin); err != nil {
		return nil, fmt.Errorf("failed to unmarshal twin: %w", err)
	}
	return &twin, nil
}

