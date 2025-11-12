package migrations

import (
	"context"
	"fmt"
	"log"
	"sort"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// Migration represents a single Neo4j schema migration.
type Migration struct {
	Version     int       // Migration version number (e.g., 1, 2, 3)
	Name        string    // Human-readable name (e.g., "add_composite_indexes")
	Description string    // Description of what the migration does
	Up          string    // Cypher query to apply the migration
	Down        string    // Cypher query to rollback the migration (optional)
	AppliedAt   time.Time // When the migration was applied
}

// MigrationRecord represents a migration record stored in Neo4j.
type MigrationRecord struct {
	Version     int       `json:"version"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	AppliedAt   time.Time `json:"applied_at"`
	Success     bool      `json:"success"`
	ErrorMsg    string    `json:"error_msg,omitempty"`
}

// Migrator handles Neo4j schema migrations.
type Migrator struct {
	driver     neo4j.DriverWithContext
	logger     *log.Logger
	migrations []Migration
}

// NewMigrator creates a new Neo4j migrator.
func NewMigrator(driver neo4j.DriverWithContext, logger *log.Logger) *Migrator {
	return &Migrator{
		driver:     driver,
		logger:     logger,
		migrations: []Migration{},
	}
}

// RegisterMigration registers a migration to be tracked.
func (m *Migrator) RegisterMigration(migration Migration) {
	m.migrations = append(m.migrations, migration)
}

// RegisterMigrations registers multiple migrations.
func (m *Migrator) RegisterMigrations(migrations []Migration) {
	m.migrations = append(m.migrations, migrations...)
}

// initMigrationTracking creates the migration tracking infrastructure in Neo4j.
func (m *Migrator) initMigrationTracking(ctx context.Context) error {
	session := m.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	// Create constraint and index for migration tracking
	queries := []string{
		// Create migration node constraint
		"CREATE CONSTRAINT migration_version_unique IF NOT EXISTS FOR (m:_Migration) REQUIRE m.version IS UNIQUE",
		
		// Create index for querying migrations
		"CREATE INDEX migration_applied_at IF NOT EXISTS FOR (m:_Migration) ON (m.applied_at)",
	}

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		for _, query := range queries {
			if _, err := tx.Run(ctx, query, nil); err != nil {
				// Ignore "already exists" errors
				if m.logger != nil {
					m.logger.Printf("Migration tracking init: %v", err)
				}
			}
		}
		return nil, nil
	})

	return err
}

// GetAppliedVersions returns a list of applied migration versions.
func (m *Migrator) GetAppliedVersions(ctx context.Context) ([]int, error) {
	session := m.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	query := `
		MATCH (m:_Migration)
		WHERE m.success = true
		RETURN m.version as version
		ORDER BY m.version
	`

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, nil)
		if err != nil {
			return nil, err
		}

		var versions []int
		for result.Next(ctx) {
			record := result.Record()
			if version, ok := record.Get("version"); ok {
				versions = append(versions, int(version.(int64)))
			}
		}

		return versions, result.Err()
	})

	if err != nil {
		return nil, err
	}

	return result.([]int), nil
}

// GetMigrationHistory returns the full migration history.
func (m *Migrator) GetMigrationHistory(ctx context.Context) ([]MigrationRecord, error) {
	session := m.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	query := `
		MATCH (m:_Migration)
		RETURN m.version as version,
		       m.name as name,
		       m.description as description,
		       m.applied_at as applied_at,
		       m.success as success,
		       m.error_msg as error_msg
		ORDER BY m.version
	`

	result, err := session.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		result, err := tx.Run(ctx, query, nil)
		if err != nil {
			return nil, err
		}

		var history []MigrationRecord
		for result.Next(ctx) {
			record := result.Record()
			
			mr := MigrationRecord{}
			if version, ok := record.Get("version"); ok {
				mr.Version = int(version.(int64))
			}
			if name, ok := record.Get("name"); ok {
				mr.Name = name.(string)
			}
			if desc, ok := record.Get("description"); ok {
				mr.Description = desc.(string)
			}
			if appliedAt, ok := record.Get("applied_at"); ok {
				if t, ok := appliedAt.(time.Time); ok {
					mr.AppliedAt = t
				}
			}
			if success, ok := record.Get("success"); ok {
				mr.Success = success.(bool)
			}
			if errMsg, ok := record.Get("error_msg"); ok && errMsg != nil {
				mr.ErrorMsg = errMsg.(string)
			}
			
			history = append(history, mr)
		}

		return history, result.Err()
	})

	if err != nil {
		return nil, err
	}

	return result.([]MigrationRecord), nil
}

// recordMigration records a migration attempt in Neo4j.
func (m *Migrator) recordMigration(ctx context.Context, migration Migration, success bool, errorMsg string) error {
	session := m.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	query := `
		MERGE (m:_Migration {version: $version})
		SET m.name = $name,
		    m.description = $description,
		    m.applied_at = datetime($applied_at),
		    m.success = $success,
		    m.error_msg = $error_msg
	`

	params := map[string]interface{}{
		"version":     migration.Version,
		"name":        migration.Name,
		"description": migration.Description,
		"applied_at":  time.Now().Format(time.RFC3339),
		"success":     success,
		"error_msg":   errorMsg,
	}

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		_, err := tx.Run(ctx, query, params)
		return nil, err
	})

	return err
}

// Migrate applies all pending migrations.
func (m *Migrator) Migrate(ctx context.Context) error {
	// Initialize migration tracking
	if err := m.initMigrationTracking(ctx); err != nil {
		return fmt.Errorf("failed to initialize migration tracking: %w", err)
	}

	// Get applied versions
	appliedVersions, err := m.GetAppliedVersions(ctx)
	if err != nil {
		return fmt.Errorf("failed to get applied versions: %w", err)
	}

	appliedMap := make(map[int]bool)
	for _, v := range appliedVersions {
		appliedMap[v] = true
	}

	// Sort migrations by version
	sort.Slice(m.migrations, func(i, j int) bool {
		return m.migrations[i].Version < m.migrations[j].Version
	})

	// Apply pending migrations
	for _, migration := range m.migrations {
		if appliedMap[migration.Version] {
			if m.logger != nil {
				m.logger.Printf("Skipping already applied migration: v%d - %s", migration.Version, migration.Name)
			}
			continue
		}

		if m.logger != nil {
			m.logger.Printf("Applying migration: v%d - %s", migration.Version, migration.Name)
		}

		if err := m.applyMigration(ctx, migration); err != nil {
			errMsg := fmt.Sprintf("failed to apply migration v%d: %v", migration.Version, err)
			m.recordMigration(ctx, migration, false, errMsg)
			return fmt.Errorf("%s", errMsg)
		}

		// Record successful migration
		if err := m.recordMigration(ctx, migration, true, ""); err != nil {
			return fmt.Errorf("failed to record migration v%d: %w", migration.Version, err)
		}

		if m.logger != nil {
			m.logger.Printf("Successfully applied migration: v%d - %s", migration.Version, migration.Name)
		}
	}

	return nil
}

// applyMigration applies a single migration.
func (m *Migrator) applyMigration(ctx context.Context, migration Migration) error {
	session := m.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		_, err := tx.Run(ctx, migration.Up, nil)
		return nil, err
	})

	return err
}

// Rollback rolls back the last N migrations.
func (m *Migrator) Rollback(ctx context.Context, steps int) error {
	if steps <= 0 {
		return fmt.Errorf("steps must be greater than 0")
	}

	// Get applied versions
	appliedVersions, err := m.GetAppliedVersions(ctx)
	if err != nil {
		return fmt.Errorf("failed to get applied versions: %w", err)
	}

	if len(appliedVersions) == 0 {
		return fmt.Errorf("no migrations to rollback")
	}

	// Sort in descending order for rollback
	sort.Sort(sort.Reverse(sort.IntSlice(appliedVersions)))

	// Rollback last N migrations
	rollbackCount := steps
	if rollbackCount > len(appliedVersions) {
		rollbackCount = len(appliedVersions)
	}

	for i := 0; i < rollbackCount; i++ {
		version := appliedVersions[i]
		
		// Find the migration
		var migration *Migration
		for _, m := range m.migrations {
			if m.Version == version {
				migration = &m
				break
			}
		}

		if migration == nil {
			return fmt.Errorf("migration v%d not found in registered migrations", version)
		}

		if migration.Down == "" {
			return fmt.Errorf("migration v%d has no rollback query", version)
		}

		if m.logger != nil {
			m.logger.Printf("Rolling back migration: v%d - %s", version, migration.Name)
		}

		if err := m.rollbackMigration(ctx, *migration); err != nil {
			return fmt.Errorf("failed to rollback migration v%d: %w", version, err)
		}

		// Remove migration record
		if err := m.removeMigrationRecord(ctx, version); err != nil {
			return fmt.Errorf("failed to remove migration record v%d: %w", version, err)
		}

		if m.logger != nil {
			m.logger.Printf("Successfully rolled back migration: v%d - %s", version, migration.Name)
		}
	}

	return nil
}

// rollbackMigration rolls back a single migration.
func (m *Migrator) rollbackMigration(ctx context.Context, migration Migration) error {
	session := m.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		_, err := tx.Run(ctx, migration.Down, nil)
		return nil, err
	})

	return err
}

// removeMigrationRecord removes a migration record from Neo4j.
func (m *Migrator) removeMigrationRecord(ctx context.Context, version int) error {
	session := m.driver.NewSession(ctx, neo4j.SessionConfig{DatabaseName: "neo4j"})
	defer session.Close(ctx)

	query := `MATCH (m:_Migration {version: $version}) DELETE m`
	params := map[string]interface{}{"version": version}

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		_, err := tx.Run(ctx, query, params)
		return nil, err
	})

	return err
}

// Status returns the current migration status.
func (m *Migrator) Status(ctx context.Context) (*MigrationStatus, error) {
	appliedVersions, err := m.GetAppliedVersions(ctx)
	if err != nil {
		return nil, err
	}

	appliedMap := make(map[int]bool)
	for _, v := range appliedVersions {
		appliedMap[v] = true
	}

	// Sort migrations by version
	sort.Slice(m.migrations, func(i, j int) bool {
		return m.migrations[i].Version < m.migrations[j].Version
	})

	status := &MigrationStatus{
		TotalMigrations:   len(m.migrations),
		AppliedMigrations: len(appliedVersions),
		PendingMigrations: 0,
		Migrations:        []MigrationStatusItem{},
	}

	for _, migration := range m.migrations {
		applied := appliedMap[migration.Version]
		if !applied {
			status.PendingMigrations++
		}

		status.Migrations = append(status.Migrations, MigrationStatusItem{
			Version:     migration.Version,
			Name:        migration.Name,
			Description: migration.Description,
			Applied:     applied,
		})
	}

	return status, nil
}

// MigrationStatus represents the overall migration status.
type MigrationStatus struct {
	TotalMigrations   int                    `json:"total_migrations"`
	AppliedMigrations int                    `json:"applied_migrations"`
	PendingMigrations int                    `json:"pending_migrations"`
	Migrations        []MigrationStatusItem  `json:"migrations"`
}

// MigrationStatusItem represents the status of a single migration.
type MigrationStatusItem struct {
	Version     int    `json:"version"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Applied     bool   `json:"applied"`
}
