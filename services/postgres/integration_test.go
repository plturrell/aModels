// +build integration

package postgres_test

import (
	"context"
	"database/sql"
	"testing"
	"time"

	"github.com/google/uuid"
	_ "github.com/jackc/pgx/v5/stdlib"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/plturrell/aModels/services/postgres/internal/db"
	postgresv1 "github.com/plturrell/aModels/services/postgres/pkg/gen/v1"
	"github.com/plturrell/aModels/services/postgres/pkg/models"
	"github.com/plturrell/aModels/services/postgres/pkg/repository"
	"github.com/plturrell/aModels/services/postgres/pkg/service"
)

const (
	testDSN  = "postgres://postgres:postgres@localhost:5432/lang_ops?sslmode=disable"
	grpcAddr = "localhost:50055"
)

func setupTestDB(t *testing.T) (*sql.DB, func()) {
	t.Helper()

	database, err := sql.Open("pgx", testDSN)
	if err != nil {
		t.Fatalf("failed to connect to test database: %v", err)
	}

	if err := database.Ping(); err != nil {
		database.Close()
		t.Fatalf("failed to ping test database: %v", err)
	}

	// Clean up before test
	_, _ = database.Exec("TRUNCATE lang_operations CASCADE")

	cleanup := func() {
		_, _ = database.Exec("TRUNCATE lang_operations CASCADE")
		database.Close()
	}

	return database, cleanup
}

func TestIntegration_RepositoryOperations(t *testing.T) {
	database, cleanup := setupTestDB(t)
	defer cleanup()

	repo := repository.NewOperationsRepository(database)
	ctx := context.Background()

	t.Run("LogAndGetOperation", func(t *testing.T) {
		op := &models.LangOperation{
			ID:          uuid.NewString(),
			LibraryType: "langchain",
			Operation:   "execute_chain",
			Input:       map[string]any{"prompt": "test"},
			Output:      map[string]any{"result": "success"},
			Status:      models.OperationStatusSuccess,
			LatencyMs:   150,
			CreatedAt:   time.Now().UTC(),
			SessionID:   "test-session",
		}

		// Log operation
		err := repo.LogOperation(ctx, op)
		if err != nil {
			t.Fatalf("failed to log operation: %v", err)
		}

		// Get operation back
		retrieved, err := repo.GetOperation(ctx, op.ID)
		if err != nil {
			t.Fatalf("failed to get operation: %v", err)
		}

		if retrieved.ID != op.ID {
			t.Errorf("expected id %s, got %s", op.ID, retrieved.ID)
		}

		if retrieved.LibraryType != op.LibraryType {
			t.Errorf("expected library_type %s, got %s", op.LibraryType, retrieved.LibraryType)
		}

		if retrieved.Status != op.Status {
			t.Errorf("expected status %v, got %v", op.Status, retrieved.Status)
		}
	})

	t.Run("ListOperations", func(t *testing.T) {
		// Log multiple operations
		for i := 0; i < 5; i++ {
			op := &models.LangOperation{
				ID:          uuid.NewString(),
				LibraryType: "langgraph",
				Operation:   "run_graph",
				Input:       map[string]any{},
				Output:      map[string]any{},
				Status:      models.OperationStatusSuccess,
				CreatedAt:   time.Now().UTC().Add(-time.Duration(i) * time.Hour),
			}
			if err := repo.LogOperation(ctx, op); err != nil {
				t.Fatalf("failed to log operation: %v", err)
			}
		}

		// List operations
		filters := repository.ListFilters{
			LibraryType: "langgraph",
			PageSize:    10,
		}

		ops, nextToken, err := repo.ListOperations(ctx, filters)
		if err != nil {
			t.Fatalf("failed to list operations: %v", err)
		}

		if len(ops) < 5 {
			t.Errorf("expected at least 5 operations, got %d", len(ops))
		}

		if nextToken != "" {
			t.Logf("next page token: %s", nextToken)
		}
	})

	t.Run("GetAnalytics", func(t *testing.T) {
		// Get analytics
		filters := repository.AnalyticsFilters{}
		summary, err := repo.GetAnalytics(ctx, filters)
		if err != nil {
			t.Fatalf("failed to get analytics: %v", err)
		}

		if summary.TotalOperations == 0 {
			t.Error("expected non-zero total operations")
		}

		t.Logf("Analytics: %d operations, %.2f%% success rate, %.2fms avg latency",
			summary.TotalOperations,
			summary.SuccessRate*100,
			summary.AverageLatency)
	})

	t.Run("CleanupOperations", func(t *testing.T) {
		// Log an old operation
		oldOp := &models.LangOperation{
			ID:          uuid.NewString(),
			LibraryType: "langchain",
			Operation:   "old_op",
			Input:       map[string]any{},
			Output:      map[string]any{},
			Status:      models.OperationStatusSuccess,
			CreatedAt:   time.Now().UTC().Add(-48 * time.Hour),
		}
		if err := repo.LogOperation(ctx, oldOp); err != nil {
			t.Fatalf("failed to log old operation: %v", err)
		}

		// Cleanup operations older than 24 hours
		cutoff := time.Now().UTC().Add(-24 * time.Hour)
		result, err := repo.CleanupOperations(ctx, cutoff)
		if err != nil {
			t.Fatalf("failed to cleanup: %v", err)
		}

		if result.Deleted < 1 {
			t.Errorf("expected at least 1 deletion, got %d", result.Deleted)
		}

		t.Logf("Cleaned up %d operations", result.Deleted)
	})
}

func TestIntegration_ServiceLayer(t *testing.T) {
	database, cleanup := setupTestDB(t)
	defer cleanup()

	repo := repository.NewOperationsRepository(database)
	svc := service.NewLangService(repo, "test-1.0.0")
	ctx := context.Background()

	t.Run("HealthCheck", func(t *testing.T) {
		resp, err := svc.HealthCheck(ctx, &postgresv1.HealthCheckRequest{})
		if err != nil {
			t.Fatalf("health check failed: %v", err)
		}

		if resp.Status != "SERVING" {
			t.Errorf("expected SERVING status, got %s", resp.Status)
		}

		if resp.Version != "test-1.0.0" {
			t.Errorf("expected version test-1.0.0, got %s", resp.Version)
		}
	})

	t.Run("LogAndGetOperation", func(t *testing.T) {
		// Log operation via service
		logReq := &postgresv1.LogLangOperationRequest{
			Operation: &postgresv1.LangOperation{
				LibraryType: "llamaindex",
				Operation:   "query_engine",
				Status:      postgresv1.OperationStatus_OPERATION_STATUS_SUCCESS,
				LatencyMs:   200,
			},
		}

		logResp, err := svc.LogLangOperation(ctx, logReq)
		if err != nil {
			t.Fatalf("failed to log operation: %v", err)
		}

		opID := logResp.Operation.Id
		if opID == "" {
			t.Fatal("expected operation id in response")
		}

		// Get operation via service
		getReq := &postgresv1.GetLangOperationRequest{Id: opID}
		getResp, err := svc.GetLangOperation(ctx, getReq)
		if err != nil {
			t.Fatalf("failed to get operation: %v", err)
		}

		if getResp.Operation.LibraryType != "llamaindex" {
			t.Errorf("expected library_type llamaindex, got %s", getResp.Operation.LibraryType)
		}
	})
}

func TestIntegration_GRPCEndToEnd(t *testing.T) {
	// Skip if gRPC server not running
	conn, err := grpc.Dial(grpcAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Skip("gRPC server not available, skipping end-to-end test")
	}
	defer conn.Close()

	client := postgresv1.NewPostgresLangServiceClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	t.Run("HealthCheck", func(t *testing.T) {
		resp, err := client.HealthCheck(ctx, &postgresv1.HealthCheckRequest{})
		if err != nil {
			t.Fatalf("health check failed: %v", err)
		}

		t.Logf("Health: %s, Version: %s", resp.Status, resp.Version)
	})

	t.Run("LogOperation", func(t *testing.T) {
		req := &postgresv1.LogLangOperationRequest{
			Operation: &postgresv1.LangOperation{
				LibraryType: "crewai",
				Operation:   "run_crew",
				Status:      postgresv1.OperationStatus_OPERATION_STATUS_SUCCESS,
				LatencyMs:   300,
			},
		}

		resp, err := client.LogLangOperation(ctx, req)
		if err != nil {
			t.Fatalf("failed to log operation: %v", err)
		}

		if resp.Operation.Id == "" {
			t.Error("expected operation id in response")
		}

		t.Logf("Logged operation: %s", resp.Operation.Id)
	})

	t.Run("ListOperations", func(t *testing.T) {
		req := &postgresv1.ListLangOperationsRequest{
			PageSize: 10,
		}

		resp, err := client.ListLangOperations(ctx, req)
		if err != nil {
			t.Fatalf("failed to list operations: %v", err)
		}

		t.Logf("Found %d operations", len(resp.Operations))
	})

	t.Run("GetAnalytics", func(t *testing.T) {
		req := &postgresv1.AnalyticsRequest{}

		resp, err := client.GetAnalytics(ctx, req)
		if err != nil {
			t.Fatalf("failed to get analytics: %v", err)
		}

		t.Logf("Analytics: %d operations, %.2f%% success rate",
			resp.TotalOperations,
			resp.SuccessRate*100)
	})
}

func TestIntegration_Concurrency(t *testing.T) {
	database, cleanup := setupTestDB(t)
	defer cleanup()

	repo := repository.NewOperationsRepository(database)
	ctx := context.Background()

	// Test concurrent writes
	const numGoroutines = 10
	const opsPerGoroutine = 5

	errCh := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(goroutineID int) {
			for j := 0; j < opsPerGoroutine; j++ {
				op := &models.LangOperation{
					ID:          uuid.NewString(),
					LibraryType: "langchain",
					Operation:   "concurrent_test",
					Input:       map[string]any{"goroutine": goroutineID, "operation": j},
					Output:      map[string]any{},
					Status:      models.OperationStatusSuccess,
					CreatedAt:   time.Now().UTC(),
				}

				if err := repo.LogOperation(ctx, op); err != nil {
					errCh <- err
					return
				}
			}
			errCh <- nil
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < numGoroutines; i++ {
		if err := <-errCh; err != nil {
			t.Errorf("goroutine error: %v", err)
		}
	}

	// Verify all operations were logged
	filters := repository.ListFilters{
		PageSize: 100,
	}
	ops, _, err := repo.ListOperations(ctx, filters)
	if err != nil {
		t.Fatalf("failed to list operations: %v", err)
	}

	expected := numGoroutines * opsPerGoroutine
	if len(ops) < expected {
		t.Errorf("expected at least %d operations, got %d", expected, len(ops))
	}

	t.Logf("Successfully logged %d concurrent operations", len(ops))
}
