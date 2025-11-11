package service

import (
	"context"
	"database/sql"
	"errors"
	"testing"
	"time"

	"github.com/plturrell/aModels/services/postgres/pkg/models"
	"github.com/plturrell/aModels/services/postgres/pkg/repository"
	postgresv1 "github.com/plturrell/aModels/services/postgres/pkg/gen/v1"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// mockRepository implements a mock OperationsRepository for testing
type mockRepository struct {
	pingFn           func(ctx context.Context) error
	logOperationFn   func(ctx context.Context, op *models.LangOperation) error
	getOperationFn   func(ctx context.Context, id string) (*models.LangOperation, error)
	listOperationsFn func(ctx context.Context, filters repository.ListFilters) ([]*models.LangOperation, string, error)
	getAnalyticsFn   func(ctx context.Context, filters repository.AnalyticsFilters) (*models.AnalyticsSummary, error)
	cleanupFn        func(ctx context.Context, olderThan time.Time) (*models.CleanupResult, error)
}

func (m *mockRepository) Ping(ctx context.Context) error {
	if m.pingFn != nil {
		return m.pingFn(ctx)
	}
	return nil
}

func (m *mockRepository) LogOperation(ctx context.Context, op *models.LangOperation) error {
	if m.logOperationFn != nil {
		return m.logOperationFn(ctx, op)
	}
	return nil
}

func (m *mockRepository) GetOperation(ctx context.Context, id string) (*models.LangOperation, error) {
	if m.getOperationFn != nil {
		return m.getOperationFn(ctx, id)
	}
	return nil, errors.New("not implemented")
}

func (m *mockRepository) ListOperations(ctx context.Context, filters repository.ListFilters) ([]*models.LangOperation, string, error) {
	if m.listOperationsFn != nil {
		return m.listOperationsFn(ctx, filters)
	}
	return nil, "", nil
}

func (m *mockRepository) GetAnalytics(ctx context.Context, filters repository.AnalyticsFilters) (*models.AnalyticsSummary, error) {
	if m.getAnalyticsFn != nil {
		return m.getAnalyticsFn(ctx, filters)
	}
	return nil, errors.New("not implemented")
}

func (m *mockRepository) CleanupOperations(ctx context.Context, olderThan time.Time) (*models.CleanupResult, error) {
	if m.cleanupFn != nil {
		return m.cleanupFn(ctx, olderThan)
	}
	return nil, errors.New("not implemented")
}

func TestHealthCheck_Success(t *testing.T) {
	mock := &mockRepository{
		pingFn: func(ctx context.Context) error {
			return nil
		},
	}

	service := NewLangService(mock, "1.0.0")
	resp, err := service.HealthCheck(context.Background(), &postgresv1.HealthCheckRequest{})

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Status != "SERVING" {
		t.Errorf("expected status SERVING, got %s", resp.Status)
	}

	if resp.Version != "1.0.0" {
		t.Errorf("expected version 1.0.0, got %s", resp.Version)
	}
}

func TestHealthCheck_DatabaseError(t *testing.T) {
	mock := &mockRepository{
		pingFn: func(ctx context.Context) error {
			return errors.New("connection refused")
		},
	}

	service := NewLangService(mock, "1.0.0")
	_, err := service.HealthCheck(context.Background(), &postgresv1.HealthCheckRequest{})

	if err == nil {
		t.Fatal("expected error, got nil")
	}

	if err.Error() != "database not reachable: connection refused" {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestLogLangOperation_Success(t *testing.T) {
	var capturedOp *models.LangOperation

	mock := &mockRepository{
		logOperationFn: func(ctx context.Context, op *models.LangOperation) error {
			capturedOp = op
			op.ID = "generated-id"
			return nil
		},
	}

	service := NewLangService(mock, "1.0.0")

	input, _ := structpb.NewStruct(map[string]interface{}{"key": "value"})
	output, _ := structpb.NewStruct(map[string]interface{}{"result": "success"})

	req := &postgresv1.LogLangOperationRequest{
		Operation: &postgresv1.LangOperation{
			LibraryType: "langchain",
			Operation:   "execute",
			Input:       input,
			Output:      output,
			Status:      postgresv1.OperationStatus_OPERATION_STATUS_SUCCESS,
			LatencyMs:   150,
		},
	}

	resp, err := service.LogLangOperation(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if capturedOp == nil {
		t.Fatal("operation was not logged")
	}

	if capturedOp.LibraryType != "langchain" {
		t.Errorf("expected library_type langchain, got %s", capturedOp.LibraryType)
	}

	if capturedOp.LatencyMs != 150 {
		t.Errorf("expected latency 150ms, got %d", capturedOp.LatencyMs)
	}

	if resp.Operation == nil {
		t.Fatal("expected operation in response")
	}

	if resp.Operation.Id != "generated-id" {
		t.Errorf("expected id generated-id, got %s", resp.Operation.Id)
	}
}

func TestLogLangOperation_NilRequest(t *testing.T) {
	mock := &mockRepository{}
	service := NewLangService(mock, "1.0.0")

	_, err := service.LogLangOperation(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error for nil request")
	}

	if err.Error() != "operation payload is required" {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestLogLangOperation_NilOperation(t *testing.T) {
	mock := &mockRepository{}
	service := NewLangService(mock, "1.0.0")

	req := &postgresv1.LogLangOperationRequest{Operation: nil}

	_, err := service.LogLangOperation(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for nil operation")
	}

	if err.Error() != "operation payload is required" {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestGetLangOperation_Success(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)

	mock := &mockRepository{
		getOperationFn: func(ctx context.Context, id string) (*models.LangOperation, error) {
			if id != "op-123" {
				return nil, sql.ErrNoRows
			}
			return &models.LangOperation{
				ID:          "op-123",
				LibraryType: "langgraph",
				Operation:   "run_graph",
				Input:       map[string]any{"param": "value"},
				Output:      map[string]any{"result": "done"},
				Status:      models.OperationStatusSuccess,
				LatencyMs:   250,
				CreatedAt:   now,
			}, nil
		},
	}

	service := NewLangService(mock, "1.0.0")

	req := &postgresv1.GetLangOperationRequest{Id: "op-123"}
	resp, err := service.GetLangOperation(context.Background(), req)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Operation.Id != "op-123" {
		t.Errorf("expected id op-123, got %s", resp.Operation.Id)
	}

	if resp.Operation.LibraryType != "langgraph" {
		t.Errorf("expected library_type langgraph, got %s", resp.Operation.LibraryType)
	}

	if resp.Operation.Status != postgresv1.OperationStatus_OPERATION_STATUS_SUCCESS {
		t.Errorf("expected status SUCCESS, got %v", resp.Operation.Status)
	}
}

func TestGetLangOperation_EmptyID(t *testing.T) {
	mock := &mockRepository{}
	service := NewLangService(mock, "1.0.0")

	req := &postgresv1.GetLangOperationRequest{Id: ""}
	_, err := service.GetLangOperation(context.Background(), req)

	if err == nil {
		t.Fatal("expected error for empty id")
	}

	if err.Error() != "id is required" {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestGetLangOperation_NotFound(t *testing.T) {
	mock := &mockRepository{
		getOperationFn: func(ctx context.Context, id string) (*models.LangOperation, error) {
			return nil, sql.ErrNoRows
		},
	}

	service := NewLangService(mock, "1.0.0")

	req := &postgresv1.GetLangOperationRequest{Id: "nonexistent"}
	_, err := service.GetLangOperation(context.Background(), req)

	if err == nil {
		t.Fatal("expected error for not found")
	}
}

func TestListLangOperations_Success(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)

	mock := &mockRepository{
		listOperationsFn: func(ctx context.Context, filters repository.ListFilters) ([]*models.LangOperation, string, error) {
			ops := []*models.LangOperation{
				{
					ID:          "op-1",
					LibraryType: "langchain",
					Operation:   "execute",
					Input:       map[string]any{},
					Output:      map[string]any{},
					Status:      models.OperationStatusSuccess,
					CreatedAt:   now,
				},
				{
					ID:          "op-2",
					LibraryType: "langchain",
					Operation:   "execute",
					Input:       map[string]any{},
					Output:      map[string]any{},
					Status:      models.OperationStatusRunning,
					CreatedAt:   now.Add(-time.Hour),
				},
			}
			return ops, "next-token", nil
		},
	}

	service := NewLangService(mock, "1.0.0")

	req := &postgresv1.ListLangOperationsRequest{
		LibraryType: "langchain",
		PageSize:    10,
	}

	resp, err := service.ListLangOperations(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(resp.Operations) != 2 {
		t.Errorf("expected 2 operations, got %d", len(resp.Operations))
	}

	if resp.NextPageToken != "next-token" {
		t.Errorf("expected next token, got %s", resp.NextPageToken)
	}

	if resp.Operations[0].Id != "op-1" {
		t.Errorf("expected first operation id op-1, got %s", resp.Operations[0].Id)
	}
}

func TestListLangOperations_WithFilters(t *testing.T) {
	var capturedFilters repository.ListFilters

	mock := &mockRepository{
		listOperationsFn: func(ctx context.Context, filters repository.ListFilters) ([]*models.LangOperation, string, error) {
			capturedFilters = filters
			return []*models.LangOperation{}, "", nil
		},
	}

	service := NewLangService(mock, "1.0.0")

	createdAfter := time.Now().Add(-24 * time.Hour)
	req := &postgresv1.ListLangOperationsRequest{
		LibraryType: "langgraph",
		SessionId:   "session-123",
		Status:      postgresv1.OperationStatus_OPERATION_STATUS_SUCCESS,
		PageSize:    25,
		PageToken:   "token-abc",
		CreatedAfter: timestamppb.New(createdAfter),
	}

	_, err := service.ListLangOperations(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if capturedFilters.LibraryType != "langgraph" {
		t.Errorf("expected library_type langgraph, got %s", capturedFilters.LibraryType)
	}

	if capturedFilters.SessionID != "session-123" {
		t.Errorf("expected session_id session-123, got %s", capturedFilters.SessionID)
	}

	if capturedFilters.Status != models.OperationStatusSuccess {
		t.Errorf("expected status success, got %v", capturedFilters.Status)
	}

	if capturedFilters.PageSize != 25 {
		t.Errorf("expected page_size 25, got %d", capturedFilters.PageSize)
	}

	if capturedFilters.PageToken != "token-abc" {
		t.Errorf("expected page_token token-abc, got %s", capturedFilters.PageToken)
	}

	if capturedFilters.CreatedAfter == nil || !capturedFilters.CreatedAfter.Equal(createdAfter) {
		t.Errorf("created_after filter not set correctly")
	}
}

func TestGetAnalytics_Success(t *testing.T) {
	now := time.Now().UTC()

	mock := &mockRepository{
		getAnalyticsFn: func(ctx context.Context, filters repository.AnalyticsFilters) (*models.AnalyticsSummary, error) {
			return &models.AnalyticsSummary{
				TotalOperations: 100,
				SuccessRate:     0.95,
				AverageLatency:  150.5,
				ErrorBreakdown:  map[string]int64{"langchain": 5},
				LibraryStats: []models.LibraryStats{
					{
						LibraryType:     "langchain",
						TotalOperations: 100,
						SuccessRate:     0.95,
						AverageLatency:  150.5,
						ErrorCount:      5,
					},
				},
				PerformanceTrend: []models.PerformanceTrend{
					{
						Date:            now,
						LibraryType:     "langchain",
						Operations:      50,
						SuccessRate:     0.96,
						AverageLatency:  145.0,
					},
				},
				GeneratedAt: now,
			}, nil
		},
	}

	service := NewLangService(mock, "1.0.0")

	req := &postgresv1.AnalyticsRequest{
		LibraryType: "langchain",
	}

	resp, err := service.GetAnalytics(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.TotalOperations != 100 {
		t.Errorf("expected total_operations 100, got %d", resp.TotalOperations)
	}

	if resp.SuccessRate != 0.95 {
		t.Errorf("expected success_rate 0.95, got %f", resp.SuccessRate)
	}

	if resp.AverageLatencyMs != 150.5 {
		t.Errorf("expected average_latency 150.5, got %f", resp.AverageLatencyMs)
	}

	if len(resp.LibraryStats) != 1 {
		t.Errorf("expected 1 library stat, got %d", len(resp.LibraryStats))
	}

	if len(resp.PerformanceTrends) != 1 {
		t.Errorf("expected 1 performance trend, got %d", len(resp.PerformanceTrends))
	}

	if resp.ErrorBreakdown["langchain"] != 5 {
		t.Errorf("expected error breakdown for langchain = 5, got %d", resp.ErrorBreakdown["langchain"])
	}
}

func TestCleanupOperations_Success(t *testing.T) {
	mock := &mockRepository{
		cleanupFn: func(ctx context.Context, olderThan time.Time) (*models.CleanupResult, error) {
			return &models.CleanupResult{Deleted: 42}, nil
		},
	}

	service := NewLangService(mock, "1.0.0")

	cutoff := time.Now().Add(-30 * 24 * time.Hour)
	req := &postgresv1.CleanupRequest{
		OlderThan: timestamppb.New(cutoff),
	}

	resp, err := service.CleanupOperations(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Deleted != 42 {
		t.Errorf("expected 42 deleted, got %d", resp.Deleted)
	}
}

func TestCleanupOperations_MissingTimestamp(t *testing.T) {
	mock := &mockRepository{}
	service := NewLangService(mock, "1.0.0")

	req := &postgresv1.CleanupRequest{OlderThan: nil}

	_, err := service.CleanupOperations(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for missing timestamp")
	}

	if err.Error() != "older_than timestamp required" {
		t.Errorf("unexpected error message: %v", err)
	}
}
