package repository

import (
	"context"
	"regexp"
	"testing"
	"time"

	sqlmock "github.com/DATA-DOG/go-sqlmock"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres/pkg/models"
)

func newTestRepo(t *testing.T) (*OperationsRepository, sqlmock.Sqlmock, func()) {
	t.Helper()

	db, mock, err := sqlmock.New(
		sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp),
		sqlmock.MonitorPingsOption(true),
	)
	if err != nil {
		t.Fatalf("failed to create sqlmock: %v", err)
	}

	repo := NewOperationsRepository(db)

	cleanup := func() {
		db.Close()
	}

	return repo, mock, cleanup
}

func TestLogOperation(t *testing.T) {
	repo, mock, cleanup := newTestRepo(t)
	defer cleanup()

	now := time.Now().UTC().Truncate(time.Second)

	op := &models.LangOperation{
		ID:           "op-123",
		LibraryType:  "langchain",
		Operation:    "execute_chain",
		Input:        map[string]any{"prompt": "hello"},
		Output:       map[string]any{"result": "world"},
		Status:       models.OperationStatusSuccess,
		Error:        "",
		LatencyMs:    1200,
		CreatedAt:    now,
		SessionID:    "session",
		UserIDHash:   "user",
		PrivacyLevel: "medium",
	}

	mock.ExpectExec(regexp.QuoteMeta(`INSERT INTO lang_operations`)).
		WithArgs(
			op.ID,
			op.LibraryType,
			op.Operation,
			sqlmock.AnyArg(),
			sqlmock.AnyArg(),
			"success",
			op.Error,
			op.LatencyMs,
			op.CreatedAt,
			sqlmock.AnyArg(),
			sqlmock.AnyArg(),
			sqlmock.AnyArg(),
			sqlmock.AnyArg(),
		).
		WillReturnResult(sqlmock.NewResult(1, 1))

	if err := repo.LogOperation(context.Background(), op); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestGetOperation(t *testing.T) {
	repo, mock, cleanup := newTestRepo(t)
	defer cleanup()

	createdAt := time.Now().UTC().Truncate(time.Second)

	rows := sqlmock.NewRows([]string{
		"id", "library_type", "operation", "input", "output", "status", "error",
		"latency_ms", "created_at", "completed_at", "session_id", "user_id_hash", "privacy_level",
	}).AddRow(
		"op-1",
		"langgraph",
		"execute_graph",
		[]byte(`{"foo":"bar"}`),
		[]byte(`{"result":"ok"}`),
		"success",
		"",
		320,
		createdAt,
		createdAt,
		"session",
		"user",
		"high",
	)

	mock.ExpectQuery(regexp.QuoteMeta(`SELECT id, library_type, operation, input, output, status, error,
               latency_ms, created_at, completed_at, session_id, user_id_hash, privacy_level
          FROM lang_operations
         WHERE id = $1`)).
		WithArgs("op-1").
		WillReturnRows(rows)

	op, err := repo.GetOperation(context.Background(), "op-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if op.ID != "op-1" || op.LibraryType != "langgraph" {
		t.Fatalf("unexpected operation returned: %+v", op)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestListOperationsPagination(t *testing.T) {
	repo, mock, cleanup := newTestRepo(t)
	defer cleanup()

	createdNow := time.Now().UTC().Truncate(time.Second)
	createdEarlier := createdNow.Add(-time.Hour)

	rows := sqlmock.NewRows([]string{
		"id", "library_type", "operation", "input", "output", "status", "error",
		"latency_ms", "created_at", "completed_at", "session_id", "user_id_hash", "privacy_level",
	}).
		AddRow("op-1", "langchain", "op", []byte(`{"a":1}`), []byte(`{"b":2}`), "running", "", 10, createdNow, nil, "s1", "u1", "low").
		AddRow("op-2", "langchain", "op", []byte(`{"a":1}`), []byte(`{"b":2}`), "success", "", 20, createdEarlier, nil, "s1", "u1", "low")

	mock.ExpectQuery(regexp.QuoteMeta(`SELECT id, library_type, operation, input, output, status, error,
               latency_ms, created_at, completed_at, session_id, user_id_hash, privacy_level
          FROM lang_operations WHERE session_id = $1 ORDER BY created_at DESC LIMIT 2`)).
		WithArgs("s1").
		WillReturnRows(rows)

	ops, token, err := repo.ListOperations(context.Background(), ListFilters{SessionID: "s1", PageSize: 1})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(ops) != 1 {
		t.Fatalf("expected 1 operation, got %d", len(ops))
	}

	if token == "" {
		t.Fatalf("expected next page token")
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestGetAnalytics(t *testing.T) {
	repo, mock, cleanup := newTestRepo(t)
	defer cleanup()

	statsQuery := `SELECT library_type,\s+COUNT\(\*\) AS total_operations,\s+COALESCE\(AVG\(latency_ms\), 0\) AS average_latency,\s+SUM\(CASE WHEN status = 'success' THEN 1 ELSE 0 END\) AS success_count,\s+SUM\(CASE WHEN status = 'error' THEN 1 ELSE 0 END\) AS error_count\s+FROM lang_operations\s+GROUP BY library_type`
	trendQuery := `SELECT date_trunc\('day', created_at\) AS date,\s+library_type,\s+COUNT\(\*\) AS operations,\s+AVG\(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END\) AS success_rate,\s+COALESCE\(AVG\(latency_ms\), 0\) AS average_latency\s+FROM lang_operations\s+GROUP BY date, library_type\s+ORDER BY date DESC\s+LIMIT 90`

	statsRows := sqlmock.NewRows([]string{"library_type", "total_operations", "average_latency", "success_count", "error_count"}).
		AddRow("langchain", int64(10), float64(100), int64(8), int64(2))

	trendRows := sqlmock.NewRows([]string{"date", "library_type", "operations", "success_rate", "average_latency"}).
		AddRow(time.Now(), "langchain", int64(5), float64(0.8), float64(90))

	mock.ExpectQuery(statsQuery).WillReturnRows(statsRows)
	mock.ExpectQuery(trendQuery).WillReturnRows(trendRows)

	summary, err := repo.GetAnalytics(context.Background(), AnalyticsFilters{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if summary.TotalOperations != 10 {
		t.Fatalf("unexpected total operations: %d", summary.TotalOperations)
	}

	if len(summary.LibraryStats) != 1 {
		t.Fatalf("expected 1 library stat, got %d", len(summary.LibraryStats))
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestCleanupOperations(t *testing.T) {
	repo, mock, cleanup := newTestRepo(t)
	defer cleanup()

	cutoff := time.Now().Add(-time.Hour)

	mock.ExpectExec(regexp.QuoteMeta(`DELETE FROM lang_operations
         WHERE created_at < $1`)).
		WithArgs(sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(0, 5))

	result, err := repo.CleanupOperations(context.Background(), cutoff)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Deleted != 5 {
		t.Fatalf("expected 5 deletions, got %d", result.Deleted)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestPing(t *testing.T) {
	repo, mock, cleanup := newTestRepo(t)
	defer cleanup()

	mock.ExpectPing()

	if err := repo.Ping(context.Background()); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}
