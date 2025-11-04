package repository

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/plturrell/aModels/services/postgres/pkg/models"
)

// OperationsRepository persists lang operations and related metadata in Postgres.
type OperationsRepository struct {
	db *sql.DB
}

// NewOperationsRepository constructs an OperationsRepository.
func NewOperationsRepository(db *sql.DB) *OperationsRepository {
	return &OperationsRepository{db: db}
}

const (
	statusRunning     = "running"
	statusSuccess     = "success"
	statusError       = "error"
	statusUnspecified = "unspecified"
)

func toStatusString(status models.OperationStatus) string {
	switch status {
	case models.OperationStatusRunning:
		return statusRunning
	case models.OperationStatusSuccess:
		return statusSuccess
	case models.OperationStatusError:
		return statusError
	default:
		return statusUnspecified
	}
}

func fromStatusString(status string) models.OperationStatus {
	switch status {
	case statusRunning:
		return models.OperationStatusRunning
	case statusSuccess:
		return models.OperationStatusSuccess
	case statusError:
		return models.OperationStatusError
	default:
		return models.OperationStatusUnspecified
	}
}

// LogOperation inserts a new lang operation row.
func (r *OperationsRepository) LogOperation(ctx context.Context, op *models.LangOperation) error {
	if op.ID == "" {
		op.ID = uuid.NewString()
	}

	inputJSON, err := json.Marshal(op.Input)
	if err != nil {
		return fmt.Errorf("marshal input: %w", err)
	}

	outputJSON, err := json.Marshal(op.Output)
	if err != nil {
		return fmt.Errorf("marshal output: %w", err)
	}

	if op.CreatedAt.IsZero() {
		op.CreatedAt = time.Now().UTC()
	}

	statusString := toStatusString(op.Status)

	query := `
        INSERT INTO lang_operations (
            id, library_type, operation, input, output, status, error,
            latency_ms, created_at, completed_at, session_id, user_id_hash, privacy_level
        ) VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7, $8, $9, $10, $11, $12, $13)
        ON CONFLICT (id) DO UPDATE SET
            library_type = EXCLUDED.library_type,
            operation = EXCLUDED.operation,
            input = EXCLUDED.input,
            output = EXCLUDED.output,
            status = EXCLUDED.status,
            error = EXCLUDED.error,
            latency_ms = EXCLUDED.latency_ms,
            created_at = EXCLUDED.created_at,
            completed_at = EXCLUDED.completed_at,
            session_id = EXCLUDED.session_id,
            user_id_hash = EXCLUDED.user_id_hash,
            privacy_level = EXCLUDED.privacy_level
    `

	_, err = r.db.ExecContext(
		ctx,
		query,
		op.ID,
		op.LibraryType,
		op.Operation,
		string(inputJSON),
		string(outputJSON),
		statusString,
		op.Error,
		op.LatencyMs,
		op.CreatedAt,
		sql.NullTime{Time: derefTime(op.CompletedAt), Valid: op.CompletedAt != nil},
		nullString(op.SessionID),
		nullString(op.UserIDHash),
		nullString(op.PrivacyLevel),
	)
	if err != nil {
		return fmt.Errorf("insert lang operation: %w", err)
	}

	return nil
}

// GetOperation retrieves an operation by id.
func (r *OperationsRepository) GetOperation(ctx context.Context, id string) (*models.LangOperation, error) {
	query := `
        SELECT id, library_type, operation, input, output, status, error,
               latency_ms, created_at, completed_at, session_id, user_id_hash, privacy_level
          FROM lang_operations
         WHERE id = $1
    `

	row := r.db.QueryRowContext(ctx, query, id)
	return scanOperation(row)
}

// ListOperations returns operations using provided filters. Pagination uses the created_at cursor encoded as RFC3339Nano string.
func (r *OperationsRepository) ListOperations(ctx context.Context, filters ListFilters) ([]*models.LangOperation, string, error) {
	baseQuery := `
        SELECT id, library_type, operation, input, output, status, error,
               latency_ms, created_at, completed_at, session_id, user_id_hash, privacy_level
          FROM lang_operations
    `

	clauses := []string{}
	args := []interface{}{}
	argPos := 1

	if filters.LibraryType != "" {
		clauses = append(clauses, fmt.Sprintf("library_type = $%d", argPos))
		args = append(args, filters.LibraryType)
		argPos++
	}

	if filters.SessionID != "" {
		clauses = append(clauses, fmt.Sprintf("session_id = $%d", argPos))
		args = append(args, filters.SessionID)
		argPos++
	}

	if filters.Status != models.OperationStatusUnspecified {
		clauses = append(clauses, fmt.Sprintf("status = $%d", argPos))
		args = append(args, toStatusString(filters.Status))
		argPos++
	}

	if filters.CreatedAfter != nil {
		clauses = append(clauses, fmt.Sprintf("created_at >= $%d", argPos))
		args = append(args, filters.CreatedAfter.UTC())
		argPos++
	}

	if filters.CreatedBefore != nil {
		clauses = append(clauses, fmt.Sprintf("created_at <= $%d", argPos))
		args = append(args, filters.CreatedBefore.UTC())
		argPos++
	}

	if filters.PageToken != "" {
		tokenTime, err := time.Parse(time.RFC3339Nano, filters.PageToken)
		if err == nil {
			clauses = append(clauses, fmt.Sprintf("created_at < $%d", argPos))
			args = append(args, tokenTime)
			argPos++
		}
	}

	query := baseQuery
	if len(clauses) > 0 {
		query += " WHERE " + joinClauses(clauses, " AND ")
	}

	query += " ORDER BY created_at DESC"

	limit := 50
	if filters.PageSize > 0 && filters.PageSize <= 500 {
		limit = filters.PageSize
	}

	query += fmt.Sprintf(" LIMIT %d", limit+1)

	rows, err := r.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, "", fmt.Errorf("list operations: %w", err)
	}
	defer rows.Close()

	operations := make([]*models.LangOperation, 0, limit)
	for rows.Next() {
		op, err := scanOperation(rows)
		if err != nil {
			return nil, "", fmt.Errorf("scan operation: %w", err)
		}
		operations = append(operations, op)
	}

	var nextToken string
	if len(operations) > limit {
		last := operations[limit]
		operations = operations[:limit]
		nextToken = last.CreatedAt.UTC().Format(time.RFC3339Nano)
	}

	return operations, nextToken, nil
}

// GetAnalytics aggregates analytics metrics for operations.
func (r *OperationsRepository) GetAnalytics(ctx context.Context, filters AnalyticsFilters) (*models.AnalyticsSummary, error) {
	clauses := []string{}
	args := []interface{}{}
	argPos := 1

	if filters.LibraryType != "" {
		clauses = append(clauses, fmt.Sprintf("library_type = $%d", argPos))
		args = append(args, filters.LibraryType)
		argPos++
	}

	if filters.StartTime != nil {
		clauses = append(clauses, fmt.Sprintf("created_at >= $%d", argPos))
		args = append(args, filters.StartTime.UTC())
		argPos++
	}

	if filters.EndTime != nil {
		clauses = append(clauses, fmt.Sprintf("created_at <= $%d", argPos))
		args = append(args, filters.EndTime.UTC())
		argPos++
	}

	where := ""
	if len(clauses) > 0 {
		where = " WHERE " + joinClauses(clauses, " AND ")
	}

	statsQuery := fmt.Sprintf(`
        SELECT library_type,
               COUNT(*) AS total_operations,
               COALESCE(AVG(latency_ms), 0) AS average_latency,
               SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_count,
               SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_count
          FROM lang_operations
          %s
         GROUP BY library_type
    `, where)

	rows, err := r.db.QueryContext(ctx, statsQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("query analytics: %w", err)
	}
	defer rows.Close()

	summary := &models.AnalyticsSummary{
		ErrorBreakdown: make(map[string]int64),
		GeneratedAt:    time.Now().UTC(),
	}

	for rows.Next() {
		var (
			libraryType    string
			totalOps       int64
			averageLatency float64
			successCount   int64
			errorCount     int64
		)

		if err := rows.Scan(&libraryType, &totalOps, &averageLatency, &successCount, &errorCount); err != nil {
			return nil, fmt.Errorf("scan analytics row: %w", err)
		}

		successRate := float64(0)
		if totalOps > 0 {
			successRate = float64(successCount) / float64(totalOps)
		}

		summary.TotalOperations += totalOps
		summary.AverageLatency += averageLatency * float64(totalOps)
		summary.SuccessRate += successRate * float64(totalOps)
		summary.ErrorBreakdown[libraryType] = errorCount

		summary.LibraryStats = append(summary.LibraryStats, models.LibraryStats{
			LibraryType:     libraryType,
			TotalOperations: totalOps,
			SuccessRate:     successRate,
			AverageLatency:  averageLatency,
			ErrorCount:      errorCount,
		})
	}

	if summary.TotalOperations > 0 {
		summary.AverageLatency = summary.AverageLatency / float64(summary.TotalOperations)
		summary.SuccessRate = summary.SuccessRate / float64(summary.TotalOperations)
	}

	trendQuery := fmt.Sprintf(`
        SELECT date_trunc('day', created_at) AS date,
               library_type,
               COUNT(*) AS operations,
               AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END) AS success_rate,
               COALESCE(AVG(latency_ms), 0) AS average_latency
          FROM lang_operations
          %s
         GROUP BY date, library_type
         ORDER BY date DESC
         LIMIT 90
    `, where)

	trendRows, err := r.db.QueryContext(ctx, trendQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("query performance trends: %w", err)
	}
	defer trendRows.Close()

	for trendRows.Next() {
		var trend models.PerformanceTrend
		if err := trendRows.Scan(&trend.Date, &trend.LibraryType, &trend.Operations, &trend.SuccessRate, &trend.AverageLatency); err != nil {
			return nil, fmt.Errorf("scan performance trend: %w", err)
		}
		summary.PerformanceTrend = append(summary.PerformanceTrend, trend)
	}

	return summary, nil
}

// CleanupOperations removes rows older than the provided timestamp.
func (r *OperationsRepository) CleanupOperations(ctx context.Context, olderThan time.Time) (*models.CleanupResult, error) {
	query := `
        DELETE FROM lang_operations
         WHERE created_at < $1
    `

	res, err := r.db.ExecContext(ctx, query, olderThan.UTC())
	if err != nil {
		return nil, fmt.Errorf("cleanup operations: %w", err)
	}

	affected, err := res.RowsAffected()
	if err != nil {
		return nil, fmt.Errorf("rows affected: %w", err)
	}

	return &models.CleanupResult{Deleted: affected}, nil
}

// Ping verifies database connectivity.
func (r *OperationsRepository) Ping(ctx context.Context) error {
	return r.db.PingContext(ctx)
}

func scanOperation(scanner interface{ Scan(dest ...any) error }) (*models.LangOperation, error) {
	op := &models.LangOperation{}
	var (
		inputJSON    []byte
		outputJSON   []byte
		statusString string
		completedAt  sql.NullTime
		sessionID    sql.NullString
		userIDHash   sql.NullString
		privacyLevel sql.NullString
	)

	if err := scanner.Scan(
		&op.ID,
		&op.LibraryType,
		&op.Operation,
		&inputJSON,
		&outputJSON,
		&statusString,
		&op.Error,
		&op.LatencyMs,
		&op.CreatedAt,
		&completedAt,
		&sessionID,
		&userIDHash,
		&privacyLevel,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, err
		}
		return nil, fmt.Errorf("scan lang operation: %w", err)
	}

	if completedAt.Valid {
		op.CompletedAt = &completedAt.Time
	}

	if sessionID.Valid {
		op.SessionID = sessionID.String
	}

	if userIDHash.Valid {
		op.UserIDHash = userIDHash.String
	}

	if privacyLevel.Valid {
		op.PrivacyLevel = privacyLevel.String
	}

	if len(inputJSON) > 0 {
		if err := json.Unmarshal(inputJSON, &op.Input); err != nil {
			return nil, fmt.Errorf("decode input json: %w", err)
		}
	} else {
		op.Input = make(map[string]any)
	}

	if len(outputJSON) > 0 {
		if err := json.Unmarshal(outputJSON, &op.Output); err != nil {
			return nil, fmt.Errorf("decode output json: %w", err)
		}
	} else {
		op.Output = make(map[string]any)
	}

	op.Status = fromStatusString(statusString)

	return op, nil
}

func joinClauses(clauses []string, sep string) string {
	switch len(clauses) {
	case 0:
		return ""
	case 1:
		return clauses[0]
	default:
		out := clauses[0]
		for i := 1; i < len(clauses); i++ {
			out += sep + clauses[i]
		}
		return out
	}
}

func derefTime(t *time.Time) time.Time {
	if t == nil {
		return time.Time{}
	}
	return t.UTC()
}

func nullString(value string) sql.NullString {
	if value == "" {
		return sql.NullString{}
	}
	return sql.NullString{String: value, Valid: true}
}
