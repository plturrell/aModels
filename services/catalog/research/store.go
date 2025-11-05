package research

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	_ "github.com/lib/pq"
)

// ReportStore persists deep research reports for later retrieval/analysis.
type ReportStore struct {
	db            *sql.DB
	logger        *log.Logger
	retentionDays int
}

// NewReportStore connects to Postgres using the provided DSN. If dsn is empty,
// it returns nil without error, allowing the caller to treat persistence as optional.
func NewReportStore(dsn string, logger *log.Logger) (*ReportStore, error) {
	if dsn == "" {
		return nil, nil
	}

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("open postgres connection: %w", err)
	}

	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping postgres: %w", err)
	}

	retentionDays := defaultRetentionDays()

	return &ReportStore{
		db:            db,
		logger:        logger,
		retentionDays: retentionDays,
	}, nil
}

// Close releases database resources.
func (s *ReportStore) Close(ctx context.Context) error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

// SaveReport persists the raw deep research response for auditing and reuse.
func (s *ReportStore) SaveReport(ctx context.Context, topic string, dataElementID string, report *ResearchReport) error {
	if s == nil || s.db == nil || report == nil || report.Report == nil {
		return nil
	}

	reportJSON, err := json.Marshal(report)
	if err != nil {
		return fmt.Errorf("marshal research report: %w", err)
	}

	summary := report.Report.Summary
	_, err = s.db.ExecContext(
		ctx,
		`INSERT INTO research_reports (topic, data_element_id, report_summary, report_json)
		 VALUES ($1, NULLIF($2, ''), $3, $4)`,
		topic,
		dataElementID,
		summary,
		reportJSON,
	)
	if err != nil {
		return fmt.Errorf("insert research report: %w", err)
	}

	if s.logger != nil {
		s.logger.Printf("Stored research report for topic=%s element=%s", topic, dataElementID)
	}

	return nil
}

// StartRetentionJob launches a background goroutine that periodically deletes
// reports older than the configured retention window. A non-positive retention
// window disables the job.
func (s *ReportStore) StartRetentionJob(ctx context.Context, interval time.Duration) {
	if s == nil || s.db == nil || s.retentionDays <= 0 || interval <= 0 {
		return
	}

	ticker := time.NewTicker(interval)
	go func() {
		defer ticker.Stop()
		s.pruneExpiredReports(ctx)
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				s.pruneExpiredReports(ctx)
			}
		}
	}()
	if s.logger != nil {
		s.logger.Printf("Research report retention job started (retention_days=%d, interval=%s)", s.retentionDays, interval)
	}
}

func (s *ReportStore) pruneExpiredReports(ctx context.Context) {
	if s == nil || s.db == nil || s.retentionDays <= 0 {
		return
	}
	result, err := s.db.ExecContext(ctx, `
		DELETE FROM research_reports
		  WHERE created_at < NOW() - ($1::int * INTERVAL '1 day')`,
		s.retentionDays,
	)
	if err != nil {
		if s.logger != nil {
			s.logger.Printf("Retention job failed: %v", err)
		}
		return
	}
	if rows, err := result.RowsAffected(); err == nil && rows > 0 {
		if s.logger != nil {
			s.logger.Printf("Retention job removed %d research report(s)", rows)
		}
	}
}

func defaultRetentionDays() int {
	const fallback = 30
	raw := strings.TrimSpace(os.Getenv("RESEARCH_REPORT_RETENTION_DAYS"))
	if raw == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return parsed
}
