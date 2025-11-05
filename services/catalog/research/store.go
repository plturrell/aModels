package research

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"

	_ "github.com/lib/pq"
)

// ReportStore persists deep research reports for later retrieval/analysis.
type ReportStore struct {
	db     *sql.DB
	logger *log.Logger
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

	return &ReportStore{
		db:     db,
		logger: logger,
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
