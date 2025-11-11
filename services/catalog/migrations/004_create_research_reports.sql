-- +goose Up
CREATE TABLE IF NOT EXISTS research_reports (
    id SERIAL PRIMARY KEY,
    topic TEXT NOT NULL,
    data_element_id TEXT,
    report_summary TEXT,
    report_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_reports_topic ON research_reports (topic);
CREATE INDEX IF NOT EXISTS idx_research_reports_element ON research_reports (data_element_id);

-- +goose Down
DROP TABLE IF EXISTS research_reports;
