-- Migration: Create break detection tables
-- Description: Tables for break detection, baseline management, and analytics

-- Break detection baselines - Store baseline snapshots for comparison
CREATE TABLE IF NOT EXISTS break_detection_baselines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    baseline_id VARCHAR(255) NOT NULL UNIQUE,
    system_name VARCHAR(100) NOT NULL, -- 'sap_fioneer', 'bcrs', 'rco', 'axiomsl'
    version VARCHAR(50) NOT NULL, -- 'current', 'v1.0.0', etc.
    snapshot_type VARCHAR(50) NOT NULL, -- 'full', 'incremental', 'point_in_time'
    snapshot_data JSONB NOT NULL, -- Actual baseline data
    metadata JSONB, -- Additional metadata about the baseline
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255),
    expires_at TIMESTAMP, -- Optional expiration
    is_active BOOLEAN NOT NULL DEFAULT true,
    CONSTRAINT unique_system_version UNIQUE(system_name, version, snapshot_type)
);

-- Break detection runs - Track detection execution history
CREATE TABLE IF NOT EXISTS break_detection_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id VARCHAR(255) NOT NULL UNIQUE,
    system_name VARCHAR(100) NOT NULL,
    baseline_id VARCHAR(255) NOT NULL,
    detection_type VARCHAR(50) NOT NULL, -- 'finance', 'capital', 'liquidity', 'regulatory'
    status VARCHAR(50) NOT NULL, -- 'running', 'completed', 'failed', 'cancelled'
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    total_breaks_detected INTEGER DEFAULT 0,
    total_records_checked INTEGER DEFAULT 0,
    configuration JSONB, -- Detection configuration
    result_summary JSONB, -- Summary of results
    error_message TEXT,
    created_by VARCHAR(255),
    workflow_instance_id VARCHAR(255), -- Link to workflow if triggered by workflow
    CONSTRAINT fk_baseline FOREIGN KEY (baseline_id) REFERENCES break_detection_baselines(baseline_id)
);

-- Break detection breaks - Store detected breaks
CREATE TABLE IF NOT EXISTS break_detection_breaks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    break_id VARCHAR(255) NOT NULL UNIQUE,
    run_id VARCHAR(255) NOT NULL,
    system_name VARCHAR(100) NOT NULL,
    detection_type VARCHAR(50) NOT NULL, -- 'finance', 'capital', 'liquidity', 'regulatory'
    break_type VARCHAR(100) NOT NULL, -- 'missing_entry', 'amount_mismatch', 'balance_break', etc.
    severity VARCHAR(20) NOT NULL, -- 'critical', 'high', 'medium', 'low'
    status VARCHAR(50) NOT NULL DEFAULT 'open', -- 'open', 'investigating', 'resolved', 'false_positive'
    
    -- Break details
    current_value JSONB, -- Current system data
    baseline_value JSONB, -- Baseline data for comparison
    difference JSONB, -- Calculated difference
    affected_entities JSONB, -- Entities affected by this break
    
    -- Analysis (from Deep Research)
    root_cause_analysis TEXT, -- Root cause explanation from Deep Research
    semantic_enrichment JSONB, -- Semantic context from Deep Research
    recommendations JSONB, -- Recommendations from Deep Research
    
    -- AI Analysis (from LocalAI)
    ai_description TEXT, -- Natural language description from LocalAI
    ai_category VARCHAR(100), -- AI-categorized break type
    ai_priority_score DECIMAL(5, 4), -- AI-calculated priority
    
    -- Search integration
    similar_breaks JSONB, -- Similar historical breaks found via search
    
    -- Metadata
    detected_at TIMESTAMP NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(255),
    resolution_notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT fk_run FOREIGN KEY (run_id) REFERENCES break_detection_runs(run_id)
);

-- Break detection analytics - Historical analytics and trends
CREATE TABLE IF NOT EXISTS break_detection_analytics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analytics_id VARCHAR(255) NOT NULL UNIQUE,
    system_name VARCHAR(100) NOT NULL,
    detection_type VARCHAR(50) NOT NULL,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    period_type VARCHAR(20) NOT NULL, -- 'hourly', 'daily', 'weekly', 'monthly'
    
    -- Metrics
    total_breaks INTEGER DEFAULT 0,
    critical_breaks INTEGER DEFAULT 0,
    high_breaks INTEGER DEFAULT 0,
    medium_breaks INTEGER DEFAULT 0,
    low_breaks INTEGER DEFAULT 0,
    
    resolved_breaks INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    average_resolution_time_seconds INTEGER,
    
    -- Break type breakdown
    break_type_counts JSONB, -- Counts by break type
    
    -- Trend analysis
    trend_direction VARCHAR(20), -- 'increasing', 'decreasing', 'stable'
    trend_confidence DECIMAL(5, 4),
    
    -- Metadata
    calculated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    CONSTRAINT unique_analytics UNIQUE(system_name, detection_type, period_start, period_end, period_type)
);

-- Break detection rules - Auto-generated and manual rules
CREATE TABLE IF NOT EXISTS break_detection_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id VARCHAR(255) NOT NULL UNIQUE,
    system_name VARCHAR(100) NOT NULL,
    detection_type VARCHAR(50) NOT NULL,
    rule_type VARCHAR(50) NOT NULL, -- 'auto_generated', 'manual', 'regulatory'
    rule_source VARCHAR(100), -- 'deep_research', 'regulatory_spec', 'manual'
    
    -- Rule definition
    rule_name VARCHAR(255) NOT NULL,
    rule_description TEXT,
    rule_condition JSONB NOT NULL, -- Condition logic
    rule_threshold JSONB, -- Threshold values
    
    -- Validation
    validation_enabled BOOLEAN NOT NULL DEFAULT true,
    validation_query TEXT, -- SQL/query for validation
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT true,
    priority INTEGER DEFAULT 0, -- Higher priority = checked first
    
    -- Metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_by VARCHAR(255),
    
    CONSTRAINT unique_rule_name UNIQUE(system_name, rule_name)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_break_detection_baselines_system_version ON break_detection_baselines(system_name, version);
CREATE INDEX IF NOT EXISTS idx_break_detection_baselines_active ON break_detection_baselines(is_active);
CREATE INDEX IF NOT EXISTS idx_break_detection_baselines_created_at ON break_detection_baselines(created_at);

CREATE INDEX IF NOT EXISTS idx_break_detection_runs_system ON break_detection_runs(system_name);
CREATE INDEX IF NOT EXISTS idx_break_detection_runs_status ON break_detection_runs(status);
CREATE INDEX IF NOT EXISTS idx_break_detection_runs_started_at ON break_detection_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_break_detection_runs_baseline ON break_detection_runs(baseline_id);
CREATE INDEX IF NOT EXISTS idx_break_detection_runs_workflow ON break_detection_runs(workflow_instance_id);

CREATE INDEX IF NOT EXISTS idx_break_detection_breaks_run ON break_detection_breaks(run_id);
CREATE INDEX IF NOT EXISTS idx_break_detection_breaks_system ON break_detection_breaks(system_name);
CREATE INDEX IF NOT EXISTS idx_break_detection_breaks_type ON break_detection_breaks(detection_type);
CREATE INDEX IF NOT EXISTS idx_break_detection_breaks_severity ON break_detection_breaks(severity);
CREATE INDEX IF NOT EXISTS idx_break_detection_breaks_status ON break_detection_breaks(status);
CREATE INDEX IF NOT EXISTS idx_break_detection_breaks_detected_at ON break_detection_breaks(detected_at);
CREATE INDEX IF NOT EXISTS idx_break_detection_breaks_break_type ON break_detection_breaks(break_type);

CREATE INDEX IF NOT EXISTS idx_break_detection_analytics_system ON break_detection_analytics(system_name);
CREATE INDEX IF NOT EXISTS idx_break_detection_analytics_type ON break_detection_analytics(detection_type);
CREATE INDEX IF NOT EXISTS idx_break_detection_analytics_period ON break_detection_analytics(period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_break_detection_analytics_calculated_at ON break_detection_analytics(calculated_at);

CREATE INDEX IF NOT EXISTS idx_break_detection_rules_system ON break_detection_rules(system_name);
CREATE INDEX IF NOT EXISTS idx_break_detection_rules_type ON break_detection_rules(detection_type);
CREATE INDEX IF NOT EXISTS idx_break_detection_rules_active ON break_detection_rules(is_active);
CREATE INDEX IF NOT EXISTS idx_break_detection_rules_priority ON break_detection_rules(priority DESC);

-- Full-text search indexes (for PostgreSQL)
-- Note: Requires pg_trgm extension for similarity search
-- CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_break_detection_breaks_ai_description_fts ON break_detection_breaks USING gin(to_tsvector('english', COALESCE(ai_description, '')));
CREATE INDEX IF NOT EXISTS idx_break_detection_breaks_root_cause_fts ON break_detection_breaks USING gin(to_tsvector('english', COALESCE(root_cause_analysis, '')));

