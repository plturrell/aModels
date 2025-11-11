-- Create langextract_audit_trail table for audit logging
CREATE TABLE IF NOT EXISTS langextract_audit_trail (
    id TEXT PRIMARY KEY,
    extraction_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    "user" TEXT NOT NULL,
    context TEXT,
    operation TEXT NOT NULL,
    request JSONB NOT NULL,
    response JSONB NOT NULL,
    processing_time_ms BIGINT NOT NULL,
    confidence FLOAT,
    schema_version TEXT,
    quality_metrics JSONB,
    resource_usage JSONB,
    metadata JSONB
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_audit_trail_extraction_id ON langextract_audit_trail(extraction_id);
CREATE INDEX IF NOT EXISTS idx_audit_trail_timestamp ON langextract_audit_trail(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_trail_user ON langextract_audit_trail("user");
CREATE INDEX IF NOT EXISTS idx_audit_trail_context ON langextract_audit_trail(context);
CREATE INDEX IF NOT EXISTS idx_audit_trail_operation ON langextract_audit_trail(operation);

-- Create composite index for common query patterns
CREATE INDEX IF NOT EXISTS idx_audit_trail_user_time ON langextract_audit_trail("user", timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_trail_context_time ON langextract_audit_trail(context, timestamp DESC);

-- Create GIN index for JSONB queries
CREATE INDEX IF NOT EXISTS idx_audit_trail_request_gin ON langextract_audit_trail USING GIN (request);
CREATE INDEX IF NOT EXISTS idx_audit_trail_response_gin ON langextract_audit_trail USING GIN (response);
CREATE INDEX IF NOT EXISTS idx_audit_trail_metadata_gin ON langextract_audit_trail USING GIN (metadata);
