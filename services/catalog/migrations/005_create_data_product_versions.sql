-- Create data_product_versions table for semantic versioning
CREATE TABLE IF NOT EXISTS data_product_versions (
    id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL,
    version TEXT NOT NULL,
    major INTEGER NOT NULL,
    minor INTEGER NOT NULL,
    patch INTEGER NOT NULL,
    pre_release TEXT,
    build_metadata TEXT,
    product_snapshot JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by TEXT,
    deprecated BOOLEAN NOT NULL DEFAULT FALSE,
    deprecated_at TIMESTAMP,
    deprecation_reason TEXT,
    metadata JSONB
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_data_product_versions_product_id ON data_product_versions(product_id);
CREATE INDEX IF NOT EXISTS idx_data_product_versions_version ON data_product_versions(version);
CREATE INDEX IF NOT EXISTS idx_data_product_versions_created_at ON data_product_versions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_data_product_versions_deprecated ON data_product_versions(deprecated);
CREATE INDEX IF NOT EXISTS idx_data_product_versions_major_minor_patch ON data_product_versions(major DESC, minor DESC, patch DESC);

-- Create unique constraint on product_id + version
CREATE UNIQUE INDEX IF NOT EXISTS idx_data_product_versions_product_version ON data_product_versions(product_id, version);

