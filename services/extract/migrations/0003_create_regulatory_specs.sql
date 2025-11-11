-- Migration: Create regulatory specification tables
-- Description: Tables for storing and managing regulatory reporting specifications (MAS 610, BCBS 239)

-- Regulatory schemas table
CREATE TABLE IF NOT EXISTS regulatory_schemas (
    id TEXT PRIMARY KEY,
    regulatory_type TEXT NOT NULL, -- "mas_610", "bcbs_239", "generic"
    version TEXT NOT NULL,
    document_source TEXT NOT NULL,
    document_version TEXT NOT NULL,
    spec_json JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by TEXT,
    is_reference BOOLEAN DEFAULT false,
    status TEXT DEFAULT 'draft', -- "draft", "approved", "deprecated"
    UNIQUE(regulatory_type, version)
);

CREATE INDEX IF NOT EXISTS idx_regulatory_schemas_type ON regulatory_schemas(regulatory_type);
CREATE INDEX IF NOT EXISTS idx_regulatory_schemas_version ON regulatory_schemas(version);
CREATE INDEX IF NOT EXISTS idx_regulatory_schemas_reference ON regulatory_schemas(regulatory_type, is_reference) WHERE is_reference = true;
CREATE INDEX IF NOT EXISTS idx_regulatory_schemas_status ON regulatory_schemas(status);

-- Schema changes table (tracks changes between versions)
CREATE TABLE IF NOT EXISTS schema_changes (
    id TEXT PRIMARY KEY,
    from_version TEXT NOT NULL,
    to_version TEXT NOT NULL,
    change_type TEXT NOT NULL, -- "added", "modified", "removed"
    field_id TEXT,
    description TEXT,
    impact TEXT, -- "low", "medium", "high"
    breaking BOOLEAN DEFAULT false,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_schema_changes_versions ON schema_changes(from_version, to_version);

-- Schema to data product mappings
CREATE TABLE IF NOT EXISTS schema_product_mappings (
    id TEXT PRIMARY KEY,
    schema_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    field_mappings JSONB NOT NULL, -- Maps regulatory fields to data product fields
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    FOREIGN KEY (schema_id) REFERENCES regulatory_schemas(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_schema_product_mappings_schema ON schema_product_mappings(schema_id);
CREATE INDEX IF NOT EXISTS idx_schema_product_mappings_product ON schema_product_mappings(product_id);

