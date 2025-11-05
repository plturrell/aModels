-- Migration: Create discoverability tables
-- Description: Tables for tags, search, and marketplace functionality

-- Tags table
CREATE TABLE IF NOT EXISTS tags (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    parent_tag_id TEXT,
    description TEXT,
    usage_count BIGINT DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    FOREIGN KEY (parent_tag_id) REFERENCES tags(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_tags_category ON tags(category);
CREATE INDEX IF NOT EXISTS idx_tags_parent ON tags(parent_tag_id);
CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);

-- Product tags junction table
CREATE TABLE IF NOT EXISTS product_tags (
    product_id TEXT NOT NULL,
    tag_id TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    source TEXT NOT NULL DEFAULT 'manual', -- 'manual', 'auto', 'suggested'
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by TEXT,
    PRIMARY KEY (product_id, tag_id),
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_product_tags_product ON product_tags(product_id);
CREATE INDEX IF NOT EXISTS idx_product_tags_tag ON product_tags(tag_id);

-- Search history table
CREATE TABLE IF NOT EXISTS search_history (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    query TEXT NOT NULL,
    result_count INTEGER,
    clicked TEXT[], -- Array of product IDs that were clicked
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_history_query ON search_history(query);
CREATE INDEX IF NOT EXISTS idx_search_history_timestamp ON search_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_search_history_user ON search_history(user_id);

-- Product usage statistics table
CREATE TABLE IF NOT EXISTS product_usage_stats (
    product_id TEXT PRIMARY KEY,
    total_views BIGINT DEFAULT 0,
    unique_viewers BIGINT DEFAULT 0,
    access_requests BIGINT DEFAULT 0,
    approved_access BIGINT DEFAULT 0,
    rejected_access BIGINT DEFAULT 0,
    average_rating FLOAT DEFAULT 0.0,
    review_count BIGINT DEFAULT 0,
    last_accessed TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_usage_stats_views ON product_usage_stats(total_views DESC);
CREATE INDEX IF NOT EXISTS idx_usage_stats_rating ON product_usage_stats(average_rating DESC);

-- Access requests table
CREATE TABLE IF NOT EXISTS access_requests (
    id TEXT PRIMARY KEY,
    product_id TEXT NOT NULL,
    requester_id TEXT NOT NULL,
    requester_team TEXT,
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
    reason TEXT,
    requested_at TIMESTAMP NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMP,
    processed_by TEXT,
    comments TEXT
);

CREATE INDEX IF NOT EXISTS idx_access_requests_product ON access_requests(product_id);
CREATE INDEX IF NOT EXISTS idx_access_requests_requester ON access_requests(requester_id);
CREATE INDEX IF NOT EXISTS idx_access_requests_status ON access_requests(status);
CREATE INDEX IF NOT EXISTS idx_access_requests_requested ON access_requests(requested_at DESC);

-- Add team and category columns to data_products if they don't exist
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'data_products' AND column_name = 'team') THEN
        ALTER TABLE data_products ADD COLUMN team TEXT;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'data_products' AND column_name = 'category') THEN
        ALTER TABLE data_products ADD COLUMN category TEXT;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'data_products' AND column_name = 'status') THEN
        ALTER TABLE data_products ADD COLUMN status TEXT DEFAULT 'draft';
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'data_products' AND column_name = 'published_at') THEN
        ALTER TABLE data_products ADD COLUMN published_at TIMESTAMP;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'data_products' AND column_name = 'usage_count') THEN
        ALTER TABLE data_products ADD COLUMN usage_count BIGINT DEFAULT 0;
    END IF;
END $$;

-- Create indexes for data_products discoverability columns
CREATE INDEX IF NOT EXISTS idx_data_products_team ON data_products(team);
CREATE INDEX IF NOT EXISTS idx_data_products_category ON data_products(category);
CREATE INDEX IF NOT EXISTS idx_data_products_status ON data_products(status);
CREATE INDEX IF NOT EXISTS idx_data_products_published ON data_products(published_at DESC);

-- Full-text search index for data products
CREATE INDEX IF NOT EXISTS idx_data_products_search ON data_products USING GIN(to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(description, '')));

