-- Migration: Create autonomous intelligence layer tables
-- Description: Tables for tracking autonomous task executions, agent performance, and learned patterns

-- Autonomous task executions
CREATE TABLE IF NOT EXISTS autonomous_task_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id VARCHAR(255) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    success BOOLEAN NOT NULL DEFAULT false,
    lessons_learned_count INTEGER DEFAULT 0,
    optimizations_applied_count INTEGER DEFAULT 0,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Agent performance tracking
CREATE TABLE IF NOT EXISTS autonomous_agent_performance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(255) NOT NULL,
    agent_type VARCHAR(100) NOT NULL,
    success_rate DECIMAL(5, 4) DEFAULT 0.0,
    failure_rate DECIMAL(5, 4) DEFAULT 0.0,
    total_executions INTEGER DEFAULT 0,
    successful_executions INTEGER DEFAULT 0,
    failed_executions INTEGER DEFAULT 0,
    last_execution_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(agent_id)
);

-- Learned patterns
CREATE TABLE IF NOT EXISTS autonomous_learned_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_id VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    context JSONB,
    success_rate DECIMAL(5, 4) DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Agent interactions
CREATE TABLE IF NOT EXISTS autonomous_agent_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_agent_id VARCHAR(255) NOT NULL,
    to_agent_id VARCHAR(255) NOT NULL,
    interaction_type VARCHAR(100) NOT NULL,
    knowledge_shared JSONB,
    outcome VARCHAR(50),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Optimizations applied
CREATE TABLE IF NOT EXISTS autonomous_optimizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    optimization_id VARCHAR(255) NOT NULL,
    optimization_type VARCHAR(100) NOT NULL,
    description TEXT,
    impact DECIMAL(5, 4),
    applied_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Knowledge base entries
CREATE TABLE IF NOT EXISTS autonomous_knowledge_base (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_id VARCHAR(255) NOT NULL UNIQUE,
    knowledge_type VARCHAR(100) NOT NULL, -- 'pattern', 'solution', 'best_practice'
    content JSONB NOT NULL,
    validation_score DECIMAL(5, 4),
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_autonomous_task_executions_task_id ON autonomous_task_executions(task_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_task_executions_task_type ON autonomous_task_executions(task_type);
CREATE INDEX IF NOT EXISTS idx_autonomous_task_executions_status ON autonomous_task_executions(status);
CREATE INDEX IF NOT EXISTS idx_autonomous_task_executions_started_at ON autonomous_task_executions(started_at);

CREATE INDEX IF NOT EXISTS idx_autonomous_agent_performance_agent_id ON autonomous_agent_performance(agent_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_agent_performance_agent_type ON autonomous_agent_performance(agent_type);

CREATE INDEX IF NOT EXISTS idx_autonomous_learned_patterns_pattern_id ON autonomous_learned_patterns(pattern_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_learned_patterns_usage_count ON autonomous_learned_patterns(usage_count);

CREATE INDEX IF NOT EXISTS idx_autonomous_agent_interactions_from_agent ON autonomous_agent_interactions(from_agent_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_agent_interactions_to_agent ON autonomous_agent_interactions(to_agent_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_agent_interactions_timestamp ON autonomous_agent_interactions(timestamp);

CREATE INDEX IF NOT EXISTS idx_autonomous_optimizations_optimization_id ON autonomous_optimizations(optimization_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_optimizations_applied_at ON autonomous_optimizations(applied_at);

CREATE INDEX IF NOT EXISTS idx_autonomous_knowledge_base_knowledge_id ON autonomous_knowledge_base(knowledge_id);
CREATE INDEX IF NOT EXISTS idx_autonomous_knowledge_base_knowledge_type ON autonomous_knowledge_base(knowledge_type);

