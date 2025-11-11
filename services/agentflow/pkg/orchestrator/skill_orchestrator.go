package orchestrator

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/log"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer1_Blockchain/agents/shared/interfaces"
)

// SkillOrchestrator manages skills for any agent with standardized patterns
type SkillOrchestrator struct {
	agentID      string
	agentName    string
	skills       map[string]interfaces.Skill
	skillMetrics map[string]*interfaces.SkillMetrics
	skillConfigs map[string]*interfaces.SkillConfig
	mu           sync.RWMutex
	active       bool
	ctx          context.Context
	cancel       context.CancelFunc
}

// NewSkillOrchestrator creates a standardized orchestrator for any agent
func NewSkillOrchestrator(agentID, agentName string) *SkillOrchestrator {
	ctx, cancel := context.WithCancel(context.Background())
	return &SkillOrchestrator{
		agentID:      agentID,
		agentName:    agentName,
		skills:       make(map[string]interfaces.Skill),
		skillMetrics: make(map[string]*interfaces.SkillMetrics),
		skillConfigs: make(map[string]*interfaces.SkillConfig),
		active:       false,
		ctx:          ctx,
		cancel:       cancel,
	}
}

// RegisterSkill registers a skill with the orchestrator
func (so *SkillOrchestrator) RegisterSkill(skill interfaces.Skill) error {
	return so.RegisterSkillWithConfig(skill, nil)
}

// RegisterSkillWithConfig registers a skill with custom configuration
func (so *SkillOrchestrator) RegisterSkillWithConfig(skill interfaces.Skill, config *interfaces.SkillConfig) error {
	so.mu.Lock()
	defer so.mu.Unlock()

	skillID := skill.ID()
	if skillID == "" {
		return fmt.Errorf("skill ID cannot be empty")
	}

	if _, exists := so.skills[skillID]; exists {
		return fmt.Errorf("skill already registered: %s", skillID)
	}

	// Validate skill
	if err := so.validateSkill(skill); err != nil {
		return fmt.Errorf("skill validation failed: %w", err)
	}

	// Register skill
	so.skills[skillID] = skill

	// Initialize metrics
	so.skillMetrics[skillID] = &interfaces.SkillMetrics{
		ExecutionCount: 0,
		SuccessCount:   0,
		FailureCount:   0,
		TotalTime:      0,
		AverageTime:    0,
		SuccessRate:    0.0,
	}

	// Set config
	if config != nil {
		so.skillConfigs[skillID] = config
	} else {
		so.skillConfigs[skillID] = so.defaultConfig()
	}

	log.Info("Skill registered",
		"agent", so.agentName,
		"skill_id", skillID,
		"skill_name", skill.Name(),
		"version", skill.Version())

	return nil
}

// ExecuteSkill executes a skill with full metrics tracking and error handling
func (so *SkillOrchestrator) ExecuteSkill(ctx context.Context, skillID string, input *interfaces.SkillInput) (*interfaces.SkillResult, error) {
	so.mu.RLock()
	skill, exists := so.skills[skillID]
	config := so.skillConfigs[skillID]
	so.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("skill not found: %s", skillID)
	}

	if !config.Enabled {
		return nil, fmt.Errorf("skill is disabled: %s", skillID)
	}

	// Validate input
	if err := skill.Validate(input); err != nil {
		so.recordFailure(skillID, fmt.Errorf("validation failed: %w", err))
		return nil, fmt.Errorf("validation failed: %w", err)
	}

	// Create execution context with timeout
	execCtx := ctx
	if config.Timeout > 0 {
		var cancel context.CancelFunc
		execCtx, cancel = context.WithTimeout(ctx, config.Timeout)
		defer cancel()
	}

	// Execute with timing
	start := time.Now()
	result, err := skill.Execute(execCtx, input)
	duration := time.Since(start)

	// Update metrics
	so.updateMetrics(skillID, duration, err)

	if err != nil {
		log.Error("Skill execution failed",
			"agent", so.agentName,
			"skill_id", skillID,
			"error", err,
			"duration", duration)
		return nil, err
	}

	log.Debug("Skill executed successfully",
		"agent", so.agentName,
		"skill_id", skillID,
		"duration", duration)

	return result, nil
}

// ExecuteSkillWithRetry executes a skill with automatic retry logic
func (so *SkillOrchestrator) ExecuteSkillWithRetry(ctx context.Context, skillID string, input *interfaces.SkillInput) (*interfaces.SkillResult, error) {
	so.mu.RLock()
	config := so.skillConfigs[skillID]
	so.mu.RUnlock()

	var lastErr error
	maxRetries := config.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 1
	}

	for attempt := 0; attempt < maxRetries; attempt++ {
		result, err := so.ExecuteSkill(ctx, skillID, input)
		if err == nil {
			return result, nil
		}

		lastErr = err
		if attempt < maxRetries-1 {
			// Exponential backoff
			backoff := time.Duration(attempt+1) * 100 * time.Millisecond
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
				continue
			}
		}
	}

	return nil, fmt.Errorf("skill execution failed after %d attempts: %w", maxRetries, lastErr)
}

// GetSkill retrieves a skill by ID
func (so *SkillOrchestrator) GetSkill(skillID string) (interfaces.Skill, bool) {
	so.mu.RLock()
	defer so.mu.RUnlock()

	skill, exists := so.skills[skillID]
	return skill, exists
}

// ListSkills returns all registered skill IDs
func (so *SkillOrchestrator) ListSkills() []string {
	so.mu.RLock()
	defer so.mu.RUnlock()

	skills := make([]string, 0, len(so.skills))
	for id := range so.skills {
		skills = append(skills, id)
	}
	return skills
}

// GetSkillInfo returns comprehensive information about a skill
func (so *SkillOrchestrator) GetSkillInfo(skillID string) (*interfaces.SkillInfo, error) {
	so.mu.RLock()
	defer so.mu.RUnlock()

	skill, exists := so.skills[skillID]
	if !exists {
		return nil, fmt.Errorf("skill not found: %s", skillID)
	}

	metrics := so.skillMetrics[skillID]
	config := so.skillConfigs[skillID]

	status := interfaces.SkillStatusActive
	if !config.Enabled {
		status = interfaces.SkillStatusDisabled
	} else if !so.active {
		status = interfaces.SkillStatusInactive
	} else if metrics.LastError != "" {
		status = interfaces.SkillStatusError
	}

	return &interfaces.SkillInfo{
		ID:           skill.ID(),
		Name:         skill.Name(),
		Description:  skill.Description(),
		Version:      skill.Version(),
		Status:       status,
		Capabilities: skill.GetCapabilities(),
		Requirements: skill.GetRequirements(),
		Metrics:      so.copyMetrics(metrics),
		Config:       config,
	}, nil
}

// GetMetrics returns metrics for a specific skill
func (so *SkillOrchestrator) GetMetrics(skillID string) (*interfaces.SkillMetrics, error) {
	so.mu.RLock()
	defer so.mu.RUnlock()

	metrics, exists := so.skillMetrics[skillID]
	if !exists {
		return nil, fmt.Errorf("metrics not found for skill: %s", skillID)
	}

	return so.copyMetrics(metrics), nil
}

// GetAllMetrics returns metrics for all skills
func (so *SkillOrchestrator) GetAllMetrics() map[string]*interfaces.SkillMetrics {
	so.mu.RLock()
	defer so.mu.RUnlock()

	result := make(map[string]*interfaces.SkillMetrics)
	for id, metrics := range so.skillMetrics {
		result[id] = so.copyMetrics(metrics)
	}
	return result
}

// UpdateSkillConfig updates the configuration for a skill
func (so *SkillOrchestrator) UpdateSkillConfig(skillID string, config *interfaces.SkillConfig) error {
	so.mu.Lock()
	defer so.mu.Unlock()

	if _, exists := so.skills[skillID]; !exists {
		return fmt.Errorf("skill not found: %s", skillID)
	}

	so.skillConfigs[skillID] = config
	log.Info("Skill configuration updated", "agent", so.agentName, "skill_id", skillID)
	return nil
}

// EnableSkill enables a skill
func (so *SkillOrchestrator) EnableSkill(skillID string) error {
	so.mu.Lock()
	defer so.mu.Unlock()

	config, exists := so.skillConfigs[skillID]
	if !exists {
		return fmt.Errorf("skill not found: %s", skillID)
	}

	config.Enabled = true
	log.Info("Skill enabled", "agent", so.agentName, "skill_id", skillID)
	return nil
}

// DisableSkill disables a skill
func (so *SkillOrchestrator) DisableSkill(skillID string) error {
	so.mu.Lock()
	defer so.mu.Unlock()

	config, exists := so.skillConfigs[skillID]
	if !exists {
		return fmt.Errorf("skill not found: %s", skillID)
	}

	config.Enabled = false
	log.Info("Skill disabled", "agent", so.agentName, "skill_id", skillID)
	return nil
}

// Start activates the orchestrator
func (so *SkillOrchestrator) Start(ctx context.Context) error {
	so.mu.Lock()
	defer so.mu.Unlock()

	so.active = true
	log.Info("Skill orchestrator started", "agent", so.agentName, "skills_count", len(so.skills))
	return nil
}

// Stop deactivates the orchestrator
func (so *SkillOrchestrator) Stop(ctx context.Context) error {
	so.mu.Lock()
	defer so.mu.Unlock()

	so.active = false
	so.cancel()
	log.Info("Skill orchestrator stopped", "agent", so.agentName)
	return nil
}

// IsActive returns the orchestrator status
func (so *SkillOrchestrator) IsActive() bool {
	so.mu.RLock()
	defer so.mu.RUnlock()

	return so.active
}

// Private helper methods

func (so *SkillOrchestrator) validateSkill(skill interfaces.Skill) error {
	if skill.ID() == "" {
		return fmt.Errorf("skill ID is required")
	}
	if skill.Name() == "" {
		return fmt.Errorf("skill name is required")
	}
	if skill.Version() == "" {
		return fmt.Errorf("skill version is required")
	}
	return nil
}

func (so *SkillOrchestrator) updateMetrics(skillID string, duration time.Duration, err error) {
	so.mu.Lock()
	defer so.mu.Unlock()

	metrics := so.skillMetrics[skillID]
	metrics.ExecutionCount++
	metrics.TotalTime += duration
	metrics.LastExecuted = time.Now()

	if err != nil {
		metrics.FailureCount++
		metrics.LastError = err.Error()
	} else {
		metrics.SuccessCount++
		metrics.LastError = ""
	}

	if metrics.ExecutionCount > 0 {
		metrics.AverageTime = metrics.TotalTime / time.Duration(metrics.ExecutionCount)
		metrics.SuccessRate = float64(metrics.SuccessCount) / float64(metrics.ExecutionCount)
	}
}

func (so *SkillOrchestrator) recordFailure(skillID string, err error) {
	so.mu.Lock()
	defer so.mu.Unlock()

	metrics := so.skillMetrics[skillID]
	metrics.FailureCount++
	metrics.LastError = err.Error()
	metrics.LastExecuted = time.Now()
}

func (so *SkillOrchestrator) copyMetrics(metrics *interfaces.SkillMetrics) *interfaces.SkillMetrics {
	return &interfaces.SkillMetrics{
		ExecutionCount: metrics.ExecutionCount,
		SuccessCount:   metrics.SuccessCount,
		FailureCount:   metrics.FailureCount,
		TotalTime:      metrics.TotalTime,
		AverageTime:    metrics.AverageTime,
		SuccessRate:    metrics.SuccessRate,
		LastExecuted:   metrics.LastExecuted,
		LastError:      metrics.LastError,
	}
}

func (so *SkillOrchestrator) defaultConfig() *interfaces.SkillConfig {
	return &interfaces.SkillConfig{
		Enabled:    true,
		Timeout:    30 * time.Second,
		MaxRetries: 3,
		RateLimit:  100,
		Settings:   make(map[string]interface{}),
	}
}
