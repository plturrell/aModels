package orchestration

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"
)

// AgentCoordinator coordinates multiple agents in workflows.
type AgentCoordinator struct {
	logger          *log.Logger
	agents          map[string]*Agent
	agentMutex      sync.RWMutex
	communicationCh chan AgentMessage
	stateManager    *AgentStateManager
	retryConfig     *RetryConfig
}

// Agent represents a single agent in a workflow.
type Agent struct {
	ID            string
	Type          string
	Status        AgentStatus
	State         map[string]any
	LastHeartbeat time.Time
	Results       []AgentResult
	ErrorCount    int
	RetryCount    int
}

// AgentStatus represents the status of an agent.
type AgentStatus string

const (
	AgentStatusIdle      AgentStatus = "idle"
	AgentStatusRunning   AgentStatus = "running"
	AgentStatusCompleted AgentStatus = "completed"
	AgentStatusFailed    AgentStatus = "failed"
	AgentStatusRetrying   AgentStatus = "retrying"
)

// AgentMessage represents a message between agents.
type AgentMessage struct {
	FromAgentID string
	ToAgentID   string
	Type        string
	Payload     map[string]any
	Timestamp   time.Time
}

// AgentResult represents the result of an agent execution.
type AgentResult struct {
	AgentID    string
	Success    bool
	Output     map[string]any
	Error      string
	Duration   time.Duration
	Timestamp  time.Time
}

// AgentStateManager manages shared state between agents.
type AgentStateManager struct {
	state map[string]any
	mu    sync.RWMutex
}

// NewAgentStateManager creates a new state manager.
func NewAgentStateManager() *AgentStateManager {
	return &AgentStateManager{
		state: make(map[string]any),
	}
}

// Get retrieves a state value.
func (asm *AgentStateManager) Get(key string) (any, bool) {
	asm.mu.RLock()
	defer asm.mu.RUnlock()
	val, exists := asm.state[key]
	return val, exists
}

// Set sets a state value.
func (asm *AgentStateManager) Set(key string, value any) {
	asm.mu.Lock()
	defer asm.mu.Unlock()
	asm.state[key] = value
}

// RetryConfig holds retry configuration.
type RetryConfig struct {
	MaxRetries      int
	InitialDelay    time.Duration
	MaxDelay        time.Duration
	BackoffMultiplier float64
}

// DefaultRetryConfig returns default retry configuration.
func DefaultRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries:       3,
		InitialDelay:     2 * time.Second,
		MaxDelay:         60 * time.Second,
		BackoffMultiplier: 2.0,
	}
}

// NewAgentCoordinator creates a new agent coordinator.
func NewAgentCoordinator(logger *log.Logger) *AgentCoordinator {
	return &AgentCoordinator{
		logger:          logger,
		agents:          make(map[string]*Agent),
		communicationCh: make(chan AgentMessage, 100),
		stateManager:    NewAgentStateManager(),
		retryConfig:     DefaultRetryConfig(),
	}
}

// RegisterAgent registers an agent with the coordinator.
func (ac *AgentCoordinator) RegisterAgent(agentID, agentType string) *Agent {
	ac.agentMutex.Lock()
	defer ac.agentMutex.Unlock()

	agent := &Agent{
		ID:            agentID,
		Type:          agentType,
		Status:        AgentStatusIdle,
		State:         make(map[string]any),
		LastHeartbeat: time.Now(),
		Results:       []AgentResult{},
	}

	ac.agents[agentID] = agent
	ac.logger.Printf("Registered agent: %s (type: %s)", agentID, agentType)

	return agent
}

// StartAgent starts an agent execution.
func (ac *AgentCoordinator) StartAgent(ctx context.Context, agentID string, task map[string]any) error {
	ac.agentMutex.Lock()
	agent, exists := ac.agents[agentID]
	if !exists {
		ac.agentMutex.Unlock()
		return fmt.Errorf("agent %s not registered", agentID)
	}
	agent.Status = AgentStatusRunning
	agent.State = task
	ac.agentMutex.Unlock()

	ac.logger.Printf("Started agent: %s", agentID)

	// Execute agent (simplified - would call actual agent execution)
	go ac.executeAgent(ctx, agentID, task)

	return nil
}

// executeAgent executes an agent with retry logic.
func (ac *AgentCoordinator) executeAgent(ctx context.Context, agentID string, task map[string]any) {
	startTime := time.Now()
	agent := ac.getAgent(agentID)
	if agent == nil {
		return
	}

	var result AgentResult
	var err error

	// Retry loop
	for attempt := 0; attempt <= ac.retryConfig.MaxRetries; attempt++ {
		if attempt > 0 {
			// Calculate backoff delay
			delay := time.Duration(float64(ac.retryConfig.InitialDelay) * 
				pow(ac.retryConfig.BackoffMultiplier, float64(attempt-1)))
			if delay > ac.retryConfig.MaxDelay {
				delay = ac.retryConfig.MaxDelay
			}

			ac.logger.Printf("Retrying agent %s (attempt %d) after %v", agentID, attempt, delay)
			agent.Status = AgentStatusRetrying
			time.Sleep(delay)
		}

		// Execute agent task (simplified)
		result, err = ac.runAgentTask(ctx, agentID, task)
		
		if err == nil {
			// Success
			result.Success = true
			result.Duration = time.Since(startTime)
			result.Timestamp = time.Now()

			ac.agentMutex.Lock()
			agent.Status = AgentStatusCompleted
			agent.Results = append(agent.Results, result)
			agent.RetryCount = attempt
			ac.agentMutex.Unlock()

			ac.logger.Printf("Agent %s completed successfully", agentID)
			return
		}

		// Error occurred
		agent.ErrorCount++
		ac.logger.Printf("Agent %s failed (attempt %d): %v", agentID, attempt+1, err)
	}

	// All retries exhausted
	result.Success = false
	result.Error = err.Error()
	result.Duration = time.Since(startTime)
	result.Timestamp = time.Now()

	ac.agentMutex.Lock()
	agent.Status = AgentStatusFailed
	agent.Results = append(agent.Results, result)
	ac.agentMutex.Unlock()

	ac.logger.Printf("Agent %s failed after %d retries", agentID, ac.retryConfig.MaxRetries)
}

// runAgentTask executes a single agent task (simplified implementation).
func (ac *AgentCoordinator) runAgentTask(ctx context.Context, agentID string, task map[string]any) (AgentResult, error) {
	// This would call the actual agent execution logic
	// For now, return a placeholder result
	
	agent := ac.getAgent(agentID)
	if agent == nil {
		return AgentResult{}, fmt.Errorf("agent not found")
	}

	// Simulate agent execution
	time.Sleep(100 * time.Millisecond)

	result := AgentResult{
		AgentID:   agentID,
		Success:   true,
		Output:    map[string]any{"status": "completed", "agent_type": agent.Type},
		Timestamp: time.Now(),
	}

	return result, nil
}

// SendMessage sends a message from one agent to another.
func (ac *AgentCoordinator) SendMessage(fromAgentID, toAgentID, messageType string, payload map[string]any) error {
	message := AgentMessage{
		FromAgentID: fromAgentID,
		ToAgentID:   toAgentID,
		Type:        messageType,
		Payload:     payload,
		Timestamp:   time.Now(),
	}

	select {
	case ac.communicationCh <- message:
		ac.logger.Printf("Message sent from %s to %s: %s", fromAgentID, toAgentID, messageType)
		return nil
	case <-time.After(5 * time.Second):
		return fmt.Errorf("message send timeout")
	}
}

// GetAgentState retrieves the current state of an agent.
func (ac *AgentCoordinator) GetAgentState(agentID string) (*Agent, error) {
	agent := ac.getAgent(agentID)
	if agent == nil {
		return nil, fmt.Errorf("agent %s not found", agentID)
	}
	return agent, nil
}

// GetAgentStatus retrieves the status of an agent.
func (ac *AgentCoordinator) GetAgentStatus(agentID string) (AgentStatus, error) {
	agent := ac.getAgent(agentID)
	if agent == nil {
		return "", fmt.Errorf("agent %s not found", agentID)
	}
	return agent.Status, nil
}

// GetAgentPerformance retrieves performance metrics for an agent.
func (ac *AgentCoordinator) GetAgentPerformance(agentID string) (map[string]any, error) {
	agent := ac.getAgent(agentID)
	if agent == nil {
		return nil, fmt.Errorf("agent %s not found", agentID)
	}

	metrics := map[string]any{
		"agent_id":     agent.ID,
		"agent_type":   agent.Type,
		"status":       agent.Status,
		"error_count":  agent.ErrorCount,
		"retry_count":  agent.RetryCount,
		"result_count": len(agent.Results),
	}

	// Calculate success rate
	if len(agent.Results) > 0 {
		successCount := 0
		for _, result := range agent.Results {
			if result.Success {
				successCount++
			}
		}
		metrics["success_rate"] = float64(successCount) / float64(len(agent.Results))
	}

	return metrics, nil
}

// GetAgent gets an agent by ID (thread-safe).
func (ac *AgentCoordinator) getAgent(agentID string) *Agent {
	ac.agentMutex.RLock()
	defer ac.agentMutex.RUnlock()
	return ac.agents[agentID]
}

// GetSharedState retrieves a value from shared state.
func (ac *AgentCoordinator) GetSharedState(key string) (any, bool) {
	return ac.stateManager.Get(key)
}

// SetSharedState sets a value in shared state.
func (ac *AgentCoordinator) SetSharedState(key string, value any) {
	ac.stateManager.Set(key, value)
}

// StartMessageProcessor starts processing agent messages.
func (ac *AgentCoordinator) StartMessageProcessor(ctx context.Context) {
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case message := <-ac.communicationCh:
				ac.handleMessage(message)
			}
		}
	}()
}

// handleMessage handles a message between agents.
func (ac *AgentCoordinator) handleMessage(message AgentMessage) {
	ac.logger.Printf("Handling message: %s -> %s (%s)", message.FromAgentID, message.ToAgentID, message.Type)

	// Update target agent state based on message
	agent := ac.getAgent(message.ToAgentID)
	if agent != nil {
		ac.agentMutex.Lock()
		if agent.State == nil {
			agent.State = make(map[string]any)
		}
		agent.State[message.Type] = message.Payload
		ac.agentMutex.Unlock()
	}
}

// Helper function
func pow(base, exp float64) float64 {
	result := 1.0
	for i := 0; i < int(exp); i++ {
		result *= base
	}
	return result
}

