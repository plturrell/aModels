package autonomous

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/plturrell/aModels/services/catalog/research"
)

// IntelligenceLayer is the core autonomous intelligence system that integrates
// Goose, Deep Research, DeepAgents, and Unified Workflow for self-learning and optimization.
type IntelligenceLayer struct {
	// Integrated services
	deepResearchClient     *research.DeepResearchClient
	deepAgentsURL         string
	unifiedWorkflowURL     string
	gooseEnabled           bool
	
	// Learning and optimization
	learningEngine         *LearningEngine
	optimizationEngine     *OptimizationEngine
	predictiveEngine      *PredictiveEngine
	governanceEngine       *GovernanceEngine
	
	// Agent coordination
	agentRegistry          *AgentRegistry
	knowledgeBase          *KnowledgeBase
	
	// HTTP client
	httpClient            *http.Client
	logger                *log.Logger
	
	// State management
	mu                    sync.RWMutex
	activeTasks           map[string]*TaskState
	performanceMetrics    *PerformanceMetrics
}

// LearningEngine enables agents to learn from each other's successes and failures.
type LearningEngine struct {
	knowledgeBase      *KnowledgeBase
	patternStore       *PatternStore
	logger             *log.Logger
}

// OptimizationEngine continuously optimizes system performance.
type OptimizationEngine struct {
	performanceMonitor *PerformanceMonitor
	optimizationRules   []OptimizationRule
	logger              *log.Logger
}

// PredictiveEngine predicts data quality issues and resource needs.
type PredictiveEngine struct {
	models              map[string]*PredictionModel
	historicalData      *HistoricalDataStore
	logger              *log.Logger
}

// GovernanceEngine ensures compliance and governance automatically.
type GovernanceEngine struct {
	policies             []GovernancePolicy
	complianceChecker    *ComplianceChecker
	logger               *log.Logger
}

// AgentRegistry manages self-learning agents and their interactions.
type AgentRegistry struct {
	agents              map[string]*Agent
	interactions        []AgentInteraction
	knowledgeSharing     *KnowledgeSharing
	mu                  sync.RWMutex
}

// Agent represents a self-learning agent in the system.
type Agent struct {
	ID                  string
	Type                string
	Capabilities        []string
	SuccessRate         float64
	FailureRate         float64
	LessonsLearned      []Lesson
	PerformanceHistory  []PerformanceSnapshot
	LastUpdated         time.Time
}

// AgentInteraction represents an interaction between agents.
type AgentInteraction struct {
	FromAgentID         string
	ToAgentID           string
	InteractionType      string
	KnowledgeShared     map[string]interface{}
	Outcome             string
	Timestamp           time.Time
}

// KnowledgeBase stores shared knowledge across agents.
type KnowledgeBase struct {
	patterns            map[string]*Pattern
	solutions           map[string]*Solution
	bestPractices       []BestPractice
	mu                  sync.RWMutex
}

// Pattern represents a learned pattern from agent operations.
type Pattern struct {
	ID                  string
	Description         string
	Context             map[string]interface{}
	SuccessRate         float64
	UsageCount          int
	LastUsed            time.Time
}

// Solution represents a successful solution to a problem.
type Solution struct {
	ID                  string
	ProblemType         string
	Solution            map[string]interface{}
	Effectiveness        float64
	UsageCount          int
}

// BestPractice represents a best practice learned from agent operations.
type BestPractice struct {
	ID                  string
	Description         string
	Context             string
	ValidationScore     float64
}

// Lesson represents a lesson learned from an agent operation.
type Lesson struct {
	ID                  string
	Type                string // "success" or "failure"
	Context             map[string]interface{}
	Insight             string
	Recommendation      string
	Timestamp           time.Time
}

// PerformanceSnapshot captures agent performance at a point in time.
type PerformanceSnapshot struct {
	Timestamp           time.Time
	SuccessRate         float64
	Latency             time.Duration
	ResourceUsage       map[string]float64
}

// TaskState tracks the state of an autonomous task.
type TaskState struct {
	TaskID              string
	Type                string
	Status              string
	StartedAt           time.Time
	Progress            float64
	Result              interface{}
	Error               error
}

// PerformanceMetrics tracks overall system performance.
type PerformanceMetrics struct {
	AverageLatency      time.Duration
	SuccessRate         float64
	ResourceEfficiency  float64
	OptimizationCount   int
	LastOptimized       time.Time
}

// NewIntelligenceLayer creates a new autonomous intelligence layer.
func NewIntelligenceLayer(
	deepResearchClient *research.DeepResearchClient,
	deepAgentsURL string,
	unifiedWorkflowURL string,
	gooseEnabled bool,
	logger *log.Logger,
) *IntelligenceLayer {
	kb := NewKnowledgeBase()
	learningEngine := NewLearningEngine(kb, logger)
	optimizationEngine := NewOptimizationEngine(logger)
	predictiveEngine := NewPredictiveEngine(logger)
	governanceEngine := NewGovernanceEngine(logger)
	agentRegistry := NewAgentRegistry(kb, logger)

	return &IntelligenceLayer{
		deepResearchClient:  deepResearchClient,
		deepAgentsURL:       deepAgentsURL,
		unifiedWorkflowURL:  unifiedWorkflowURL,
		gooseEnabled:        gooseEnabled,
		learningEngine:      learningEngine,
		optimizationEngine:  optimizationEngine,
		predictiveEngine:    predictiveEngine,
		governanceEngine:    governanceEngine,
		agentRegistry:       agentRegistry,
		knowledgeBase:       kb,
		httpClient:          &http.Client{Timeout: 300 * time.Second},
		logger:              logger,
		activeTasks:         make(map[string]*TaskState),
		performanceMetrics:  &PerformanceMetrics{},
	}
}

// NewLearningEngine creates a new learning engine.
func NewLearningEngine(kb *KnowledgeBase, logger *log.Logger) *LearningEngine {
	return &LearningEngine{
		knowledgeBase: kb,
		patternStore: NewPatternStore(),
		logger:       logger,
	}
}

// NewOptimizationEngine creates a new optimization engine.
func NewOptimizationEngine(logger *log.Logger) *OptimizationEngine {
	return &OptimizationEngine{
		performanceMonitor: NewPerformanceMonitor(),
		optimizationRules:   []OptimizationRule{},
		logger:              logger,
	}
}

// NewPredictiveEngine creates a new predictive engine.
func NewPredictiveEngine(logger *log.Logger) *PredictiveEngine {
	return &PredictiveEngine{
		models:         make(map[string]*PredictionModel),
		historicalData: NewHistoricalDataStore(),
		logger:         logger,
	}
}

// NewGovernanceEngine creates a new governance engine.
func NewGovernanceEngine(logger *log.Logger) *GovernanceEngine {
	return &GovernanceEngine{
		policies:          []GovernancePolicy{},
		complianceChecker: NewComplianceChecker(),
		logger:            logger,
	}
}

// NewAgentRegistry creates a new agent registry.
func NewAgentRegistry(kb *KnowledgeBase, logger *log.Logger) *AgentRegistry {
	return &AgentRegistry{
		agents:          make(map[string]*Agent),
		interactions:    []AgentInteraction{},
		knowledgeSharing: NewKnowledgeSharing(kb),
	}
}

// NewKnowledgeBase creates a new knowledge base.
func NewKnowledgeBase() *KnowledgeBase {
	return &KnowledgeBase{
		patterns:      make(map[string]*Pattern),
		solutions:     make(map[string]*Solution),
		bestPractices: []BestPractice{},
	}
}

// ExecuteAutonomousTask executes a task autonomously using integrated services.
func (il *IntelligenceLayer) ExecuteAutonomousTask(ctx context.Context, task *AutonomousTask) (*TaskResult, error) {
	taskState := &TaskState{
		TaskID:    task.ID,
		Type:      task.Type,
		Status:    "started",
		StartedAt: time.Now(),
		Progress:  0.0,
	}

	il.mu.Lock()
	il.activeTasks[task.ID] = taskState
	il.mu.Unlock()

	defer func() {
		il.mu.Lock()
		delete(il.activeTasks, task.ID)
		il.mu.Unlock()
	}()

	// Step 1: Use Deep Research to understand context
	taskState.Progress = 0.1
	var researchResult interface{}
	researchReport, err := il.deepResearchClient.ResearchMetadata(ctx, task.Query, true, true)
	if err != nil {
		il.logger.Printf("Deep Research failed (non-fatal): %v", err)
		// Continue without research if Deep Research is unavailable
		researchResult = nil
	} else {
		researchResult = researchReport
	}

	// Step 2: Use DeepAgents to plan and decompose task
	taskState.Progress = 0.3
	plan, err := il.planWithDeepAgents(ctx, task, researchResult)
	if err != nil {
		return nil, fmt.Errorf("planning failed: %w", err)
	}

	// Step 3: Execute plan using Unified Workflow
	taskState.Progress = 0.5
	result, err := il.executePlanWithUnifiedWorkflow(ctx, plan, task)
	if err != nil {
		return nil, fmt.Errorf("execution failed: %w", err)
	}

	// Step 4: Learn from execution
	taskState.Progress = 0.8
	il.learnFromExecution(task, result, err)

	// Step 5: Optimize based on results
	taskState.Progress = 0.9
	il.optimizeBasedOnResults(task, result)

	taskState.Progress = 1.0
	taskState.Status = "completed"
	taskState.Result = result

	return &TaskResult{
		TaskID:    task.ID,
		Success:   err == nil,
		Result:    result,
		Learned:   il.learningEngine.GetRecentLessons(),
		Optimized: il.optimizationEngine.GetRecentOptimizations(),
	}, nil
}

// planWithDeepAgents uses DeepAgents to plan and decompose the task.
func (il *IntelligenceLayer) planWithDeepAgents(ctx context.Context, task *AutonomousTask, researchResult interface{}) (*ExecutionPlan, error) {
	// Build planning prompt
	prompt := fmt.Sprintf(`Plan and decompose this autonomous task:

Task: %s
Type: %s
Context: %v

Research Context: %v

Please create a detailed plan with:
1. Task decomposition into subtasks
2. Agent selection and coordination
3. Resource requirements
4. Expected outcomes
5. Risk assessment

Use the planning and todo tools to break this down.`, task.Description, task.Type, task.Context, researchResult)

	// Call DeepAgents
	reqBody := map[string]interface{}{
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": prompt,
			},
		},
	}

	reqData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("%s/invoke", il.deepAgentsURL), 
		bytes.NewReader(reqData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := il.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	var deepAgentsResponse map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&deepAgentsResponse); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	// Extract plan from DeepAgents response
	plan := il.extractPlanFromDeepAgents(deepAgentsResponse)
	return plan, nil
}

// executePlanWithUnifiedWorkflow executes the plan using the unified workflow.
func (il *IntelligenceLayer) executePlanWithUnifiedWorkflow(ctx context.Context, plan *ExecutionPlan, task *AutonomousTask) (interface{}, error) {
	// Build unified workflow request based on plan
	workflowReq := map[string]interface{}{
		"unified_request": map[string]interface{}{
			"workflow_mode": plan.WorkflowMode,
		},
	}

	// Add knowledge graph request if needed
	if plan.NeedsKnowledgeGraph {
		workflowReq["unified_request"].(map[string]interface{})["knowledge_graph_request"] = map[string]interface{}{
			"query": task.Query,
			"context": task.Context,
		}
	}

	// Add orchestration request if needed
	if plan.NeedsOrchestration {
		workflowReq["unified_request"].(map[string]interface{})["orchestration_request"] = map[string]interface{}{
			"chain_name": plan.OrchestrationChain,
			"inputs":     plan.OrchestrationInputs,
		}
	}

	// Add agentflow request if needed
	if plan.NeedsAgentFlow {
		workflowReq["unified_request"].(map[string]interface{})["agentflow_request"] = map[string]interface{}{
			"flow_id": plan.AgentFlowID,
			"inputs":   plan.AgentFlowInputs,
		}
	}

	// Execute unified workflow
	reqData, err := json.Marshal(workflowReq)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("%s/unified/process", il.unifiedWorkflowURL),
		bytes.NewReader(reqData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := il.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return result, nil
}

// learnFromExecution learns from task execution results.
func (il *IntelligenceLayer) learnFromExecution(task *AutonomousTask, result interface{}, err error) {
	// Extract lessons learned
	lesson := &Lesson{
		ID:        fmt.Sprintf("lesson_%s_%d", task.ID, time.Now().Unix()),
		Type:      "success",
		Context:   task.Context,
		Timestamp: time.Now(),
	}

	if err != nil {
		lesson.Type = "failure"
		lesson.Insight = fmt.Sprintf("Task failed: %v", err)
		lesson.Recommendation = "Review error and adjust approach"
	} else {
		lesson.Insight = "Task completed successfully"
		lesson.Recommendation = "Apply similar approach to similar tasks"
	}

	// Store lesson in learning engine
	il.learningEngine.RecordLesson(lesson)

	// Update agent performance
	if task.AgentID != "" {
		il.agentRegistry.RecordOutcome(task.AgentID, err == nil)
	}

	// Extract patterns
	pattern := il.learningEngine.ExtractPattern(task, result, err)
	if pattern != nil {
		il.knowledgeBase.AddPattern(pattern)
	}
}

// optimizeBasedOnResults optimizes system based on execution results.
func (il *IntelligenceLayer) optimizeBasedOnResults(task *AutonomousTask, result interface{}) {
	// Check for optimization opportunities
	optimizations := il.optimizationEngine.IdentifyOptimizations(task, result)
	for _, opt := range optimizations {
		il.optimizationEngine.ApplyOptimization(opt)
	}

	// Update performance metrics
	il.performanceMetrics.LastOptimized = time.Now()
	il.performanceMetrics.OptimizationCount++
}

// AutonomousTask represents a task for autonomous execution.
type AutonomousTask struct {
	ID          string
	Type        string
	Description string
	Query       string
	Context     map[string]interface{}
	AgentID     string
	Priority    int
}

// TaskResult represents the result of an autonomous task.
type TaskResult struct {
	TaskID    string
	Success   bool
	Result    interface{}
	Learned   []Lesson
	Optimized []Optimization
}

// ExecutionPlan represents a plan for task execution.
type ExecutionPlan struct {
	WorkflowMode         string
	NeedsKnowledgeGraph  bool
	NeedsOrchestration   bool
	NeedsAgentFlow       bool
	OrchestrationChain   string
	OrchestrationInputs  map[string]interface{}
	AgentFlowID          string
	AgentFlowInputs       map[string]interface{}
	Subtasks              []Subtask
}

// Subtask represents a subtask in the execution plan.
type Subtask struct {
	ID          string
	Description string
	AgentID     string
	Order       int
}

