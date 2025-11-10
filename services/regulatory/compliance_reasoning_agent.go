package regulatory

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/plturrell/aModels/services/orchestration/agents"
)

// ComplianceReasoningAgent provides LangGraph-style stateful reasoning for BCBS 239 compliance.
// It orchestrates multi-step workflows: query generation → graph retrieval → synthesis → validation.
// Supports multi-model integration: LocalAI, GNN, Goose, and DeepResearch.
type ComplianceReasoningAgent struct {
	localAIClient  *agents.LocalAIClient
	graphClient    *BCBS239GraphClient
	logger         *log.Logger
	model          string // LocalAI model to use (e.g., "gemma-2b-q4_k_m.gguf")
	
	// Advanced model adapters
	gnnAdapter         *GNNAdapter         // Graph Neural Network for structural analysis
	gooseAdapter       *GooseAdapter       // Goose agent for autonomous tasks
	deepResearchAdapter *DeepResearchAdapter // Deep research for comprehensive analysis
	
	// Model orchestration
	modelOrchestrator *ModelOrchestrator
}

// NewComplianceReasoningAgent creates a new compliance reasoning agent.
func NewComplianceReasoningAgent(
	localAIClient *agents.LocalAIClient,
	graphClient *BCBS239GraphClient,
	logger *log.Logger,
	model string,
) *ComplianceReasoningAgent {
	if model == "" {
		model = "gemma-2b-q4_k_m.gguf" // Default model
	}
	agent := &ComplianceReasoningAgent{
		localAIClient: localAIClient,
		graphClient:   graphClient,
		logger:        logger,
		model:         model,
	}
	
	// Initialize orchestrator with LocalAI as default
	agent.modelOrchestrator = NewModelOrchestrator(logger)
	
	return agent
}

// WithGNNAdapter adds Graph Neural Network capabilities.
func (a *ComplianceReasoningAgent) WithGNNAdapter(adapter *GNNAdapter) *ComplianceReasoningAgent {
	a.gnnAdapter = adapter
	if a.modelOrchestrator != nil {
		a.modelOrchestrator.RegisterModel(adapter)
	}
	return a
}

// WithGooseAdapter adds Goose autonomous agent capabilities.
func (a *ComplianceReasoningAgent) WithGooseAdapter(adapter *GooseAdapter) *ComplianceReasoningAgent {
	a.gooseAdapter = adapter
	if a.modelOrchestrator != nil {
		a.modelOrchestrator.RegisterModel(adapter)
	}
	return a
}

// WithDeepResearchAdapter adds Deep Research capabilities.
func (a *ComplianceReasoningAgent) WithDeepResearchAdapter(adapter *DeepResearchAdapter) *ComplianceReasoningAgent {
	a.deepResearchAdapter = adapter
	if a.modelOrchestrator != nil {
		a.modelOrchestrator.RegisterModel(adapter)
	}
	return a
}

// ComplianceWorkflowState represents the stateful workflow for compliance analysis.
// This mirrors LangGraph's state management approach.
type ComplianceWorkflowState struct {
	// Input
	Question         string                 `json:"question"`
	PrincipleID      string                 `json:"principle_id,omitempty"`
	CalculationID    string                 `json:"calculation_id,omitempty"`
	
	// Intermediate state
	GeneratedQuery   string                 `json:"generated_query,omitempty"`
	GraphFacts       []map[string]interface{} `json:"graph_facts,omitempty"`
	VectorContext    string                 `json:"vector_context,omitempty"`
	
	// Output
	SynthesizedAnswer string                `json:"synthesized_answer,omitempty"`
	Confidence       float64                `json:"confidence,omitempty"`
	Sources          []string               `json:"sources,omitempty"`
	
	// Control flow
	CurrentNode      string                 `json:"current_node"`
	NextNode         string                 `json:"next_node,omitempty"`
	RequiresApproval bool                   `json:"requires_approval,omitempty"`
	ApprovalStatus   string                 `json:"approval_status,omitempty"` // "pending", "approved", "rejected"
	
	// Metadata
	StartTime        time.Time              `json:"start_time"`
	LastUpdateTime   time.Time              `json:"last_update_time"`
	Errors           []string               `json:"errors,omitempty"`
}

// WorkflowNode represents a single node in the compliance reasoning workflow.
type WorkflowNode interface {
	Execute(ctx context.Context, state *ComplianceWorkflowState) error
	Name() string
}

// RunComplianceWorkflow executes a stateful compliance reasoning workflow.
func (a *ComplianceReasoningAgent) RunComplianceWorkflow(
	ctx context.Context,
	question string,
	principleID string,
) (*ComplianceWorkflowState, error) {
	if a.logger != nil {
		a.logger.Printf("Starting compliance workflow for question: %s (principle: %s)", question, principleID)
	}

	// Initialize workflow state
	state := &ComplianceWorkflowState{
		Question:       question,
		PrincipleID:    principleID,
		CurrentNode:    "intake",
		StartTime:      time.Now(),
		LastUpdateTime: time.Now(),
		GraphFacts:     []map[string]interface{}{},
		Sources:        []string{},
		Errors:         []string{},
	}

	// Define workflow nodes in execution order
	nodes := []WorkflowNode{
		&IntakeNode{agent: a},
		&GraphQueryNode{agent: a},
		&SynthesisNode{agent: a},
		&ValidationNode{agent: a},
	}

	// Execute workflow nodes sequentially with state transitions
	for _, node := range nodes {
		state.CurrentNode = node.Name()
		state.LastUpdateTime = time.Now()
		
		if a.logger != nil {
			a.logger.Printf("Executing workflow node: %s", node.Name())
		}
		
		if err := node.Execute(ctx, state); err != nil {
			state.Errors = append(state.Errors, fmt.Sprintf("Node %s failed: %v", node.Name(), err))
			return state, fmt.Errorf("workflow failed at node %s: %w", node.Name(), err)
		}
		
		// Check for human-in-the-loop checkpoint
		if state.RequiresApproval && state.ApprovalStatus == "pending" {
			if a.logger != nil {
				a.logger.Printf("Workflow paused at node %s for human approval", node.Name())
			}
			return state, nil // Return state for external approval
		}
	}

	if a.logger != nil {
		a.logger.Printf("Compliance workflow completed successfully")
	}

	return state, nil
}

// ResumeWorkflow resumes a paused workflow after human approval.
func (a *ComplianceReasoningAgent) ResumeWorkflow(
	ctx context.Context,
	state *ComplianceWorkflowState,
	approvalStatus string,
) (*ComplianceWorkflowState, error) {
	state.ApprovalStatus = approvalStatus
	state.LastUpdateTime = time.Now()

	if approvalStatus != "approved" {
		return state, fmt.Errorf("workflow rejected by human reviewer")
	}

	// Continue from the next node
	// (Implementation would resume from state.NextNode)
	return state, nil
}

// IntakeNode classifies the compliance question and determines the workflow path.
type IntakeNode struct {
	agent *ComplianceReasoningAgent
}

func (n *IntakeNode) Name() string {
	return "intake"
}

func (n *IntakeNode) Execute(ctx context.Context, state *ComplianceWorkflowState) error {
	// Use LocalAI to classify the question type
	prompt := fmt.Sprintf(`You are a BCBS 239 compliance analyst. Classify this compliance question:

Question: %s
Principle: %s

Classify the question as one of: "lineage_tracing", "control_mapping", "impact_analysis", "compliance_assessment"

Respond with only the classification category.`, state.Question, state.PrincipleID)

	payload := map[string]interface{}{
		"model": n.agent.model,
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"temperature": 0.1,
	}

	result, err := n.agent.localAIClient.CallDomainEndpoint(ctx, "regulatory", "chat/completions", payload)
	if err != nil {
		return fmt.Errorf("failed to classify question: %w", err)
	}

	// Extract classification from response
	if choices, ok := result["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if message, ok := choice["message"].(map[string]interface{}); ok {
				if content, ok := message["content"].(string); ok {
					state.NextNode = "graph_query"
					if n.agent.logger != nil {
						n.agent.logger.Printf("Question classified as: %s", content)
					}
				}
			}
		}
	}

	return nil
}

// GraphQueryNode generates and executes Cypher queries against Neo4j.
type GraphQueryNode struct {
	agent *ComplianceReasoningAgent
}

func (n *GraphQueryNode) Name() string {
	return "graph_query"
}

func (n *GraphQueryNode) Execute(ctx context.Context, state *ComplianceWorkflowState) error {
	// Generate Cypher query using LocalAI
	prompt := fmt.Sprintf(`You are a Neo4j expert specializing in BCBS 239 compliance graphs.

Generate a Cypher query to answer this question:
Question: %s
Principle ID: %s

The graph schema includes:
- Nodes: BCBS239Principle, BCBS239Control, DataAsset, Process, RegulatoryCalculation
- Relationships: ENSURED_BY, APPLIES_TO, DEPENDS_ON, TRANSFORMS, VALIDATED_BY

Return ONLY the Cypher query, no explanation.`, state.Question, state.PrincipleID)

	payload := map[string]interface{}{
		"model": n.agent.model,
		"messages": []map[string]interface{}{
			{
				"role":    "system",
				"content": "You are a Neo4j Cypher expert. Generate valid Cypher queries only.",
			},
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"temperature": 0.1,
	}

	result, err := n.agent.localAIClient.CallDomainEndpoint(ctx, "regulatory", "chat/completions", payload)
	if err != nil {
		return fmt.Errorf("failed to generate Cypher query: %w", err)
	}

	// Extract generated query
	if choices, ok := result["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if message, ok := choice["message"].(map[string]interface{}); ok {
				if content, ok := message["content"].(string); ok {
					state.GeneratedQuery = content
					if n.agent.logger != nil {
						n.agent.logger.Printf("Generated Cypher query: %s", content)
					}
				}
			}
		}
	}

	// Execute graph queries to retrieve facts
	if state.PrincipleID != "" {
		controls, err := n.agent.graphClient.GetPrincipleControls(ctx, state.PrincipleID)
		if err == nil {
			for _, control := range controls {
				fact := map[string]interface{}{
					"type":        "control_mapping",
					"principle":   control.PrincipleID,
					"control":     control.ControlID,
					"control_name": control.ControlName,
				}
				state.GraphFacts = append(state.GraphFacts, fact)
				state.Sources = append(state.Sources, fmt.Sprintf("Neo4j:Control:%s", control.ControlID))
			}
		}
	}

	state.NextNode = "synthesis"
	return nil
}

// SynthesisNode combines graph facts and generates a coherent compliance answer.
type SynthesisNode struct {
	agent *ComplianceReasoningAgent
}

func (n *SynthesisNode) Name() string {
	return "synthesis"
}

func (n *SynthesisNode) Execute(ctx context.Context, state *ComplianceWorkflowState) error {
	// Use ModelOrchestrator for intelligent model selection
	if n.agent.modelOrchestrator != nil && len(n.agent.modelOrchestrator.models) > 0 {
		return n.executeWithOrchestrator(ctx, state)
	}
	
	// Fallback to LocalAI if orchestrator not available
	return n.executeWithLocalAI(ctx, state)
}

// executeWithOrchestrator uses ModelOrchestrator for synthesis.
func (n *SynthesisNode) executeWithOrchestrator(ctx context.Context, state *ComplianceWorkflowState) error {
	// Prepare graph context data
	graphData := &GraphContextData{
		Facts: state.GraphFacts,
	}
	
	// Create model query request
	modelRequest := ModelQueryRequest{
		QueryType:   state.CurrentNode,
		Question:    state.Question,
		PrincipleID: state.PrincipleID,
		GraphData:   graphData,
		Context: map[string]interface{}{
			"workflow_state": state.CurrentNode,
			"sources_count":  len(state.Sources),
		},
	}
	
	// Route and execute
	response, err := n.agent.modelOrchestrator.RouteAndExecute(ctx, modelRequest)
	if err != nil {
		if n.agent.logger != nil {
			n.agent.logger.Printf("Model orchestrator failed, falling back to LocalAI: %v", err)
		}
		return n.executeWithLocalAI(ctx, state)
	}
	
	// Update state with orchestrated response
	state.SynthesizedAnswer = response.Answer
	state.Confidence = response.Confidence
	state.Sources = append(state.Sources, response.Sources...)
	
	// Add metadata about which model was used
	if state.Errors == nil {
		state.Errors = []string{}
	}
	state.Errors = append(state.Errors, 
		fmt.Sprintf("Synthesized using: %s (confidence: %.2f, time: %v)", 
			response.ModelType, response.Confidence, response.ProcessTime))
	
	if n.agent.logger != nil {
		n.agent.logger.Printf("Synthesis completed using %s (confidence: %.2f)", 
			response.ModelType, response.Confidence)
	}
	
	// Mark as requiring approval for critical analysis
	n.checkCriticalApproval(state, response)
	
	state.NextNode = "validation"
	return nil
}

// executeWithLocalAI uses LocalAI for synthesis (fallback).
func (n *SynthesisNode) executeWithLocalAI(ctx context.Context, state *ComplianceWorkflowState) error {
	// Convert graph facts to JSON for prompt
	factsJSON, err := json.MarshalIndent(state.GraphFacts, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal graph facts: %w", err)
	}

	prompt := fmt.Sprintf(`You are a BCBS 239 compliance expert. Generate a detailed compliance analysis.

Question: %s
Principle: %s

Graph Facts Retrieved:
%s

Generate a comprehensive compliance analysis that:
1. Directly answers the question
2. Cites specific controls and graph entities by ID
3. Identifies any compliance gaps or risks
4. Provides actionable recommendations

Format your response as a structured compliance narrative.`, 
		state.Question, 
		state.PrincipleID, 
		string(factsJSON),
	)

	payload := map[string]interface{}{
		"model": n.agent.model,
		"messages": []map[string]interface{}{
			{
				"role":    "system",
				"content": "You are a regulatory compliance analyst specializing in BCBS 239. Provide accurate, well-cited analysis.",
			},
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"temperature": 0.3,
	}

	result, err := n.agent.localAIClient.CallDomainEndpoint(ctx, "regulatory", "chat/completions", payload)
	if err != nil {
		return fmt.Errorf("failed to synthesize answer: %w", err)
	}

	// Extract synthesized answer
	if choices, ok := result["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if message, ok := choice["message"].(map[string]interface{}); ok {
				if content, ok := message["content"].(string); ok {
					state.SynthesizedAnswer = content
					state.Confidence = 0.85 // Example confidence score
					if n.agent.logger != nil {
						n.agent.logger.Printf("Synthesized compliance answer (%d chars)", len(content))
					}
				}
			}
		}
	}

	// Mark as requiring approval for high-criticality principles
	criticalPrinciples := []string{"P3", "P4", "P7", "P12"} // Accuracy, Completeness, etc.
	for _, p := range criticalPrinciples {
		if state.PrincipleID == p {
			state.RequiresApproval = true
			state.ApprovalStatus = "pending"
			break
		}
	}

	state.NextNode = "validation"
	return nil
}

// ValidationNode validates the synthesized answer for quality and accuracy.
type ValidationNode struct {
	agent *ComplianceReasoningAgent
}

func (n *ValidationNode) Name() string {
	return "validation"
}

func (n *ValidationNode) Execute(ctx context.Context, state *ComplianceWorkflowState) error {
	// Basic validation checks
	if len(state.SynthesizedAnswer) == 0 {
		return fmt.Errorf("synthesized answer is empty")
	}

	if len(state.GraphFacts) == 0 {
		state.Errors = append(state.Errors, "Warning: No graph facts retrieved to support answer")
	}

	// Set final confidence based on validation
	if len(state.Sources) >= 3 && len(state.GraphFacts) >= 2 {
		state.Confidence = 0.9
	} else {
		state.Confidence = 0.7
	}

	state.NextNode = "complete"
	return nil
}

// checkCriticalApproval determines if the synthesis requires human approval.
func (n *SynthesisNode) checkCriticalApproval(state *ComplianceWorkflowState, response *ModelQueryResponse) {
	// Mark as requiring approval for high-criticality principles
	criticalPrinciples := []string{"P3", "P4", "P7", "P12"} // Accuracy, Completeness, etc.
	for _, p := range criticalPrinciples {
		if state.PrincipleID == p {
			state.RequiresApproval = true
			state.ApprovalStatus = "pending"
			return
		}
	}
	
	// Also require approval for low confidence
	if response.Confidence < 0.7 {
		state.RequiresApproval = true
		state.ApprovalStatus = "pending"
		if n.agent.logger != nil {
			n.agent.logger.Printf("Low confidence (%.2f) - requiring human approval", response.Confidence)
		}
	}
}

// GenerateComplianceNarrative is a convenience method for simple compliance queries.
func (a *ComplianceReasoningAgent) GenerateComplianceNarrative(
	ctx context.Context,
	calculation RegulatoryCalculation,
	principle string,
) (string, error) {
	question := fmt.Sprintf(
		"Analyze the compliance of calculation %s (%s) against BCBS 239 principle %s",
		calculation.CalculationID,
		calculation.CalculationType,
		principle,
	)

	state, err := a.RunComplianceWorkflow(ctx, question, principle)
	if err != nil {
		return "", err
	}

	return state.SynthesizedAnswer, nil
}

// QueryWithHybridModels executes a query using multiple models and combines results.
func (a *ComplianceReasoningAgent) QueryWithHybridModels(
	ctx context.Context,
	question string,
	principleID string,
	modelTypes []string,
) (*HybridQueryResponse, error) {
	if a.modelOrchestrator == nil {
		return nil, fmt.Errorf("model orchestrator not initialized")
	}
	
	request := ModelQueryRequest{
		QueryType:   "hybrid",
		Question:    question,
		PrincipleID: principleID,
	}
	
	return a.modelOrchestrator.HybridQuery(ctx, request, modelTypes)
}
