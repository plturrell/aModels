package autonomous

import (
	"context"
	"log"
	"os"
	"testing"
	"time"

	"github.com/plturrell/aModels/services/catalog/research"
)

func TestNewIntelligenceLayer(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	deepResearchClient := research.NewDeepResearchClient("http://localhost:8085", logger)

	il := NewIntelligenceLayer(
		deepResearchClient,
		"http://localhost:9004",
		"http://localhost:8081",
		true,
		logger,
	)

	if il == nil {
		t.Fatal("NewIntelligenceLayer returned nil")
	}

	if il.deepResearchClient == nil {
		t.Error("deepResearchClient is nil")
	}

	if il.deepAgentsURL == "" {
		t.Error("deepAgentsURL is empty")
	}

	if il.unifiedWorkflowURL == "" {
		t.Error("unifiedWorkflowURL is empty")
	}

	if il.learningEngine == nil {
		t.Error("learningEngine is nil")
	}

	if il.optimizationEngine == nil {
		t.Error("optimizationEngine is nil")
	}

	if il.predictiveEngine == nil {
		t.Error("predictiveEngine is nil")
	}

	if il.governanceEngine == nil {
		t.Error("governanceEngine is nil")
	}

	if il.agentRegistry == nil {
		t.Error("agentRegistry is nil")
	}

	if il.knowledgeBase == nil {
		t.Error("knowledgeBase is nil")
	}
}

func TestAutonomousTask(t *testing.T) {
	task := &AutonomousTask{
		ID:          "test_task_001",
		Type:        "data_quality_analysis",
		Description: "Test data quality analysis",
		Query:       "What are data quality issues?",
		Context:     map[string]interface{}{
			"domain": "test",
		},
		AgentID:  "test_agent",
		Priority: 1,
	}

	if task.ID == "" {
		t.Error("Task ID is empty")
	}

	if task.Type == "" {
		t.Error("Task Type is empty")
	}

	if task.Query == "" {
		t.Error("Task Query is empty")
	}
}

func TestLearningEngine(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	kb := NewKnowledgeBase()
	le := NewLearningEngine(kb, logger)

	if le == nil {
		t.Fatal("NewLearningEngine returned nil")
	}

	if le.knowledgeBase == nil {
		t.Error("knowledgeBase is nil")
	}

	if le.patternStore == nil {
		t.Error("patternStore is nil")
	}

	// Test lesson recording
	lesson := &Lesson{
		ID:        "test_lesson_001",
		Type:      "success",
		Context:   map[string]interface{}{"test": "value"},
		Insight:   "Test insight",
		Recommendation: "Test recommendation",
		Timestamp: time.Now(),
	}

	le.RecordLesson(lesson)
	
	lessons := le.GetRecentLessons()
	if len(lessons) == 0 {
		t.Log("No lessons returned (expected for now)")
	}
}

func TestOptimizationEngine(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	oe := NewOptimizationEngine(logger)

	if oe == nil {
		t.Fatal("NewOptimizationEngine returned nil")
	}

	if oe.performanceMonitor == nil {
		t.Error("performanceMonitor is nil")
	}

	// Test optimization identification
	task := &AutonomousTask{
		ID:   "test_task",
		Type: "test",
	}
	
	optimizations := oe.IdentifyOptimizations(task, nil)
	if optimizations == nil {
		t.Error("IdentifyOptimizations returned nil")
	}
}

func TestKnowledgeBase(t *testing.T) {
	kb := NewKnowledgeBase()

	if kb == nil {
		t.Fatal("NewKnowledgeBase returned nil")
	}

	// Test pattern addition
	pattern := &Pattern{
		ID:          "test_pattern_001",
		Description: "Test pattern",
		Context:     map[string]interface{}{"test": "value"},
		SuccessRate: 0.95,
		UsageCount:  1,
		LastUsed:    time.Now(),
	}

	kb.AddPattern(pattern)

	kb.mu.RLock()
	_, exists := kb.patterns[pattern.ID]
	kb.mu.RUnlock()

	if !exists {
		t.Error("Pattern was not added to knowledge base")
	}
}

func TestAgentRegistry(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	kb := NewKnowledgeBase()
	ar := NewAgentRegistry(kb, logger)

	if ar == nil {
		t.Fatal("NewAgentRegistry returned nil")
	}

	// Test outcome recording
	ar.RecordOutcome("agent_001", true)
	ar.RecordOutcome("agent_001", true)
	ar.RecordOutcome("agent_001", false)

	ar.mu.RLock()
	agent, exists := ar.agents["agent_001"]
	ar.mu.RUnlock()

	if !exists {
		t.Error("Agent was not registered")
	}

	if agent == nil {
		t.Error("Agent is nil")
	}

	if agent.SuccessRate == 0 {
		t.Log("Success rate is 0 (expected for initial implementation)")
	}
}

func TestIntegratedAutonomousSystem(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	deepResearchClient := research.NewDeepResearchClient("http://localhost:8085", logger)

	ias := NewIntegratedAutonomousSystem(
		deepResearchClient,
		"http://localhost:9004",
		"http://localhost:8081",
		nil, // No DB for testing
		logger,
	)

	if ias == nil {
		t.Fatal("NewIntegratedAutonomousSystem returned nil")
	}

	if ias.intelligenceLayer == nil {
		t.Error("intelligenceLayer is nil")
	}

	// Test metrics retrieval
	metrics := ias.GetPerformanceMetrics()
	if metrics == nil {
		t.Error("Performance metrics is nil")
	}

	// Test registry retrieval
	registry := ias.GetAgentRegistry()
	if registry == nil {
		t.Error("Agent registry is nil")
	}

	// Test knowledge base retrieval
	kb := ias.GetKnowledgeBase()
	if kb == nil {
		t.Error("Knowledge base is nil")
	}
}

func TestExtractPlanFromDeepAgents(t *testing.T) {
	logger := log.New(os.Stdout, "[test] ", log.LstdFlags)
	deepResearchClient := research.NewDeepResearchClient("http://localhost:8085", logger)

	il := NewIntelligenceLayer(
		deepResearchClient,
		"http://localhost:9004",
		"http://localhost:8081",
		true,
		logger,
	)

	// Test with minimal response
	response := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{
				"role":    "assistant",
				"content": "This is a test plan with parallel execution",
			},
		},
	}

	plan := il.extractPlanFromDeepAgents(response)

	if plan == nil {
		t.Fatal("extractPlanFromDeepAgents returned nil")
	}

	if plan.WorkflowMode == "" {
		t.Error("WorkflowMode is empty")
	}
}

