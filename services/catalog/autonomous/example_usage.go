package autonomous

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/plturrell/aModels/services/catalog/research"
)

// ExampleUsage demonstrates how to use the Autonomous Intelligence Layer.
func ExampleUsage() {
	// Initialize logger
	logger := log.New(os.Stdout, "[autonomous] ", log.LstdFlags)

	// Initialize Deep Research client
	deepResearchURL := os.Getenv("DEEP_RESEARCH_URL")
	if deepResearchURL == "" {
		deepResearchURL = "http://localhost:8085"
	}
	deepResearchClient := research.NewDeepResearchClient(deepResearchURL, logger)

	// Initialize Autonomous Intelligence Layer
	deepAgentsURL := os.Getenv("DEEPAGENTS_URL")
	if deepAgentsURL == "" {
		deepAgentsURL = "http://deepagents-service:9004"
	}
	unifiedWorkflowURL := os.Getenv("GRAPH_SERVICE_URL")
	if unifiedWorkflowURL == "" {
		unifiedWorkflowURL = "http://graph-service:8081"
	}

	intelligenceLayer := NewIntelligenceLayer(
		deepResearchClient,
		deepAgentsURL,
		unifiedWorkflowURL,
		true, // Goose enabled
		logger,
	)

	// Example 1: Execute autonomous data quality analysis
	ctx := context.Background()
	task := &AutonomousTask{
		ID:          fmt.Sprintf("task_%d", time.Now().Unix()),
		Type:        "data_quality_analysis",
		Description: "Analyze data quality for customer data",
		Query:       "What are the data quality issues for customer data?",
		Context: map[string]interface{}{
			"domain":   "customer",
			"priority":  "high",
			"include_lineage": true,
			"include_quality":  true,
		},
		Priority: 1,
	}

	fmt.Println("Executing autonomous task:", task.ID)
	result, err := intelligenceLayer.ExecuteAutonomousTask(ctx, task)
	if err != nil {
		logger.Printf("Task execution failed: %v", err)
		return
	}

	fmt.Printf("Task completed successfully: %v\n", result.Success)
	fmt.Printf("Lessons learned: %d\n", len(result.Learned))
	fmt.Printf("Optimizations applied: %d\n", len(result.Optimized))

	// Example 2: Register an agent and track performance
	agentID := "data_ingestion_agent_001"
	intelligenceLayer.agentRegistry.RecordOutcome(agentID, true)
	intelligenceLayer.agentRegistry.RecordOutcome(agentID, true)
	intelligenceLayer.agentRegistry.RecordOutcome(agentID, false)

	// Example 3: Add a learned pattern to knowledge base
	pattern := &Pattern{
		ID:          fmt.Sprintf("pattern_%d", time.Now().Unix()),
		Description: "Pattern for successful data quality analysis",
		Context: map[string]interface{}{
			"task_type": "data_quality_analysis",
			"domain":    "customer",
		},
		SuccessRate: 0.95,
		UsageCount:  1,
		LastUsed:    time.Now(),
	}
	intelligenceLayer.knowledgeBase.AddPattern(pattern)

	// Example 4: Get performance metrics
	metrics := intelligenceLayer.performanceMetrics
	fmt.Printf("Performance Metrics:\n")
	fmt.Printf("  Optimization Count: %d\n", metrics.OptimizationCount)
	fmt.Printf("  Last Optimized: %v\n", metrics.LastOptimized)

	// Example 5: Record a lesson learned
	lesson := &Lesson{
		ID:        fmt.Sprintf("lesson_%d", time.Now().Unix()),
		Type:      "success",
		Context:   task.Context,
		Insight:   "Data quality analysis works best with full lineage context",
		Recommendation: "Always include lineage when analyzing data quality",
		Timestamp: time.Now(),
	}
	intelligenceLayer.learningEngine.RecordLesson(lesson)
}

// ExampleHTTPRequest demonstrates making an HTTP request to the autonomous API.
func ExampleHTTPRequest() {
	// Example JSON request body
	requestBody := map[string]interface{}{
		"task_id":     fmt.Sprintf("task_%d", time.Now().Unix()),
		"type":        "data_quality_analysis",
		"description": "Analyze data quality for customer data",
		"query":       "What are the data quality issues for customer data?",
		"context": map[string]interface{}{
			"domain":   "customer",
			"priority": "high",
		},
		"agent_id":  "data_quality_agent_001",
		"priority":  1,
	}

	jsonData, err := json.MarshalIndent(requestBody, "", "  ")
	if err != nil {
		log.Printf("Error marshaling request: %v", err)
		return
	}

	fmt.Println("Example HTTP Request to POST /api/autonomous/execute:")
	fmt.Println(string(jsonData))
}

// ExampleResponse demonstrates the expected response format.
func ExampleResponse() {
	response := &TaskResult{
		TaskID:  "task_1234567890",
		Success: true,
		Result: map[string]interface{}{
			"data_quality_issues": []string{
				"Missing values in customer_name",
				"Duplicate customer IDs",
			},
			"recommendations": []string{
				"Implement data validation rules",
				"Add uniqueness constraints",
			},
		},
		Learned: []Lesson{
			{
				ID:        "lesson_001",
				Type:      "success",
				Insight:   "Data quality analysis works best with full lineage context",
				Recommendation: "Always include lineage when analyzing data quality",
				Timestamp: time.Now(),
			},
		},
		Optimized: []Optimization{
			{
				ID:          "opt_001",
				Type:        "query_optimization",
				Description: "Optimized knowledge graph query",
				Impact:      0.15, // 15% improvement
				Timestamp:   time.Now(),
			},
		},
	}

	jsonData, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		log.Printf("Error marshaling response: %v", err)
		return
	}

	fmt.Println("Example HTTP Response:")
	fmt.Println(string(jsonData))
}

