package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/plturrell/aModels/services/graph"
	"github.com/plturrell/aModels/services/orchestration/agents"
	"github.com/plturrell/aModels/services/regulatory"
)

// Example: Using BCBS239 Audit Pipeline for Compliance Insights

func main() {
	ctx := context.Background()
	logger := log.New(os.Stdout, "[Example] ", log.LstdFlags)

	// Example 1: Basic Audit with GNN
	logger.Println("=== Example 1: Basic Audit ===")
	basicAuditExample(ctx, logger)

	// Example 2: Audit with Deep Research
	logger.Println("\n=== Example 2: Deep Research Analysis ===")
	deepResearchExample(ctx, logger)

	// Example 3: Audit with Goose Autonomous Remediation
	logger.Println("\n=== Example 3: Goose Auto-Remediation ===")
	gooseRemediationExample(ctx, logger)

	// Example 4: Full Pipeline (All Models)
	logger.Println("\n=== Example 4: Full Multi-Model Pipeline ===")
	fullPipelineExample(ctx, logger)
}

// Example 1: Basic compliance audit using GNN + Graph
func basicAuditExample(ctx context.Context, logger *log.Logger) {
	// Setup (in production, these would come from config)
	driver := setupNeo4j(ctx)
	defer driver.Close(ctx)

	graphClient := graph.NewNeo4jGraphClient(driver, logger)
	bcbs239GraphClient := regulatory.NewBCBS239GraphClient(driver, graphClient, logger)

	localAIClient := agents.NewLocalAIClient("http://localhost:8080", nil, logger)

	// Create reasoning agent (GNN enabled by default)
	reasoningAgent := regulatory.NewComplianceReasoningAgent(
		localAIClient,
		bcbs239GraphClient,
		logger,
		"gemma-2b-q4_k_m.gguf",
	).WithGNNAdapter(regulatory.NewGNNAdapter("http://localhost:8081", logger))

	// Create pipeline
	pipeline := regulatory.NewBCBS239AuditPipeline(
		reasoningAgent,
		bcbs239GraphClient,
		logger,
	)

	// Run basic audit
	request := regulatory.AuditRequest{
		AuditID:          "basic-audit-001",
		Principles:       []string{"P3", "P4"},
		Scope:            "quick",
		ReportPeriod:     "2024-Q4",
		UseGoose:         false, // No Goose for basic audit
		UseDeepResearch:  false, // No research for quick audit
		GenerateInsights: true,
	}

	result, err := pipeline.RunAudit(ctx, request)
	if err != nil {
		logger.Fatalf("Audit failed: %v", err)
	}

	// Display results
	logger.Printf("Audit completed: %s", result.ComplianceStatus)
	logger.Printf("Overall score: %.2f%%", result.OverallScore*100)
	
	for _, audit := range result.PrincipleAudits {
		logger.Printf("  %s: %.1f%% (%d gaps identified)",
			audit.PrincipleID, audit.ComplianceScore*100, audit.ControlsMissing)
	}
}

// Example 2: Deep Research for comprehensive regulatory analysis
func deepResearchExample(ctx context.Context, logger *log.Logger) {
	driver := setupNeo4j(ctx)
	defer driver.Close(ctx)

	graphClient := graph.NewNeo4jGraphClient(driver, logger)
	bcbs239GraphClient := regulatory.NewBCBS239GraphClient(driver, graphClient, logger)

	localAIClient := agents.NewLocalAIClient("http://localhost:8080", nil, logger)

	// Create reasoning agent WITH Deep Research
	reasoningAgent := regulatory.NewComplianceReasoningAgent(
		localAIClient,
		bcbs239GraphClient,
		logger,
		"gemma-2b-q4_k_m.gguf",
	).
		WithGNNAdapter(regulatory.NewGNNAdapter("http://localhost:8081", logger)).
		WithDeepResearchAdapter(regulatory.NewDeepResearchAdapter("http://localhost:8083", logger))

	pipeline := regulatory.NewBCBS239AuditPipeline(
		reasoningAgent,
		bcbs239GraphClient,
		logger,
	)

	// Run audit with deep research enabled
	request := regulatory.AuditRequest{
		AuditID:          "research-audit-001",
		Principles:       []string{"P3", "P4", "P7"},
		Scope:            "full",
		ReportPeriod:     "2024-Q4",
		UseGoose:         false,
		UseDeepResearch:  true, // ENABLE Deep Research
		GenerateInsights: true,
	}

	result, err := pipeline.RunAudit(ctx, request)
	if err != nil {
		logger.Fatalf("Audit failed: %v", err)
	}

	// Display research findings
	logger.Printf("Research reports generated: %d", len(result.ResearchReports))
	
	for _, report := range result.ResearchReports {
		logger.Printf("\nResearch: %s", report.Topic)
		logger.Printf("  Confidence: %.1f%%", report.Confidence*100)
		logger.Printf("  Sources: %d", len(report.Sources))
		logger.Printf("  Summary: %s", truncate(report.Summary, 200))
	}

	// Show insights derived from research
	for _, insight := range result.Insights {
		logger.Printf("\nInsight [%s]: %s", insight.Severity, insight.Title)
		logger.Printf("  %s", insight.Description)
	}
}

// Example 3: Goose autonomous remediation script generation
func gooseRemediationExample(ctx context.Context, logger *log.Logger) {
	driver := setupNeo4j(ctx)
	defer driver.Close(ctx)

	graphClient := graph.NewNeo4jGraphClient(driver, logger)
	bcbs239GraphClient := regulatory.NewBCBS239GraphClient(driver, graphClient, logger)

	localAIClient := agents.NewLocalAIClient("http://localhost:8080", nil, logger)

	// Create reasoning agent WITH Goose
	reasoningAgent := regulatory.NewComplianceReasoningAgent(
		localAIClient,
		bcbs239GraphClient,
		logger,
		"gemma-2b-q4_k_m.gguf",
	).
		WithGNNAdapter(regulatory.NewGNNAdapter("http://localhost:8081", logger)).
		WithGooseAdapter(regulatory.NewGooseAdapter("http://localhost:8082", logger))

	pipeline := regulatory.NewBCBS239AuditPipeline(
		reasoningAgent,
		bcbs239GraphClient,
		logger,
	)

	// Run audit with Goose auto-remediation
	request := regulatory.AuditRequest{
		AuditID:          "goose-audit-001",
		Principles:       []string{"P3", "P4"},
		Scope:            "targeted",
		ReportPeriod:     "2024-Q4",
		UseGoose:         true, // ENABLE Goose
		UseDeepResearch:  false,
		GenerateInsights: true,
		AutoRemediate:    true, // Let Goose generate scripts
	}

	result, err := pipeline.RunAudit(ctx, request)
	if err != nil {
		logger.Fatalf("Audit failed: %v", err)
	}

	// Display Goose-generated remediation plans
	logger.Printf("Remediation plans generated: %d", len(result.Recommendations))
	
	for i, remedy := range result.Recommendations {
		logger.Printf("\nRemediation %d [%s priority]:", i+1, remedy.Priority)
		logger.Printf("  Principle: %s", remedy.PrincipleID)
		logger.Printf("  Gap: %s", remedy.Gap)
		logger.Printf("  Estimated Effort: %s", remedy.EstimatedEffort)
		
		if remedy.AutomationScript != "" {
			logger.Printf("  ✓ Goose generated automation script:")
			logger.Printf("    %s", truncate(remedy.AutomationScript, 500))
		}
	}

	// Show all generated scripts
	logger.Printf("\nTotal scripts generated by Goose: %d", len(result.GooseGeneratedScripts))
}

// Example 4: Full pipeline with all models (production scenario)
func fullPipelineExample(ctx context.Context, logger *log.Logger) {
	driver := setupNeo4j(ctx)
	defer driver.Close(ctx)

	graphClient := graph.NewNeo4jGraphClient(driver, logger)
	bcbs239GraphClient := regulatory.NewBCBS239GraphClient(driver, graphClient, logger)

	localAIClient := agents.NewLocalAIClient("http://localhost:8080", nil, logger)

	// Create fully-loaded reasoning agent (ALL models)
	reasoningAgent := regulatory.NewComplianceReasoningAgent(
		localAIClient,
		bcbs239GraphClient,
		logger,
		"gemma-2b-q4_k_m.gguf",
	).
		WithGNNAdapter(regulatory.NewGNNAdapter("http://localhost:8081", logger)).
		WithGooseAdapter(regulatory.NewGooseAdapter("http://localhost:8082", logger)).
		WithDeepResearchAdapter(regulatory.NewDeepResearchAdapter("http://localhost:8083", logger))

	pipeline := regulatory.NewBCBS239AuditPipeline(
		reasoningAgent,
		bcbs239GraphClient,
		logger,
	)

	// Full comprehensive audit
	request := regulatory.AuditRequest{
		AuditID:          "full-audit-001",
		Principles:       []string{"P3", "P4", "P7", "P12"}, // All critical principles
		Scope:            "full",
		ReportPeriod:     "2024-Q4",
		UseGoose:         true,         // Autonomous remediation
		UseDeepResearch:  true,         // Comprehensive research
		GenerateInsights: true,         // Generate actionable insights
		AutoRemediate:    true,         // Auto-generate scripts
	}

	logger.Printf("Running full multi-model audit...")
	logger.Printf("  Models: GNN + LocalAI + Goose + DeepResearch")
	logger.Printf("  Principles: %v", request.Principles)

	result, err := pipeline.RunAudit(ctx, request)
	if err != nil {
		logger.Fatalf("Audit failed: %v", err)
	}

	// Comprehensive results
	logger.Printf("\n=== AUDIT SUMMARY ===")
	logger.Printf("Overall Compliance: %s (%.1f%%)", result.ComplianceStatus, result.OverallScore*100)
	logger.Printf("Duration: %v", result.EndTime.Sub(result.StartTime))
	logger.Printf("Models Used: %v", result.ModelsUsed)

	// Principle scores
	logger.Printf("\n=== PRINCIPLE SCORES ===")
	for _, audit := range result.PrincipleAudits {
		status := "✅"
		if audit.Status == "non_compliant" {
			status = "❌"
		} else if audit.Status == "partially_compliant" {
			status = "⚠️"
		}
		logger.Printf("%s %s - %s: %.1f%%",
			status, audit.PrincipleID, audit.PrincipleName, audit.ComplianceScore*100)
	}

	// Key insights
	logger.Printf("\n=== KEY INSIGHTS (%d) ===", len(result.Insights))
	for _, insight := range result.Insights {
		logger.Printf("[%s] %s", insight.Severity, insight.Title)
		logger.Printf("  Affected: %v", insight.AffectedPrinciples)
		logger.Printf("  %s", insight.Description)
	}

	// Research reports
	logger.Printf("\n=== RESEARCH REPORTS (%d) ===", len(result.ResearchReports))
	for _, report := range result.ResearchReports {
		logger.Printf("• %s (%.1f%% confidence, %d sources)",
			report.Topic, report.Confidence*100, len(report.Sources))
	}

	// Remediation plans
	logger.Printf("\n=== REMEDIATION PLANS (%d) ===", len(result.Recommendations))
	criticalCount := 0
	for _, remedy := range result.Recommendations {
		if remedy.Priority == "critical" {
			criticalCount++
		}
	}
	logger.Printf("  Critical: %d", criticalCount)
	logger.Printf("  Automated scripts: %d", len(result.GooseGeneratedScripts))

	// Performance breakdown
	logger.Printf("\n=== PERFORMANCE ===")
	for step, duration := range result.ProcessingTime {
		logger.Printf("  %s: %v", step, duration)
	}

	// Export example (JSON)
	logger.Printf("\n=== EXPORT ===")
	logger.Printf("Results can be exported to:")
	logger.Printf("  • JSON: audit-results.json")
	logger.Printf("  • PDF: audit-report.pdf")
	logger.Printf("  • Dashboard: /compliance/audits/%s", result.AuditID)
}

// Helper: Setup Neo4j connection
func setupNeo4j(ctx context.Context) neo4j.DriverWithContext {
	neo4jURL := os.Getenv("NEO4J_URL")
	if neo4jURL == "" {
		neo4jURL = "bolt://localhost:7687"
	}

	driver, err := neo4j.NewDriverWithContext(
		neo4jURL,
		neo4j.BasicAuth("neo4j", "password", ""),
	)
	if err != nil {
		log.Fatalf("Failed to connect to Neo4j: %v", err)
	}

	if err := driver.VerifyConnectivity(ctx); err != nil {
		log.Fatalf("Neo4j connectivity failed: %v", err)
	}

	return driver
}

// Helper: Truncate string for display
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
