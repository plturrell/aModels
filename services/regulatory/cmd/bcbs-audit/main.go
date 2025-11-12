package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	graphneo4j "github.com/plturrell/aModels/services/graph/pkg/clients/neo4j"
	"github.com/plturrell/aModels/services/orchestration/agents"
	"github.com/plturrell/aModels/services/regulatory"
)

// BCBS Audit CLI - Executable pipeline for compliance auditing and insights
func main() {
	// CLI flags
	var (
		auditID         = flag.String("audit-id", "", "Unique audit identifier (required)")
		principles      = flag.String("principles", "P3,P4,P7,P12", "Comma-separated principle IDs to audit")
		scope           = flag.String("scope", "full", "Audit scope: full, quick, or targeted")
		reportPeriod    = flag.String("period", "2024-Q4", "Reporting period")
		useGoose        = flag.Bool("goose", true, "Enable Goose for autonomous remediation")
		useDeepResearch = flag.Bool("research", true, "Enable Deep Research for comprehensive analysis")
		autoRemediate   = flag.Bool("auto-remediate", false, "Auto-generate remediation scripts with Goose")
		outputFormat    = flag.String("output", "json", "Output format: json, summary, or detailed")
		
		// Service URLs
		neo4jURL        = flag.String("neo4j", os.Getenv("NEO4J_URL"), "Neo4j connection URL")
		neo4jUser       = flag.String("neo4j-user", os.Getenv("NEO4J_USER"), "Neo4j username")
		neo4jPass       = flag.String("neo4j-pass", os.Getenv("NEO4J_PASSWORD"), "Neo4j password")
		localaiURL      = flag.String("localai", os.Getenv("LOCALAI_URL"), "LocalAI server URL")
		gnnURL          = flag.String("gnn", os.Getenv("GNN_SERVICE_URL"), "GNN training service URL")
		gooseURL        = flag.String("goose-url", os.Getenv("GOOSE_SERVER_URL"), "Goose server URL")
		deepagentsURL   = flag.String("deepagents", os.Getenv("DEEPAGENTS_URL"), "DeepAgents service URL")
	)
	
	flag.Parse()
	
	// Validate required flags
	if *auditID == "" {
		log.Fatal("Error: --audit-id is required")
	}
	
	// Set defaults from environment
	if *neo4jURL == "" {
		*neo4jURL = "bolt://localhost:7687"
	}
	if *localaiURL == "" {
		*localaiURL = "http://localhost:8080"
	}
	if *gnnURL == "" {
		*gnnURL = "http://localhost:8081"
	}
	if *gooseURL == "" {
		*gooseURL = "http://localhost:8082"
	}
	if *deepagentsURL == "" {
		*deepagentsURL = "http://localhost:8083"
	}
	
	logger := log.New(os.Stdout, "[BCBS-Audit] ", log.LstdFlags)
	ctx := context.Background()
	
	logger.Printf("Starting BCBS239 Audit Pipeline: %s", *auditID)
	logger.Printf("Scope: %s | Period: %s | Principles: %s", *scope, *reportPeriod, *principles)
	
	// Initialize Neo4j driver
	driver, err := neo4j.NewDriverWithContext(
		*neo4jURL,
		neo4j.BasicAuth(*neo4jUser, *neo4jPass, ""),
	)
	if err != nil {
		log.Fatalf("Failed to connect to Neo4j: %v", err)
	}
	defer driver.Close(ctx)
	
	// Verify connectivity
	if err := driver.VerifyConnectivity(ctx); err != nil {
		log.Fatalf("Neo4j connectivity check failed: %v", err)
	}
	logger.Println("✓ Connected to Neo4j")
	
	// Setup compliance stack
	graphClient := graphneo4j.NewNeo4jGraphClient(driver, logger)
	bcbs239GraphClient := regulatory.NewBCBS239GraphClient(driver, graphClient, logger)
	
	localAIClient := agents.NewLocalAIClient(*localaiURL, nil, logger)
	
	// Create reasoning agent with multi-model support
	reasoningAgent := regulatory.NewComplianceReasoningAgent(
		localAIClient,
		bcbs239GraphClient,
		logger,
		"gemma-2b-q4_k_m.gguf",
	)
	
	// Wire model adapters based on flags
	if *gnnURL != "" {
		gnnAdapter := regulatory.NewGNNAdapter(*gnnURL, logger)
		reasoningAgent.WithGNNAdapter(gnnAdapter)
		logger.Println("✓ GNN adapter enabled")
	}
	
	if *useGoose && *gooseURL != "" {
		gooseAdapter := regulatory.NewGooseAdapter(*gooseURL, logger)
		reasoningAgent.WithGooseAdapter(gooseAdapter)
		logger.Println("✓ Goose autonomous agent enabled")
	}
	
	if *useDeepResearch && *deepagentsURL != "" {
		deepResearchAdapter := regulatory.NewDeepResearchAdapter(*deepagentsURL, logger)
		reasoningAgent.WithDeepResearchAdapter(deepResearchAdapter)
		logger.Println("✓ Deep Research agent enabled")
	}
	
	// Create audit pipeline
	pipeline := regulatory.NewBCBS239AuditPipeline(
		reasoningAgent,
		bcbs239GraphClient,
		logger,
	)
	
	// Parse principles
	principleList := strings.Split(*principles, ",")
	for i, p := range principleList {
		principleList[i] = strings.TrimSpace(p)
	}
	
	// Execute audit
	logger.Printf("Executing audit with %d models enabled...", countEnabledModels(*useGoose, *useDeepResearch))
	
	request := regulatory.AuditRequest{
		AuditID:          *auditID,
		Principles:       principleList,
		Scope:            *scope,
		ReportPeriod:     *reportPeriod,
		UseGoose:         *useGoose,
		UseDeepResearch:  *useDeepResearch,
		GenerateInsights: true,
		AutoRemediate:    *autoRemediate,
	}
	
	result, err := pipeline.RunAudit(ctx, request)
	if err != nil {
		log.Fatalf("Audit failed: %v", err)
	}
	
	// Output results
	switch *outputFormat {
	case "json":
		outputJSON(result)
	case "summary":
		outputSummary(result, logger)
	case "detailed":
		outputDetailed(result, logger)
	default:
		outputSummary(result, logger)
	}
	
	logger.Printf("Audit completed successfully: %s", result.ComplianceStatus)
}

func countEnabledModels(goose, research bool) int {
	count := 2 // LocalAI + GNN always enabled
	if goose {
		count++
	}
	if research {
		count++
	}
	return count
}

func outputJSON(result *regulatory.AuditResult) {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(result); err != nil {
		log.Fatalf("Failed to encode JSON: %v", err)
	}
}

func outputSummary(result *regulatory.AuditResult, logger *log.Logger) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Printf("BCBS239 Audit Summary: %s\n", result.AuditID)
	fmt.Println(strings.Repeat("=", 80))
	
	fmt.Printf("\nOverall Compliance: %s (Score: %.2f%%)\n", 
		strings.ToUpper(result.ComplianceStatus), result.OverallScore*100)
	fmt.Printf("Duration: %v\n", result.EndTime.Sub(result.StartTime))
	fmt.Printf("Models Used: %v\n", result.ModelsUsed)
	
	fmt.Println("\n" + strings.Repeat("-", 80))
	fmt.Println("Principle Audit Results:")
	fmt.Println(strings.Repeat("-", 80))
	
	for _, audit := range result.PrincipleAudits {
		status := getStatusEmoji(audit.Status)
		fmt.Printf("%s %s - %s: %.1f%% (%d controls, %d gaps)\n",
			status,
			audit.PrincipleID,
			audit.PrincipleName,
			audit.ComplianceScore*100,
			audit.ControlsCovered,
			audit.ControlsMissing,
		)
	}
	
	if len(result.Insights) > 0 {
		fmt.Println("\n" + strings.Repeat("-", 80))
		fmt.Printf("Key Insights: %d found\n", len(result.Insights))
		fmt.Println(strings.Repeat("-", 80))
		
		for _, insight := range result.Insights {
			emoji := getSeverityEmoji(insight.Severity)
			fmt.Printf("%s [%s] %s: %s\n",
				emoji,
				strings.ToUpper(insight.Severity),
				insight.Title,
				insight.Description,
			)
		}
	}
	
	if len(result.Recommendations) > 0 {
		fmt.Println("\n" + strings.Repeat("-", 80))
		fmt.Printf("Remediation Plans: %d generated\n", len(result.Recommendations))
		fmt.Println(strings.Repeat("-", 80))
		
		for i, remedy := range result.Recommendations {
			fmt.Printf("%d. [%s] %s - %s\n",
				i+1,
				strings.ToUpper(remedy.Priority),
				remedy.PrincipleID,
				remedy.Gap,
			)
			if remedy.AutomationScript != "" {
				fmt.Printf("   ✓ Automation script generated by Goose\n")
			}
		}
	}
	
	if len(result.ResearchReports) > 0 {
		fmt.Println("\n" + strings.Repeat("-", 80))
		fmt.Printf("Research Reports: %d generated by Deep Research Agent\n", len(result.ResearchReports))
		fmt.Println(strings.Repeat("-", 80))
		
		for _, report := range result.ResearchReports {
			fmt.Printf("• %s (Confidence: %.1f%%, Sources: %d)\n",
				report.Topic,
				report.Confidence*100,
				len(report.Sources),
			)
		}
	}
	
	fmt.Println("\n" + strings.Repeat("=", 80))
}

func outputDetailed(result *regulatory.AuditResult, logger *log.Logger) {
	outputSummary(result, logger)
	
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("DETAILED AUDIT RESULTS")
	fmt.Println(strings.Repeat("=", 80))
	
	for _, audit := range result.PrincipleAudits {
		fmt.Printf("\n%s - %s\n", audit.PrincipleID, audit.PrincipleName)
		fmt.Println(strings.Repeat("-", 40))
		fmt.Printf("Status: %s (%.1f%%)\n", audit.Status, audit.ComplianceScore*100)
		fmt.Printf("Analyzed by: %s\n", audit.AnalyzedBy)
		
		if len(audit.Evidence) > 0 {
			fmt.Println("\nEvidence:")
			for _, ev := range audit.Evidence {
				fmt.Printf("  • %s\n", ev)
			}
		}
		
		if len(audit.Gaps) > 0 {
			fmt.Println("\nIdentified Gaps:")
			for _, gap := range audit.Gaps {
				fmt.Printf("  ⚠ %s\n", gap)
			}
		}
	}
	
	if len(result.ResearchReports) > 0 {
		fmt.Println("\n" + strings.Repeat("=", 80))
		fmt.Println("RESEARCH REPORTS (Deep Research Agent)")
		fmt.Println(strings.Repeat("=", 80))
		
		for _, report := range result.ResearchReports {
			fmt.Printf("\n%s\n", report.Topic)
			fmt.Println(strings.Repeat("-", 40))
			fmt.Printf("Confidence: %.1f%%\n", report.Confidence*100)
			fmt.Printf("Sources: %d\n", len(report.Sources))
			fmt.Printf("\nSummary:\n%s\n", report.Summary)
			
			if len(report.KeyFindings) > 0 {
				fmt.Println("\nKey Findings:")
				for _, finding := range report.KeyFindings {
					fmt.Printf("  • %s\n", finding)
				}
			}
		}
	}
	
	if len(result.GooseGeneratedScripts) > 0 {
		fmt.Println("\n" + strings.Repeat("=", 80))
		fmt.Println("GOOSE-GENERATED REMEDIATION SCRIPTS")
		fmt.Println(strings.Repeat("=", 80))
		
		for i, script := range result.GooseGeneratedScripts {
			fmt.Printf("\nScript %d:\n", i+1)
			fmt.Println(strings.Repeat("-", 40))
			fmt.Println(script)
		}
	}
}

func getStatusEmoji(status string) string {
	switch status {
	case "compliant":
		return "✅"
	case "partially_compliant":
		return "⚠️"
	case "non_compliant":
		return "❌"
	default:
		return "•"
	}
}

func getSeverityEmoji(severity string) string {
	switch severity {
	case "critical":
		return "🔴"
	case "high":
		return "🟠"
	case "medium":
		return "🟡"
	case "low":
		return "🟢"
	default:
		return "•"
	}
}
