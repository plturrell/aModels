# BCBS 239 Compliance Framework

A comprehensive regulatory compliance framework integrating **LangGraph-style orchestration**, **Neo4j knowledge graphs**, and **LocalAI reasoning** for BCBS 239 (Principles for effective risk data aggregation and risk reporting).

## 🏗️ Architecture Overview

This framework implements the architecture proposed for BCBS 239 compliance, combining **multi-model AI orchestration**:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    BCBS239Reporting (Orchestrator)                   │
│  • Coordinates multi-step compliance workflow                        │
│  • Manages state transitions (draft → validated → approved)          │
│  • Enforces human checkpoints for critical reports                   │
└────────────┬──────────────────────────────────────┬──────────────────┘
             │                                      │
    ┌────────▼────────┐                  ┌──────────▼─────────────────┐
    │ Calculation     │                  │ Compliance Reasoning Agent │
    │ Engine          │                  │ (Multi-Model Orchestrator) │
    │                 │                  │                            │
    │ • Computes      │                  │  ┌──────────────────────┐ │
    │   metrics       │                  │  │  ModelOrchestrator   │ │
    │ • Auto-emits    │                  │  │  • Intelligent       │ │
    │   to Neo4j      │                  │  │    routing           │ │
    └────────┬────────┘                  │  │  • Fallback chains   │ │
             │                            │  └──────────┬───────────┘ │
             │                            │             │              │
             │                            │  ┌──────────▼──────┐      │
             │                            │  │   LocalAI       │      │
             │                            │  │  • General LLM  │      │
             │                            │  └─────────────────┘      │
             │                            │  ┌────────────────┐       │
             │                            │  │   GNN Adapter   │      │
             │                            │  │ • Structural    │      │
             │                            │  │   analysis      │      │
             │                            │  │ • Embeddings    │      │
             │                            │  └─────────────────┘      │
             │                            │  ┌──────────────────┐     │
             │                            │  │  Goose Adapter    │    │
             │                            │  │ • Autonomous tasks│    │
             │                            │  │ • Multi-step      │    │
             │                            │  └──────────────────────┘ │
             │                            │  ┌────────────────────┐   │
             │                            │  │ DeepResearch Adapter│  │
             │                            │  │ • Comprehensive     │  │
             │                            │  │   analysis          │  │
             │                            │  └────────────────────┘   │
             │                            └─────────────┬──────────────┘
             │                                          │
    ┌────────▼──────────────────────────────────────────▼──────────────┐
    │          Neo4j Knowledge Graph (BCBS239GraphClient)              │
    │  • 14 BCBS 239 Principles (nodes)                                │
    │  • Controls ensuring each principle                               │
    │  • Data assets with lineage (DEPENDS_ON edges)                   │
    │  • Calculations validated by controls                             │
    │  • Cypher query templates for compliance analysis                │
    └───────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Role | Key Features |
|-----------|------|--------------|
| **BCBS239GraphSchema** | Neo4j schema manager | 14 BCBS principles, constraints, indexes, Cypher templates |
| **BCBS239GraphClient** | Graph operations | Lineage tracing, control mapping, impact analysis, gap detection |
| **RegulatoryCalculationEngine** | Calculation executor | Metric computation, auto-emit to graph, framework-specific logic |
| **ComplianceReasoningAgent** | Multi-model LangGraph orchestrator | Intelligent routing to LocalAI, GNN, Goose, or DeepResearch |
| **ModelOrchestrator** | Model router & fallback manager | Rule-based routing, performance tracking, hybrid queries |
| **GNNAdapter** | Graph Neural Network interface | Structural analysis, pattern recognition, embeddings |
| **GooseAdapter** | Autonomous agent interface | Multi-step task execution, code generation |
| **DeepResearchAdapter** | Research agent interface | Comprehensive analysis, multi-source validation |
| **BCBS239Reporting** | Report generator | Graph-enriched reports, AI narratives, human checkpoints |

---

## 🤖 Multi-Model AI Integration

The framework now supports **intelligent model orchestration** with automatic routing based on query type:

### Supported Models

#### 1. **LocalAI** (Default)
- **Best for**: General compliance questions, narrative generation
- **Capabilities**: Text generation, chat completion, prompt-based reasoning
- **Usage**: Always available as fallback

#### 2. **GNN (Graph Neural Networks)**
- **Best for**: Structural analysis, pattern recognition, lineage tracing
- **Capabilities**: 
  - Graph embeddings for similarity search
  - Anomaly detection in control networks
  - Link prediction for missing relationships
  - Node classification by compliance domain
- **Auto-selected for**: Queries containing "pattern", "structure", "similar", "anomaly", "predict"

#### 3. **Goose (Autonomous Agent)**
- **Best for**: Multi-step autonomous tasks, workflow orchestration
- **Capabilities**:
  - Code generation for compliance scripts
  - Multi-step task planning and execution
  - Tool integration and MCP server interaction
  - Documentation generation
- **Auto-selected for**: Queries containing "workflow", "automate", "generate code", "create", "build"

#### 4. **Deep Research Agent**
- **Best for**: Comprehensive regulatory research, cross-reference validation
- **Capabilities**:
  - Multi-source analysis of regulatory documents
  - Best practice recommendations
  - Compliance gap identification
  - Citation tracking and validation
- **Auto-selected for**: Queries containing "research", "comprehensive", "detailed analysis", "regulatory document"

---

## 🚀 Quick Start

### 1. Initialize Neo4j Schema

```go
import (
    "context"
    "log"
    
    "github.com/neo4j/neo4j-go-driver/v5/neo4j"
    "github.com/plturrell/aModels/services/regulatory"
)

// Connect to Neo4j
driver, err := neo4j.NewDriverWithContext(
    "bolt://localhost:7687",
    neo4j.BasicAuth("neo4j", "password", ""),
)
if err != nil {
    log.Fatal(err)
}
defer driver.Close(ctx)

// Initialize BCBS 239 schema
schema := regulatory.NewBCBS239GraphSchema(driver)
if err := schema.InitializeSchema(ctx); err != nil {
    log.Fatalf("Failed to initialize schema: %v", err)
}

// Seed the 14 BCBS 239 principles
if err := schema.SeedBCBS239Principles(ctx); err != nil {
    log.Fatalf("Failed to seed principles: %v", err)
}
```

### 2. Configure the Compliance Stack

```go
import (
    "github.com/plturrell/aModels/services/graph"
    "github.com/plturrell/aModels/services/orchestration/agents"
)

logger := log.New(os.Stdout, "[BCBS239] ", log.LstdFlags)

// Setup graph client
graphClient := graph.NewNeo4jGraphClient(driver, logger)
bcbs239GraphClient := regulatory.NewBCBS239GraphClient(driver, graphClient, logger)

// Setup LocalAI client
localAIClient := agents.NewLocalAIClient(
    "http://localhost:8080", // LocalAI server URL
    nil, // Use default HTTP client
    logger,
)

// Create reasoning agent
reasoningAgent := regulatory.NewComplianceReasoningAgent(
    localAIClient,
    bcbs239GraphClient,
    logger,
    "gemma-2b-q4_k_m.gguf", // Model for compliance reasoning
)

// Create calculation engine with Neo4j integration
calcEngine := regulatory.NewRegulatoryCalculationEngine(logger).
    WithBCBS239GraphClient(bcbs239GraphClient)

// Setup reporting system
reporting := regulatory.NewBCBS239Reporting(
    nil, // extractor (optional)
    calcEngine,
    regulatory.NewReportValidator(logger),
    regulatory.NewOutputTracer(logger),
    logger,
).
WithGraphClient(bcbs239GraphClient).
WithReasoningAgent(reasoningAgent)
```

### 2a. Configure Multi-Model AI (Optional but Recommended)

Add GNN, Goose, and DeepResearch capabilities for enhanced compliance analysis:

```go
// Create model adapters
gnnAdapter := regulatory.NewGNNAdapter(
    "http://training-service:8080", // GNN training service
    logger,
)

gooseAdapter := regulatory.NewGooseAdapter(
    "http://goose-server:8081", // Goose agent server
    logger,
)

deepResearchAdapter := regulatory.NewDeepResearchAdapter(
    "http://deepagents:8082", // Deep research service
    logger,
)

// Wire multi-model capabilities into reasoning agent
reasoningAgent.
    WithGNNAdapter(gnnAdapter).
    WithGooseAdapter(gooseAdapter).
    WithDeepResearchAdapter(deepResearchAdapter)

// Now the agent will intelligently route queries:
// - Structural questions → GNN
// - Comprehensive research → DeepResearch
// - Workflow automation → Goose
// - General questions → LocalAI (fallback)
```

### 2b. Using Hybrid Multi-Model Queries

Query multiple models simultaneously and combine results:

```go
// Execute hybrid query across GNN + DeepResearch
hybridResponse, err := reasoningAgent.QueryWithHybridModels(
    ctx,
    "What are the structural patterns in P3 (Accuracy) compliance controls?",
    "P3",
    []string{"GNN", "DeepResearch"}, // Models to use
)

if err == nil {
    fmt.Printf("Combined Analysis:\n%s\n", hybridResponse.CombinedAnswer)
    fmt.Printf("Average Confidence: %.2f\n", hybridResponse.AverageConfidence)
    fmt.Printf("Sources: %v\n", hybridResponse.Sources)
    
    // Access individual model responses
    for _, resp := range hybridResponse.ModelResponses {
        fmt.Printf("\n%s Analysis (%.2f confidence):\n%s\n", 
            resp.ModelType, resp.Confidence, resp.Answer)
    }
}
```

### 3. Generate a BCBS 239 Report

```go
// Simple report generation
report, err := reporting.GenerateReport(ctx, regulatory.BCBS239ReportRequest{
    ReportPeriod: "2024-Q4",
    Metrics:      []string{"risk_data_aggregation", "accuracy_validation"},
    GeneratedBy:  "compliance.officer@bank.com",
})

if err != nil {
    log.Fatalf("Report generation failed: %v", err)
}

// Report includes:
// - Calculated metrics (auto-persisted to Neo4j)
// - Graph-derived lineage insights
// - AI-generated compliance narrative
// - Compliance area assessments
log.Printf("Report %s generated: %s", report.ReportID, report.Status)
```

### 4. Human-in-the-Loop Approval

For critical reports (non-compliant areas, gaps detected), the workflow pauses for approval:

```go
// Request report with mandatory approval
report, err := reporting.GenerateReport(ctx, regulatory.BCBS239ReportRequest{
    ReportPeriod:     "2024-Q4",
    Metrics:          []string{"risk_data_aggregation"},
    GeneratedBy:      "system",
    RequiresApproval: true, // Force human checkpoint
})

if report.Status == "pending_approval" {
    log.Printf("Report %s awaiting approval", report.ReportID)
    
    // Present to compliance officer for review
    // ... external approval workflow ...
    
    // Approve the report
    err = reporting.ApproveReport(
        ctx,
        report.ReportID,
        "compliance.officer@bank.com",
        "Reviewed and approved for submission",
    )
}
```

---

## 📊 Workflow Details

### End-to-End Compliance Workflow

```
1. Calculate Metrics
   └─> RegulatoryCalculationEngine.CalculateRegulatoryMetrics()
       • Computes BCBS 239 metrics (accuracy, completeness, timeliness)
       • Auto-emits to Neo4j via BCBS239GraphClient.UpsertCalculationWithLineage()
       • Links calculations to source assets and controls

2. Retrieve Graph Insights
   └─> BCBS239Reporting.enrichWithGraphInsights()
       • Traces data lineage for each calculation
       • Identifies non-compliant areas (missing controls)
       • Analyzes downstream impact

3. Generate AI Narrative
   └─> ComplianceReasoningAgent.GenerateComplianceNarrative()
       • LocalAI classifies the compliance question
       • Generates Cypher queries to retrieve relevant graph facts
       • Synthesizes coherent compliance analysis with citations

4. Human Checkpoint (if critical)
   └─> BCBS239Reporting.isCriticalReport()
       • Pauses workflow if:
         - Any compliance area is non-compliant
         - Graph insights reveal gaps
         - RequiresApproval = true
       • Stores pending state for external approval

5. Validate & Finalize
   └─> ReportValidator.ValidateBCBS239Report()
       • Ensures all required sections present
       • Checks compliance area completeness
       • Assigns final status: validated | validation_failed
```

---

## 🧪 Testing

### Run Unit Tests
```bash
cd /home/aModels/services/regulatory
go test -v -short
```

### Run Integration Tests (requires Neo4j and LocalAI)
```bash
go test -v
```

### Test Coverage
```bash
go test -v -cover -coverprofile=coverage.out
go tool cover -html=coverage.out
```

---

## 🗂️ Neo4j Graph Schema

### Node Types

| Node Label | Properties | Description |
|------------|-----------|-------------|
| `BCBS239Principle` | `principle_id`, `principle_name`, `compliance_area`, `priority` | One of 14 BCBS principles |
| `BCBS239Control` | `control_id`, `control_name`, `control_type` | Control ensuring principle compliance |
| `DataAsset` | `asset_id`, `asset_type`, `source_system` | Tables, reports, data fields |
| `RegulatoryCalculation` | `calculation_id`, `calculation_type`, `result`, `status` | Computed metrics |
| `Process` | `process_id`, `process_name` | Business workflows |

### Relationship Types

| Relationship | Source → Target | Meaning |
|--------------|----------------|---------|
| `ENSURED_BY` | Principle → Control | Control implements principle |
| `APPLIES_TO` | Control → Process/Asset | Control validates entity |
| `DEPENDS_ON` | Calculation → DataAsset | Calculation sources data from asset |
| `TRANSFORMS` | Process → DataAsset | Process creates/modifies asset |
| `VALIDATED_BY` | Calculation → Control | Calculation verified by control |

### Example Cypher Queries

**Find all controls ensuring Principle 3 (Accuracy):**
```cypher
MATCH (p:BCBS239Principle {principle_id: 'P3'})
      -[:ENSURED_BY]->(c:BCBS239Control)
      -[:APPLIES_TO]->(target)
RETURN p, c, target
```

**Trace lineage for a calculation:**
```cypher
MATCH path = (calc:RegulatoryCalculation {calculation_id: 'BCBS239-RDA-2024-Q4'})
             -[:DEPENDS_ON|SOURCE_FROM*1..5]->(asset:DataAsset)
RETURN path
ORDER BY length(path) DESC
```

**Identify non-compliant principles (missing controls):**
```cypher
MATCH (p:BCBS239Principle)
WHERE NOT EXISTS {
    MATCH (p)-[:ENSURED_BY]->(:BCBS239Control)
}
RETURN p.principle_id, p.principle_name
```

---

## 🔍 Compliance Reasoning Agent

The `ComplianceReasoningAgent` implements a LangGraph-style stateful workflow for answering compliance questions:

### Workflow Nodes

1. **IntakeNode** - Classifies the question type (lineage_tracing | control_mapping | impact_analysis)
2. **GraphQueryNode** - Generates and executes Cypher queries
3. **SynthesisNode** - Combines graph facts into coherent narrative
4. **ValidationNode** - Validates answer quality and confidence

### Example Usage

```go
agent := regulatory.NewComplianceReasoningAgent(
    localAIClient,
    bcbs239GraphClient,
    logger,
    "gemma-2b-q4_k_m.gguf",
)

state, err := agent.RunComplianceWorkflow(
    ctx,
    "What controls ensure data accuracy for regulatory calculations?",
    "P3", // Principle 3: Accuracy and Integrity
)

if state.RequiresApproval && state.ApprovalStatus == "pending" {
    // Workflow paused - human review required
}

fmt.Printf("Answer: %s\n", state.SynthesizedAnswer)
fmt.Printf("Confidence: %.2f\n", state.Confidence)
fmt.Printf("Sources: %v\n", state.Sources)
```

---

## 📋 BCBS 239 Principles Reference

### Governance & Infrastructure (P1-P2)
- **P1: Governance** - Strong governance for risk data and reporting
- **P2: Data Architecture** - Robust IT infrastructure supporting aggregation

### Risk Data Aggregation (P3-P6)
- **P3: Accuracy & Integrity** - Accurate, reliable risk data
- **P4: Completeness** - Capture all material risk data
- **P5: Timeliness** - Generate data in timely manner
- **P6: Adaptability** - Meet ad hoc reporting requests

### Risk Reporting (P7-P11)
- **P7: Accuracy** - Reports convey risk precisely
- **P8: Comprehensiveness** - Cover all material risk areas
- **P9: Clarity** - Clear, concise, understandable
- **P10: Frequency** - Distributed with appropriate frequency
- **P11: Distribution** - Reach relevant parties securely

### Supervisory Review (P12-P14)
- **P12: Supervisory Reporting** - Accurate supervisory reports
- **P13: Remediation Plans** - Banks develop remediation when needed
- **P14: Home-Host Coordination** - Supervisors share information

---

## 🛠️ Advanced Configuration

### Custom Calculation Types

Extend the calculation engine to emit custom lineage:

```go
func (rce *RegulatoryCalculationEngine) emitBCBS239ToGraph(
    ctx context.Context,
    calculations []RegulatoryCalculation,
) error {
    for _, calc := range calculations {
        var sourceAssets []string
        var controlIDs []string
        
        switch calc.CalculationType {
        case "custom_metric":
            sourceAssets = []string{"asset-custom-source"}
            controlIDs = []string{"control-custom-p3"}
        }
        
        rce.bcbs239GraphClient.UpsertCalculationWithLineage(
            ctx, calc, sourceAssets, controlIDs,
        )
    }
    return nil
}
```

### Custom Approval Logic

Override the critical report detector:

```go
func (b *BCBS239Reporting) isCriticalReport(report *BCBS239Report) bool {
    // Custom logic: require approval for all P3, P4 related reports
    for _, calc := range report.Calculations {
        if strings.Contains(calc.CalculationType, "accuracy") ||
           strings.Contains(calc.CalculationType, "completeness") {
            return true
        }
    }
    return false
}
```

---

## 📚 API Reference

### Key Types

```go
// BCBS239ReportRequest - Input for report generation
type BCBS239ReportRequest struct {
    ReportPeriod     string   // e.g., "2024-Q4"
    Metrics          []string // e.g., ["risk_data_aggregation"]
    GeneratedBy      string   // User/system identifier
    RequiresApproval bool     // Force human checkpoint
}

// BCBS239Report - Output compliance report
type BCBS239Report struct {
    ReportID             string
    Status               string // draft | pending_approval | validated
    Calculations         []RegulatoryCalculation
    ComplianceAreas      []BCBS239ComplianceArea
    GraphInsights        []GraphInsight
    AIGeneratedNarrative string
    ApprovalRequired     bool
}

// ComplianceWorkflowState - LangGraph-style state
type ComplianceWorkflowState struct {
    Question         string
    PrincipleID      string
    GraphFacts       []map[string]interface{}
    SynthesizedAnswer string
    Confidence       float64
    RequiresApproval bool
}
```

---

## 🔍 BCBS239 Audit Pipeline

For **automated compliance auditing and insights**, use the executable audit pipeline:

### Quick Start

```bash
cd /home/aModels/services/regulatory

# Build the audit CLI
go build -o bcbs-audit ./cmd/bcbs-audit

# Run comprehensive audit
./bcbs-audit \
  --audit-id "Q4-2024-audit" \
  --principles "P3,P4,P7,P12" \
  --goose \
  --research \
  --auto-remediate \
  --output detailed
```

### What the Pipeline Does

1. **Audits** each BCBS239 principle against Neo4j graph controls
2. **Deep Research** analyzes gaps with multi-source regulatory validation
3. **Goose generates** production-ready remediation scripts automatically
4. **Produces** actionable insights with prioritized recommendations

### Goose Autonomous Remediation

**Goose** generates production-ready automation scripts for identified gaps:

```python
# Example: Auto-generated by Goose for P3 (Accuracy) gap

def accuracy_validation_control(driver, calculation_id):
    """Automated control for BCBS239 P3 - validates calculations"""
    with driver.session() as session:
        # Retrieve calculation lineage
        result = session.run("""
            MATCH (calc:RegulatoryCalculation {calculation_id: $calc_id})
                  -[:DEPENDS_ON]->(asset:DataAsset)
            RETURN calc, asset
        """, calc_id=calculation_id)
        
        # Validate data integrity and record results
        # ... (full implementation with error handling)
```

### Deep Research Insights

**Deep Research Agent** provides comprehensive regulatory analysis:

```
Research Report: BCBS 239 P4 Compliance
Confidence: 92% | Sources: 8

KEY FINDINGS:
1. Automated Data Reconciliation [Basel Committee 2019 Survey]
   - 87% of compliant banks use automated reconciliation
   - Recommendation: Implement daily completeness checks

2. Cross-System Data Lineage [FSI Working Paper No. 28]
   - End-to-end lineage reduces gaps by 45%
   - Graph databases recommended for complex lineage

3. Exception Management Framework [BCBS 239 Guide]
   - Formal exception tracking required
   - Real-time alerting for completeness thresholds

CITATIONS: [8 authoritative sources]
```

### Usage Examples

```bash
# Quick gap analysis
./bcbs-audit --audit-id "gaps-001" --principles "P3,P4" --scope quick

# Research-focused analysis
./bcbs-audit --audit-id "research-001" --research --output detailed

# Autonomous remediation
./bcbs-audit --audit-id "auto-fix" --goose --auto-remediate

# Full multi-model pipeline
./bcbs-audit \
  --audit-id "full-001" \
  --goose \
  --research \
  --auto-remediate \
  --output json > results.json
```

**See:** `AUDIT_PIPELINE_USAGE.md` for detailed examples and use cases.

---

## 🔗 Integration Checklist

### Core Framework (Required)
- [x] Neo4j schema initialized with 14 BCBS 239 principles
- [x] LocalAI server running (http://localhost:8080)
- [x] Calculation engine wired to emit lineage to Neo4j
- [x] Compliance reasoning agent configured with model
- [x] Human approval workflow integrated
- [x] Unit and integration tests passing

### Multi-Model AI (Optional but Recommended)
- [x] GNN adapter created for structural analysis
- [x] Goose adapter created for autonomous tasks
- [x] DeepResearch adapter created for comprehensive analysis
- [x] ModelOrchestrator with intelligent routing
- [x] Hybrid query support for multi-model analysis
- [ ] GNN training service deployed (http://training-service:8080)
- [ ] Goose server deployed (http://goose-server:8081)
- [ ] DeepAgents service deployed (http://deepagents:8082)
- [ ] Model performance monitoring dashboard

### Production Deployment
- [ ] Production Neo4j cluster configured
- [ ] LocalAI GPU acceleration enabled
- [ ] Approval UI/API endpoints deployed
- [ ] Observability (LangSmith/traces) integrated
- [ ] Model failover and fallback tested
- [ ] Multi-model load balancing configured

---

## 📖 Further Reading

- [BCBS 239 Original Document](https://www.bis.org/publ/bcbs239.htm)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Neo4j Graph Data Science](https://neo4j.com/docs/graph-data-science/)
- [LocalAI Documentation](https://localai.io/)

---

## 🤝 Contributing

This framework is part of the `aModels` regulatory compliance suite. For changes:
1. Add tests covering new functionality
2. Update this README with API changes
3. Ensure Neo4j schema migrations are backward-compatible
4. Document any new Cypher query templates

---

**Built with:** Go • Neo4j • LocalAI • LangGraph patterns  
**License:** Internal use only  
**Maintainer:** Regulatory Compliance Team
