package embeddings

import (
	"github.com/plturrell/aModels/services/extract/pkg/graph"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
)

// generateEmbedding generates embedding for SQL query (legacy function, kept for backward compatibility)
func generateEmbedding(ctx context.Context, sql string) ([]float32, error) {
	return generateSQLEmbedding(ctx, sql)
}

// generateSemanticEmbedding generates semantic embedding using sap-rpt-1-oss for text queries
// Phase 3: Uses connection pooling (handled by Python script)
// Phase 10: Enhanced with LNN terminology layer
func generateSemanticEmbedding(ctx context.Context, text string) ([]float32, error) {
	cmd := exec.CommandContext(ctx, "python3", "./scripts/embeddings/embed_sap_rpt.py",
		"--artifact-type", "text",
		"--text", text,
	)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("failed to generate semantic embedding: %w, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("failed to generate semantic embedding: %w", err)
	}

	var embedding []float32
	if err := json.Unmarshal(output, &embedding); err != nil {
		return nil, fmt.Errorf("failed to unmarshal semantic embedding: %w", err)
	}

	// Phase 10: Apply terminology enhancement if available
	if globalTerminologyLearner != nil {
		enhanced, err := globalTerminologyLearner.EnhanceEmbedding(ctx, text, embedding, "sap_rpt")
		if err == nil {
			return enhanced, nil
		}
		// Fallback to original embedding if enhancement fails
	}

	return embedding, nil
}

// globalTerminologyLearner is a global reference to the terminology learner (set during initialization)
var globalTerminologyLearner *TerminologyLearner

// SetGlobalTerminologyLearner sets the global terminology learner (Phase 10).
func SetGlobalTerminologyLearner(learner *TerminologyLearner) {
	globalTerminologyLearner = learner
}

// GetGlobalTerminologyLearner returns the global terminology learner (Phase 10).
func GetGlobalTerminologyLearner() *TerminologyLearner {
	return globalTerminologyLearner
}

// generateSQLEmbedding generates embedding for SQL query
// Phase 10: Enhanced with LNN terminology layer
func generateSQLEmbedding(ctx context.Context, sql string) ([]float32, error) {
	cmd := exec.CommandContext(ctx, "python3", "./scripts/embeddings/embed.py", "--artifact-type", "sql", "--sql", sql)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("failed to generate SQL embedding: %w, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("failed to generate SQL embedding: %w", err)
	}

	var embedding []float32
	if err := json.Unmarshal(output, &embedding); err != nil {
		return nil, fmt.Errorf("failed to unmarshal SQL embedding: %w", err)
	}

	// Phase 10: Apply terminology enhancement if available
	if globalTerminologyLearner != nil {
		enhanced, err := globalTerminologyLearner.EnhanceEmbedding(ctx, sql, embedding, "relational_transformer")
		if err == nil {
			return enhanced, nil
		}
		// Fallback to original embedding if enhancement fails
	}

	return embedding, nil
}

// generateTableEmbedding generates embedding for table schema
// Returns both RelationalTransformer and sap-rpt-1-oss embeddings if available
func generateTableEmbedding(ctx context.Context, node graph.Node) ([]float32, []float32, error) {
	// Extract columns from node properties or edges
	columns := []map[string]any{}
	if node.Props != nil {
		// Try to get columns from properties
		if cols, ok := node.Props["columns"].([]map[string]any); ok {
			columns = cols
		}
	}

	// If no columns in properties, create minimal column info from node label
	if len(columns) == 0 {
		columns = []map[string]any{
			{"name": node.Label, "type": "string"},
		}
	}

	columnsJSON, err := json.Marshal(columns)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal columns: %w", err)
	}

	metadataJSON := "{}"
	if node.Props != nil {
		metadataBytes, err := json.Marshal(node.Props)
		if err == nil {
			metadataJSON = string(metadataBytes)
		}
	}

	// Generate RelationalTransformer embedding
	cmd := exec.CommandContext(ctx, "python3", "./scripts/embeddings/embed.py",
		"--artifact-type", "table",
		"--table-name", node.Label,
		"--columns", string(columnsJSON),
		"--metadata", metadataJSON,
	)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, nil, fmt.Errorf("failed to generate table embedding: %w, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, nil, fmt.Errorf("failed to generate table embedding: %w", err)
	}

	var relationalEmbedding []float32
	if err := json.Unmarshal(output, &relationalEmbedding); err != nil {
		return nil, nil, fmt.Errorf("failed to unmarshal table embedding: %w", err)
	}

	// Phase 10: Apply terminology enhancement to relational embedding
	if globalTerminologyLearner != nil {
		enhanced, err := globalTerminologyLearner.EnhanceEmbedding(ctx, node.Label, relationalEmbedding, "relational_transformer")
		if err == nil {
			relationalEmbedding = enhanced
		}
	}

	// Try to generate sap-rpt-1-oss semantic embedding if enabled
	var semanticEmbedding []float32
	if os.Getenv("USE_SAP_RPT_EMBEDDINGS") == "true" {
		cmdSemantic := exec.CommandContext(ctx, "python3", "./scripts/embeddings/embed_sap_rpt.py",
			"--artifact-type", "table",
			"--table-name", node.Label,
			"--columns", string(columnsJSON),
		)

		outputSemantic, err := cmdSemantic.Output()
		if err == nil {
			var semantic []float32
			if err := json.Unmarshal(outputSemantic, &semantic); err == nil && len(semantic) > 0 {
				semanticEmbedding = semantic

				// Phase 10: Apply terminology enhancement to semantic embedding
				if globalTerminologyLearner != nil {
					enhanced, err := globalTerminologyLearner.EnhanceEmbedding(ctx, node.Label, semanticEmbedding, "sap_rpt")
					if err == nil {
						semanticEmbedding = enhanced
					}
				}
			}
		}
		// Non-fatal: continue with relational embedding if semantic fails
	}

	return relationalEmbedding, semanticEmbedding, nil
}

// generateTableEmbeddingLegacy generates single embedding (for backward compatibility)
func generateTableEmbeddingLegacy(ctx context.Context, node graph.Node) ([]float32, error) {
	relational, _, err := generateTableEmbedding(ctx, node)
	return relational, err
}

// generateColumnEmbedding generates embedding for column definition
func generateColumnEmbedding(ctx context.Context, node graph.Node) ([]float32, error) {
	columnType := "string"
	if node.Props != nil {
		if t, ok := node.Props["type"].(string); ok && t != "" {
			columnType = t
		}
	}

	metadataJSON := "{}"
	if node.Props != nil {
		metadataBytes, err := json.Marshal(node.Props)
		if err == nil {
			metadataJSON = string(metadataBytes)
		}
	}

	cmd := exec.CommandContext(ctx, "python3", "./scripts/embeddings/embed.py",
		"--artifact-type", "column",
		"--column-name", node.Label,
		"--column-type", columnType,
		"--metadata", metadataJSON,
	)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("failed to generate column embedding: %w, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("failed to generate column embedding: %w", err)
	}

	var embedding []float32
	if err := json.Unmarshal(output, &embedding); err != nil {
		return nil, fmt.Errorf("failed to unmarshal column embedding: %w", err)
	}

	return embedding, nil
}

func generateSignavioProcessEmbedding(ctx context.Context, summary SignavioProcessSummary) ([]float32, error) {
	var builder strings.Builder
	builder.WriteString(fmt.Sprintf("Signavio process %s (%s)", summary.Name, summary.ID))
	if summary.SourceFile != "" {
		builder.WriteString(fmt.Sprintf(" sourced from %s", summary.SourceFile))
	}
	builder.WriteString(fmt.Sprintf(" with %d elements.", summary.ElementCount))

	if len(summary.ElementTypes) > 0 {
		typeCounts := make([]string, 0, len(summary.ElementTypes))
		for elementType, count := range summary.ElementTypes {
			typeCounts = append(typeCounts, fmt.Sprintf("%s:%d", elementType, count))
		}
		sort.Strings(typeCounts)
		builder.WriteString(" Element types: ")
		builder.WriteString(strings.Join(typeCounts, ", "))
		builder.WriteString(".")
	}

	maxElements := len(summary.Elements)
	if maxElements > 20 {
		maxElements = 20
	}
	for i := 0; i < maxElements; i++ {
		elem := summary.Elements[i]
		builder.WriteString(fmt.Sprintf(" Step %d: %s (%s).", i+1, elem.Name, elem.Type))
	}
	if len(summary.Elements) > maxElements {
		builder.WriteString(fmt.Sprintf(" ... %d additional steps omitted.", len(summary.Elements)-maxElements))
	}

	return generateSemanticEmbedding(ctx, builder.String())
}

// generateJobEmbedding generates embedding for Control-M job
func generateJobEmbedding(ctx context.Context, job ControlMJob) ([]float32, error) {
	conditions := []string{}
	for _, inCond := range job.InConds {
		conditions = append(conditions, inCond.Name)
	}
	for _, outCond := range job.OutConds {
		conditions = append(conditions, outCond.Name)
	}

	conditionsJSON, err := json.Marshal(conditions)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal conditions: %w", err)
	}

	metadata := job.Properties()
	metadataJSON := "{}"
	if metadata != nil {
		metadataBytes, err := json.Marshal(metadata)
		if err == nil {
			metadataJSON = string(metadataBytes)
		}
	}

	cmd := exec.CommandContext(ctx, "python3", "./scripts/embeddings/embed.py",
		"--artifact-type", "job",
		"--job-name", job.JobName,
		"--command", job.Command,
		"--conditions", string(conditionsJSON),
		"--metadata", metadataJSON,
	)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("failed to generate job embedding: %w, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("failed to generate job embedding: %w", err)
	}

	var embedding []float32
	if err := json.Unmarshal(output, &embedding); err != nil {
		return nil, fmt.Errorf("failed to unmarshal job embedding: %w", err)
	}

	return embedding, nil
}

// generateSequenceEmbedding generates embedding for table process sequence
func generateSequenceEmbedding(ctx context.Context, sequence TableProcessSequence) ([]float32, error) {
	tablesJSON, err := json.Marshal(sequence.Tables)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tables: %w", err)
	}

	metadata := map[string]any{
		"sequence_id":   sequence.SequenceID,
		"source_type":   sequence.SourceType,
		"source_file":   sequence.SourceFile,
		"sequence_type": sequence.SequenceType,
		"order":         sequence.Order,
	}
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal metadata: %w", err)
	}

	cmd := exec.CommandContext(ctx, "python3", "./scripts/embeddings/embed.py",
		"--artifact-type", "sequence",
		"--sequence-id", sequence.SequenceID,
		"--tables", string(tablesJSON),
		"--order", fmt.Sprintf("%d", sequence.Order),
		"--metadata", string(metadataJSON),
	)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("failed to generate sequence embedding: %w, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("failed to generate sequence embedding: %w", err)
	}

	var embedding []float32
	if err := json.Unmarshal(output, &embedding); err != nil {
		return nil, fmt.Errorf("failed to unmarshal sequence embedding: %w", err)
	}

	return embedding, nil
}

// generatePetriNetEmbedding generates embedding for Petri net workflow
func generatePetriNetEmbedding(ctx context.Context, petriNet *PetriNet) ([]float32, error) {
	// Convert PetriNet struct to JSON for Python script
	petriNetJSON, err := json.Marshal(petriNet)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Petri net: %w", err)
	}

	cmd := exec.CommandContext(ctx, "python3", "./scripts/embeddings/embed.py",
		"--artifact-type", "petri_net",
		"--petri-net", string(petriNetJSON),
	)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("failed to generate Petri net embedding: %w, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("failed to generate Petri net embedding: %w", err)
	}

	var embedding []float32
	if err := json.Unmarshal(output, &embedding); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Petri net embedding: %w", err)
	}

	return embedding, nil
}
