package mappers

import (
	"ai_benchmarks/scripts/factory"
	"fmt"
	"strconv"
	"strings"
)

// HellaSwagTask defines the structure for a HellaSwag completion task.
// Matches the actual HellaSwag dataset format with adversarial filtering.
type HellaSwagTask struct {
	IndActivity string   `json:"ind_activity,omitempty"` // ActivityNet category
	IndWikiHow  string   `json:"ind_wikihow,omitempty"`  // WikiHow category
	Context     string   `json:"ctx"`                    // 2-sentence context
	ContextA    string   `json:"ctx_a,omitempty"`        // First sentence
	ContextB    string   `json:"ctx_b,omitempty"`        // Second sentence
	Endings     []string `json:"endings"`                // 4 endings (1 gold + 3 AF)
	Label       int      `json:"label"`                  // Correct ending index
	SourceID    string   `json:"source_id,omitempty"`    // Original video/article ID
	Split       string   `json:"split,omitempty"`        // train/val/test
	SplitType   string   `json:"split_type,omitempty"`   // in-domain/zero-shot
}

// HellaSwagMapper transforms data from the real HellaSwag dataset.
// IMPORTANT: This expects properly formatted HellaSwag data with adversarial filtering.
// Do NOT use this for generating synthetic examples - use the actual dataset.
type HellaSwagMapper struct{}

func (m *HellaSwagMapper) Map(data factory.SourceData) ([]factory.BenchmarkTask, error) {
	row, ok := data.Content.(map[string]string)
	if !ok {
		return nil, fmt.Errorf("invalid data type for HellaSwagMapper")
	}

	// Extract context (must be 2-sentence for proper difficulty)
	context := row["ctx"]
	if context == "" {
		context = row["context"]
	}
	if context == "" {
		return nil, fmt.Errorf("no context field found")
	}

	// Parse endings - MUST have 4 endings from adversarial filtering
	endings := parseEndings(row)
	if len(endings) != 4 {
		return nil, fmt.Errorf("HellaSwag requires exactly 4 endings (1 gold + 3 adversarial), got %d", len(endings))
	}

	// Parse label
	label := 0
	if labelStr := row["label"]; labelStr != "" {
		if parsed, err := strconv.Atoi(labelStr); err == nil {
			label = parsed
		}
	}

	task := HellaSwagTask{
		IndActivity: row["ind_activity"],
		IndWikiHow:  row["ind_wikihow"],
		Context:     context,
		ContextA:    row["ctx_a"],
		ContextB:    row["ctx_b"],
		Endings:     endings,
		Label:       label,
		SourceID:    row["source_id"],
		Split:       row["split"],
		SplitType:   row["split_type"],
	}

	return []factory.BenchmarkTask{task}, nil
}

// parseEndings extracts the 4 endings from various possible formats
func parseEndings(row map[string]string) []string {
	// Try comma-separated endings
	if endingsStr := row["endings"]; endingsStr != "" {
		endings := strings.Split(endingsStr, "|")
		if len(endings) == 4 {
			return endings
		}
	}

	// Try individual ending fields
	endings := []string{}
	for i := 0; i < 4; i++ {
		key := fmt.Sprintf("ending%d", i)
		if ending := row[key]; ending != "" {
			endings = append(endings, ending)
		}
	}

	if len(endings) == 4 {
		return endings
	}

	// Try alternative naming
	for _, suffix := range []string{"0", "1", "2", "3"} {
		if ending := row["ending_"+suffix]; ending != "" {
			endings = append(endings, ending)
		}
	}

	return endings
}
