// +build !notelemetry

package monitoring

import (
	"context"
	"crypto/sha256"
	"fmt"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
)

// TelemetryPromptTokens records the tokenized structure of a prompt for observability.
type TelemetryPromptTokens struct {
	// PromptID is a unique identifier for this prompt instance.
	PromptID string
	// Tokens is the hierarchical token tree produced by TokenAwarePromptValue.
	Tokens []llms.Token
	// TemplateType indicates the prompt type (template, chat, few-shot).
	TemplateType string
	// VariableCount is the number of resolved variables.
	VariableCount int
}

// TokensToMap converts token trees into a telemetry-friendly map.
func TokensToMap(tokens []llms.Token) []map[string]any {
	var recurse func([]llms.Token) []map[string]any
	recurse = func(ts []llms.Token) []map[string]any {
		out := make([]map[string]any, 0, len(ts))
		for _, t := range ts {
			entry := map[string]any{
				"type":  t.Type,
				"value": t.Value,
			}
			if len(t.Metadata) > 0 {
				entry["metadata"] = t.Metadata
			}
			if len(t.Children) > 0 {
				entry["children"] = recurse(t.Children)
			}
			out = append(out, entry)
		}
		return out
	}
	return recurse(tokens)
}

// LogPromptTokens emits a telemetry record containing the tokenized prompt.
func (c *TelemetryClient) LogPromptTokens(ctx context.Context, promptID, templateType string, tokens []llms.Token, variableCount int) error {
	return c.Log(ctx, TelemetryRecord{
		Operation: "prompt_tokens",
		Input: map[string]any{
			"prompt_id":      promptID,
			"template_type":  templateType,
			"variable_count": variableCount,
			"tokens":         TokensToMap(tokens),
		},
	})
}

// PromptID generates a deterministic ID for a prompt instance.
func PromptID(template string, vars map[string]any) string {
	h := sha256.New()
	h.Write([]byte(template))
	for k, v := range vars {
		h.Write([]byte(fmt.Sprintf("%s=%v", k, v)))
	}
	return fmt.Sprintf("%x", h.Sum(nil))[:12]
}
