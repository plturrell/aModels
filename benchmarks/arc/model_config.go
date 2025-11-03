package arc

import "time"

// ModelConfig stores model-specific HINTS for ARC tasks
// These are initial suggestions, but the LNN will learn better values through training
type ModelConfig struct {
	Name             string
	ExpectedAccuracy float64 // Target accuracy (for evaluation only)
	TimeoutSeconds   int     // Resource constraint
	MaxComplexity    string  // Complexity hint
	// The following are INITIAL VALUES only - LNN will optimize these
	ReasoningDepth    int
	MCTSRollouts      int
	UseFastHeuristics bool
	EnableSynthesis   bool
	TransformMask     int
	MemoryBudgetMB    int
	BatchSize         int
}

// ModelRegistry maintains configurations for different models
var ModelRegistry = map[string]ModelConfig{
	"GemmaVault": {
		Name:              "GemmaVault",
		ExpectedAccuracy:  0.85,
		TimeoutSeconds:    300,
		MaxComplexity:     "high",
		ReasoningDepth:    5,
		MCTSRollouts:      1000,
		UseFastHeuristics: false,
		EnableSynthesis:   true,
		TransformMask:     255, // All transforms
		MemoryBudgetMB:    4096,
		BatchSize:         16,
	},
	"Phi-Mini-3.5": {
		Name:              "Phi-Mini-3.5",
		ExpectedAccuracy:  0.65,
		TimeoutSeconds:    180,
		MaxComplexity:     "medium",
		ReasoningDepth:    3,
		MCTSRollouts:      200,
		UseFastHeuristics: true,
		EnableSynthesis:   false,
		TransformMask:     63, // Core transforms only
		MemoryBudgetMB:    1024,
		BatchSize:         8,
	},
}

// GetModelConfig retrieves configuration for a specific model
func GetModelConfig(modelName string) ModelConfig {
	if cfg, ok := ModelRegistry[modelName]; ok {
		return cfg
	}
	// Return default configuration
	return ModelConfig{
		Name:              "default",
		ExpectedAccuracy:  0.50,
		TimeoutSeconds:    120,
		MaxComplexity:     "low",
		ReasoningDepth:    2,
		MCTSRollouts:      100,
		UseFastHeuristics: true,
		EnableSynthesis:   false,
		TransformMask:     31, // Basic transforms only
		MemoryBudgetMB:    512,
		BatchSize:         4,
	}
}

// GetTimeout returns the timeout duration for a model
func (mc ModelConfig) GetTimeout() time.Duration {
	return time.Duration(mc.TimeoutSeconds) * time.Second
}

// ToParams converts ModelConfig to parameter map for RunOptions
func (mc ModelConfig) ToParams() map[string]float64 {
	params := make(map[string]float64)

	params["arc_depth"] = float64(mc.ReasoningDepth)
	params["mcts_rollouts"] = float64(mc.MCTSRollouts)
	params["arc_mask"] = float64(mc.TransformMask)

	if mc.EnableSynthesis {
		params["arc_synthesis"] = 1.0
	} else {
		params["arc_synthesis"] = 0.0
	}

	if mc.UseFastHeuristics {
		params["palette_soft"] = 1.0
	} else {
		params["palette_soft"] = 0.0
	}

	return params
}

// AutoDetectModel attempts to detect model type from model string
func AutoDetectModel(modelStr string) string {
	modelLower := ""
	for _, c := range modelStr {
		if c >= 'A' && c <= 'Z' {
			modelLower += string(c + 32)
		} else {
			modelLower += string(c)
		}
	}

	// Check for known model patterns (only Gemma and Phi exist)
	if contains(modelLower, "gemma") {
		return "GemmaVault"
	}
	if contains(modelLower, "phi") {
		return "Phi-Mini-3.5"
	}

	return "default"
}

func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// MergeParams merges model config params with user-provided params
// User params take precedence
func MergeParams(modelParams, userParams map[string]float64) map[string]float64 {
	merged := make(map[string]float64)

	// Start with model defaults
	for k, v := range modelParams {
		merged[k] = v
	}

	// Override with user params
	for k, v := range userParams {
		merged[k] = v
	}

	return merged
}
