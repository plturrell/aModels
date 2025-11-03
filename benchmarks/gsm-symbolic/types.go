package gsmsymbolic

// GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs
// Paper: https://machinelearning.apple.com/research/gsm-symbolic
// GitHub: https://github.com/apple/ml-gsm-symbolic

// Example represents a GSM-Symbolic math word problem
type Example struct {
	ID               int    `json:"id"`                // Template ID
	Instance         int    `json:"instance"`          // Instance ID (0-49)
	Question         string `json:"question"`          // The math word problem
	Answer           string `json:"answer"`            // Step-by-step solution with #### final_answer
	OriginalID       int    `json:"original_id"`       // GSM8K test set ID
	OriginalQuestion string `json:"original_question"` // Original GSM8K question
	OriginalAnswer   string `json:"original_answer"`   // Original GSM8K answer
	Canary           string `json:"canary"`            // Canary GUID for contamination detection
}

// Variant represents different GSM-Symbolic dataset variants
type Variant string

const (
	VariantSymbolic   Variant = "GSM_symbolic"    // Base symbolic variant
	VariantSymbolicP1 Variant = "GSM_symbolic_p1" // Perturbation 1
	VariantSymbolicP2 Variant = "GSM_symbolic_p2" // Perturbation 2
)

// InstanceGroup groups multiple instances of the same template
type InstanceGroup struct {
	TemplateID int
	Instances  []Example
}

// PerformanceMetrics tracks performance across instances
type PerformanceMetrics struct {
	TemplateID    int
	InstanceCount int
	CorrectCount  int
	Accuracy      float64
	Variance      float64 // Variance in performance across instances
	MinAccuracy   float64
	MaxAccuracy   float64
}
