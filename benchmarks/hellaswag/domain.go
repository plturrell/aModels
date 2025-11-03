package hellaswag

// Domain handling for HellaSwag
// Implements ActivityNet vs WikiHow distinction and zero-shot evaluation

// Domain represents the source of HellaSwag data
type Domain string

const (
	ActivityNet Domain = "activitynet"
	WikiHow     Domain = "wikihow"
)

// SplitType represents evaluation split
type SplitType string

const (
	InDomain SplitType = "in-domain"
	ZeroShot SplitType = "zero-shot"
)

// DomainMetadata stores domain-specific information
type DomainMetadata struct {
	Domain       Domain
	Category     string // Fine-grained category
	SplitType    SplitType
	SourceID     string
	VideoID      string // For ActivityNet
	ArticleTitle string // For WikiHow
}

// DomainClassifier determines the domain of an example
type DomainClassifier struct {
	ActivityNetCategories map[string]bool
	WikiHowCategories     map[string]bool
}

// NewDomainClassifier creates a classifier with known categories
func NewDomainClassifier() *DomainClassifier {
	return &DomainClassifier{
		ActivityNetCategories: activityNetCategories(),
		WikiHowCategories:     wikiHowCategories(),
	}
}

func (dc *DomainClassifier) Classify(category string) Domain {
	if dc.ActivityNetCategories[category] {
		return ActivityNet
	}
	if dc.WikiHowCategories[category] {
		return WikiHow
	}
	// Default to ActivityNet if unknown
	return ActivityNet
}

// DomainEvaluator tracks performance by domain
type DomainEvaluator struct {
	ActivityNetCorrect int
	ActivityNetTotal   int
	WikiHowCorrect     int
	WikiHowTotal       int
	InDomainCorrect    int
	InDomainTotal      int
	ZeroShotCorrect    int
	ZeroShotTotal      int
}

// Update adds a result to the evaluator
func (de *DomainEvaluator) Update(metadata DomainMetadata, correct bool) {
	// Update by domain
	switch metadata.Domain {
	case ActivityNet:
		de.ActivityNetTotal++
		if correct {
			de.ActivityNetCorrect++
		}
	case WikiHow:
		de.WikiHowTotal++
		if correct {
			de.WikiHowCorrect++
		}
	}

	// Update by split type
	switch metadata.SplitType {
	case InDomain:
		de.InDomainTotal++
		if correct {
			de.InDomainCorrect++
		}
	case ZeroShot:
		de.ZeroShotTotal++
		if correct {
			de.ZeroShotCorrect++
		}
	}
}

// GetMetrics returns evaluation metrics
func (de *DomainEvaluator) GetMetrics() map[string]float64 {
	metrics := make(map[string]float64)

	if de.ActivityNetTotal > 0 {
		metrics["activitynet_accuracy"] = float64(de.ActivityNetCorrect) / float64(de.ActivityNetTotal)
	}
	if de.WikiHowTotal > 0 {
		metrics["wikihow_accuracy"] = float64(de.WikiHowCorrect) / float64(de.WikiHowTotal)
	}
	if de.InDomainTotal > 0 {
		metrics["in_domain_accuracy"] = float64(de.InDomainCorrect) / float64(de.InDomainTotal)
	}
	if de.ZeroShotTotal > 0 {
		metrics["zero_shot_accuracy"] = float64(de.ZeroShotCorrect) / float64(de.ZeroShotTotal)
	}

	// Calculate generalization gap
	if de.InDomainTotal > 0 && de.ZeroShotTotal > 0 {
		inDomainAcc := float64(de.InDomainCorrect) / float64(de.InDomainTotal)
		zeroShotAcc := float64(de.ZeroShotCorrect) / float64(de.ZeroShotTotal)
		metrics["generalization_gap"] = inDomainAcc - zeroShotAcc
	}

	return metrics
}

// ActivityNet categories from the paper
func activityNetCategories() map[string]bool {
	return map[string]bool{
		"Painting":                 true,
		"Doing motocross":          true,
		"BMX":                      true,
		"Surfing":                  true,
		"Skateboarding":            true,
		"Playing pool":             true,
		"Doing a powerbomb":        true,
		"Breakdancing":             true,
		"Playing beach volleyball": true,
		"Doing kickboxing":         true,
		// Add more as needed
	}
}

// WikiHow categories from the paper
func wikiHowCategories() map[string]bool {
	return map[string]bool{
		"Home and Garden":              true,
		"Health":                       true,
		"Hobbies and Crafts":           true,
		"Food and Entertaining":        true,
		"Personal Care and Style":      true,
		"Travel":                       true,
		"Education and Communications": true,
		"Relationships":                true,
		"Cars & Other Vehicles":        true,
		"Finance and Business":         true,
		// Add more as needed
	}
}

// ContextLengthAnalyzer validates the Goldilocks zone (2 sentences)
type ContextLengthAnalyzer struct {
	OneSentenceCount int
	TwoSentenceCount int
	ThreePlusCount   int
}

func (cla *ContextLengthAnalyzer) Analyze(context string) int {
	// Count sentences (simple heuristic)
	sentences := 0
	for _, char := range context {
		if char == '.' || char == '!' || char == '?' {
			sentences++
		}
	}

	// Update counts
	switch {
	case sentences == 1:
		cla.OneSentenceCount++
	case sentences == 2:
		cla.TwoSentenceCount++
	default:
		cla.ThreePlusCount++
	}

	return sentences
}

func (cla *ContextLengthAnalyzer) GetDistribution() map[string]float64 {
	total := cla.OneSentenceCount + cla.TwoSentenceCount + cla.ThreePlusCount
	if total == 0 {
		return map[string]float64{}
	}

	return map[string]float64{
		"one_sentence":  float64(cla.OneSentenceCount) / float64(total),
		"two_sentences": float64(cla.TwoSentenceCount) / float64(total),
		"three_plus":    float64(cla.ThreePlusCount) / float64(total),
	}
}
