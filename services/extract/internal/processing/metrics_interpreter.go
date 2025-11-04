package processing

import (
	"fmt"
	"log"
)

// MetricsThresholds defines thresholds for information theory metrics
type MetricsThresholds struct {
	// Entropy thresholds
	LowEntropyThreshold  float64 // Warn if entropy < this (default: 1.0)
	HighEntropyThreshold float64 // Info if entropy > this (default: 4.0)
	
	// KL Divergence thresholds
	WarningKLThreshold float64 // Warn if KL divergence > this (default: 0.5)
	ErrorKLThreshold   float64 // Error if KL divergence > this (default: 1.0)
	
	// Data quality thresholds
	MinColumnCount int // Minimum columns required for meaningful metrics (default: 5)
}

// DefaultMetricsThresholds returns sensible default thresholds
func DefaultMetricsThresholds() MetricsThresholds {
	return MetricsThresholds{
		LowEntropyThreshold:  1.0,
		HighEntropyThreshold: 4.0,
		WarningKLThreshold:   0.5,
		ErrorKLThreshold:     1.0,
		MinColumnCount:       5,
	}
}

// MetricsInterpretation represents the interpretation of metrics
type MetricsInterpretation struct {
	// Metric values
	MetadataEntropy   float64
	KLDivergence      float64
	ColumnCount       int
	ActualDistribution map[string]float64
	IdealDistribution  map[string]float64
	
	// Interpretation results
	QualityScore      float64 // 0.0-1.0, higher is better
	QualityLevel      string  // "excellent", "good", "fair", "poor", "critical"
	Issues            []string
	Recommendations   []string
	ShouldReject      bool
	ShouldWarn        bool
	ShouldRetry       bool
	
	// Processing suggestions
	ProcessingStrategy string // "standard", "enhanced", "simplified", "skip"
	NeedsValidation    bool
	NeedsReview        bool
}

// InterpretMetrics analyzes metrics and provides actionable interpretation
func InterpretMetrics(
	metadataEntropy float64,
	klDivergence float64,
	columnCount int,
	actualDistribution map[string]float64,
	idealDistribution map[string]float64,
	thresholds MetricsThresholds,
	logger *log.Logger,
) MetricsInterpretation {
	interpretation := MetricsInterpretation{
		MetadataEntropy:    metadataEntropy,
		KLDivergence:       klDivergence,
		ColumnCount:        columnCount,
		ActualDistribution: actualDistribution,
		IdealDistribution:   idealDistribution,
		Issues:             []string{},
		Recommendations:    []string{},
	}
	
	// Calculate quality score (0.0-1.0)
	// Higher entropy is generally good (more diversity), but not too high
	// Lower KL divergence is better (closer to ideal)
	entropyScore := 0.5
	if metadataEntropy >= thresholds.LowEntropyThreshold && metadataEntropy <= thresholds.HighEntropyThreshold {
		entropyScore = 1.0
	} else if metadataEntropy < thresholds.LowEntropyThreshold {
		entropyScore = metadataEntropy / thresholds.LowEntropyThreshold * 0.5
	} else {
		// Very high entropy might indicate inconsistency
		entropyScore = 1.0 - (metadataEntropy-thresholds.HighEntropyThreshold)/10.0
		if entropyScore < 0.3 {
			entropyScore = 0.3
		}
	}
	
	klScore := 1.0
	if klDivergence > thresholds.ErrorKLThreshold {
		klScore = 0.0
	} else if klDivergence > thresholds.WarningKLThreshold {
		klScore = 1.0 - (klDivergence-thresholds.WarningKLThreshold)/(thresholds.ErrorKLThreshold-thresholds.WarningKLThreshold)*0.5
	}
	
	columnScore := 1.0
	if columnCount < thresholds.MinColumnCount {
		columnScore = float64(columnCount) / float64(thresholds.MinColumnCount)
	}
	
	interpretation.QualityScore = (entropyScore*0.4 + klScore*0.4 + columnScore*0.2)
	
	// Determine quality level
	if interpretation.QualityScore >= 0.9 {
		interpretation.QualityLevel = "excellent"
	} else if interpretation.QualityScore >= 0.7 {
		interpretation.QualityLevel = "good"
	} else if interpretation.QualityScore >= 0.5 {
		interpretation.QualityLevel = "fair"
	} else if interpretation.QualityScore >= 0.3 {
		interpretation.QualityLevel = "poor"
	} else {
		interpretation.QualityLevel = "critical"
	}
	
	// Identify issues
	if columnCount < thresholds.MinColumnCount {
		interpretation.Issues = append(interpretation.Issues, 
			fmt.Sprintf("Low column count (%d) - insufficient data for reliable metrics", columnCount))
		interpretation.ShouldWarn = true
	}
	
	if metadataEntropy < thresholds.LowEntropyThreshold {
		interpretation.Issues = append(interpretation.Issues,
			fmt.Sprintf("Low metadata entropy (%.3f) - schema has low diversity, may indicate homogeneous data types", metadataEntropy))
		interpretation.ShouldWarn = true
		interpretation.NeedsReview = true
	}
	
	if metadataEntropy > thresholds.HighEntropyThreshold {
		interpretation.Issues = append(interpretation.Issues,
			fmt.Sprintf("High metadata entropy (%.3f) - schema has high diversity, may indicate inconsistent data types", metadataEntropy))
		interpretation.ShouldWarn = true
		interpretation.NeedsValidation = true
	}
	
	if klDivergence > thresholds.WarningKLThreshold {
		interpretation.Issues = append(interpretation.Issues,
			fmt.Sprintf("High KL divergence (%.3f) - data type distribution deviates significantly from ideal", klDivergence))
		interpretation.ShouldWarn = true
		interpretation.NeedsValidation = true
	}
	
	if klDivergence > thresholds.ErrorKLThreshold {
		interpretation.Issues = append(interpretation.Issues,
			fmt.Sprintf("Very high KL divergence (%.3f) - data type distribution is highly abnormal, data quality concerns", klDivergence))
		interpretation.ShouldWarn = true
		interpretation.ShouldReject = true
		interpretation.NeedsReview = true
	}
	
	// Generate recommendations
	if metadataEntropy < thresholds.LowEntropyThreshold {
		interpretation.Recommendations = append(interpretation.Recommendations,
			"Consider enriching schema with additional data types or sources")
	}
	
	if klDivergence > thresholds.WarningKLThreshold {
		interpretation.Recommendations = append(interpretation.Recommendations,
			"Review data sources for type consistency and data quality issues")
		interpretation.Recommendations = append(interpretation.Recommendations,
			"Consider adjusting ideal_distribution to match actual data patterns")
	}
	
	if columnCount < thresholds.MinColumnCount {
		interpretation.Recommendations = append(interpretation.Recommendations,
			"Include more columns or sources to improve metric reliability")
	}
	
	// Determine processing strategy
	if interpretation.QualityLevel == "critical" || interpretation.ShouldReject {
		interpretation.ProcessingStrategy = "skip"
	} else if interpretation.QualityLevel == "poor" {
		interpretation.ProcessingStrategy = "simplified"
	} else if interpretation.NeedsValidation {
		interpretation.ProcessingStrategy = "enhanced"
	} else {
		interpretation.ProcessingStrategy = "standard"
	}
	
	// Log interpretation
	if logger != nil {
		logger.Printf("Metrics interpretation: quality=%s (score=%.2f), strategy=%s, issues=%d, recommendations=%d",
			interpretation.QualityLevel,
			interpretation.QualityScore,
			interpretation.ProcessingStrategy,
			len(interpretation.Issues),
			len(interpretation.Recommendations))
		
		if len(interpretation.Issues) > 0 {
			for _, issue := range interpretation.Issues {
				logger.Printf("  Issue: %s", issue)
			}
		}
		if len(interpretation.Recommendations) > 0 {
			for _, rec := range interpretation.Recommendations {
				logger.Printf("  Recommendation: %s", rec)
			}
		}
	}
	
	return interpretation
}

// ShouldProcessGraph determines if graph processing should proceed based on metrics
func ShouldProcessGraph(interpretation MetricsInterpretation) bool {
	return !interpretation.ShouldReject
}

// GetProcessingFlags returns flags for how to process the graph
func GetProcessingFlags(interpretation MetricsInterpretation) map[string]bool {
	return map[string]bool{
		"skip_validation":    !interpretation.NeedsValidation,
		"skip_review":        !interpretation.NeedsReview,
		"enhanced_processing": interpretation.ProcessingStrategy == "enhanced",
		"simplified_processing": interpretation.ProcessingStrategy == "simplified",
		"skip_processing":     interpretation.ProcessingStrategy == "skip",
	}
}

