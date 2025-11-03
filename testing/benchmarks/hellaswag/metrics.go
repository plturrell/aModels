package hellaswag

import (
	"fmt"
	"sync"
	"time"
)

// FilterMetrics tracks adversarial filtering effectiveness
type FilterMetrics struct {
	mu sync.RWMutex

	// Generation metrics
	TotalGenerated     int
	SuccessfulFiltered int
	FailedFiltered     int
	AverageConfusion   float64

	// Performance metrics
	GenerationTime time.Duration
	FilteringTime  time.Duration
	TotalTime      time.Duration

	// Quality metrics
	HighQualityCount   int // Confusion > 0.7
	MediumQualityCount int // Confusion 0.4-0.7
	LowQualityCount    int // Confusion < 0.4

	// Distribution
	ConfusionDistribution map[string]int // Buckets: 0-0.2, 0.2-0.4, etc.

	// Errors
	Errors []error
}

func NewFilterMetrics() *FilterMetrics {
	return &FilterMetrics{
		ConfusionDistribution: make(map[string]int),
	}
}

func (m *FilterMetrics) RecordGeneration(count int, duration time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.TotalGenerated += count
	m.GenerationTime += duration
}

func (m *FilterMetrics) RecordFiltering(success bool, confusion float64, duration time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if success {
		m.SuccessfulFiltered++

		// Update average confusion
		total := m.AverageConfusion * float64(m.SuccessfulFiltered-1)
		m.AverageConfusion = (total + confusion) / float64(m.SuccessfulFiltered)

		// Quality buckets
		if confusion > 0.7 {
			m.HighQualityCount++
		} else if confusion > 0.4 {
			m.MediumQualityCount++
		} else {
			m.LowQualityCount++
		}

		// Distribution
		bucket := fmt.Sprintf("%.1f-%.1f", float64(int(confusion*10))/10, float64(int(confusion*10)+1)/10)
		m.ConfusionDistribution[bucket]++
	} else {
		m.FailedFiltered++
	}

	m.FilteringTime += duration
}

func (m *FilterMetrics) RecordError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.Errors = append(m.Errors, err)
}

func (m *FilterMetrics) GetSummary() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	summary := make(map[string]interface{})

	// Basic stats
	summary["total_generated"] = m.TotalGenerated
	summary["successful_filtered"] = m.SuccessfulFiltered
	summary["failed_filtered"] = m.FailedFiltered
	summary["success_rate"] = float64(m.SuccessfulFiltered) / float64(m.TotalGenerated)

	// Quality
	summary["average_confusion"] = m.AverageConfusion
	summary["high_quality_pct"] = float64(m.HighQualityCount) / float64(m.SuccessfulFiltered)
	summary["medium_quality_pct"] = float64(m.MediumQualityCount) / float64(m.SuccessfulFiltered)
	summary["low_quality_pct"] = float64(m.LowQualityCount) / float64(m.SuccessfulFiltered)

	// Performance
	summary["generation_time_ms"] = m.GenerationTime.Milliseconds()
	summary["filtering_time_ms"] = m.FilteringTime.Milliseconds()
	summary["total_time_ms"] = (m.GenerationTime + m.FilteringTime).Milliseconds()

	if m.SuccessfulFiltered > 0 {
		summary["avg_time_per_distractor_ms"] = (m.GenerationTime + m.FilteringTime).Milliseconds() / int64(m.SuccessfulFiltered)
	}

	// Distribution
	summary["confusion_distribution"] = m.ConfusionDistribution

	// Errors
	summary["error_count"] = len(m.Errors)

	return summary
}

func (m *FilterMetrics) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.TotalGenerated = 0
	m.SuccessfulFiltered = 0
	m.FailedFiltered = 0
	m.AverageConfusion = 0.0
	m.GenerationTime = 0
	m.FilteringTime = 0
	m.TotalTime = 0
	m.HighQualityCount = 0
	m.MediumQualityCount = 0
	m.LowQualityCount = 0
	m.ConfusionDistribution = make(map[string]int)
	m.Errors = nil
}

// MetricsCollector wraps filter with metrics collection
type MetricsCollector struct {
	Filter  *AdversarialFilter
	Metrics *FilterMetrics
}

func NewMetricsCollector(filter *AdversarialFilter) *MetricsCollector {
	return &MetricsCollector{
		Filter:  filter,
		Metrics: NewFilterMetrics(),
	}
}

func (mc *MetricsCollector) GenerateDistractors(context, goldEnding string, numDistractors int) ([]FilteredEnding, error) {
	startTotal := time.Now()

	// Generation phase
	startGen := time.Now()
	candidates, err := mc.Filter.Generator.Generate(context, mc.Filter.Config.NumCandidates*numDistractors)
	genDuration := time.Since(startGen)
	mc.Metrics.RecordGeneration(len(candidates), genDuration)

	if err != nil {
		mc.Metrics.RecordError(err)
		return nil, err
	}

	// Filtering phase
	startFilter := time.Now()
	scored := make([]FilteredEnding, 0, len(candidates))

	for _, candidate := range candidates {
		if isTooSimilar(candidate, goldEnding) {
			continue
		}

		score, err := mc.Filter.Discriminator.Score(context, candidate)
		if err != nil {
			mc.Metrics.RecordError(err)
			continue
		}

		if score >= mc.Filter.Config.MinBERTConfusion {
			scored = append(scored, FilteredEnding{
				Text:      candidate,
				BERTScore: score,
			})
			mc.Metrics.RecordFiltering(true, score, time.Since(startFilter))
		} else {
			mc.Metrics.RecordFiltering(false, score, time.Since(startFilter))
		}
	}

	filterDuration := time.Since(startFilter)
	mc.Metrics.FilteringTime += filterDuration

	// Sort and select
	sortByConfusion(scored)

	if len(scored) < numDistractors {
		err := fmt.Errorf("insufficient distractors: got %d, need %d", len(scored), numDistractors)
		mc.Metrics.RecordError(err)
		return scored, err
	}

	result := scored[:numDistractors]
	for i := range result {
		result[i].ConfusionRank = i + 1
	}

	mc.Metrics.TotalTime = time.Since(startTotal)

	return result, nil
}

func (mc *MetricsCollector) GetMetrics() *FilterMetrics {
	return mc.Metrics
}

func (mc *MetricsCollector) PrintSummary() {
	summary := mc.Metrics.GetSummary()

	fmt.Println("\n=== Adversarial Filtering Metrics ===")
	fmt.Printf("Total Generated: %d\n", summary["total_generated"])
	fmt.Printf("Successfully Filtered: %d\n", summary["successful_filtered"])
	fmt.Printf("Success Rate: %.2f%%\n", summary["success_rate"].(float64)*100)
	fmt.Printf("\nQuality Distribution:\n")
	fmt.Printf("  High (>0.7): %.1f%%\n", summary["high_quality_pct"].(float64)*100)
	fmt.Printf("  Medium (0.4-0.7): %.1f%%\n", summary["medium_quality_pct"].(float64)*100)
	fmt.Printf("  Low (<0.4): %.1f%%\n", summary["low_quality_pct"].(float64)*100)
	fmt.Printf("\nPerformance:\n")
	fmt.Printf("  Generation Time: %dms\n", summary["generation_time_ms"])
	fmt.Printf("  Filtering Time: %dms\n", summary["filtering_time_ms"])
	if avgTime, ok := summary["avg_time_per_distractor_ms"]; ok {
		fmt.Printf("  Avg per Distractor: %dms\n", avgTime)
	}
	fmt.Printf("\nErrors: %d\n", summary["error_count"])
}
