package digitaltwin

import (
	"context"
	"fmt"
	"log"
	"time"
)

// StressTester performs stress testing on digital twins.
type StressTester struct {
	twinManager   *TwinManager
	loadGenerator *LoadGenerator
	metricsCollector *MetricsCollector
	analyzer      *StressAnalyzer
	logger        *log.Logger
	tests         map[string]*StressTest
}

// StressTest represents a stress test execution.
type StressTest struct {
	ID            string
	TwinID        string
	Status        string // "running", "completed", "failed"
	StartTime     time.Time
	EndTime       time.Time
	Config        StressTestConfig
	Results       StressTestResults
	Metrics       []MetricSample
}

// StressTestConfig configures a stress test.
type StressTestConfig struct {
	Duration       time.Duration
	LoadProfile    LoadProfile
	RampUpTime     time.Duration
	RampDownTime   time.Duration
	TargetRPS      float64 // Requests per second
	MaxConcurrency int
	Metrics        []string
}

// LoadProfile defines how load changes over time.
type LoadProfile struct {
	Type     string // "constant", "linear", "exponential", "spike", "custom"
	Stages   []LoadStage
}

// LoadStage represents a stage in the load profile.
type LoadStage struct {
	Duration    time.Duration
	TargetRPS   float64
	Concurrency int
}

// StressTestResults contains the results of a stress test.
type StressTestResults struct {
	TotalRequests    int64
	SuccessfulRequests int64
	FailedRequests   int64
	AverageLatency   time.Duration
	P50Latency       time.Duration
	P95Latency       time.Duration
	P99Latency       time.Duration
	MaxLatency       time.Duration
	Throughput       float64 // Requests per second
	ErrorRate        float64
	ResourceUsage    ResourceUsage
	Bottlenecks      []Bottleneck
	Recommendations  []string
}

// ResourceUsage tracks resource utilization during stress test.
type ResourceUsage struct {
	CPUUsage    []float64
	MemoryUsage []float64
	DiskIO      []float64
	NetworkIO   []float64
	MaxCPU      float64
	MaxMemory   float64
	PeakTime    time.Time
}

// MetricSample represents a metric sample at a point in time.
type MetricSample struct {
	Timestamp   time.Time
	Metric      string
	Value       float64
	Tags        map[string]string
}

// NewStressTester creates a new stress tester.
func NewStressTester(twinManager *TwinManager, logger *log.Logger) *StressTester {
	return &StressTester{
		twinManager:     twinManager,
		loadGenerator:   NewLoadGenerator(logger),
		metricsCollector: NewMetricsCollector(logger),
		analyzer:        NewStressAnalyzer(logger),
		logger:          logger,
		tests:           make(map[string]*StressTest),
	}
}

// RunStressTest runs a stress test on a twin.
func (st *StressTester) RunStressTest(ctx context.Context, req StressTestRequest) (*StressTest, error) {
	// Get twin
	twin, err := st.twinManager.GetTwin(ctx, req.TwinID)
	if err != nil {
		return nil, fmt.Errorf("twin not found: %w", err)
	}

	// Create test
	test := &StressTest{
		ID:        fmt.Sprintf("stress-%s-%d", req.TwinID, time.Now().UnixNano()),
		TwinID:    req.TwinID,
		Status:    "running",
		StartTime: time.Now(),
		Config:    req.Config,
		Results:   StressTestResults{},
		Metrics:   []MetricSample{},
	}

	st.tests[test.ID] = test

	// Update twin state
	twin.State.Status = "stress_testing"
	st.twinManager.UpdateTwinState(ctx, req.TwinID, twin.State)

	// Run stress test
	go st.executeStressTest(ctx, test)

	if st.logger != nil {
		st.logger.Printf("Started stress test %s for twin %s", test.ID, req.TwinID)
	}

	return test, nil
}

// executeStressTest executes a stress test.
func (st *StressTester) executeStressTest(ctx context.Context, test *StressTest) {
	defer func() {
		test.Status = "completed"
		test.EndTime = time.Now()

		// Analyze results
		st.analyzer.Analyze(test)

		// Update twin state
		twin, err := st.twinManager.GetTwin(ctx, test.TwinID)
		if err == nil {
			twin.State.Status = "active"
			st.twinManager.UpdateTwinState(ctx, test.TwinID, twin.State)
		}
	}()

	// Start metrics collection
	metricsCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	
	go st.metricsCollector.Collect(metricsCtx, test.ID, test.Config.Metrics, func(sample MetricSample) {
		test.Metrics = append(test.Metrics, sample)
	})

	// Generate load
	loadCtx, loadCancel := context.WithTimeout(ctx, test.Config.Duration)
	defer loadCancel()

	st.loadGenerator.GenerateLoad(loadCtx, test, func(request RequestResult) {
		test.Results.TotalRequests++
		if request.Success {
			test.Results.SuccessfulRequests++
		} else {
			test.Results.FailedRequests++
		}

		// Update latency metrics
		st.updateLatencyMetrics(test, request.Latency)
	})

	// Calculate final metrics
	st.calculateFinalMetrics(test)
}

// updateLatencyMetrics updates latency percentiles.
func (st *StressTester) updateLatencyMetrics(test *StressTest, latency time.Duration) {
	// Simplified percentile calculation
	// In production, would use proper percentile tracking
	if test.Results.MaxLatency < latency {
		test.Results.MaxLatency = latency
	}

	// Update average
	totalLatency := test.Results.AverageLatency * time.Duration(test.Results.SuccessfulRequests-1) + latency
	test.Results.AverageLatency = totalLatency / time.Duration(test.Results.SuccessfulRequests)

	// Approximate percentiles (simplified)
	if test.Results.SuccessfulRequests%2 == 0 {
		test.Results.P50Latency = test.Results.AverageLatency
	}
	if test.Results.SuccessfulRequests%20 == 0 {
		test.Results.P95Latency = test.Results.AverageLatency * 2
	}
	if test.Results.SuccessfulRequests%100 == 0 {
		test.Results.P99Latency = test.Results.AverageLatency * 3
	}
}

// calculateFinalMetrics calculates final metrics from test results.
func (st *StressTester) calculateFinalMetrics(test *StressTest) {
	duration := test.EndTime.Sub(test.StartTime)
	if duration > 0 {
		test.Results.Throughput = float64(test.Results.SuccessfulRequests) / duration.Seconds()
	}

	if test.Results.TotalRequests > 0 {
		test.Results.ErrorRate = float64(test.Results.FailedRequests) / float64(test.Results.TotalRequests) * 100.0
	}

	// Calculate resource usage peaks
	if len(test.Metrics) > 0 {
		for _, metric := range test.Metrics {
			if metric.Metric == "cpu" && metric.Value > test.Results.ResourceUsage.MaxCPU {
				test.Results.ResourceUsage.MaxCPU = metric.Value
				test.Results.ResourceUsage.PeakTime = metric.Timestamp
			}
			if metric.Metric == "memory" && metric.Value > test.Results.ResourceUsage.MaxMemory {
				test.Results.ResourceUsage.MaxMemory = metric.Value
			}
		}
	}
}

// GetStressTest retrieves a stress test by ID.
func (st *StressTester) GetStressTest(id string) (*StressTest, error) {
	test, exists := st.tests[id]
	if !exists {
		return nil, fmt.Errorf("stress test not found: %s", id)
	}
	return test, nil
}

// StressTestRequest represents a request to run a stress test.
type StressTestRequest struct {
	TwinID string
	Config StressTestConfig
}

// RequestResult represents the result of a single request.
type RequestResult struct {
	Success bool
	Latency time.Duration
	Error   string
}

// LoadGenerator generates load for stress testing.
type LoadGenerator struct {
	logger *log.Logger
}

// NewLoadGenerator creates a new load generator.
func NewLoadGenerator(logger *log.Logger) *LoadGenerator {
	return &LoadGenerator{
		logger: logger,
	}
}

// GenerateLoad generates load according to the load profile.
func (lg *LoadGenerator) GenerateLoad(ctx context.Context, test *StressTest, callback func(RequestResult)) {
	profile := test.Config.LoadProfile

	// Execute load stages
	for _, stage := range profile.Stages {
		stageCtx, cancel := context.WithTimeout(ctx, stage.Duration)
		
		// Generate requests at target RPS
		interval := time.Duration(float64(time.Second) / stage.TargetRPS)
		ticker := time.NewTicker(interval)
		
		for stageCtx.Err() == nil {
			select {
			case <-stageCtx.Done():
				ticker.Stop()
				cancel()
				return
			case <-ticker.C:
				// Generate request
				start := time.Now()
				success := lg.executeRequest(test)
				latency := time.Since(start)

				callback(RequestResult{
					Success: success,
					Latency: latency,
				})
			}
		}
		cancel()
	}
}

// executeRequest executes a single request (simplified).
func (lg *LoadGenerator) executeRequest(test *StressTest) bool {
	// Simulate request execution
	time.Sleep(time.Millisecond * 10)
	return true // Simplified - would actually call twin
}

// MetricsCollector collects metrics during stress testing.
type MetricsCollector struct {
	logger *log.Logger
}

// NewMetricsCollector creates a new metrics collector.
func NewMetricsCollector(logger *log.Logger) *MetricsCollector {
	return &MetricsCollector{
		logger: logger,
	}
}

// Collect collects metrics during stress testing.
func (mc *MetricsCollector) Collect(ctx context.Context, testID string, metrics []string, callback func(MetricSample)) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Collect metrics
			for _, metric := range metrics {
				value := mc.collectMetric(metric)
				callback(MetricSample{
					Timestamp: time.Now(),
					Metric:    metric,
					Value:     value,
					Tags:      map[string]string{"test_id": testID},
				})
			}
		}
	}
}

// collectMetric collects a single metric (simplified).
func (mc *MetricsCollector) collectMetric(name string) float64 {
	// In production, would collect actual metrics
	return 50.0 // Placeholder
}

// StressAnalyzer analyzes stress test results.
type StressAnalyzer struct {
	logger *log.Logger
}

// NewStressAnalyzer creates a new stress analyzer.
func NewStressAnalyzer(logger *log.Logger) *StressAnalyzer {
	return &StressAnalyzer{
		logger: logger,
	}
}

// Analyze analyzes stress test results and identifies issues.
func (sa *StressAnalyzer) Analyze(test *StressTest) {
	// Identify bottlenecks
	if test.Results.ErrorRate > 1.0 {
		test.Results.Bottlenecks = append(test.Results.Bottlenecks, Bottleneck{
			Component:    "overall",
			Type:         "reliability",
			Severity:     "high",
			Description:  fmt.Sprintf("High error rate: %.2f%%", test.Results.ErrorRate),
			Impact:       test.Results.ErrorRate,
			Recommendation: "Review error handling and system capacity",
		})
	}

	if test.Results.AverageLatency > 1*time.Second {
		test.Results.Bottlenecks = append(test.Results.Bottlenecks, Bottleneck{
			Component:    "performance",
			Type:         "latency",
			Severity:     "medium",
			Description:  fmt.Sprintf("High average latency: %v", test.Results.AverageLatency),
			Impact:       float64(test.Results.AverageLatency.Milliseconds()),
			Recommendation: "Optimize critical paths and reduce processing time",
		})
	}

	// Generate recommendations
	if test.Results.ResourceUsage.MaxCPU > 80.0 {
		test.Results.Recommendations = append(test.Results.Recommendations,
			"Consider scaling horizontally to reduce CPU usage")
	}

	if test.Results.ErrorRate > 5.0 {
		test.Results.Recommendations = append(test.Results.Recommendations,
			"Increase system capacity or implement rate limiting")
	}
}

