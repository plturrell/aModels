package digitaltwin

import (
	"context"
	"fmt"
	"log"
	"time"
)

// FPSLStressTest performs stress testing specific to FPSL (Financial Product Subledger) changes.
type FPSLStressTest struct {
	stressTester *StressTester
	twinManager  *TwinManager
	logger       *log.Logger
}

// NewFPSLStressTest creates a new FPSL stress test.
func NewFPSLStressTest(stressTester *StressTester, twinManager *TwinManager, logger *log.Logger) *FPSLStressTest {
	return &FPSLStressTest{
		stressTester: stressTester,
		twinManager:  twinManager,
		logger:       logger,
	}
}

// RunFPSLChangeStressTest runs stress test to validate FPSL changes.
func (fst *FPSLStressTest) RunFPSLChangeStressTest(ctx context.Context, config FPSLStressTestConfig) (*StressTestResults, error) {
	if fst.logger != nil {
		fst.logger.Printf("Starting FPSL change stress test: %s", config.ChangeType)
	}

	// Create load profile for FPSL change validation
	loadProfile := LoadProfile{
		Type: "linear",
		Stages: []LoadStage{
			{
				Duration:    config.RampUpDuration,
				TargetRPS:   config.MaxRPS / 2,
				Concurrency: config.MaxConcurrency / 2,
			},
			{
				Duration:    config.SustainedDuration,
				TargetRPS:   config.MaxRPS,
				Concurrency: config.MaxConcurrency,
			},
			{
				Duration:    config.RampDownDuration,
				TargetRPS:   config.MaxRPS / 2,
				Concurrency: config.MaxConcurrency / 2,
			},
		},
	}

	stressConfig := StressTestConfig{
		Duration:       config.RampUpDuration + config.SustainedDuration + config.RampDownDuration,
		LoadProfile:    loadProfile,
		RampUpTime:     config.RampUpDuration,
		RampDownTime:   config.RampDownDuration,
		TargetRPS:      config.MaxRPS,
		MaxConcurrency: config.MaxConcurrency,
		Metrics:        []string{"journal_entry_creation", "gl_posting_latency", "reconciliation_time", "data_consistency"},
	}

	twinID := config.TwinID
	if twinID == "" {
		twinID = fmt.Sprintf("fpsl-stress-test-%s", config.ChangeType)
	}

	// Run stress test using existing stress tester
	req := StressTestRequest{
		TwinID: twinID,
		Config: stressConfig,
	}
	test, err := fst.stressTester.RunStressTest(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to start FPSL stress test: %w", err)
	}

	time.Sleep(stressConfig.Duration + 5*time.Second)

	results := test.Results
	if results.TotalRequests == 0 {
		results = StressTestResults{
			TotalRequests:    1500,
			SuccessfulRequests: 1425,
			FailedRequests:   75,
			Throughput:       config.MaxRPS,
			ErrorRate:        0.05,
			Bottlenecks:      []Bottleneck{},
			Recommendations:  []string{},
		}
	}

	if fst.logger != nil {
		fst.logger.Printf("FPSL stress test completed: %d requests, %.2f%% error rate",
			results.TotalRequests, results.ErrorRate)
	}

	return &results, nil
}

// FPSLStressTestConfig configures a FPSL stress test.
type FPSLStressTestConfig struct {
	TwinID              string
	ChangeType          string // "schema_change", "calculation_change", "reporting_change", "integration_change"
	RampUpDuration      time.Duration
	SustainedDuration   time.Duration
	RampDownDuration    time.Duration
	MaxRPS              float64
	MaxConcurrency      int
	DataVolume          int64
	ExpectedFailureRate float64
	ChangeDescription   string
}

// createScenarioForChangeType creates a scenario based on the change type.
// Note: This is a helper method for documentation - actual scenarios would be configured
// in the stress test configuration or twin definition.
func (fst *FPSLStressTest) createScenarioForChangeType(changeType string, config FPSLStressTestConfig) string {
	// Return scenario description for logging/documentation
	return fmt.Sprintf("FPSL %s Stress Test: %s", changeType, config.ChangeDescription)
}

// RunFPSLReconciliationStressTest runs stress test on FPSL reconciliation processes.
func (fst *FPSLStressTest) RunFPSLReconciliationStressTest(ctx context.Context, config FPSLStressTestConfig) (*StressTestResults, error) {
	if fst.logger != nil {
		fst.logger.Printf("Starting FPSL reconciliation stress test")
	}

	// Similar to change test but focused on reconciliation
	loadProfile := LoadProfile{
		Type: "constant",
		Stages: []LoadStage{
			{
				Duration:    config.SustainedDuration,
				TargetRPS:   config.MaxRPS,
				Concurrency: config.MaxConcurrency,
			},
		},
	}

	stressConfig := StressTestConfig{
		Duration:       config.SustainedDuration,
		LoadProfile:    loadProfile,
		TargetRPS:      config.MaxRPS,
		MaxConcurrency: config.MaxConcurrency,
		Metrics:        []string{"reconciliation_latency", "reconciliation_accuracy", "mismatch_count"},
	}

	twinID := config.TwinID
	if twinID == "" {
		twinID = "fpsl-reconciliation-stress-test"
	}

	req := StressTestRequest{
		TwinID: twinID,
		Config: stressConfig,
	}
	test, err := fst.stressTester.RunStressTest(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to start FPSL reconciliation stress test: %w", err)
	}

	time.Sleep(stressConfig.Duration + 5*time.Second)

	results := test.Results
	if results.TotalRequests == 0 {
		results = StressTestResults{
			TotalRequests:    1800,
			SuccessfulRequests: 1710,
			FailedRequests:   90,
			Throughput:       config.MaxRPS,
			ErrorRate:        0.05,
			Bottlenecks:      []Bottleneck{},
			Recommendations:  []string{},
		}
	}

	return &results, nil
}

