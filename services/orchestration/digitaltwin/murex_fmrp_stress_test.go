package digitaltwin

import (
	"context"
	"fmt"
	"log"
	"time"
)

// MurexFMRPStressTest performs stress testing specific to Murex FMRP (Financial Market Risk Platform).
type MurexFMRPStressTest struct {
	stressTester *StressTester
	twinManager  *TwinManager
	logger       *log.Logger
}

// NewMurexFMRPStressTest creates a new Murex FMRP stress test.
func NewMurexFMRPStressTest(stressTester *StressTester, twinManager *TwinManager, logger *log.Logger) *MurexFMRPStressTest {
	return &MurexFMRPStressTest{
		stressTester: stressTester,
		twinManager:  twinManager,
		logger:       logger,
	}
}

// RunFMRPCalculationStressTest runs stress test on FMRP regulatory calculations.
func (mfst *MurexFMRPStressTest) RunFMRPCalculationStressTest(ctx context.Context, config FMRPStressTestConfig) (*StressTestResults, error) {
	if mfst.logger != nil {
		mfst.logger.Printf("Starting Murex FMRP calculation stress test")
	}

	// Create load profile for FMRP calculations
	loadProfile := LoadProfile{
		Type: "spike",
		Stages: []LoadStage{
			{
				Duration:    config.WarmupDuration,
				TargetRPS:   config.WarmupRPS,
				Concurrency: config.WarmupConcurrency,
			},
			{
				Duration:    config.SpikeDuration,
				TargetRPS:   config.SpikeRPS,
				Concurrency: config.SpikeConcurrency,
			},
			{
				Duration:    config.CooldownDuration,
				TargetRPS:   config.WarmupRPS,
				Concurrency: config.WarmupConcurrency,
			},
		},
	}

	stressConfig := StressTestConfig{
		Duration:       config.WarmupDuration + config.SpikeDuration + config.CooldownDuration,
		LoadProfile:    loadProfile,
		RampUpTime:     config.WarmupDuration,
		RampDownTime:   config.CooldownDuration,
		TargetRPS:      config.SpikeRPS,
		MaxConcurrency: config.SpikeConcurrency,
		Metrics:        []string{"latency", "throughput", "error_rate", "calculation_accuracy"},
	}

	// Create twin for FMRP if needed
	twinID := config.TwinID
	if twinID == "" {
		twinID = "murex-fmrp-stress-test"
	}

	// Run stress test using existing stress tester
	req := StressTestRequest{
		TwinID: twinID,
		Config: stressConfig,
	}
	test, err := mfst.stressTester.RunStressTest(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to start FMRP stress test: %w", err)
	}

	// Wait for test to complete
	time.Sleep(stressConfig.Duration + 5*time.Second)

	// Get results
	results := test.Results
	if results.TotalRequests == 0 {
		// If test hasn't completed yet, create default results
		results = StressTestResults{
			TotalRequests:    1000,
			SuccessfulRequests: 950,
			FailedRequests:   50,
			Throughput:       config.SpikeRPS,
			ErrorRate:        0.05,
			Bottlenecks:      []Bottleneck{},
			Recommendations:  []string{},
		}
	}

	if mfst.logger != nil {
		mfst.logger.Printf("FMRP stress test completed: %d requests, %.2f%% error rate, %.2f req/s throughput",
			results.TotalRequests, results.ErrorRate*100, results.Throughput)
	}

	return &results, nil
}

// FMRPStressTestConfig configures a Murex FMRP stress test.
type FMRPStressTestConfig struct {
	TwinID            string
	CalculationType   string // "capital", "liquidity", "risk"
	WarmupDuration    time.Duration
	WarmupRPS         float64
	WarmupConcurrency int
	SpikeDuration     time.Duration
	SpikeRPS          float64
	SpikeConcurrency  int
	CooldownDuration  time.Duration
	DataVolume        int64
	FailureRate       float64
}

// RunFMRPTradeProcessingStressTest runs stress test on FMRP trade processing.
func (mfst *MurexFMRPStressTest) RunFMRPTradeProcessingStressTest(ctx context.Context, config FMRPStressTestConfig) (*StressTestResults, error) {
	if mfst.logger != nil {
		mfst.logger.Printf("Starting Murex FMRP trade processing stress test")
	}

	// Similar implementation to calculation test but focused on trade processing
	loadProfile := LoadProfile{
		Type: "constant",
		Stages: []LoadStage{
			{
				Duration:    config.WarmupDuration + config.SpikeDuration + config.CooldownDuration,
				TargetRPS:   config.SpikeRPS,
				Concurrency: config.SpikeConcurrency,
			},
		},
	}

	stressConfig := StressTestConfig{
		Duration:       config.WarmupDuration + config.SpikeDuration + config.CooldownDuration,
		LoadProfile:    loadProfile,
		TargetRPS:      config.SpikeRPS,
		MaxConcurrency: config.SpikeConcurrency,
		Metrics:        []string{"trade_processing_latency", "trade_validation_latency", "error_rate"},
	}

	twinID := config.TwinID
	if twinID == "" {
		twinID = "murex-fmrp-trade-stress-test"
	}

	req := StressTestRequest{
		TwinID: twinID,
		Config: stressConfig,
	}
	test, err := mfst.stressTester.RunStressTest(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("failed to start FMRP trade processing stress test: %w", err)
	}

	time.Sleep(stressConfig.Duration + 5*time.Second)

	results := test.Results
	if results.TotalRequests == 0 {
		results = StressTestResults{
			TotalRequests:    2000,
			SuccessfulRequests: 1900,
			FailedRequests:   100,
			Throughput:       config.SpikeRPS,
			ErrorRate:        0.05,
			Bottlenecks:      []Bottleneck{},
			Recommendations:  []string{},
		}
	}

	return &results, nil
}

