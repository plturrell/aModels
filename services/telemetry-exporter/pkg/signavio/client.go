package signavio

// This file re-exports the SignavioClient from the testing package
// for use in the telemetry-exporter service.

import (
	"github.com/plturrell/aModels/services/testing"
)

// Re-export types and functions for convenience
type SignavioClient = testing.SignavioClient
type SignavioTelemetryRecord = testing.SignavioTelemetryRecord
type SignavioToolUsage = testing.SignavioToolUsage
type SignavioLLMCall = testing.SignavioLLMCall
type SignavioProcessStep = testing.SignavioProcessStep
type PromptMetrics = testing.PromptMetrics

var NewSignavioClient = testing.NewSignavioClient

