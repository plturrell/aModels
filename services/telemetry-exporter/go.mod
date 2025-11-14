module github.com/plturrell/aModels/services/telemetry-exporter

go 1.24

require (
	go.opentelemetry.io/otel v1.38.0
	go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp v1.38.0
	go.opentelemetry.io/otel/sdk v1.38.0
	go.opentelemetry.io/proto/otlp v1.7.1
	github.com/plturrell/aModels/services/testing v0.0.0
	google.golang.org/protobuf v1.36.10
)

replace github.com/plturrell/aModels/services/testing => ../testing

// Use local copy of pkg/localai to avoid broken dependency in published version
// This ensures the testing service uses the correct localai package
replace github.com/plturrell/aModels/pkg/localai => ../../pkg/localai

// Use local copy of pkg/catalog/flightcatalog to avoid broken dependency
// The published version declares itself as ai_benchmarks instead of github.com/plturrell/aModels
replace github.com/plturrell/aModels/pkg/catalog/flightcatalog => ../../pkg/catalog/flightcatalog

replace google.golang.org/protobuf => google.golang.org/protobuf v1.36.10
