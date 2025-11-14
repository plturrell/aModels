module github.com/plturrell/aModels/services/testing

go 1.24

require (
	github.com/lib/pq v1.10.9
	github.com/plturrell/aModels/pkg/localai v0.0.0-00010101000000-000000000000
)

// Use local copy of pkg/localai to avoid broken dependency in published version
// The published version v0.0.0-20251110082742-7f127d35aa81 has incorrect import:
// ai_benchmarks/pkg/catalog/flightcatalog instead of github.com/plturrell/aModels/pkg/catalog/flightcatalog
replace github.com/plturrell/aModels/pkg/localai => ../../pkg/localai

// Use local copy of pkg/catalog/flightcatalog to avoid broken dependency
// The published version declares itself as ai_benchmarks instead of github.com/plturrell/aModels
replace github.com/plturrell/aModels/pkg/catalog/flightcatalog => ../../pkg/catalog/flightcatalog
