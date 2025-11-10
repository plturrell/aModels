module github.com/plturrell/aModels/services/telemetry-exporter

go 1.18

require github.com/plturrell/aModels/services/testing v0.0.0

require github.com/plturrell/aModels/pkg/localai v0.0.0-20251110082742-7f127d35aa81 // indirect

replace github.com/plturrell/aModels/services/testing => ../testing

// Fix localai import path issue
replace github.com/plturrell/aModels/pkg/localai => ../../pkg/localai
