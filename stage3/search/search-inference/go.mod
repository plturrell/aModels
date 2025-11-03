module github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Search/search-inference

go 1.25.3

require (
	github.com/SAP/go-hdb v1.14.6
	github.com/elastic/go-elasticsearch/v8 v8.19.0
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK v0.0.0
	github.com/redis/go-redis/v9 v9.14.1
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/elastic/elastic-transport-go/v8 v8.7.0 // indirect
	github.com/go-logr/logr v1.4.2 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	go.opentelemetry.io/otel v1.28.0 // indirect
	go.opentelemetry.io/otel/metric v1.28.0 // indirect
	go.opentelemetry.io/otel/trace v1.28.0 // indirect
	golang.org/x/text v0.30.0 // indirect
)

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK => ../../agenticAiETH_layer4_AgentSDK

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI => ../../agenticAiETH_layer4_LocalAI

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA => ../../agenticAiETH_layer4_HANA

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Extract => ../../agenticAiETH_layer4_Extract

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models => ../../agenticAiETH_layer4_Models
