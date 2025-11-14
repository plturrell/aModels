module github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Search/search-inference

go 1.24

require (
	github.com/elastic/go-elasticsearch/v7 v7.17.10
	github.com/mattn/go-sqlite3 v1.14.22
	github.com/redis/go-redis/v9 v9.0.0
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/stretchr/testify v1.9.0 // indirect
)


// Exclude HANA driver - only used with hana build tag, which requires Go 1.24+
exclude github.com/SAP/go-hdb v1.14.9

replace github.com/SAP/go-hdb => ../../infrastructure/third_party/go-hdb
// Use elasticsearch v7 which doesn't require OTEL
replace github.com/elastic/go-elasticsearch/v7 => github.com/elastic/go-elasticsearch/v7 v7.17.10
