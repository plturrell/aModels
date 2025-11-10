module github.com/plturrell/aModels/services/extract

go 1.24.3

require (
	github.com/Chahine-tech/sql-parser-go v0.0.0-20250711162409-da324d384ca3
	github.com/DATA-DOG/go-sqlmock v1.5.0
	github.com/SAP/go-hdb v1.14.9
	github.com/alicebob/miniredis/v2 v2.31.1
	github.com/apache/arrow-go/v18 v18.4.1
	github.com/lib/pq v1.10.9
	github.com/mattn/go-sqlite3 v1.14.32
	github.com/neo4j/neo4j-go-driver/v5 v5.28.4
	github.com/plturrell/aModels/services/postgres v0.0.0
	github.com/pressly/goose/v3 v3.21.1
	github.com/redis/go-redis/v9 v9.16.0
	google.golang.org/grpc v1.76.0
	google.golang.org/protobuf v1.36.8
)

replace github.com/plturrell/aModels/services/postgres => ../postgres

replace github.com/SAP/go-hdb => ../../infrastructure/third_party/go-hdb

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration => ../../infrastructure/third_party/orchestration

replace github.com/Chahine-tech/sql-parser-go => ../../infrastructure/third_party/sql-parser-go

replace google.golang.org/protobuf => google.golang.org/protobuf v1.34.2

require (
	github.com/alicebob/gopher-json v0.0.0-20200520072559-a9ecdc9d1d3a // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/goccy/go-json v0.10.5 // indirect
	github.com/google/flatbuffers v25.9.23+incompatible // indirect
	github.com/klauspost/compress v1.18.1 // indirect
	github.com/klauspost/cpuid/v2 v2.3.0 // indirect
	github.com/mfridman/interpolate v0.0.2 // indirect
	github.com/pierrec/lz4/v4 v4.1.22 // indirect
	github.com/sethvargo/go-retry v0.2.4 // indirect
	github.com/stretchr/testify v1.11.1 // indirect
	github.com/yuin/gopher-lua v1.1.0 // indirect
	github.com/zeebo/xxh3 v1.0.2 // indirect
	go.opentelemetry.io/otel v1.38.0 // indirect
	go.opentelemetry.io/otel/sdk/metric v1.38.0 // indirect
	go.uber.org/multierr v1.11.0 // indirect
	golang.org/x/exp v0.0.0-20250620022241-b7579e27df2b // indirect
	golang.org/x/mod v0.29.0 // indirect
	golang.org/x/net v0.46.0 // indirect
	golang.org/x/sync v0.17.0 // indirect
	golang.org/x/sys v0.37.0 // indirect
	golang.org/x/telemetry v0.0.0-20251008203120-078029d740a8 // indirect
	golang.org/x/text v0.30.0 // indirect
	golang.org/x/tools v0.38.0 // indirect
	golang.org/x/xerrors v0.0.0-20240903120638-7835f813f4da // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250825161204-c5933d9347a5 // indirect
)
