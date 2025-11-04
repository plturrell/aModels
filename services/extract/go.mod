module github.com/plturrell/aModels/services/extract

go 1.25.3

require (
	github.com/Chahine-tech/sql-parser-go v0.0.0-20250711162409-da324d384ca3
	github.com/DATA-DOG/go-sqlmock v1.5.0
	github.com/SAP/go-hdb v1.14.9
	github.com/alicebob/miniredis/v2 v2.31.1
	github.com/apache/arrow/go/v16 v16.0.0
	github.com/lib/pq v1.10.9
	github.com/mattn/go-sqlite3 v1.14.32
	github.com/neo4j/neo4j-go-driver/v5 v5.28.4
	github.com/plturrell/aModels/services/postgres v0.0.0
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration v0.0.0-00010101000000-000000000000
	github.com/redis/go-redis/v9 v9.16.0
	google.golang.org/grpc v1.76.0
	google.golang.org/protobuf v1.36.10
)

replace github.com/plturrell/aModels/services/postgres => ../postgres

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration => ../../infrastructure/third_party/orchestration

replace github.com/Chahine-tech/sql-parser-go => ../../infrastructure/third_party/sql-parser-go

replace github.com/apache/arrow/go/v16 => ../../infrastructure/third_party/go-arrow

require (
	github.com/Code-Hex/go-generics-cache v1.3.1 // indirect
	github.com/Masterminds/goutils v1.1.1 // indirect
	github.com/Masterminds/semver/v3 v3.2.0 // indirect
	github.com/Masterminds/sprig/v3 v3.2.3 // indirect
	github.com/alicebob/gopher-json v0.0.0-20200520072559-a9ecdc9d1d3a // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/dlclark/regexp2 v1.10.0 // indirect
	github.com/dustin/go-humanize v1.0.1 // indirect
	github.com/goccy/go-json v0.10.5 // indirect
	github.com/google/flatbuffers v25.9.23+incompatible // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/goph/emperror v0.17.2 // indirect
	github.com/huandu/xstrings v1.3.3 // indirect
	github.com/imdario/mergo v0.3.13 // indirect
	github.com/json-iterator/go v1.1.12 // indirect
	github.com/klauspost/compress v1.18.1 // indirect
	github.com/klauspost/cpuid/v2 v2.3.0 // indirect
	github.com/mitchellh/copystructure v1.0.0 // indirect
	github.com/mitchellh/reflectwalk v1.0.0 // indirect
	github.com/modern-go/concurrent v0.0.0-20180306012644-bacd9c7ef1dd // indirect
	github.com/modern-go/reflect2 v1.0.2 // indirect
	github.com/nikolalohinski/gonja v1.5.3 // indirect
	github.com/pelletier/go-toml/v2 v2.2.4 // indirect
	github.com/pierrec/lz4/v4 v4.1.22 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/pkoukk/tiktoken-go v0.1.6 // indirect
	github.com/shopspring/decimal v1.3.1 // indirect
	github.com/sirupsen/logrus v1.9.3 // indirect
	github.com/spf13/cast v1.10.0 // indirect
	github.com/yargevad/filepathx v1.0.0 // indirect
	github.com/yuin/gopher-lua v1.1.0 // indirect
	github.com/zeebo/xxh3 v1.0.2 // indirect
	go.starlark.net v0.0.0-20230302034142-4b1e35fe2254 // indirect
	golang.org/x/crypto v0.43.0 // indirect
	golang.org/x/exp v0.0.0-20250620022241-b7579e27df2b // indirect
	golang.org/x/mod v0.28.0 // indirect
	golang.org/x/net v0.46.0 // indirect
	golang.org/x/sync v0.17.0 // indirect
	golang.org/x/sys v0.37.0 // indirect
	golang.org/x/telemetry v0.0.0-20250908211612-aef8a434d053 // indirect
	golang.org/x/text v0.30.0 // indirect
	golang.org/x/tools v0.37.0 // indirect
	golang.org/x/xerrors v0.0.0-20240903120638-7835f813f4da // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250825161204-c5933d9347a5 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)
