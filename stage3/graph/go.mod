module github.com/langchain-ai/langgraph-go

go 1.23

require (
	github.com/ethereum/go-ethereum v1.16.5
	github.com/fsnotify/fsnotify v1.9.0
	github.com/lib/pq v1.10.9
	github.com/mattn/go-sqlite3 v1.14.32
	github.com/neo4j/neo4j-go-driver/v5 v5.28.4
	github.com/plturrell/agenticAiETH/agenticAiETH_layer1_Blockchain v0.0.0
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK v0.0.0
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Extract v0.0.0
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA v0.0.0
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths v0.0.0
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres v0.0.0
	github.com/redis/go-redis/v9 v9.16.0
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/Microsoft/go-winio v0.6.2 // indirect
	github.com/SAP/go-hdb v1.14.9 // indirect
	github.com/apapsch/go-jsonmerge/v2 v2.0.0 // indirect
	github.com/bits-and-blooms/bitset v1.24.0 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/consensys/gnark-crypto v0.19.0 // indirect
	github.com/crate-crypto/go-eth-kzg v1.4.0 // indirect
	github.com/crate-crypto/go-ipa v0.0.0-20240724233137-53bbb0ceb27a // indirect
	github.com/deckarep/golang-set/v2 v2.8.0 // indirect
	github.com/decred/dcrd/dcrec/secp256k1/v4 v4.4.0 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/ethereum/c-kzg-4844/v2 v2.1.5 // indirect
	github.com/ethereum/go-verkle v0.2.2 // indirect
	github.com/go-ole/go-ole v1.3.0 // indirect
	github.com/goccy/go-json v0.10.5 // indirect
	github.com/google/flatbuffers v25.9.23+incompatible // indirect
	github.com/gorilla/websocket v1.5.4-0.20250319132907-e064f32e3674 // indirect
	github.com/holiman/uint256 v1.3.2 // indirect
	github.com/joho/godotenv v1.5.1 // indirect
	github.com/klauspost/compress v1.18.1 // indirect
	github.com/klauspost/cpuid/v2 v2.3.0 // indirect
	github.com/pierrec/lz4/v4 v4.1.22 // indirect
	github.com/shirou/gopsutil v3.21.11+incompatible // indirect
	github.com/supranational/blst v0.3.16 // indirect
	github.com/tklauser/go-sysconf v0.3.15 // indirect
	github.com/tklauser/numcpus v0.10.0 // indirect
	github.com/yusufpapurcu/wmi v1.2.4 // indirect
	github.com/zeebo/xxh3 v1.0.2 // indirect
	golang.org/x/crypto v0.43.0 // indirect
	golang.org/x/exp v0.0.0-20250620022241-b7579e27df2b // indirect
	golang.org/x/mod v0.29.0 // indirect
	golang.org/x/net v0.46.0 // indirect
	golang.org/x/sync v0.17.0 // indirect
	golang.org/x/sys v0.37.0 // indirect
	golang.org/x/telemetry v0.0.0-20251008203120-078029d740a8 // indirect
	golang.org/x/text v0.30.0 // indirect
	golang.org/x/tools v0.38.0 // indirect
	golang.org/x/xerrors v0.0.0-20240903120638-7835f813f4da // indirect
	gonum.org/v1/gonum v0.16.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250825161204-c5933d9347a5 // indirect
)

require (
	github.com/apache/arrow/go/v16 v16.0.0
	github.com/google/uuid v1.6.0
	google.golang.org/grpc v1.76.0
	google.golang.org/protobuf v1.36.10
)

replace github.com/redis/go-redis/v9 => ../third_party/go-redis

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer1_Blockchain => ../agenticAiETH_layer1_Blockchain

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA => ../agenticAiETH_layer4_HANA

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK => ../agenticAiETH_layer4_AgentSDK

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Extract => ../agenticAiETH_layer4_Extract

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres => ../agenticAiETH_layer4_Postgres

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths => ../agenticAiETH_layer4_Models/maths

replace github.com/apache/arrow/go/v16 => ../third_party/go-arrow
