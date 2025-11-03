module github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentFlow

go 1.25.3

require (
	github.com/google/uuid v1.6.0
	github.com/joho/godotenv v1.5.1
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK v0.0.0
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI v0.0.0
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres v0.0.0
	golang.org/x/time v0.14.0
	google.golang.org/grpc v1.76.0
	google.golang.org/protobuf v1.36.10
)

require (
	github.com/SAP/go-hdb v1.14.6 // indirect
	github.com/bits-and-blooms/bitset v1.24.0 // indirect
	github.com/consensys/gnark-crypto v0.19.0 // indirect
	github.com/crate-crypto/go-eth-kzg v1.4.0 // indirect
	github.com/crate-crypto/go-ipa v0.0.0-20240724233137-53bbb0ceb27a // indirect
	github.com/deckarep/golang-set/v2 v2.8.0 // indirect
	github.com/ethereum/go-verkle v0.2.2 // indirect
	github.com/go-skynet/go-llama.cpp v0.0.0-20240314183750-6a8041ef6b46 // indirect
	github.com/goccy/go-json v0.10.5 // indirect
	github.com/google/flatbuffers v25.9.23+incompatible // indirect
	github.com/gorilla/websocket v1.5.4-0.20250319132907-e064f32e3674 // indirect
	github.com/holiman/uint256 v1.3.2 // indirect
	github.com/klauspost/compress v1.18.1 // indirect
	github.com/pierrec/lz4/v4 v4.1.22 // indirect
	github.com/shirou/gopsutil v3.21.11+incompatible // indirect
	github.com/tklauser/go-sysconf v0.3.15 // indirect
	github.com/zeebo/xxh3 v1.0.2 // indirect
	golang.org/x/crypto v0.43.0 // indirect
	golang.org/x/exp v0.0.0-20250620022241-b7579e27df2b // indirect
	golang.org/x/net v0.46.0 // indirect
	golang.org/x/sync v0.17.0 // indirect
	golang.org/x/sys v0.37.0 // indirect
	golang.org/x/text v0.30.0 // indirect
	golang.org/x/xerrors v0.0.0-20240903120638-7835f813f4da // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250825161204-c5933d9347a5 // indirect
)

require (
	github.com/apache/arrow/go/v16 v16.0.0
	github.com/ethereum/go-ethereum v1.16.5
	github.com/plturrell/agenticAiETH/agenticAiETH_layer1_Blockchain v0.0.0
)

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI => ../agenticAiETH_layer4_LocalAI

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer1_Blockchain => ../agenticAiETH_layer1_Blockchain

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres => ../agenticAiETH_layer4_Postgres

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK => ../agenticAiETH_layer4_AgentSDK

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA => ../agenticAiETH_layer4_HANA

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/glove => ../agenticAiETH_layer4_Training/models/glove

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece => ../agenticAiETH_layer4_Training/models/sentencepiece

replace github.com/apache/arrow/go/v16 => ../third_party/go-arrow
