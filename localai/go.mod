module github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI

go 1.25.3

require (
	github.com/SAP/go-hdb v1.14.6
	github.com/ethereum/go-ethereum v1.16.5
	github.com/go-skynet/go-llama.cpp v0.0.0-20240314183750-6a8041ef6b46
	github.com/plturrell/agenticAiETH/agenticAiETH_layer1_Blockchain v0.0.0
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths v0.0.0
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/glove v0.0.0-00010101000000-000000000000
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece v0.0.0-00010101000000-000000000000
	golang.org/x/time v0.14.0
	google.golang.org/grpc v1.76.0
	google.golang.org/protobuf v1.36.10
)

require (
	github.com/Microsoft/go-winio v0.6.2 // indirect
	github.com/bits-and-blooms/bitset v1.24.0 // indirect
	github.com/consensys/gnark-crypto v0.19.0 // indirect
	github.com/crate-crypto/go-eth-kzg v1.4.0 // indirect
	github.com/crate-crypto/go-ipa v0.0.0-20240724233137-53bbb0ceb27a // indirect
	github.com/deckarep/golang-set/v2 v2.8.0 // indirect
	github.com/decred/dcrd/dcrec/secp256k1/v4 v4.4.0 // indirect
	github.com/ethereum/c-kzg-4844/v2 v2.1.5 // indirect
	github.com/ethereum/go-verkle v0.2.2 // indirect
	github.com/go-ole/go-ole v1.3.0 // indirect
	github.com/gorilla/websocket v1.5.4-0.20250319132907-e064f32e3674 // indirect
	github.com/holiman/uint256 v1.3.2 // indirect
	github.com/shirou/gopsutil v3.21.11+incompatible // indirect
	github.com/supranational/blst v0.3.16 // indirect
	github.com/tklauser/go-sysconf v0.3.15 // indirect
	github.com/tklauser/numcpus v0.10.0 // indirect
	github.com/yusufpapurcu/wmi v1.2.4 // indirect
	golang.org/x/crypto v0.43.0 // indirect
	golang.org/x/net v0.46.0 // indirect
	golang.org/x/sync v0.17.0 // indirect
	golang.org/x/sys v0.37.0 // indirect
	golang.org/x/text v0.30.0 // indirect
	gonum.org/v1/gonum v0.16.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250825161204-c5933d9347a5 // indirect
)

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer1_Blockchain => ../agenticAiETH_layer1_Blockchain

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/glove => ../agenticAiETH_layer4_Training/models/glove

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece => ../agenticAiETH_layer4_Training/models/sentencepiece

replace github.com/go-skynet/go-llama.cpp => ../third_party/go-llama.cpp

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK => ../agenticAiETH_layer4_AgentSDK

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths => ../agenticAiETH_layer4_Models/maths
