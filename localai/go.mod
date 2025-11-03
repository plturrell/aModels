module github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI

go 1.23

require (
    github.com/go-skynet/go-llama.cpp v0.0.0-20240314183750-6a8041ef6b46
    github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/glove v0.0.0-00010101000000-000000000000
    github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece v0.0.0-00010101000000-000000000000
    golang.org/x/time v0.5.0
    google.golang.org/grpc v1.70.0
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
    // pruned ethereum-related indirects by removing go-ethereum/log usage
	github.com/go-ole/go-ole v1.3.0 // indirect
	github.com/gorilla/websocket v1.5.4-0.20250319132907-e064f32e3674 // indirect
	github.com/holiman/uint256 v1.3.2 // indirect
	github.com/shirou/gopsutil v3.21.11+incompatible // indirect
	github.com/supranational/blst v0.3.16 // indirect
	github.com/tklauser/go-sysconf v0.3.15 // indirect
	github.com/tklauser/numcpus v0.10.0 // indirect
	github.com/yusufpapurcu/wmi v1.2.4 // indirect
	golang.org/x/crypto v0.22.0 // indirect
	golang.org/x/net v0.24.0 // indirect
	golang.org/x/sync v0.6.0 // indirect
	golang.org/x/sys v0.19.0 // indirect
	golang.org/x/text v0.14.0 // indirect
	gonum.org/v1/gonum v0.16.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250825161204-c5933d9347a5 // indirect
)

// Removed replace directives for unavailable agenticAiETH dependencies
// replace github.com/plturrell/agenticAiETH/agenticAiETH_layer1_Blockchain => ../agenticAiETH_layer1_Blockchain

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/glove => ../models/glove

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece => ../models/sentencepiece

// Removed - third_party not available: replace github.com/go-skynet/go-llama.cpp => ../third_party/go-llama.cpp

// Removed - AgentSDK not available: replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK => ../agenticAiETH_layer4_AgentSDK
// Removed - maths not needed: replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths => ../agenticAiETH_layer4_Models/maths
