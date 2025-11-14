module github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI

go 1.24

require (
	github.com/SAP/go-hdb v1.14.9
	github.com/ethereum/go-ethereum v1.16.7
	github.com/go-skynet/go-llama.cpp v0.0.0-20240314183750-6a8041ef6b46
	github.com/lib/pq v1.10.9
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths v0.0.0-00010101000000-000000000000
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/glove v0.0.0-00010101000000-000000000000
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece v0.0.0-00010101000000-000000000000
	github.com/redis/go-redis/v9 v9.5.1
	golang.org/x/time v0.9.0
	google.golang.org/grpc v1.70.0
	google.golang.org/protobuf v1.36.10
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
)

require (
	github.com/holiman/uint256 v1.3.2 // indirect
	golang.org/x/net v0.44.0 // indirect
	golang.org/x/sys v0.36.0 // indirect
	golang.org/x/text v0.30.0 // indirect
	gonum.org/v1/gonum v0.16.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20250825161204-c5933d9347a5 // indirect
)


replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/glove => ../../models/glove

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece => ../../models/sentencepiece

