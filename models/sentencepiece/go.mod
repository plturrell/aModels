module github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece

go 1.24.0

require (
	github.com/SAP/go-hdb v1.14.9
	google.golang.org/protobuf v1.36.10
)

require golang.org/x/text v0.30.0 // indirect

replace github.com/SAP/go-hdb => ../../infrastructure/third_party/go-hdb
