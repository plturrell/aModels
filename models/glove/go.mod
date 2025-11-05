module github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/glove

go 1.24.0

require (
	github.com/SAP/go-hdb v1.14.9
	github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece v0.0.0-00010101000000-000000000000
)

require (
	golang.org/x/text v0.30.0 // indirect
	google.golang.org/protobuf v1.36.10 // indirect
)

replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece => ../sentencepiece
replace github.com/SAP/go-hdb => ../../infrastructure/third_party/go-hdb
