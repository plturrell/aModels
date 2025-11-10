module github.com/plturrell/aModels/services/runtime

go 1.23

require (
	github.com/gorilla/websocket v1.5.3
	github.com/plturrell/aModels/services/framework v0.0.0
	github.com/plturrell/aModels/services/stdlib v0.0.0
	github.com/plturrell/aModels/services/plot v0.0.0
)

replace github.com/plturrell/aModels/services/framework => ../framework
replace github.com/plturrell/aModels/services/stdlib => ../stdlib
replace github.com/plturrell/aModels/services/plot => ../plot
