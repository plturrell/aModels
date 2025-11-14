module github.com/plturrell/aModels/services/catalog

go 1.24

require (
	github.com/golang-jwt/jwt/v5 v5.3.0
	github.com/google/uuid v1.6.0
	github.com/gorilla/websocket v1.5.3
	github.com/lib/pq v1.10.9
	github.com/mattn/go-sqlite3 v1.14.32
	github.com/neo4j/neo4j-go-driver/v5 v5.28.4
	github.com/plturrell/aModels/services/extract v0.0.0-00010101000000-000000000000
	github.com/plturrell/aModels/services/orchestration v0.0.0-00010101000000-000000000000
	github.com/pressly/goose/v3 v3.21.1
	github.com/prometheus/client_golang v1.23.2
	github.com/redis/go-redis/v9 v9.16.0
	golang.org/x/time v0.14.0
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/mfridman/interpolate v0.0.2 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/prometheus/client_model v0.6.2 // indirect
	github.com/prometheus/common v0.66.1 // indirect
	github.com/prometheus/procfs v0.16.1 // indirect
	github.com/sethvargo/go-retry v0.2.4 // indirect
	go.uber.org/multierr v1.11.0 // indirect
	go.yaml.in/yaml/v2 v2.4.2 // indirect
	golang.org/x/sync v0.17.0 // indirect
	golang.org/x/sys v0.37.0 // indirect
	google.golang.org/protobuf v1.36.10 // indirect
)

replace github.com/plturrell/aModels/services/orchestration => ../orchestration

replace github.com/pressly/goose/v3 => ./third_party/goose

replace github.com/plturrell/aModels/services/extract => ../extract

replace github.com/golang-jwt/jwt/v5 => github.com/golang-jwt/jwt/v5 v5.2.1
