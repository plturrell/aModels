.PHONY: build run test clean

build:
	mkdir -p bin
	GOWORK=off go build -tags noflightcatalog -o bin/telemetry-exporter ./cmd/telemetry-exporter

run:
	go run ./cmd/telemetry-exporter

test:
	go test ./...

clean:
	rm -f bin/telemetry-exporter

