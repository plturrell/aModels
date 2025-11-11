.PHONY: build run test clean

build:
	go build -o bin/testing-service ./cmd/testing-service

run: build
	./bin/testing-service

test:
	go test ./...

clean:
	rm -f bin/testing-service

