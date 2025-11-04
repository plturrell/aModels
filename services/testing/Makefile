.PHONY: build run test clean

build:
	go build -o bin/testing-service ./main.go

run: build
	./bin/testing-service

test:
	go test ./...

clean:
	rm -f bin/testing-service

