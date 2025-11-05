.PHONY: migrate-up migrate-down migrate-status migrate-create build test

# Migration commands
MIGRATIONS_DIR := migrations
NEO4J_URI := $(shell echo $${NEO4J_URI:-bolt://localhost:7687})
NEO4J_USERNAME := $(shell echo $${NEO4J_USERNAME:-neo4j})
NEO4J_PASSWORD := $(shell echo $${NEO4J_PASSWORD:-password})

migrate-up:
	@echo "Running migrations..."
	@GOWORK=off go run ./cmd/migrate up

migrate-down:
	@echo "Rolling back migrations..."
	@echo "Warning: This will rollback all migrations. Use with caution."

migrate-status:
	@echo "Checking migration status..."
	@GOWORK=off go run ./cmd/migrate status

migrate-create:
	@echo "Usage: make migrate-create NAME=migration_name"
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME is required"; \
		exit 1; \
	fi
	@mkdir -p $(MIGRATIONS_DIR)
	@timestamp=$$(date +%Y%m%d%H%M%S); \
	cat > $(MIGRATIONS_DIR)/$${timestamp}_$(NAME).cypher << 'EOF'; \
-- +goose Up
-- TODO: Add migration logic here


-- +goose Down
-- TODO: Add rollback logic here

EOF
	@echo "Created migration: $(MIGRATIONS_DIR)/$${timestamp}_$(NAME).cypher"

build:
	@echo "Building catalog service..."
	@GOWORK=off go build -o catalog-service .

test:
	@echo "Running tests..."
	@GOWORK=off go test ./...

run:
	@echo "Running catalog service..."
	@GOWORK=off go run main.go
