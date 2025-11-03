.DEFAULT_GOAL := build

GO ?= go
POSTGRES_DSN ?= postgres://user@localhost:5432/postgres?sslmode=disable
GRPC_PORT ?= 50055
POSTGRES_LANG_SERVICE_ADDR ?= localhost:50055

.PHONY: build run test postgres-start postgres-stop postgres-migrate postgres-setup healthcheck clean

build: postgres-setup
	@echo "[make] building Go binaries"
	GOWORK=off $(GO) build ./...

run: postgres-setup
	@echo "[make] starting Postgres layer4 service on $(GRPC_PORT)"
	AUTO_SETUP_POSTGRES=false AUTO_APPLY_MIGRATIONS=false \
	POSTGRES_DSN="$(POSTGRES_DSN)" GRPC_PORT="$(GRPC_PORT)" \
	./scripts/run_layer4_postgres_service.sh

healthcheck:
	@echo "[make] waiting for Postgres layer4 service health"
	GOWORK=off $(GO) run ./cmd/healthcheck -wait=10s -timeout=2s -addr="$(POSTGRES_LANG_SERVICE_ADDR)"

test:
	@echo "[make] running unit tests"
	./scripts/run_tests.sh

postgres-setup: postgres-start postgres-migrate

postgres-start:
	@if [ "$(POSTGRES_DSN)" = "postgres://user@localhost:5432/postgres?sslmode=disable" ]; then \
		echo "[make] ensuring local Postgres is running"; \
		POSTGRES_DSN="$(POSTGRES_DSN)" ./scripts/start_local_postgres.sh; \
	else \
		echo "[make] skipping local Postgres bootstrap (custom POSTGRES_DSN)"; \
	fi

postgres-migrate:
	@echo "[make] applying migrations"
	POSTGRES_DSN="$(POSTGRES_DSN)" ./scripts/apply_migrations.sh

postgres-stop:
	@echo "[make] stopping local Postgres"
	./scripts/stop_local_postgres.sh

clean:
	@echo "[make] cleaning build artifacts"
	rm -rf .build
