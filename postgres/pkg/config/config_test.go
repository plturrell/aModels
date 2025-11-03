package config

import (
	"os"
	"testing"
)

func TestLoadSuccess(t *testing.T) {
	t.Setenv("POSTGRES_DSN", "postgres://user:pass@localhost:5432/db?sslmode=disable")
	t.Setenv("GRPC_PORT", "19000")
	t.Setenv("POSTGRES_MAX_OPEN_CONN", "40")
	t.Setenv("POSTGRES_MAX_IDLE_CONN", "15")
	t.Setenv("POSTGRES_CONN_MAX_LIFETIME_MINUTES", "60")
	t.Setenv("SHUTDOWN_GRACE_PERIOD_SECONDS", "5")
	t.Setenv("SERVICE_VERSION", "test-version")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg.GRPCPort != 19000 {
		t.Fatalf("expected port 19000, got %d", cfg.GRPCPort)
	}

	if cfg.PostgresDSN != "postgres://user:pass@localhost:5432/db?sslmode=disable" {
		t.Fatalf("unexpected DSN: %s", cfg.PostgresDSN)
	}

	if cfg.MaxOpenConnections != 40 || cfg.MaxIdleConnections != 15 {
		t.Fatalf("unexpected connection pool settings: %d/%d", cfg.MaxOpenConnections, cfg.MaxIdleConnections)
	}

	if cfg.ConnectionMaxLifetime.Minutes() != 60 {
		t.Fatalf("unexpected connection lifetime: %v", cfg.ConnectionMaxLifetime)
	}

	if cfg.ShutdownGracePeriod.Seconds() != 5 {
		t.Fatalf("unexpected shutdown grace: %v", cfg.ShutdownGracePeriod)
	}

	if cfg.ServiceVersion != "test-version" {
		t.Fatalf("unexpected service version: %s", cfg.ServiceVersion)
	}
}

func TestLoadMissingDSN(t *testing.T) {
	os.Unsetenv("POSTGRES_DSN")

	if _, err := Load(); err == nil {
		t.Fatal("expected error when DSN missing")
	}
}
