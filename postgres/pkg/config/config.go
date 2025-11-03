package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config captures runtime configuration for the Postgres layer service.
type Config struct {
	GRPCPort              int
	PostgresDSN           string
	MaxOpenConnections    int
	MaxIdleConnections    int
	ConnectionMaxLifetime time.Duration
	ShutdownGracePeriod   time.Duration
	ServiceVersion        string
	FlightAddr            string
	FlightMaxRows         int
	ExtractFlightAddr     string
}

const (
	defaultGRPCPort             = 50055
	defaultMaxOpenConnections   = 20
	defaultMaxIdleConnections   = 10
	defaultConnMaxLifetimeMin   = 30
	defaultShutdownGraceSeconds = 15
	defaultServiceVersion       = "0.1.0"
	defaultFlightAddr           = ":8825"
	defaultFlightMaxRows        = 200
)

// Load reads configuration values from environment variables, applying defaults when missing.
func Load() (*Config, error) {
	cfg := &Config{}

	if portStr, ok := os.LookupEnv("GRPC_PORT"); ok {
		port, err := strconv.Atoi(portStr)
		if err != nil {
			return nil, fmt.Errorf("invalid GRPC_PORT: %w", err)
		}
		cfg.GRPCPort = port
	} else {
		cfg.GRPCPort = defaultGRPCPort
	}

	cfg.PostgresDSN = os.Getenv("POSTGRES_DSN")
	if cfg.PostgresDSN == "" {
		return nil, fmt.Errorf("POSTGRES_DSN environment variable is required")
	}

	if maxOpenStr, ok := os.LookupEnv("POSTGRES_MAX_OPEN_CONN"); ok {
		value, err := strconv.Atoi(maxOpenStr)
		if err != nil {
			return nil, fmt.Errorf("invalid POSTGRES_MAX_OPEN_CONN: %w", err)
		}
		cfg.MaxOpenConnections = value
	} else {
		cfg.MaxOpenConnections = defaultMaxOpenConnections
	}

	if maxIdleStr, ok := os.LookupEnv("POSTGRES_MAX_IDLE_CONN"); ok {
		value, err := strconv.Atoi(maxIdleStr)
		if err != nil {
			return nil, fmt.Errorf("invalid POSTGRES_MAX_IDLE_CONN: %w", err)
		}
		cfg.MaxIdleConnections = value
	} else {
		cfg.MaxIdleConnections = defaultMaxIdleConnections
	}

	if lifetimeStr, ok := os.LookupEnv("POSTGRES_CONN_MAX_LIFETIME_MINUTES"); ok {
		value, err := strconv.Atoi(lifetimeStr)
		if err != nil {
			return nil, fmt.Errorf("invalid POSTGRES_CONN_MAX_LIFETIME_MINUTES: %w", err)
		}
		cfg.ConnectionMaxLifetime = time.Duration(value) * time.Minute
	} else {
		cfg.ConnectionMaxLifetime = defaultConnMaxLifetimeMin * time.Minute
	}

	if shutdownStr, ok := os.LookupEnv("SHUTDOWN_GRACE_PERIOD_SECONDS"); ok {
		value, err := strconv.Atoi(shutdownStr)
		if err != nil {
			return nil, fmt.Errorf("invalid SHUTDOWN_GRACE_PERIOD_SECONDS: %w", err)
		}
		cfg.ShutdownGracePeriod = time.Duration(value) * time.Second
	} else {
		cfg.ShutdownGracePeriod = defaultShutdownGraceSeconds * time.Second
	}

	if version := os.Getenv("SERVICE_VERSION"); version != "" {
		cfg.ServiceVersion = version
	} else {
		cfg.ServiceVersion = defaultServiceVersion
	}

	if addr := strings.TrimSpace(os.Getenv("FLIGHT_ADDR")); addr != "" {
		cfg.FlightAddr = addr
	} else {
		cfg.FlightAddr = defaultFlightAddr
	}

	if rowsStr, ok := os.LookupEnv("FLIGHT_MAX_ROWS"); ok {
		value, err := strconv.Atoi(rowsStr)
		if err != nil {
			return nil, fmt.Errorf("invalid FLIGHT_MAX_ROWS: %w", err)
		}
		cfg.FlightMaxRows = value
	} else {
		cfg.FlightMaxRows = defaultFlightMaxRows
	}

	cfg.ExtractFlightAddr = strings.TrimSpace(os.Getenv("EXTRACT_FLIGHT_ADDR"))

	return cfg, nil
}
