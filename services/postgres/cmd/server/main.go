package main

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"

	"github.com/plturrell/aModels/services/postgres/internal/db"
	"github.com/plturrell/aModels/services/postgres/pkg/config"
	"github.com/plturrell/aModels/services/postgres/pkg/flight"
	postgresv1 "github.com/plturrell/aModels/services/postgres/pkg/gen/v1"
	"github.com/plturrell/aModels/services/postgres/pkg/repository"
	"github.com/plturrell/aModels/services/postgres/pkg/service"
)

const (
	serviceName = "postgres-lang-service"
)

func main() {
	// Configure structured logging
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339})

	log.Info().Msg("starting postgres lang service")

	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatal().Err(err).Msg("failed to load configuration")
	}

	log.Info().
		Int("grpc_port", cfg.GRPCPort).
		Str("version", cfg.ServiceVersion).
		Str("flight_addr", cfg.FlightAddr).
		Msg("configuration loaded")

	// Validate configuration
	if err := validateConfig(cfg); err != nil {
		log.Fatal().Err(err).Msg("invalid configuration")
	}

	// Initialize database
	database, err := db.New(
		cfg.PostgresDSN,
		cfg.MaxOpenConnections,
		cfg.MaxIdleConnections,
		cfg.ConnectionMaxLifetime,
	)
	if err != nil {
		log.Fatal().Err(err).Msg("failed to initialize database")
	}
	defer database.Close()

	log.Info().
		Int("max_open_conns", cfg.MaxOpenConnections).
		Int("max_idle_conns", cfg.MaxIdleConnections).
		Dur("conn_max_lifetime", cfg.ConnectionMaxLifetime).
		Msg("database connection established")

	// Initialize repository and service
	repo := repository.NewOperationsRepository(database)
	langService := service.NewLangService(repo, cfg.ServiceVersion)

	// Start Apache Arrow Flight server (optional)
	var flightServer *flight.Server
	if cfg.FlightAddr != "" {
		flightServer, err = flight.New(cfg.FlightAddr, repo, cfg.FlightMaxRows)
		if err != nil {
			log.Fatal().Err(err).Msg("failed to create flight server")
		}

		if err := flightServer.Start(); err != nil {
			log.Fatal().Err(err).Msg("failed to start flight server")
		}

		log.Info().
			Str("addr", cfg.FlightAddr).
			Int("max_rows", cfg.FlightMaxRows).
			Msg("arrow flight server started")
	}

	// Create gRPC server
	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(loggingInterceptor),
	)

	// Register services
	postgresv1.RegisterPostgresLangServiceServer(grpcServer, langService)
	
	// Register health check service
	healthServer := health.NewServer()
	grpc_health_v1.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus(serviceName, grpc_health_v1.HealthCheckResponse_SERVING)

	// Enable reflection for grpcurl and similar tools
	reflection.Register(grpcServer)

	// Start gRPC listener
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.GRPCPort))
	if err != nil {
		log.Fatal().Err(err).Int("port", cfg.GRPCPort).Msg("failed to create listener")
	}

	// Start gRPC server in goroutine
	serverErrors := make(chan error, 1)
	go func() {
		log.Info().
			Str("addr", listener.Addr().String()).
			Msg("grpc server listening")
		serverErrors <- grpcServer.Serve(listener)
	}()

	// Setup graceful shutdown
	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)

	// Block until shutdown signal or server error
	select {
	case err := <-serverErrors:
		log.Fatal().Err(err).Msg("grpc server error")
	case sig := <-shutdown:
		log.Info().
			Str("signal", sig.String()).
			Msg("shutdown signal received")

		// Mark service as not serving
		healthServer.SetServingStatus(serviceName, grpc_health_v1.HealthCheckResponse_NOT_SERVING)

		// Create shutdown context with timeout
		ctx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownGracePeriod)
		defer cancel()

		// Graceful shutdown
		log.Info().
			Dur("grace_period", cfg.ShutdownGracePeriod).
			Msg("starting graceful shutdown")

		stopped := make(chan struct{})
		go func() {
			grpcServer.GracefulStop()
			close(stopped)
		}()

		select {
		case <-ctx.Done():
			log.Warn().Msg("shutdown grace period exceeded, forcing stop")
			grpcServer.Stop()
		case <-stopped:
			log.Info().Msg("graceful shutdown completed")
		}

		// Stop flight server if running
		if flightServer != nil {
			if err := flightServer.Stop(); err != nil {
				log.Error().Err(err).Msg("error stopping flight server")
			} else {
				log.Info().Msg("flight server stopped")
			}
		}
	}
}

// validateConfig performs startup validation of configuration
func validateConfig(cfg *config.Config) error {
	if cfg.GRPCPort < 1 || cfg.GRPCPort > 65535 {
		return fmt.Errorf("invalid grpc port: %d (must be 1-65535)", cfg.GRPCPort)
	}

	if cfg.MaxOpenConnections < 1 {
		return fmt.Errorf("invalid max open connections: %d (must be >= 1)", cfg.MaxOpenConnections)
	}

	if cfg.MaxIdleConnections < 0 {
		return fmt.Errorf("invalid max idle connections: %d (must be >= 0)", cfg.MaxIdleConnections)
	}

	if cfg.MaxIdleConnections > cfg.MaxOpenConnections {
		return fmt.Errorf("max idle connections (%d) cannot exceed max open connections (%d)",
			cfg.MaxIdleConnections, cfg.MaxOpenConnections)
	}

	if cfg.ConnectionMaxLifetime < time.Minute {
		return fmt.Errorf("connection max lifetime too short: %s (minimum 1 minute)", cfg.ConnectionMaxLifetime)
	}

	if cfg.ShutdownGracePeriod < time.Second {
		return fmt.Errorf("shutdown grace period too short: %s (minimum 1 second)", cfg.ShutdownGracePeriod)
	}

	if cfg.FlightMaxRows < 1 {
		return fmt.Errorf("invalid flight max rows: %d (must be >= 1)", cfg.FlightMaxRows)
	}

	return nil
}

// loggingInterceptor provides structured logging for all gRPC requests
func loggingInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (interface{}, error) {
	start := time.Now()

	// Call the handler
	resp, err := handler(ctx, req)

	// Log the request
	duration := time.Since(start)
	logger := log.With().
		Str("method", info.FullMethod).
		Dur("duration_ms", duration).
		Logger()

	if err != nil {
		logger.Error().Err(err).Msg("request failed")
	} else {
		logger.Info().Msg("request completed")
	}

	return resp, err
}
