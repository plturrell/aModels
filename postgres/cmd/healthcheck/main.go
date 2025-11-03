package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	postgresv1 "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Postgres/pkg/gen/v1"
)

func main() {
	addr := flag.String("addr", envString("POSTGRES_LANG_SERVICE_ADDR", "localhost:50055"), "Postgres Lang gRPC address")
	timeout := flag.Duration("timeout", envDuration("HEALTHCHECK_TIMEOUT", 5*time.Second), "per-attempt timeout")
	wait := flag.Duration("wait", envDuration("HEALTHCHECK_WAIT", 0), "total time to keep retrying (0=single attempt)")
	flag.Parse()

	deadline := time.Now().Add(*wait)

	for {
		ctx, cancel := context.WithTimeout(context.Background(), *timeout)
		conn, err := grpc.DialContext(ctx, *addr, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
		if err == nil {
			client := postgresv1.NewPostgresLangServiceClient(conn)
			_, callErr := client.HealthCheck(ctx, &postgresv1.HealthCheckRequest{})
			cancel()
			_ = conn.Close()
			if callErr == nil {
				fmt.Printf("Postgres Lang service healthy at %s\n", *addr)
				return
			}
			err = callErr
		} else {
			cancel()
		}

		if *wait == 0 || time.Now().After(deadline) {
			log.Fatalf("health check failed against %s: %v", *addr, err)
		}

		time.Sleep(250 * time.Millisecond)
	}
}

func envString(key, fallback string) string {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		return v
	}
	return fallback
}

func envDuration(key string, fallback time.Duration) time.Duration {
	if v, ok := os.LookupEnv(key); ok && v != "" {
		d, err := time.ParseDuration(v)
		if err == nil {
			return d
		}
		log.Printf("[healthcheck] warning: could not parse %s=%s as duration: %v", key, v, err)
	}
	return fallback
}
