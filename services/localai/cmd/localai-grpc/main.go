package main

import (
	"context"
	"log"
	"net"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	localaiv1 "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/api/localai/v1"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/grpcserver"
	"google.golang.org/grpc"
)

func main() {
	cfg := grpcserver.Config{
		Endpoint:    envDefault("LOCALAI_HTTP_BASE_URL", "http://127.0.0.1:8080"),
		Model:       envDefault("LOCALAI_MODEL", "auto"),
		Temperature: envFloat("LOCALAI_TEMPERATURE", 0.7),
		MaxTokens:   envInt("LOCALAI_MAX_TOKENS", 512),
		APIKey:      os.Getenv("LOCALAI_API_KEY"),
	}

	server, err := grpcserver.New(cfg)
	if err != nil {
		log.Fatalf("init grpc server: %v", err)
	}

	addr := envDefault("LOCALAI_GRPC_ADDR", ":50061")
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("listen %s: %v", addr, err)
	}

	s := grpc.NewServer()
	localaiv1.RegisterChatServiceServer(s, server)

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	go func() {
		<-ctx.Done()
		stopped := make(chan struct{})
		go func() {
			s.GracefulStop()
			close(stopped)
		}()
		select {
		case <-stopped:
		case <-time.After(5 * time.Second):
			s.Stop()
		}
	}()

	log.Printf("LocalAI gRPC bridge listening on %s (backend %s)", addr, cfg.Endpoint)
	if err := s.Serve(lis); err != nil && !strings.Contains(err.Error(), "use of closed network connection") {
		log.Fatalf("serve failed: %v", err)
	}
}

func envDefault(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if parsed, err := strconv.ParseFloat(v, 64); err == nil {
			return parsed
		}
	}
	return def
}

func envInt(key string, def int32) int32 {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil {
			return int32(parsed)
		}
	}
	return def
}
