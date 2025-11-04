package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
)

func main() {
	postgresDSN := flag.String("postgres", os.Getenv("POSTGRES_DSN"), "PostgreSQL DSN")
	redisURL := flag.String("redis", os.Getenv("REDIS_URL"), "Redis URL")
	redisKey := flag.String("redis-key", "localai:domains:config", "Redis key for domain configs")
	syncInterval := flag.Duration("interval", 30*time.Second, "Sync interval")
	flag.Parse()

	if *postgresDSN == "" {
		log.Fatal("POSTGRES_DSN or -postgres flag required")
	}
	if *redisURL == "" {
		log.Fatal("REDIS_URL or -redis flag required")
	}

	log.Printf("🔄 Starting domain config sync service")
	log.Printf("   PostgreSQL: %s", maskDSN(*postgresDSN))
	log.Printf("   Redis: %s", *redisURL)
	log.Printf("   Sync interval: %v", *syncInterval)

	// Initialize stores
	pgStore, err := domain.NewPostgresConfigStore(*postgresDSN)
	if err != nil {
		log.Fatalf("❌ Failed to connect to PostgreSQL: %v", err)
	}
	defer pgStore.Close()

	redisLoader, err := domain.NewRedisConfigLoader(*redisURL, *redisKey)
	if err != nil {
		log.Fatalf("❌ Failed to connect to Redis: %v", err)
	}
	defer redisLoader.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Initial sync
	log.Printf("🔄 Performing initial sync...")
	if err := pgStore.SyncToRedis(ctx, redisLoader); err != nil {
		log.Fatalf("❌ Initial sync failed: %v", err)
	}
	log.Printf("✅ Initial sync complete")

	// Periodic sync
	ticker := time.NewTicker(*syncInterval)
	defer ticker.Stop()

	// Handle shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	for {
		select {
		case <-ctx.Done():
			return
		case <-sigChan:
			log.Printf("🛑 Shutdown signal received")
			return
		case <-ticker.C:
			if err := pgStore.SyncToRedis(ctx, redisLoader); err != nil {
				log.Printf("⚠️  Sync failed: %v", err)
			} else {
				log.Printf("✅ Sync complete")
			}
		}
	}
}

func maskDSN(dsn string) string {
	// Mask password in DSN for logging
	// Simple implementation - in production, use proper URL parsing
	if len(dsn) > 20 {
		return dsn[:20] + "..."
	}
	return dsn
}

