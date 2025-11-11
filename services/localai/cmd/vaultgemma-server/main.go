package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	// Catalog watching requires AgentSDK which is not available in aModels repo
	// Removed dependency on agenticAiETH_layer4_AgentSDK
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/gguf"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/server"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/transformers"
	llama "github.com/go-skynet/go-llama.cpp"
	"golang.org/x/time/rate"
)

func main() {
	modelPath := flag.String("model", "../../models/vaultgemma-1b-transformers", "Model path")
	port := flag.String("port", "8080", "Server port")
	configPath := flag.String("config", os.Getenv("DOMAIN_CONFIG_PATH"), "Domain configuration file")
	if *configPath == "" {
		*configPath = "config/domains.json"
	}
	flag.Parse()

	log.Printf("🚀 VaultGemma Server - Pure Go Implementation")
	log.Printf("%s", "============================================================")
	log.Printf("📦 Model path: %s", *modelPath)
	log.Printf("📋 Config path: %s", *configPath)
	log.Printf("⚡ Using agenticAiETH tensor operations")
	log.Printf("🔒 Differential privacy built-in")

	disableFallbackEnv := strings.ToLower(strings.TrimSpace(os.Getenv("DISABLE_VAULTGEMMA_FALLBACK")))
	disableFallback := disableFallbackEnv == "1" || disableFallbackEnv == "true" || disableFallbackEnv == "yes"

	// Phase 4: Lazy loading - models will be loaded on first use
	// Only load the default VaultGemma model if lazy loading is disabled
	enableLazyLoading := os.Getenv("ENABLE_LAZY_LOADING") != "0" // Default to enabled
	var model *ai.VaultGemma
	
	if !enableLazyLoading {
		// Legacy behavior: load model at startup
		if disableFallback {
			log.Printf("\n⏭️  DISABLE_VAULTGEMMA_FALLBACK enabled — skipping VaultGemma safetensor load.")
		} else {
			log.Printf("\n📥 Loading VaultGemma model (lazy loading disabled)...")
			stopSpinner := startLoadingSpinner("⏳ Loading VaultGemma weights")
			loadedModel, err := ai.LoadVaultGemmaFromSafetensors(*modelPath)
			stopSpinner()
			if err != nil {
				log.Printf("❌ Failed to load model: %v", err)
				log.Printf("⚠️  Continuing with stubbed inference responses")
			} else {
				model = loadedModel
			}

			if model != nil {
				log.Printf("✅ Model loaded successfully!")
				log.Printf("   - Layers: %d", model.Config.NumLayers)
				log.Printf("   - Hidden size: %d", model.Config.HiddenSize)
				log.Printf("   - Vocab size: %d", model.Config.VocabSize)
				log.Printf("   - Attention heads: %d", model.Config.NumHeads)
			}
		}
	} else {
		log.Printf("\n🚀 Lazy loading enabled - models will be loaded on first use")
		if !disableFallback {
			log.Printf("📝 VaultGemma model will be loaded from: %s", *modelPath)
		}
	}

	// Load domain configurations (Redis or file-based)
	domainManager := domain.NewDomainManager()
	configLoader, err := domain.NewConfigLoader()
	if err != nil {
		log.Printf("⚠️  Failed to create config loader: %v", err)
		log.Printf("⚠️  Falling back to file-based config")
		configLoader = domain.NewFileConfigLoader(*configPath)
	}

	if err := configLoader.LoadDomainConfigs(context.Background(), domainManager); err != nil {
		log.Printf("⚠️  Failed to load domain configs: %v", err)
		log.Printf("⚠️  Continuing with single model mode")
	} else {
		log.Printf("✅ Loaded domain configs from %s", domain.GetConfigSource())
	}

	// Initialize model registries
	models := make(map[string]*ai.VaultGemma)
	if model != nil {
		models["general"] = model
		models["vaultgemma"] = model
	}
	ggufModels := make(map[string]*gguf.Model)
	transformerClients := make(map[string]*transformers.Client)

	// Phase 4: Register models for lazy loading instead of loading them
	log.Printf("\n🎯 Registering models for lazy loading...")

	if domainManager != nil {
		configs := domainManager.ListDomainConfigs()
		for name, cfg := range configs {
			if cfg == nil {
				continue
			}

			cfgModelPath := strings.TrimSpace(cfg.ModelPath)
			if cfgModelPath == "" && !strings.EqualFold(cfg.BackendType, "hf-transformers") {
				continue
			}
			lowerPath := strings.ToLower(cfgModelPath)

			if strings.EqualFold(cfg.BackendType, "hf-transformers") {
				if cfg.TransformersConfig == nil {
					log.Printf("⚠️  Transformers config missing for domain %s", name)
					continue
				}
				timeout := time.Duration(cfg.TransformersConfig.TimeoutSeconds) * time.Second
				client := transformers.NewClient(cfg.TransformersConfig.Endpoint, cfg.TransformersConfig.ModelName, timeout)
				transformerClients[name] = client
				if model != nil && !disableFallback {
					if _, exists := models[name]; !exists {
						models[name] = model
					}
				}
				log.Printf("✅ Transformers backend ready for domain %s -> %s", name, cfg.TransformersConfig.ModelName)
				continue
			}

			if strings.HasSuffix(lowerPath, ".gguf") {
				// Phase 4: Register GGUF model for lazy loading
				log.Printf("📝 Registered GGUF model for domain %s: %s (will load on first use)", name, cfgModelPath)
				// Model will be loaded via modelCache when needed
				if model != nil && !disableFallback {
					if _, exists := models[name]; !exists {
						models[name] = model
					}
				}
				continue
			}

			if disableFallback {
				continue
			}

			// Phase 4: Register safetensors model for lazy loading
			if cfgModelPath == *modelPath {
				// Default model already handled above
				if model != nil {
					models[name] = model
				}
			} else {
				log.Printf("📝 Registered safetensors model for domain %s: %s (will load on first use)", name, cfgModelPath)
				// Model will be loaded via modelCache when needed
			}
		}
	}

	vgServer := server.NewVaultGemmaServer(
		models,
		ggufModels,
		transformerClients,
		domainManager,
		rate.NewLimiter(rate.Every(time.Second), 10),
		"2.0.0",
	)

	// Phase 4: Register models in cache for lazy loading
	if enableLazyLoading && vgServer.modelCache != nil {
		log.Printf("\n📝 Registering models in cache for lazy loading...")
		
		// Register default VaultGemma model if not disabled
		if !disableFallback && *modelPath != "" {
			vgServer.modelCache.RegisterSafetensorModel("general", *modelPath)
			vgServer.modelCache.RegisterSafetensorModel("vaultgemma", *modelPath)
			log.Printf("✅ Registered default model: %s", *modelPath)
		}

		// Collect domains for preloading
		preloadDomains := []string{}
		
		// Register domain-specific models
		if domainManager != nil {
			configs := domainManager.ListDomainConfigs()
			for name, cfg := range configs {
				if cfg == nil {
					continue
				}

				cfgModelPath := strings.TrimSpace(cfg.ModelPath)
				if cfgModelPath == "" {
					continue
				}

				lowerPath := strings.ToLower(cfgModelPath)

				// Register GGUF models
				if strings.HasSuffix(lowerPath, ".gguf") {
					vgServer.modelCache.RegisterGGUFModel(name, cfgModelPath)
					log.Printf("✅ Registered GGUF model for domain %s: %s", name, cfgModelPath)
					continue
				}

				// Register safetensors models (skip if it's the default model path)
				if cfgModelPath != *modelPath && !strings.EqualFold(cfg.BackendType, "hf-transformers") {
					vgServer.modelCache.RegisterSafetensorModel(name, cfgModelPath)
					log.Printf("✅ Registered safetensors model for domain %s: %s", name, cfgModelPath)
				}

				// Register transformers clients
				if strings.EqualFold(cfg.BackendType, "hf-transformers") && cfg.TransformersConfig != nil {
					vgServer.modelCache.RegisterTransformerClient(name, cfg.TransformersConfig)
				}
			}
		}
		log.Printf("✅ Model registration complete - models will load on first use")
		
		// Preload frequently used models if configured
		preloadEnv := os.Getenv("PRELOAD_MODELS")
		if preloadEnv != "" {
			// Parse comma-separated list of domains to preload
			envDomains := strings.Split(preloadEnv, ",")
			for _, d := range envDomains {
				d = strings.TrimSpace(d)
				if d != "" {
					preloadDomains = append(preloadDomains, d)
				}
			}
		}
		
		// Also preload default models if configured
		if os.Getenv("PRELOAD_DEFAULT_MODELS") == "1" || os.Getenv("PRELOAD_DEFAULT_MODELS") == "true" {
			if !disableFallback {
				preloadDomains = append(preloadDomains, "general", "vaultgemma")
			}
		}
		
		// Preload models in background
		if len(preloadDomains) > 0 {
			log.Printf("\n🚀 Preloading %d models in background: %v", len(preloadDomains), preloadDomains)
			ctx := context.Background()
			for _, domain := range preloadDomains {
				vgServer.modelCache.PreloadModel(ctx, domain)
			}
		}
	}

	// Load domain-specific configurations
	domains := domainManager.ListDomains()
	if len(domains) > 0 {
		log.Printf("\n🎯 Configuring %d agent domains...", len(domains))
		log.Printf("✅ Agent domains ready for routing")
	}

	flightAddr := strings.TrimSpace(os.Getenv("AGENTSDK_FLIGHT_ADDR"))
	vgServer.SetFlightAddr(flightAddr)
	// Catalog watching disabled - AgentSDK not available in standalone aModels repo
	if flightAddr != "" {
		log.Printf("ℹ️  Catalog watching disabled (AgentSDK not available). AGENTSDK_FLIGHT_ADDR will be ignored.")
	}

	// Setup routes
	http.HandleFunc("/v1/chat/completions", server.EnableCORS(vgServer.RateLimitMiddleware(vgServer.HandleChat)))
	http.HandleFunc("/v1/chat/completions/stream", server.EnableCORS(vgServer.RateLimitMiddleware(vgServer.HandleStreamingChat)))
	http.HandleFunc("/v1/chat/completions/function-calling", server.EnableCORS(vgServer.RateLimitMiddleware(vgServer.HandleFunctionCalling)))
	http.HandleFunc("/v1/models", server.EnableCORS(vgServer.HandleModels))
	http.HandleFunc("/v1/embeddings", server.EnableCORS(vgServer.RateLimitMiddleware(vgServer.HandleEmbeddings)))
	http.HandleFunc("/v1/domains", server.EnableCORS(vgServer.HandleListDomains))

	// Phase 3: Domain lifecycle management API
	// Initialize PostgreSQL and Redis stores if available
	var postgresStore *domain.PostgresConfigStore
	var redisLoader *domain.RedisConfigLoader

	postgresDSN := os.Getenv("POSTGRES_DSN")
	if postgresDSN != "" {
		if store, err := domain.NewPostgresConfigStore(postgresDSN); err == nil {
			postgresStore = store
			log.Printf("✅ PostgreSQL config store initialized")
		} else {
			log.Printf("⚠️  Failed to initialize PostgreSQL store: %v", err)
		}
	}

	redisURL := os.Getenv("REDIS_URL")
	if redisURL != "" {
		if loader, err := domain.NewRedisConfigLoader(redisURL, "localai:domains:config"); err == nil {
			redisLoader = loader
			log.Printf("✅ Redis config loader initialized")
		} else {
			log.Printf("⚠️  Failed to initialize Redis loader: %v", err)
		}
	}

	if postgresStore != nil || redisLoader != nil {
		lifecycleManager := domain.NewLifecycleManager(domainManager, postgresStore, redisLoader)
		lifecycleAPI := domain.NewDomainLifecycleAPI(lifecycleManager)
		http.HandleFunc("/v1/domains/create", server.EnableCORS(lifecycleAPI.HandleCreateDomain))
		http.HandleFunc("/v1/domains/list", server.EnableCORS(lifecycleAPI.HandleListDomains))
		// Note: Update, archive, delete would need domain ID in path - handled via mux or query params
		log.Printf("✅ Domain lifecycle API enabled")
	}
	http.HandleFunc("/v1/domain-registry", server.EnableCORS(vgServer.HandleDomainRegistry))
	http.HandleFunc("/v1/agent-catalog", server.EnableCORS(handleAgentCatalog(vgServer)))
	http.HandleFunc("/health", server.EnableCORS(vgServer.HandleHealth))
	http.HandleFunc("/metrics", server.EnableCORS(vgServer.HandleMetrics))
	
	// Phase 3: Performance profiling endpoints
	if vgServer.profiler != nil {
		http.HandleFunc("/debug/stats", server.EnableCORS(vgServer.profiler.HandleProfilingStats))
	}
	http.HandleFunc("/debug/pprof", server.EnableCORS(server.HandlePprofRedirect))
	http.HandleFunc("/debug/pprof/", server.EnableCORS(http.DefaultServeMux.ServeHTTP))

	// Serve web UI
	http.Handle("/", http.FileServer(http.Dir("web")))

	addr := ":" + *port
	log.Printf("\n✅ Server ready on http://localhost%s", addr)
	log.Printf("   🌐 Web UI:  http://localhost%s/", addr)
	log.Printf("   Health:   http://localhost%s/health", addr)
	log.Printf("   Metrics:  http://localhost%s/metrics", addr)
	log.Printf("   Models:   http://localhost%s/v1/models", addr)
	log.Printf("   Domains:  http://localhost%s/v1/domains", addr)
	log.Printf("   Registry: http://localhost%s/v1/domain-registry", addr)
	log.Printf("   Chat:     http://localhost%s/v1/chat/completions", addr)
	log.Printf("\n🎯 Production Features:")
	// Phase 4: Count registered models (lazy loading)
	totalRegistered := 0
	if vgServer.modelCache != nil {
		stats := vgServer.modelCache.GetStats()
		if safetensorCount, ok := stats["safetensor_models"].(int); ok {
			totalRegistered += safetensorCount
		}
		if ggufCount, ok := stats["gguf_models"].(int); ok {
			totalRegistered += ggufCount
		}
	}
	log.Printf("   ✓ Models registered: %d (Transformers: %d)", totalRegistered, len(transformerClients))
	log.Printf("   ✓ Agent domains: %d", len(domains))
	log.Printf("   ✓ Auto domain detection")
	log.Printf("   ✓ Rate limiting: 10 req/sec")
	log.Printf("   ✓ Request validation")
	log.Printf("   ✓ Context timeouts: 30s")
	log.Printf("   ✓ CORS enabled")
	log.Printf("   ✓ Metrics endpoint")
	log.Printf("   ✓ Health monitoring")
	log.Printf("\n🚀 Ready for multi-domain inference!")

	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

func handleAgentCatalog(vgServer *server.VaultGemmaServer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Agent catalog disabled - AgentSDK not available
		if cached, updated := vgServer.AgentCatalogSnapshot(); cached != nil {
			sendAgentCatalogResponse(w, cached, updated)
			return
		}
		http.Error(w, "agent catalog not available (AgentSDK dependency removed)", http.StatusServiceUnavailable)
	}
}

func sendAgentCatalogResponse(w http.ResponseWriter, cat *server.AgentCatalog, updated time.Time) {
	response := map[string]any{}
	if cat != nil {
		response["suites"] = cat.Suites
		response["tools"] = cat.Tools
	} else {
		response["suites"] = []server.AgentSuite{}
		response["tools"] = []server.AgentTool{}
	}
	if !updated.IsZero() {
		response["updated_at"] = updated.Format(time.RFC3339)
	}
	respondJSON(w, response)
}

// convertCatalog removed - was dependent on AgentSDK flightcatalog package

func respondJSON(w http.ResponseWriter, data any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(data); err != nil {
		http.Error(w, "failed to encode response", http.StatusInternalServerError)
	}
}

func startLoadingSpinner(message string) func() {
	stop := make(chan struct{})
	go func() {
		ticker := time.NewTicker(200 * time.Millisecond)
		defer ticker.Stop()

		symbols := []rune{'|', '/', '-', '\\'}
		idx := 0
		for {
			select {
			case <-stop:
				fmt.Printf("\r%s ✔\n", message)
				return
			case <-ticker.C:
				fmt.Printf("\r%s %c", message, symbols[idx])
				idx = (idx + 1) % len(symbols)
			}
		}
	}()

	return func() {
		close(stop)
	}
}
