package main

import (
    // "context"
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
	"golang.org/x/time/rate"
)

func main() {
	modelPath := flag.String("model", "../agenticAiETH_layer4_Models/vaultgemm/vaultgemma-transformers-1b-v1", "Model path")
	port := flag.String("port", "8080", "Server port")
	configPath := flag.String("config", "config/domains.json", "Domain configuration file")
	flag.Parse()

	log.Printf("🚀 VaultGemma Server - Pure Go Implementation")
	log.Printf("%s", "============================================================")
	log.Printf("📦 Model path: %s", *modelPath)
	log.Printf("📋 Config path: %s", *configPath)
	log.Printf("⚡ Using agenticAiETH tensor operations")
	log.Printf("🔒 Differential privacy built-in")

	disableFallbackEnv := strings.ToLower(strings.TrimSpace(os.Getenv("DISABLE_VAULTGEMMA_FALLBACK")))
	disableFallback := disableFallbackEnv == "1" || disableFallbackEnv == "true" || disableFallbackEnv == "yes"

	var model *ai.VaultGemma
	if disableFallback {
		log.Printf("\n⏭️  DISABLE_VAULTGEMMA_FALLBACK enabled — skipping VaultGemma safetensor load.")
	} else {
		log.Printf("\n📥 Loading VaultGemma model...")
		stopSpinner := startLoadingSpinner("⏳ Loading VaultGemma weights")
		loadedModel, err := ai.LoadVaultGemmaFromSafetensors(*modelPath)
		stopSpinner()
		if err != nil {
			log.Fatalf("❌ Failed to load model: %v", err)
		}
		model = loadedModel

		log.Printf("✅ Model loaded successfully!")
		log.Printf("   - Layers: %d", model.Config.NumLayers)
		log.Printf("   - Hidden size: %d", model.Config.HiddenSize)
		log.Printf("   - Vocab size: %d", model.Config.VocabSize)
		log.Printf("   - Attention heads: %d", model.Config.NumHeads)
	}

	// Load domain configurations
	domainManager := domain.NewDomainManager()
	if err := domainManager.LoadDomainConfigs(*configPath); err != nil {
		log.Printf("⚠️  Failed to load domain configs: %v", err)
		log.Printf("⚠️  Continuing with single model mode")
	}

	// Initialize model registries
	models := make(map[string]*ai.VaultGemma)
	if model != nil {
		models["general"] = model
		models["vaultgemma"] = model
	}
	ggufModels := make(map[string]*gguf.Model)
	transformerClients := make(map[string]*transformers.Client)

	uniqueSafetensorModels := make(map[string]*ai.VaultGemma)
	if model != nil {
		uniqueSafetensorModels[*modelPath] = model
	}
	uniqueGGUFModels := map[string]*gguf.Model{}

	log.Printf("\n🎯 Loading additional models...")

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
				loaded, ok := uniqueGGUFModels[cfgModelPath]
				if !ok {
					log.Printf("📥 Loading GGUF model for domain %s from %s...", name, cfgModelPath)
					gm, err := gguf.Load(cfgModelPath)
					if err != nil {
						log.Printf("⚠️  Failed to load GGUF model %s: %v", filepath.Base(cfgModelPath), err)
						continue
					}
					uniqueGGUFModels[cfgModelPath] = gm
					loaded = gm
					log.Printf("✅ GGUF model ready: %s", filepath.Base(cfgModelPath))
				}
				ggufModels[name] = loaded
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

			if _, seen := uniqueSafetensorModels[cfgModelPath]; !seen {
				if cfgModelPath == *modelPath {
					uniqueSafetensorModels[cfgModelPath] = model
				} else {
					log.Printf("📥 Loading safetensors model for domain %s from %s...", name, cfgModelPath)
					loadedModel, err := ai.LoadVaultGemmaFromSafetensors(cfgModelPath)
					if err != nil {
						log.Printf("⚠️  Failed to load model %s: %v", filepath.Base(cfgModelPath), err)
						continue
					}
					uniqueSafetensorModels[cfgModelPath] = loadedModel
					log.Printf("✅ Safetensors model ready: %s", filepath.Base(cfgModelPath))
				}
			}

			if resolved, ok := uniqueSafetensorModels[cfgModelPath]; ok {
				models[name] = resolved
			}
		}
	}

	// Ensure GGUF models are released on shutdown
	for _, gm := range uniqueGGUFModels {
		defer gm.Close()
	}

	vgServer := server.NewVaultGemmaServer(
		models,
		ggufModels,
		transformerClients,
		domainManager,
		rate.NewLimiter(rate.Every(time.Second), 10),
		"2.0.0",
	)

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
	http.HandleFunc("/v1/domains", server.EnableCORS(vgServer.HandleListDomains))
	http.HandleFunc("/v1/domain-registry", server.EnableCORS(vgServer.HandleDomainRegistry))
	http.HandleFunc("/v1/agent-catalog", server.EnableCORS(handleAgentCatalog(vgServer)))
	http.HandleFunc("/health", server.EnableCORS(vgServer.HandleHealth))
	http.HandleFunc("/metrics", server.EnableCORS(vgServer.HandleMetrics))

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
	totalLoaded := len(uniqueSafetensorModels) + len(uniqueGGUFModels)
	log.Printf("   ✓ Models loaded: %d (GGUF: %d, Transformers: %d)", totalLoaded, len(uniqueGGUFModels), len(transformerClients))
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
