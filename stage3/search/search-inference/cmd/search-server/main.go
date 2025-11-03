package main

import (
	"context"
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	// Removed AgentSDK dependencies - catalog watching disabled
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Search/search-inference/internal/catalog/flightcatalog"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Search/search-inference/pkg/search"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Search/search-inference/pkg/server"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Search/search-inference/pkg/storage"
)

func main() {
	hanaDefault := os.Getenv("HANA_DSN")
	privacyDefault := os.Getenv("SEARCH_PRIVACY_LEVEL")
	if privacyDefault == "" {
		privacyDefault = string(storage.PrivacyLevelMedium)
	}
	esAddrDefault := os.Getenv("ELASTICSEARCH_ADDRESSES")
	esCloudDefault := os.Getenv("ELASTICSEARCH_CLOUD_ID")
	esIndexDefault := os.Getenv("ELASTICSEARCH_INDEX")
	esAPIKeyDefault := os.Getenv("ELASTICSEARCH_API_KEY")
	esUserDefault := os.Getenv("ELASTICSEARCH_USERNAME")
	esPassDefault := os.Getenv("ELASTICSEARCH_PASSWORD")
	redisAddrDefault := os.Getenv("REDIS_ADDR")
	redisPasswordDefault := os.Getenv("REDIS_PASSWORD")
	redisDBDefault := 0
	if v := os.Getenv("REDIS_DB"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil {
			redisDBDefault = parsed
		}
	}
	esTimeoutDefault := 10 * time.Second
	if v := os.Getenv("ELASTICSEARCH_TIMEOUT"); v != "" {
		if dur, err := time.ParseDuration(v); err == nil {
			esTimeoutDefault = dur
		}
	}

	modelPath := flag.String("model", "../agenticAiETH_layer4_Training/models/vaultgemma-transformers-1b-v1", "Path to baseline VaultGemma model")
	localAIURL := flag.String("localai", "", "LocalAI base URL")
	localAIKey := flag.String("localai-key", "", "LocalAI API key")
	port := flag.String("port", "7070", "HTTP server port")
	hanaDSN := flag.String("hana-dsn", hanaDefault, "SAP HANA DSN for canonical storage")
	privacyLevel := flag.String("privacy-level", privacyDefault, "Privacy level (low|medium|high)")
	esAddrs := flag.String("es-addrs", esAddrDefault, "Comma-separated Elasticsearch addresses")
	esCloudID := flag.String("es-cloud-id", esCloudDefault, "Elastic Cloud ID")
	esIndex := flag.String("es-index", esIndexDefault, "Elasticsearch index name")
	esAPIKey := flag.String("es-api-key", esAPIKeyDefault, "Elasticsearch API key")
	esUsername := flag.String("es-username", esUserDefault, "Elasticsearch username")
	esPassword := flag.String("es-password", esPassDefault, "Elasticsearch password")
	esTimeout := flag.Duration("es-timeout", esTimeoutDefault, "Elasticsearch request timeout")
	redisAddr := flag.String("redis-addr", redisAddrDefault, "Redis address for embedding cache")
	redisPassword := flag.String("redis-password", redisPasswordDefault, "Redis password")
	redisDB := flag.Int("redis-db", redisDBDefault, "Redis database index")
	flag.Parse()

	if *localAIURL == "" {
		if envURL := os.Getenv("LOCALAI_BASE_URL"); envURL != "" {
			*localAIURL = envURL
		}
	}

	if *localAIKey == "" {
		if envKey := os.Getenv("LOCALAI_API_KEY"); envKey != "" {
			*localAIKey = envKey
		}
	}

	log.Printf("🔍 Search Inference Server")
	log.Printf("===========================")
	log.Printf("📦 Model path: %s", *modelPath)
	log.Printf("🌐 LocalAI URL: %s", *localAIURL)
	if *hanaDSN != "" {
		log.Printf("🛢️  HANA DSN configured")
	} else {
		log.Printf("⚠️  HANA DSN not set - persistence falls back to Elasticsearch source data")
	}
	if trimmed := strings.TrimSpace(*esAddrs); trimmed != "" {
		log.Printf("🧠 Elasticsearch addresses: %s", trimmed)
	} else if strings.TrimSpace(*esCloudID) != "" {
		log.Printf("🧠 Elasticsearch Cloud ID configured")
	} else {
		log.Printf("⚠️  Elasticsearch address not specified, defaulting to http://localhost:9200")
	}
	if addr := strings.TrimSpace(*redisAddr); addr != "" {
		log.Printf("🗄️  Redis cache: %s", addr)
	}

	searchModel, err := search.NewSearchModelWithLocalAI(*modelPath, *localAIURL, *localAIKey)
	if err != nil {
		log.Fatalf("failed to initialize search model: %v", err)
	}
	defer searchModel.Close()

	addresses := []string{}
	if trimmed := strings.TrimSpace(*esAddrs); trimmed != "" {
		for _, part := range strings.Split(trimmed, ",") {
			p := strings.TrimSpace(part)
			if p != "" {
				addresses = append(addresses, p)
			}
		}
	}

	privacy := storage.PrivacyLevel(strings.ToLower(strings.TrimSpace(*privacyLevel)))
	switch privacy {
	case storage.PrivacyLevelLow, storage.PrivacyLevelMedium, storage.PrivacyLevelHigh:
	default:
		privacy = storage.PrivacyLevelMedium
	}

	serviceCfg := search.SearchServiceConfig{
		ModelPath:     *modelPath,
		LocalAIURL:    *localAIURL,
		LocalAIKey:    *localAIKey,
		HANADSN:       strings.TrimSpace(*hanaDSN),
		PrivacyLevel:  privacy,
		ExistingModel: searchModel,
		Elasticsearch: search.ElasticsearchConfig{
			Addresses:      addresses,
			APIKey:         strings.TrimSpace(*esAPIKey),
			Username:       strings.TrimSpace(*esUsername),
			Password:       strings.TrimSpace(*esPassword),
			CloudID:        strings.TrimSpace(*esCloudID),
			Index:          strings.TrimSpace(*esIndex),
			RequestTimeout: *esTimeout,
			AllowMissing:   true,
		},
	}

	if addr := strings.TrimSpace(*redisAddr); addr != "" {
		serviceCfg.Redis = &search.RedisConfig{
			Addr:     addr,
			Password: strings.TrimSpace(*redisPassword),
			DB:       *redisDB,
		}
	}

	searchService, err := search.NewSearchServiceWithConfig(serviceCfg)
	if err != nil {
		log.Fatalf("failed to initialize search service: %v", err)
	}
	defer searchService.Close()

	flightAddr := strings.TrimSpace(os.Getenv("AGENTSDK_FLIGHT_ADDR"))
	searchService.SetFlightAddr(flightAddr)
	// Catalog watching disabled - AgentSDK not available in standalone aModels repo
	var catalogWatcher interface{} = nil
	if flightAddr != "" {
		log.Printf("ℹ️  Catalog watching disabled (AgentSDK not available). AGENTSDK_FLIGHT_ADDR will be ignored.")
	}

	srv := server.NewSearchServer(searchService.Model(), searchService)

	http.HandleFunc("/health", srv.HandleHealth)
	http.HandleFunc("/v1/embed", srv.HandleEmbed)
	http.HandleFunc("/v1/rerank", srv.HandleRerank)
	http.HandleFunc("/v1/search", srv.HandleSearch)
	http.HandleFunc("/v1/documents", srv.HandleAddDocument)
	http.HandleFunc("/v1/documents/batch", srv.HandleAddDocuments)
	http.HandleFunc("/v1/model", srv.HandleModelInfo)
	http.HandleFunc("/v1/agent-catalog", handleAgentCatalog(searchService))
	http.HandleFunc("/v1/agent-catalog/stats", handleAgentCatalogStats(searchService))
	http.HandleFunc("/", server.ServeStatic("web"))

	log.Printf("✅ Ready on http://localhost:%s", *port)
	log.Printf("   Agent catalog: http://localhost:%s/v1/agent-catalog", *port)
	if err := http.ListenAndServe(":"+*port, nil); err != nil {
		log.Fatalf("server stopped: %v", err)
	}
}

func handleAgentCatalog(service *search.SearchService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if service == nil {
			http.Error(w, "search service not configured", http.StatusServiceUnavailable)
			return
		}
		// Agent catalog disabled - AgentSDK not available
		if cached, updated := service.AgentCatalogSnapshot(); cached != nil {
			sendSearchCatalogResponse(w, cached, updated)
			return
		}
		http.Error(w, "agent catalog not available (AgentSDK dependency removed)", http.StatusServiceUnavailable)
	}
}

func sendSearchCatalogResponse(w http.ResponseWriter, cat *search.AgentCatalog, updated time.Time) {
	response := map[string]any{}
	if cat != nil {
		response["suites"] = cat.Suites
		response["tools"] = cat.Tools
		enrichment := search.EnrichCatalog(cat)
		if enrichment.Summary != "" {
			response["agent_catalog_summary"] = enrichment.Summary
		}
		if enrichment.Prompt != "" {
			response["agent_catalog_context"] = enrichment.Prompt
		}
		if enrichment.Stats.SuiteCount > 0 || enrichment.Stats.UniqueToolCount > 0 {
			response["agent_catalog_stats"] = enrichment.Stats
		}
		if len(enrichment.Implementations) > 0 {
			response["agent_catalog_matrix"] = enrichment.Implementations
		}
		if len(enrichment.UniqueTools) > 0 {
			response["agent_catalog_unique_tools"] = enrichment.UniqueTools
		}
		if len(enrichment.StandaloneTools) > 0 {
			response["agent_catalog_tool_details"] = enrichment.StandaloneTools
		}
	} else {
		response["suites"] = []search.AgentSuite{}
		response["tools"] = []search.AgentTool{}
	}
	if !updated.IsZero() {
		response["updated_at"] = updated.Format(time.RFC3339)
	}
	respondJSON(w, response)
}

func handleAgentCatalogStats(service *search.SearchService) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if service == nil {
			http.Error(w, "search service not configured", http.StatusServiceUnavailable)
			return
		}
		catalog, updated := service.AgentCatalogSnapshot()
		if catalog == nil {
			http.Error(w, "no catalog cached", http.StatusServiceUnavailable)
			return
		}
		enrichment := search.EnrichCatalog(catalog)
		payload := map[string]any{
			"stats":           enrichment.Stats,
			"summary":         enrichment.Summary,
			"context":         enrichment.Prompt,
			"implementations": enrichment.Implementations,
			"unique_tools":    enrichment.UniqueTools,
			"tool_details":    enrichment.StandaloneTools,
			"log_sources":     []string{"watcher_push", "watcher_initial", "fallback_initial", "handler_refresh"},
		}
		if !updated.IsZero() {
			payload["updated_at"] = updated.Format(time.RFC3339)
		}
		respondJSON(w, payload)
	}
}

// convertWatchCatalog removed - was dependent on AgentSDK catalogwatch package

func convertCatalog(cat flightcatalog.Catalog) search.AgentCatalog {
	suites := make([]search.AgentSuite, 0, len(cat.Suites))
	for _, suite := range cat.Suites {
		suites = append(suites, search.AgentSuite{
			Name:           suite.Name,
			ToolNames:      append([]string(nil), suite.ToolNames...),
			ToolCount:      int(suite.ToolCount),
			Implementation: suite.Implementation,
			Version:        suite.Version,
			AttachedAt:     suite.AttachedAt,
		})
	}

	tools := make([]search.AgentTool, 0, len(cat.Tools))
	for _, tool := range cat.Tools {
		tools = append(tools, search.AgentTool{
			Name:        tool.Name,
			Description: tool.Description,
		})
	}

	result := search.AgentCatalog{Suites: suites, Tools: tools}
	result.Normalize()
	return result
}

func logCatalogEvent(source string, enrichment interface{}) {
	// Catalog logging disabled - catalogprompt not available
	if e, ok := enrichment.(interface{ Stats() interface{} }); ok {
		log.Printf("[catalog:%s] enrichment stats available", source)
	} else {
		log.Printf("[catalog:%s] catalog event (enrichment disabled)", source)
	}
}

func respondJSON(w http.ResponseWriter, data any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(data); err != nil {
		http.Error(w, "failed to encode response", http.StatusInternalServerError)
	}
}
