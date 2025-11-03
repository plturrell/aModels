package localai

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	localserver "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/server"
	"golang.org/x/time/rate"
)

// Config captures the inputs required to bootstrap the LocalAI server.
type Config struct {
	ModelsDir     string
	DomainsConfig string
	Port          int
}

// Defaults tries to discover the agenticAiETH workspace and fills in sensible defaults.
func Defaults() Config {
	root := detectAgenticRoot()
	return Config{
		ModelsDir:     filepath.Join(root, "agenticAiETH_layer4_Models"),
		DomainsConfig: filepath.Join(root, "agenticAiETH_layer4_LocalAI", "config", "domains.json"),
		Port:          8080,
	}
}

// Start launches the LocalAI HTTP server and blocks until the context is cancelled.
func Start(ctx context.Context, cfg Config, logger *slog.Logger) error {
	if cfg.ModelsDir == "" {
		return errors.New("models directory must be provided")
	}
	if cfg.DomainsConfig == "" {
		return errors.New("domains configuration path must be provided")
	}

	modelsDir, err := filepath.Abs(cfg.ModelsDir)
	if err != nil {
		return fmt.Errorf("resolve models directory: %w", err)
	}
	domainConfigPath, err := filepath.Abs(cfg.DomainsConfig)
	if err != nil {
		return fmt.Errorf("resolve domains config: %w", err)
	}

	if logger == nil {
		logger = slog.New(slog.NewTextHandler(os.Stdout, nil))
	}

	logger.Info("starting LocalAI server", "models_dir", modelsDir, "domains_config", domainConfigPath, "port", cfg.Port)

	domainManager := domain.NewDomainManager()
	if err := domainManager.LoadDomainConfigs(domainConfigPath); err != nil {
		return fmt.Errorf("load domain configs: %w", err)
	}

	models, err := loadModels(modelsDir, domainConfigPath, domainManager, logger)
	if err != nil {
		return err
	}

	limiter := rate.NewLimiter(rate.Every(time.Second), 10)
	vaultServer := localserver.NewVaultGemmaServer(models, nil, nil, domainManager, limiter, "2.0.0")

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", localserver.EnableCORS(vaultServer.RateLimitMiddleware(vaultServer.HandleChat)))
	mux.HandleFunc("/v1/chat/completions/stream", localserver.EnableCORS(vaultServer.RateLimitMiddleware(vaultServer.HandleStreamingChat)))
	mux.HandleFunc("/v1/chat/completions/function-calling", localserver.EnableCORS(vaultServer.RateLimitMiddleware(vaultServer.HandleFunctionCalling)))
	mux.HandleFunc("/v1/models", localserver.EnableCORS(vaultServer.HandleModels))
	mux.HandleFunc("/v1/domains", localserver.EnableCORS(vaultServer.HandleListDomains))
	mux.HandleFunc("/v1/domain-registry", localserver.EnableCORS(vaultServer.HandleDomainRegistry))
	mux.HandleFunc("/metrics", localserver.EnableCORS(vaultServer.HandleMetrics))
	mux.HandleFunc("/health", localserver.EnableCORS(vaultServer.HandleHealth))
	mux.Handle("/", http.FileServer(http.Dir(filepath.Join(filepath.Dir(domainConfigPath), "..", "web"))))

	httpServer := &http.Server{
		Addr:              fmt.Sprintf(":%d", cfg.Port),
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	errCh := make(chan error, 1)
	go func() {
		errCh <- httpServer.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(shutdownCtx); err != nil {
			return fmt.Errorf("shutdown localai server: %w", err)
		}
		return nil
	case err := <-errCh:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	}
}

func loadModels(modelsDir, domainConfigPath string, manager *domain.DomainManager, logger *slog.Logger) (map[string]*ai.VaultGemma, error) {
	configDir := filepath.Dir(domainConfigPath)
	defaultModelsRoot := filepath.Clean(filepath.Join(configDir, "..", "agenticAiETH_layer4_Models"))

	cache := map[string]*ai.VaultGemma{}
	models := map[string]*ai.VaultGemma{}

	relativeToModelsDir := func(path string) string {
		if rel, err := filepath.Rel(defaultModelsRoot, path); err == nil && !strings.HasPrefix(rel, "..") {
			return filepath.Join(modelsDir, rel)
		}
		return path
	}

	configs := manager.ListDomainConfigs()
	for domainName, cfg := range configs {
		resolvedPath := cfg.ModelPath
		if !filepath.IsAbs(resolvedPath) {
			resolvedPath = filepath.Clean(filepath.Join(configDir, resolvedPath))
		}
		resolvedPath = relativeToModelsDir(resolvedPath)

		model, ok := cache[resolvedPath]
		if !ok {
			loaded, err := ai.LoadVaultGemmaFromSafetensors(resolvedPath)
			if err != nil {
				logger.Warn("failed to load model for domain", "domain", domainName, "path", resolvedPath, "err", err)
				continue
			}
			model = loaded
			cache[resolvedPath] = model

			for _, alias := range deriveAliases(resolvedPath) {
				models[alias] = model
			}
		}

		models[domainName] = model
	}

	if len(models) == 0 {
		return nil, errors.New("no models were loaded from the provided configuration")
	}

	return models, nil
}

func deriveAliases(modelPath string) []string {
	aliases := []string{}
	base := strings.ToLower(filepath.Base(modelPath))
	switch {
	case strings.Contains(base, "phi"):
		aliases = append(aliases, "phi-3.5-mini")
	case strings.Contains(base, "granite"):
		aliases = append(aliases, "granite-4.0-h-micro")
	case strings.Contains(base, "vaultgemma"):
		aliases = append(aliases, "vaultgemma", "general")
	}
	aliases = append(aliases, base)

	unique := make([]string, 0, len(aliases))
	seen := map[string]struct{}{}
	for _, alias := range aliases {
		alias = strings.TrimSpace(alias)
		if alias == "" {
			continue
		}
		if _, ok := seen[alias]; ok {
			continue
		}
		seen[alias] = struct{}{}
		unique = append(unique, alias)
	}
	return unique
}

func detectAgenticRoot() string {
	candidates := []string{}
	if rootEnv := os.Getenv("AGENTICAIETH_ROOT"); rootEnv != "" {
		candidates = append(candidates, rootEnv)
	}
	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates,
			filepath.Join(home, "Library", "CloudStorage", "Dropbox", "agenticAiETH"),
			filepath.Join(home, "Dropbox", "agenticAiETH"),
		)
	}

	seen := map[string]struct{}{}
	for _, candidate := range candidates {
		clean := filepath.Clean(candidate)
		if _, ok := seen[clean]; ok {
			continue
		}
		seen[clean] = struct{}{}
		if info, err := os.Stat(clean); err == nil && info.IsDir() {
			return clean
		}
	}
	return ""
}
