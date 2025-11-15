package discovery

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"time"
)

// ServiceDiscovery handles automatic discovery of Gitea and Glean services
type ServiceDiscovery struct {
	logger *log.Logger
}

// NewServiceDiscovery creates a new service discovery instance
func NewServiceDiscovery(logger *log.Logger) *ServiceDiscovery {
	return &ServiceDiscovery{
		logger: logger,
	}
}

// DiscoverServices automatically discovers and configures Gitea and Glean services
func (sd *ServiceDiscovery) DiscoverServices() {
	sd.logger.Println("[discovery] Starting service discovery...")

	// Discover Gitea
	giteaURL := sd.discoverGitea()
	if giteaURL != "" {
		if os.Getenv("GITEA_URL") == "" {
			os.Setenv("GITEA_URL", giteaURL)
			sd.logger.Printf("[discovery] Auto-discovered Gitea: %s", giteaURL)
		}
	}

	// Discover Glean
	gleanAvailable := sd.discoverGlean()
	if gleanAvailable {
		if os.Getenv("GLEAN_REALTIME_ENABLE") == "" {
			os.Setenv("GLEAN_REALTIME_ENABLE", "true")
			sd.logger.Println("[discovery] Auto-enabled Glean real-time export")
		}
	}

	// Set defaults if not configured
	sd.setDefaults()
}

// discoverGitea attempts to discover Gitea service
func (sd *ServiceDiscovery) discoverGitea() string {
	candidates := []string{
		"http://gitea:3000",     // Docker network
		"http://localhost:3000", // Local
		"http://127.0.0.1:3000", // Explicit localhost
	}

	for _, url := range candidates {
		if sd.isReachable(url + "/api/healthz") {
			return url
		}
	}

	return ""
}

// discoverGlean checks if Glean CLI is available
func (sd *ServiceDiscovery) discoverGlean() bool {
	// Check if glean command exists
	_, err := exec.LookPath("glean")
	if err != nil {
		return false
	}

	// Verify glean can be executed
	cmd := exec.Command("glean", "--help")
	if err := cmd.Run(); err != nil {
		return false
	}

	return true
}

// isReachable checks if a URL is reachable
func (sd *ServiceDiscovery) isReachable(url string) bool {
	client := &http.Client{
		Timeout: 2 * time.Second,
	}

	resp, err := client.Get(url)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK
}

// setDefaults sets default configuration values
func (sd *ServiceDiscovery) setDefaults() {
	// Glean export directory
	if os.Getenv("GLEAN_EXPORT_DIR") == "" {
		exportDir := "./data/glean/exports"
		os.Setenv("GLEAN_EXPORT_DIR", exportDir)
		if err := os.MkdirAll(exportDir, 0755); err == nil {
			sd.logger.Printf("[discovery] Created default Glean export directory: %s", exportDir)
		}
	}

	// Glean worker count
	if os.Getenv("GLEAN_REALTIME_WORKERS") == "" {
		os.Setenv("GLEAN_REALTIME_WORKERS", "2")
	}

	// Gitea webhook secret check
	if os.Getenv("GITEA_WEBHOOK_SECRET") == "" {
		sd.logger.Println("[discovery] WARNING: GITEA_WEBHOOK_SECRET not set. Webhook signature verification disabled.")
	}
}

// ValidateConfiguration validates the current configuration
func (sd *ServiceDiscovery) ValidateConfiguration() []string {
	var issues []string

	// Check Gitea configuration
	giteaURL := os.Getenv("GITEA_URL")
	if giteaURL == "" {
		issues = append(issues, "GITEA_URL not configured")
	} else if !sd.isReachable(giteaURL + "/api/healthz") {
		issues = append(issues, fmt.Sprintf("Gitea not reachable at %s", giteaURL))
	}

	giteaToken := os.Getenv("GITEA_TOKEN")
	if giteaToken == "" {
		issues = append(issues, "GITEA_TOKEN not configured (Gitea API will not work)")
	}

	// Check Glean configuration
	if os.Getenv("GLEAN_REALTIME_ENABLE") == "true" {
		if os.Getenv("GLEAN_DB_NAME") == "" {
			issues = append(issues, "GLEAN_REALTIME_ENABLE=true but GLEAN_DB_NAME not set")
		}

		if !sd.discoverGlean() {
			issues = append(issues, "GLEAN_REALTIME_ENABLE=true but glean CLI not found")
		}

		exportDir := os.Getenv("GLEAN_EXPORT_DIR")
		if exportDir != "" {
			if _, err := os.Stat(exportDir); os.IsNotExist(err) {
				issues = append(issues, fmt.Sprintf("GLEAN_EXPORT_DIR does not exist: %s", exportDir))
			}
		}
	}

	return issues
}

// GetConfigurationStatus returns a summary of the current configuration
func (sd *ServiceDiscovery) GetConfigurationStatus() map[string]interface{} {
	status := map[string]interface{}{
		"gitea": map[string]interface{}{
			"url":            os.Getenv("GITEA_URL"),
			"token_set":      os.Getenv("GITEA_TOKEN") != "",
			"webhook_secret": os.Getenv("GITEA_WEBHOOK_SECRET") != "",
			"reachable":      false,
		},
		"glean": map[string]interface{}{
			"enabled":       os.Getenv("GLEAN_REALTIME_ENABLE") == "true",
			"db_name":       os.Getenv("GLEAN_DB_NAME"),
			"export_dir":    os.Getenv("GLEAN_EXPORT_DIR"),
			"workers":       os.Getenv("GLEAN_REALTIME_WORKERS"),
			"cli_available": sd.discoverGlean(),
		},
	}

	// Check Gitea reachability
	giteaURL := os.Getenv("GITEA_URL")
	if giteaURL != "" {
		status["gitea"].(map[string]interface{})["reachable"] = sd.isReachable(giteaURL + "/api/healthz")
	}

	return status
}

// PrintConfigurationStatus prints the configuration status to console
func (sd *ServiceDiscovery) PrintConfigurationStatus() {
	sd.logger.Println("=== Service Configuration Status ===")

	status := sd.GetConfigurationStatus()

	// Print Gitea status
	giteaStatus := status["gitea"].(map[string]interface{})
	sd.logger.Printf("Gitea:")
	sd.logger.Printf("  URL: %v", giteaStatus["url"])
	sd.logger.Printf("  Token Set: %v", giteaStatus["token_set"])
	sd.logger.Printf("  Webhook Secret: %v", giteaStatus["webhook_secret"])
	sd.logger.Printf("  Reachable: %v", giteaStatus["reachable"])

	// Print Glean status
	gleanStatus := status["glean"].(map[string]interface{})
	sd.logger.Printf("Glean:")
	sd.logger.Printf("  Enabled: %v", gleanStatus["enabled"])
	sd.logger.Printf("  DB Name: %v", gleanStatus["db_name"])
	sd.logger.Printf("  Export Dir: %v", gleanStatus["export_dir"])
	sd.logger.Printf("  Workers: %v", gleanStatus["workers"])
	sd.logger.Printf("  CLI Available: %v", gleanStatus["cli_available"])

	// Print validation issues
	issues := sd.ValidateConfiguration()
	if len(issues) > 0 {
		sd.logger.Println("Configuration Issues:")
		for _, issue := range issues {
			sd.logger.Printf("  ⚠️  %s", issue)
		}
	} else {
		sd.logger.Println("✅ Configuration valid")
	}

	sd.logger.Println("====================================")
}
