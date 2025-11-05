package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/plturrell/aModels/services/sap-bdc"
)

func main() {
	logger := log.New(os.Stdout, "[sap-bdc-service] ", log.LstdFlags|log.Lmsgprefix)

	// Load configuration
	cfg := sapbdc.Config{
		BaseURL:       os.Getenv("SAP_BDC_BASE_URL"),
		APIToken:      os.Getenv("SAP_BDC_API_TOKEN"),
		FormationID:   os.Getenv("SAP_BDC_FORMATION_ID"),
		DatasphereURL: os.Getenv("SAP_DATASPHERE_URL"),
	}

	if cfg.BaseURL == "" {
		logger.Fatal("SAP_BDC_BASE_URL environment variable is required")
	}
	if cfg.APIToken == "" {
		logger.Fatal("SAP_BDC_API_TOKEN environment variable is required")
	}
	if cfg.FormationID == "" {
		logger.Fatal("SAP_BDC_FORMATION_ID environment variable is required")
	}

	// Create client and service
	client := sapbdc.NewClient(cfg, logger)
	service := sapbdc.NewService(client, logger)

	// Setup HTTP handlers
	mux := http.NewServeMux()

	// Health check
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	})

	// Extract endpoint
	mux.HandleFunc("/extract", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req sapbdc.ExtractRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
			return
		}

		response, err := service.Extract(r.Context(), req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	})

	// List data products
	mux.HandleFunc("/data-products", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		products, err := client.ListDataProducts(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"data_products": products,
		})
	})

	// List intelligent applications
	mux.HandleFunc("/intelligent-applications", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		apps, err := client.ListIntelligentApplications(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"intelligent_applications": apps,
		})
	})

	// Get formation
	mux.HandleFunc("/formation", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		formation, err := client.GetFormation(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(formation)
	})

	port := os.Getenv("PORT")
	if port == "" {
		port = "8083"
	}

	logger.Printf("SAP BDC service starting on port %s", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		logger.Fatalf("Failed to start server: %v", err)
	}
}

