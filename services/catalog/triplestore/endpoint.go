package triplestore

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
)

// SPARQLEndpoint provides an HTTP endpoint for SPARQL queries.
type SPARQLEndpoint struct {
	sparqlClient *SPARQLClient
	logger       *log.Logger
}

// NewSPARQLEndpoint creates a new SPARQL endpoint.
func NewSPARQLEndpoint(sparqlClient *SPARQLClient, logger *log.Logger) *SPARQLEndpoint {
	return &SPARQLEndpoint{
		sparqlClient: sparqlClient,
		logger:       logger,
	}
}

// HandleSPARQL handles SPARQL query requests via HTTP.
func (e *SPARQLEndpoint) HandleSPARQL(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost && r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var query string
	var format string

	if r.Method == http.MethodPost {
		contentType := r.Header.Get("Content-Type")
		if strings.Contains(contentType, "application/sparql-query") {
			// SPARQL query in body
			body, err := io.ReadAll(r.Body)
			if err != nil {
				http.Error(w, fmt.Sprintf("Failed to read request body: %v", err), http.StatusBadRequest)
				return
			}
			query = string(body)
		} else if strings.Contains(contentType, "application/x-www-form-urlencoded") {
			// Form-encoded
			if err := r.ParseForm(); err != nil {
				http.Error(w, fmt.Sprintf("Failed to parse form: %v", err), http.StatusBadRequest)
				return
			}
			query = r.FormValue("query")
			format = r.FormValue("format")
		} else {
			// Try JSON
			var req struct {
				Query  string `json:"query"`
				Format string `json:"format,omitempty"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err == nil {
				query = req.Query
				format = req.Format
			}
		}
	} else {
		// GET request - query parameter
		query = r.URL.Query().Get("query")
		format = r.URL.Query().Get("format")
	}

	if query == "" {
		http.Error(w, "SPARQL query is required", http.StatusBadRequest)
		return
	}

	// Default format
	if format == "" {
		format = "json"
	}

	// Execute query
	result, err := e.sparqlClient.ExecuteQuery(r.Context(), query)
	if err != nil {
		http.Error(w, fmt.Sprintf("Query execution failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Format response based on requested format
	switch format {
	case "json":
		w.Header().Set("Content-Type", "application/sparql-results+json")
		if err := e.writeJSONResult(w, result); err != nil {
			if e.logger != nil {
				e.logger.Printf("Failed to write JSON result: %v", err)
			}
		}
	case "xml":
		w.Header().Set("Content-Type", "application/sparql-results+xml")
		if err := e.writeXMLResult(w, result); err != nil {
			if e.logger != nil {
				e.logger.Printf("Failed to write XML result: %v", err)
			}
		}
	default:
		w.Header().Set("Content-Type", "application/sparql-results+json")
		if err := e.writeJSONResult(w, result); err != nil {
			if e.logger != nil {
				e.logger.Printf("Failed to write JSON result: %v", err)
			}
		}
	}
}

// writeJSONResult writes SPARQL results in JSON format (SPARQL 1.1 Results Format).
func (e *SPARQLEndpoint) writeJSONResult(w http.ResponseWriter, result *QueryResult) error {
	response := map[string]any{
		"head": map[string]any{
			"vars": result.Variables,
		},
		"results": map[string]any{
			"bindings": result.Bindings,
		},
	}

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(response)
}

// writeXMLResult writes SPARQL results in XML format (simplified).
func (e *SPARQLEndpoint) writeXMLResult(w http.ResponseWriter, result *QueryResult) error {
	// Simplified XML output
	xml := `<?xml version="1.0"?>` + "\n"
	xml += `<sparql xmlns="http://www.w3.org/2005/sparql-results#">` + "\n"
	xml += `  <head>` + "\n"
	for _, varName := range result.Variables {
		xml += fmt.Sprintf(`    <variable name="%s"/>`+"\n", varName)
	}
	xml += `  </head>` + "\n"
	xml += `  <results>` + "\n"
	for _, binding := range result.Bindings {
		xml += `    <result>` + "\n"
		for varName, value := range binding {
			xml += fmt.Sprintf(`      <binding name="%s">`, varName) + "\n"
			xml += fmt.Sprintf(`        <uri>%s</uri>`, value) + "\n"
			xml += `      </binding>` + "\n"
		}
		xml += `    </result>` + "\n"
	}
	xml += `  </results>` + "\n"
	xml += `</sparql>` + "\n"

	w.WriteHeader(http.StatusOK)
	_, err := w.Write([]byte(xml))
	return err
}

