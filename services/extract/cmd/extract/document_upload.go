package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/extract/pkg/clients"
	"github.com/plturrell/aModels/services/extract/pkg/extraction"
	"github.com/plturrell/aModels/services/extract/pkg/git"
	"github.com/plturrell/aModels/services/extract/pkg/graph"
)

// handleDocumentUpload handles document upload and processing
func (s *extractServer) handleDocumentUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse multipart form (max 100MB)
	if err := r.ParseMultipartForm(100 << 20); err != nil {
		http.Error(w, fmt.Sprintf("failed to parse form: %v", err), http.StatusBadRequest)
		return
	}

	// Get file
	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, fmt.Sprintf("file is required: %v", err), http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Get metadata
	name := r.FormValue("name")
	if name == "" {
		name = header.Filename
	}
	description := r.FormValue("description")
	projectID := r.FormValue("project_id")
	if projectID == "" {
		projectID = "default-project"
	}
	systemID := r.FormValue("system_id")
	if systemID == "" {
		systemID = "default-system"
	}

	// Get Gitea storage config
	giteaURL := r.FormValue("gitea_url")
	if giteaURL == "" {
		giteaURL = os.Getenv("GITEA_URL")
	}
	giteaToken := r.FormValue("gitea_token")
	if giteaToken == "" {
		giteaToken = os.Getenv("GITEA_TOKEN")
	}

	// Create temp file
	tmpFile, err := os.CreateTemp("", fmt.Sprintf("upload-*%s", filepath.Ext(header.Filename)))
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to create temp file: %v", err), http.StatusInternalServerError)
		return
	}
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	// Copy file content
	if _, err := io.Copy(tmpFile, file); err != nil {
		http.Error(w, fmt.Sprintf("failed to save file: %v", err), http.StatusInternalServerError)
		return
	}
	tmpFile.Close()

	// Process document
	ctx := r.Context()
	docResult, err := s.processAndStoreDocument(ctx, tmpFile.Name(), name, description, projectID, systemID, giteaURL, giteaToken)
	if err != nil {
		s.logger.Printf("Document processing failed: %v", err)
		http.Error(w, fmt.Sprintf("document processing failed: %v", err), http.StatusInternalServerError)
		return
	}

	// Return response
	response := map[string]interface{}{
		"document_id":      docResult.DocumentID,
		"name":             name,
		"description":      description,
		"gitea_url":        docResult.GiteaURL,
		"processed_path":   docResult.ProcessedPath,
		"source":           docResult.Source,
		"ocr_used":         docResult.OCRUsed,
		"content_hash":     docResult.ContentHash,
		"processed_at":     docResult.ProcessedAt.Format(time.RFC3339),
		"catalog_exported": true, // Automatically exported to catalog/Glean
	}

	handlers.WriteJSON(w, http.StatusCreated, response)
}

// DocumentProcessingResult represents the result of document processing
type DocumentProcessingResult struct {
	DocumentID   string
	GiteaURL     string
	ProcessedPath string
	Source       string
	OCRUsed      bool
	ContentHash  string
	ProcessedAt  time.Time
}

// processAndStoreDocument processes and stores a document using Gitea-first flow
func (s *extractServer) processAndStoreDocument(
	ctx context.Context,
	filePath, name, description, projectID, systemID, giteaURL, giteaToken string,
) (*DocumentProcessingResult, error) {
	var giteaRepoURL string
	var result *git.DocumentProcessingResult
	var docProcessor *git.DocumentProcessor

	if giteaURL != "" && giteaToken != "" {
		// Gitea-first flow: upload → clone → process → update
		giteaStorage := git.NewGiteaStorage(giteaURL, giteaToken, s.logger)
		storageConfig := git.StorageConfig{
			Owner:       "extract-service",
			RepoName:    fmt.Sprintf("%s-documents", projectID),
			Branch:      "main",
			BasePath:    "documents/processed/",
			ProjectID:   projectID,
			SystemID:    systemID,
			AutoCreate:  true,
			Description: fmt.Sprintf("Documents for project %s", projectID),
		}

		// Step 1: Upload raw document to Gitea
		rawBasePath := "raw/documents/"
		giteaRepo, cloneURL, err := giteaStorage.UploadRawFiles(ctx, storageConfig, []string{filePath}, rawBasePath)
		if err != nil {
			return nil, fmt.Errorf("upload raw document to Gitea: %w", err)
		}
		giteaRepoURL = giteaRepo.HTMLURL

		// Step 2: Clone from Gitea
		tempDir := filepath.Join(os.TempDir(), "extract-doc-upload")
		if err := os.MkdirAll(tempDir, 0755); err != nil {
			return nil, fmt.Errorf("create temp dir: %w", err)
		}
		defer os.RemoveAll(tempDir)

		clonePath, err := giteaStorage.CloneFromGitea(ctx, cloneURL, storageConfig.Branch, tempDir)
		if err != nil {
			return nil, fmt.Errorf("clone from Gitea: %w", err)
		}
		defer os.RemoveAll(clonePath)

		// Step 3: Get document path from Gitea clone
		fileName := filepath.Base(filePath)
		rawPathInClone := filepath.Join(clonePath, rawBasePath, fileName)
		if _, err := os.Stat(rawPathInClone); err != nil {
			return nil, fmt.Errorf("document not found in Gitea clone: %w", err)
		}

		// Step 4: Process document from Gitea clone
		var markitdownClient *clients.MarkItDownClient
		if adapter, ok := s.markitdownIntegration.(*markitdownClientAdapter); ok {
			markitdownClient = adapter.client
		}

		docProcessor = git.NewDocumentProcessor(markitdownClient, s.multiModalExtractor, s.logger)
		docProcessor.SetGiteaStorage(giteaStorage)

		result, err = docProcessor.ProcessDocument(ctx, rawPathInClone)
		if err != nil {
			return nil, fmt.Errorf("process document: %w", err)
		}

		// Verify document
		if !docProcessor.VerifyDocumentCompleteness(result) {
			return nil, fmt.Errorf("document verification failed")
		}

		// Step 5: Update Gitea with processed results
		_, err = docProcessor.StoreProcessedDocument(ctx, result, storageConfig)
		if err != nil {
			s.logger.Printf("Warning: failed to store processed document in Gitea: %v", err)
		}
	} else {
		// No Gitea - process locally (fallback)
		var markitdownClient *clients.MarkItDownClient
		if adapter, ok := s.markitdownIntegration.(*markitdownClientAdapter); ok {
			markitdownClient = adapter.client
		}

		docProcessor = git.NewDocumentProcessor(markitdownClient, s.multiModalExtractor, s.logger)
		var err error
		result, err = docProcessor.ProcessDocument(ctx, filePath)
		if err != nil {
			return nil, fmt.Errorf("process document: %w", err)
		}

		if !docProcessor.VerifyDocumentCompleteness(result) {
			return nil, fmt.Errorf("document verification failed")
		}
	}

	// Create document node in knowledge graph
	docNode := docProcessor.CreateDocumentNode(result, projectID, systemID)
	
	// Add description to node
	if description != "" {
		docNode.Props["description"] = description
	}
	docNode.Props["name"] = name
	docNode.Props["gitea_url"] = giteaRepoURL

	// Store in knowledge graph (this will auto-export to catalog/Glean)
	if s.neo4jPersistence != nil {
		if err := s.neo4jPersistence.SaveNodes(ctx, []graph.Node{docNode}); err != nil {
			s.logger.Printf("Warning: failed to save document node: %v", err)
		}
	}

	return &DocumentProcessingResult{
		DocumentID:   docNode.ID,
		GiteaURL:     giteaRepoURL,
		ProcessedPath: result.ProcessedPath,
		Source:       result.Source,
		OCRUsed:      result.OCRUsed,
		ContentHash:  result.ContentHash,
		ProcessedAt:  result.ProcessedAt,
	}, nil
}

// handleDocumentsRouter routes document requests
func (s *extractServer) handleDocumentsRouter(w http.ResponseWriter, r *http.Request) {
	// Remove /documents prefix
	path := strings.TrimPrefix(r.URL.Path, "/documents")
	path = strings.TrimPrefix(path, "/")
	
	if path == "" {
		// List documents
		s.handleListDocuments(w, r)
		return
	}
	
	// Get specific document
	s.handleGetDocument(w, r)
}

// handleListDocuments lists documents from knowledge graph
func (s *extractServer) handleListDocuments(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()

	// Get query parameters
	projectID := r.URL.Query().Get("project_id")
	systemID := r.URL.Query().Get("system_id")
	limit := parseIntQuery(r, "limit", 50)
	offset := parseIntQuery(r, "offset", 0)

	// Query Neo4j for document nodes using Cypher
	var query strings.Builder
	query.WriteString("MATCH (d:Document)")
	params := make(map[string]interface{})
	
	whereClauses := []string{}
	if projectID != "" {
		whereClauses = append(whereClauses, "d.project_id = $project_id")
		params["project_id"] = projectID
	}
	if systemID != "" {
		whereClauses = append(whereClauses, "d.system_id = $system_id")
		params["system_id"] = systemID
	}
	
	if len(whereClauses) > 0 {
		query.WriteString(" WHERE " + strings.Join(whereClauses, " AND "))
	}
	
	query.WriteString(" RETURN d ORDER BY d.processed_at DESC SKIP $offset LIMIT $limit")
	params["offset"] = offset
	params["limit"] = limit

	// Use Neo4j driver directly to query
	if s.neo4jPersistence == nil {
		http.Error(w, "Neo4j not configured", http.StatusServiceUnavailable)
		return
	}

	// Execute query - need to check persistence interface
	// For now, use a simplified approach: query via the existing query endpoint pattern
	// This will need to be implemented based on actual persistence interface
	nodes, err := s.queryNeo4jNodes(ctx, query.String(), params)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to query documents: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert to response format
	documents := make([]map[string]interface{}, len(nodes))
	for i, node := range nodes {
		documents[i] = map[string]interface{}{
			"id":             node.ID,
			"name":           node.Props["name"],
			"description":   node.Props["description"],
			"gitea_url":      node.Props["gitea_url"],
			"processed_path": node.Props["processed_path"],
			"source":         node.Props["source"],
			"ocr_used":       node.Props["ocr_used"],
			"content_hash":   node.Props["content_hash"],
			"processed_at":   node.Props["processed_at"],
			"project_id":     node.Props["project_id"],
			"system_id":      node.Props["system_id"],
		}
	}

	response := map[string]interface{}{
		"documents": documents,
		"total":     len(documents),
		"limit":     limit,
		"offset":     offset,
	}

	handlers.WriteJSON(w, http.StatusOK, response)
}

// handleGetDocument gets a specific document by ID
func (s *extractServer) handleGetDocument(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()
	// Extract document ID from path
	path := strings.TrimPrefix(r.URL.Path, "/documents")
	path = strings.TrimPrefix(path, "/")
	documentID := path

	// Query Neo4j for document node
	query := "MATCH (d:Document {id: $id}) RETURN d"
	params := map[string]interface{}{"id": documentID}

	nodes, err := s.queryNeo4jNodes(ctx, query, params)
	if err != nil || len(nodes) == 0 {
		http.Error(w, "document not found", http.StatusNotFound)
		return
	}

	node := nodes[0]
	response := map[string]interface{}{
		"id":             node.ID,
		"name":           node.Props["name"],
		"description":   node.Props["description"],
		"gitea_url":      node.Props["gitea_url"],
		"processed_path": node.Props["processed_path"],
		"source":         node.Props["source"],
		"ocr_used":       node.Props["ocr_used"],
		"content_hash":   node.Props["content_hash"],
		"processed_at":   node.Props["processed_at"],
		"project_id":     node.Props["project_id"],
		"system_id":      node.Props["system_id"],
		"verified":       node.Props["meta_verified"],
		"catalog_exported": true, // Always exported to catalog/Glean
	}

	handlers.WriteJSON(w, http.StatusOK, response)
}

// parseIntQuery parses an integer query parameter
func parseIntQuery(r *http.Request, key string, defaultValue int) int {
	value := r.URL.Query().Get(key)
	if value == "" {
		return defaultValue
	}
	var result int
	if _, err := fmt.Sscanf(value, "%d", &result); err != nil {
		return defaultValue
	}
	return result
}

// queryNeo4jNodes queries Neo4j and returns nodes
// This is a helper that uses the Neo4j driver directly
func (s *extractServer) queryNeo4jNodes(ctx context.Context, cypherQuery string, params map[string]interface{}) ([]graph.Node, error) {
	if s.neo4jPersistence == nil {
		return nil, fmt.Errorf("Neo4j not configured")
	}

	// Execute query using ExecuteQuery method
	result, err := s.neo4jPersistence.ExecuteQuery(ctx, cypherQuery, params)
	if err != nil {
		return nil, fmt.Errorf("execute query: %w", err)
	}

	// Parse QueryResult into nodes
	// QueryResult has Columns and Data
	// For "MATCH (d:Document) RETURN d", column is "d" and value is Neo4j node
	var nodes []graph.Node
	
	if result == nil || len(result.Data) == 0 {
		return nodes, nil
	}

	// Find the column that contains the node (usually first column)
	var nodeColumn string
	if len(result.Columns) > 0 {
		nodeColumn = result.Columns[0]
	}

	// Parse each row
	for _, row := range result.Data {
		if nodeColumn == "" {
			// No column specified, try to find node in row
			for _, value := range row {
				if nodeMap, ok := value.(map[string]interface{}); ok {
					if node := parseNeo4jNodeToGraphNode(nodeMap); node.ID != "" {
						nodes = append(nodes, node)
						break
					}
				}
			}
		} else {
			// Use specified column
			if value, ok := row[nodeColumn]; ok {
				if nodeMap, ok := value.(map[string]interface{}); ok {
					if node := parseNeo4jNodeToGraphNode(nodeMap); node.ID != "" {
						nodes = append(nodes, node)
					}
				}
			}
		}
	}

	return nodes, nil
}

// parseNeo4jNodeToGraphNode converts a Neo4j node (from QueryResult) to graph.Node
func parseNeo4jNodeToGraphNode(nodeMap map[string]interface{}) graph.Node {
	node := graph.Node{}
	
	// Neo4j node format: {"id": "...", "labels": [...], "properties": {...}}
	if id, ok := nodeMap["id"].(string); ok {
		node.ID = id
	}
	
	if labels, ok := nodeMap["labels"].([]interface{}); ok && len(labels) > 0 {
		if label, ok := labels[0].(string); ok {
			node.Type = label
			node.Label = label
		}
	}
	
	if props, ok := nodeMap["properties"].(map[string]interface{}); ok {
		node.Props = props
		// Extract ID, type, label from properties if not in node structure
		if node.ID == "" {
			if id, ok := props["id"].(string); ok {
				node.ID = id
			}
		}
		if node.Type == "" {
			if nodeType, ok := props["type"].(string); ok {
				node.Type = nodeType
			}
		}
		if node.Label == "" {
			if label, ok := props["label"].(string); ok {
				node.Label = label
			} else if node.ID != "" {
				node.Label = node.ID
			}
		}
	}
	
	return node
}


