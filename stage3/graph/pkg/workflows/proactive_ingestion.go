package workflows

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/langchain-ai/langgraph-go/internal/catalog/flightcatalog"
	extractgrpcclient "github.com/langchain-ai/langgraph-go/pkg/clients/extractgrpc"
	extractpersist "github.com/langchain-ai/langgraph-go/pkg/persistence/extract"
	"github.com/langchain-ai/langgraph-go/pkg/stategraph"
	extractpb "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Extract/gen/extractpb"
)

// EdgeSpec describes a directed connection between two nodes for helper builders.
type EdgeSpec struct {
	From  string
	To    string
	Label string
}

// GraphOptions configures runtime dependencies for the proactive ingestion workflow.
type GraphOptions struct {
	SearchServiceURL string
	ExtractHTTPURL   string
	ExtractGRPC      *extractgrpcclient.Client
}

type extractServiceExtraction struct {
	ExtractionClass string         `json:"extraction_class"`
	ExtractionText  string         `json:"extraction_text"`
	Attributes      map[string]any `json:"attributes,omitempty"`
}

type extractServiceResponse struct {
	Entities    map[string][]string        `json:"entities"`
	Extractions []extractServiceExtraction `json:"extractions"`
}

var extractHTTPClient = &http.Client{
	Timeout: 30 * time.Second,
}

var agentSDKHTTPClient = &http.Client{
	Timeout: 20 * time.Second,
}

// BuildGraph constructs a compiled state graph using the supplied node handlers and edges.
func BuildGraph(entry, exit string, nodes map[string]stategraph.NodeFunc, edges []EdgeSpec) (*stategraph.CompiledStateGraph, error) {
	if len(nodes) == 0 {
		return nil, fmt.Errorf("graph builder: at least one node must be provided")
	}
	builder := stategraph.New()
	for id, handler := range nodes {
		if err := builder.AddNode(id, handler); err != nil {
			return nil, err
		}
	}
	for _, edge := range edges {
		if err := builder.AddEdge(edge.From, edge.To, stategraph.WithEdgeLabel(edge.Label)); err != nil {
			return nil, err
		}
	}
	if entry != "" {
		builder.SetEntryPoint(entry)
	}
	if exit != "" {
		builder.SetFinishPoint(exit)
	}
	return builder.Compile()
}

// wrapStateFunc adapts map-based node handlers into the runtime-agnostic
// signature used by the stategraph engine.
func wrapStateFunc(fn func(context.Context, map[string]any) (map[string]any, error)) stategraph.NodeFunc {
	return func(ctx context.Context, input any) (any, error) {
		state, ok := input.(map[string]any)
		if !ok || state == nil {
			state = make(map[string]any)
		}
		return fn(ctx, state)
	}
}

// ReadFileNode returns a node that reads a file.
func ReadFileNode() stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		filePath, ok := state["file_path"].(string)
		if !ok {
			return nil, fmt.Errorf("file_path not found in state")
		}

		log.Printf("Reading file: %s", filePath)
		content, err := ioutil.ReadFile(filePath)
		if err != nil {
			return nil, err
		}

		newState := make(map[string]any)
		for k, v := range state {
			newState[k] = v
		}
		newState["document_content"] = string(content)

		return newState, nil
	})
}

// ExtractEntitiesNode returns a node that extracts entities from a document.
func ExtractEntitiesNode(searchServiceURL, extractServiceURL string, extractGRPC *extractgrpcclient.Client) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		documentContent, ok := state["document_content"].(string)
		if !ok {
			return nil, fmt.Errorf("document_content not found in state")
		}

		log.Println("Extracting entities from document...")

		var (
			entities         map[string]any
			extractionSource string
			rawExtractions   []extractServiceExtraction
		)

		if extractGRPC != nil {
			if resp, err := extractGRPC.Extract(ctx, documentContent); err != nil {
				log.Printf("extract gRPC unavailable: %v", err)
			} else if resp != nil {
				if grpcEntities := protoEntitiesToState(resp.GetEntities()); len(grpcEntities) > 0 {
					entities = grpcEntities
					extractionSource = "extract-grpc"
				}
				if protoExt := protoExtractionsToRecords(resp.GetExtractions()); len(protoExt) > 0 {
					rawExtractions = protoExt
				}
			}
		}

		if entities == nil && extractServiceURL != "" && extractServiceURL != "offline" {
			if remoteEntities, serviceExtractions, err := requestEntityExtraction(ctx, extractServiceURL, documentContent); err != nil {
				log.Printf("extract service unavailable: %v", err)
			} else {
				entities = remoteEntities
				extractionSource = "extract-service"
				rawExtractions = serviceExtractions
			}
		}

		if entities == nil && shouldUseTool(state, "search_documents") {
			if remoteEntities, toolSource, err := fetchEntitiesViaMCP(ctx, state, documentContent, "search_documents"); err == nil {
				entities = remoteEntities
				extractionSource = toolSource
			}
		}

		if entities == nil && searchServiceURL != "offline" && searchServiceURL != "" {
			prompt := "Please extract the key entities (people, projects, dates, locations) from the following document. Return the result as a JSON object with keys 'people', 'projects', 'dates', and 'locations'.\n\n" + documentContent

			requestBody, err := json.Marshal(map[string]string{
				"query": prompt,
			})
			if err == nil {
				resp, err := http.Post(searchServiceURL+"/v1/ai-search", "application/json", bytes.NewBuffer(requestBody))
				if err == nil {
					defer resp.Body.Close()
					if resp.StatusCode == http.StatusOK {
						var result map[string]any
						if err := json.NewDecoder(resp.Body).Decode(&result); err == nil {
							var parsed map[string]any
							if raw, ok := result["response"].(string); ok {
								if err := json.Unmarshal([]byte(raw), &parsed); err == nil {
									entities = parsed
									extractionSource = "search-service"
								} else {
									log.Printf("Could not parse entities from LLM response: %v", err)
								}
							}
						}
					} else {
						log.Printf("AI extraction service returned status %s, falling back to heuristic extraction", resp.Status)
					}
				} else {
					log.Printf("AI extraction service unavailable: %v", err)
				}
			}
		}

		if entities == nil {
			extractionSource = "heuristic"
			entities = heuristicEntityExtraction(documentContent)
		} else {
			ensureEntityStateKeys(entities)
		}

		newState := make(map[string]any)
		for k, v := range state {
			newState[k] = v
		}
		newState["entities"] = entities
		records := extractionRecordsFromService(rawExtractions, extractionSource)
		if len(records) == 0 {
			records = extractionRecordsFromEntities(entities, extractionSource)
		}
		newState["extraction_records"] = records
		newState["extraction_source"] = extractionSource

		log.Println("Entities extracted successfully")
		return newState, nil
	})
}

func requestEntityExtraction(ctx context.Context, serviceURL, documentContent string) (map[string]any, []extractServiceExtraction, error) {
	endpoint := strings.TrimRight(serviceURL, "/") + "/extract"
	payload := map[string]any{
		"document": documentContent,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal extract payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewBuffer(body))
	if err != nil {
		return nil, nil, fmt.Errorf("build extract request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := extractHTTPClient.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("request extract service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := ioutil.ReadAll(resp.Body)
		return nil, nil, fmt.Errorf("extract service responded with %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}

	var svcResp extractServiceResponse
	if err := json.NewDecoder(resp.Body).Decode(&svcResp); err != nil {
		return nil, nil, fmt.Errorf("decode extract service response: %w", err)
	}

	if len(svcResp.Entities) == 0 && len(svcResp.Extractions) > 0 {
		svcResp.Entities = groupExtractsByClass(svcResp.Extractions)
	}

	result := make(map[string]any, len(svcResp.Entities))
	for key, values := range svcResp.Entities {
		cloned := append([]string(nil), values...)
		result[key] = cloned
	}
	ensureEntityStateKeys(result)
	return result, svcResp.Extractions, nil
}

func shouldUseTool(state map[string]any, toolName string) bool {
	if state == nil {
		return false
	}
	toolName = strings.ToLower(strings.TrimSpace(toolName))
	if toolName == "" {
		return false
	}
	if catalogPtr, ok := state["agent_catalog"].(*flightcatalog.Catalog); ok && catalogPtr != nil {
		for _, suite := range catalogPtr.Suites {
			for _, tool := range suite.ToolNames {
				if strings.EqualFold(tool, toolName) {
					return true
				}
			}
		}
	}
	if catalogVal, ok := state["agent_catalog"].(flightcatalog.Catalog); ok {
		for _, suite := range catalogVal.Suites {
			for _, tool := range suite.ToolNames {
				if strings.EqualFold(tool, toolName) {
					return true
				}
			}
		}
	}
	if genericTools, ok := state["agent_tools"].([]any); ok {
		for _, raw := range genericTools {
			switch t := raw.(type) {
			case string:
				if strings.EqualFold(t, toolName) {
					return true
				}
			case map[string]any:
				if name, ok := t["name"].(string); ok && strings.EqualFold(name, toolName) {
					return true
				}
			}
		}
	}
	return false
}

func fetchEntitiesViaMCP(ctx context.Context, state map[string]any, documentContent, toolName string) (map[string]any, string, error) {
	entities, source, err := fetchEntitiesViaAgentSDK(ctx, documentContent, toolName)
	if err == nil {
		return entities, source, nil
	}
	log.Printf("agent sdk tool %s unavailable: %v; falling back to heuristics", toolName, err)
	fallback := heuristicEntityExtraction(documentContent)
	ensureEntityStateKeys(fallback)
	return fallback, "heuristic", nil
}

func fetchEntitiesViaAgentSDK(ctx context.Context, documentContent, toolName string) (map[string]any, string, error) {
	addr := strings.TrimSpace(os.Getenv("AGENTSDK_HTTP_ADDR"))
	if addr == "" {
		return nil, "", fmt.Errorf("AGENTSDK_HTTP_ADDR not configured")
	}
	base := normaliseAgentSDKAddr(addr)
	if base == "" {
		return nil, "", fmt.Errorf("invalid AGENTSDK_HTTP_ADDR: %s", addr)
	}
	trimmed := documentContent
	if len(trimmed) > 1600 {
		trimmed = trimmed[:1600]
	}
	prompt := fmt.Sprintf("Use the tool %q with JSON input {\\\"query\\\": %q, \\\"top_k\\\": 5}. Return only the JSON emitted by the tool.", toolName, trimmed)
	requestPayload := map[string]any{
		"prompt": prompt,
	}
	body, err := json.Marshal(requestPayload)
	if err != nil {
		return nil, "", err
	}
	endpoint := fmt.Sprintf("%s/agenticAiETH.agentsdk.v1.AgentService/ProcessTask", base)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := agentSDKHTTPClient.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, "", fmt.Errorf("agent sdk responded %s: %s", resp.Status, strings.TrimSpace(string(b)))
	}
	var parsed struct {
		Content string `json:"content"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, "", err
	}
	if parsed.Content == "" {
		return nil, "", fmt.Errorf("agent sdk returned empty content")
	}
	var entities map[string]any
	if err := json.Unmarshal([]byte(parsed.Content), &entities); err != nil {
		// Fallback to heuristics so we still produce an answer.
		entities = heuristicEntityExtraction(documentContent)
	} else {
		ensureEntityStateKeys(entities)
	}
	return entities, "agent-sdk:" + toolName, nil
}

func normaliseAgentSDKAddr(raw string) string {
	if strings.HasPrefix(raw, "http://") || strings.HasPrefix(raw, "https://") {
		return strings.TrimRight(raw, "/")
	}
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}
	if strings.HasPrefix(trimmed, ":") {
		return "http://127.0.0.1" + trimmed
	}
	return "http://" + strings.TrimRight(trimmed, "/")
}

func groupExtractsByClass(extractions []extractServiceExtraction) map[string][]string {
	grouped := map[string][]string{}
	for _, ext := range extractions {
		class := strings.ToLower(strings.TrimSpace(ext.ExtractionClass))
		text := strings.TrimSpace(ext.ExtractionText)
		if class == "" || text == "" {
			continue
		}
		if !containsString(grouped[class], text) {
			grouped[class] = append(grouped[class], text)
		}
	}
	return grouped
}

func containsString(values []string, candidate string) bool {
	for _, v := range values {
		if v == candidate {
			return true
		}
	}
	return false
}

func protoEntitiesToState(source map[string]*extractpb.EntityList) map[string]any {
	if len(source) == 0 {
		return nil
	}
	out := make(map[string]any, len(source))
	for key, list := range source {
		if list == nil {
			continue
		}
		values := append([]string(nil), list.GetValues()...)
		out[key] = values
	}
	return out
}

func protoExtractionsToRecords(results []*extractpb.ExtractionResult) []extractServiceExtraction {
	if len(results) == 0 {
		return nil
	}
	records := make([]extractServiceExtraction, 0, len(results))
	for _, res := range results {
		if res == nil {
			continue
		}
		attrs := map[string]any{}
		if s := res.GetAttributes(); s != nil {
			for k, v := range s.AsMap() {
				attrs[k] = v
			}
		}
		if start := res.GetStartIndex(); start != nil {
			attrs["start_index"] = start.GetValue()
		}
		if end := res.GetEndIndex(); end != nil {
			attrs["end_index"] = end.GetValue()
		}
		records = append(records, extractServiceExtraction{
			ExtractionClass: res.GetExtractionClass(),
			ExtractionText:  res.GetExtractionText(),
			Attributes:      attrs,
		})
	}
	return records
}

func ensureEntityStateKeys(entities map[string]any) {
	if entities == nil {
		return
	}
	for _, key := range []string{"people", "projects", "dates", "locations"} {
		if _, ok := entities[key]; !ok {
			entities[key] = []string{}
		}
	}
}

// PersistEntitiesNode writes extraction outputs to configured persistence targets.
func PersistEntitiesNode() stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		recordIface, ok := state["extraction_records"]
		if !ok {
			log.Println("No extraction records present; skipping persistence")
			return state, nil
		}

		recordMaps := normalizeRecordSlice(recordIface)
		if len(recordMaps) == 0 {
			log.Println("Extraction records empty; skipping persistence")
			return state, nil
		}

		filePath, _ := state["file_path"].(string)
		if filePath == "" {
			filePath = "unknown"
		}

		records := make([]extractpersist.Record, 0, len(recordMaps))
		now := time.Now().UTC()
		for _, rm := range recordMaps {
			class, _ := rm["class"].(string)
			text, _ := rm["text"].(string)
			source, _ := rm["source"].(string)
			attrs, _ := rm["attributes"].(map[string]any)
			if class == "" || text == "" {
				continue
			}
			records = append(records, extractpersist.Record{
				FilePath:    filePath,
				Class:       strings.ToLower(class),
				Text:        strings.TrimSpace(text),
				Attributes:  attrs,
				Source:      source,
				ExtractedAt: now,
			})
		}

		if len(records) == 0 {
			log.Println("Extraction records filtered to zero entries; skipping persistence")
			return state, nil
		}

		summary, err := extractpersist.Persist(ctx, records)
		if err != nil {
			return nil, err
		}

		newState := make(map[string]any, len(state)+2)
		for k, v := range state {
			newState[k] = v
		}
		newState["persistence_summary"] = summary.Targets
		newState["persistence_records"] = summary.Records

		log.Printf("Persisted %d records across backends", summary.Records)
		return newState, nil
	})
}

func extractionRecordsFromService(extractions []extractServiceExtraction, source string) []map[string]any {
	records := make([]map[string]any, 0, len(extractions))
	for _, ext := range extractions {
		class := strings.ToLower(strings.TrimSpace(ext.ExtractionClass))
		text := strings.TrimSpace(ext.ExtractionText)
		if class == "" || text == "" {
			continue
		}
		records = append(records, map[string]any{
			"class":      class,
			"text":       text,
			"source":     source,
			"attributes": ext.Attributes,
		})
	}
	return records
}

func extractionRecordsFromEntities(entities map[string]any, source string) []map[string]any {
	records := []map[string]any{}
	if entities == nil {
		return records
	}
	for class, value := range entities {
		entries, ok := value.([]string)
		if !ok {
			continue
		}
		for _, text := range entries {
			trimmed := strings.TrimSpace(text)
			if trimmed == "" {
				continue
			}
			records = append(records, map[string]any{
				"class":  class,
				"text":   trimmed,
				"source": source,
			})
		}
	}
	return records
}

func normalizeRecordSlice(value any) []map[string]any {
	switch v := value.(type) {
	case []map[string]any:
		return v
	case []any:
		records := make([]map[string]any, 0, len(v))
		for _, item := range v {
			if m, ok := item.(map[string]any); ok {
				records = append(records, m)
			}
		}
		return records
	default:
		return nil
	}
}

type trainingGenerationResponse struct {
	Success  bool     `json:"success"`
	Mode     string   `json:"mode"`
	Manifest string   `json:"manifest"`
	Files    []string `json:"files"`
}

func GenerateTrainingArtifactsNode(extractServiceURL string) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		if extractServiceURL == "" || extractServiceURL == "offline" {
			return state, nil
		}
		if !strings.EqualFold(os.Getenv("ENABLE_DOCUMENT_TRAINING_EXPORT"), "true") {
			return state, nil
		}
		filePath, ok := state["file_path"].(string)
		if !ok || strings.TrimSpace(filePath) == "" {
			log.Println("Training export enabled but file_path missing; skipping")
			return state, nil
		}
		if !isSupportedDocument(filePath) {
			return state, nil
		}

		payload := map[string]any{
			"mode": "document",
			"document": map[string]any{
				"inputs": []string{filePath},
			},
		}
		body, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal training payload: %w", err)
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(extractServiceURL, "/")+"/generate/training", bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("build training request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := extractHTTPClient.Do(req)
		if err != nil {
			log.Printf("document training export failed: %v", err)
			return state, nil
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			msg, _ := ioutil.ReadAll(io.LimitReader(resp.Body, 1024))
			log.Printf("document training export returned %s: %s", resp.Status, strings.TrimSpace(string(msg)))
			return state, nil
		}

		var genResp trainingGenerationResponse
		if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
			log.Printf("document training export decode failed: %v", err)
			return state, nil
		}
		if !genResp.Success {
			log.Printf("document training export unsuccessful for %s", filePath)
			return state, nil
		}

		newState := make(map[string]any, len(state)+1)
		for k, v := range state {
			newState[k] = v
		}
		newState["training_artifacts"] = map[string]any{
			"manifest": genResp.Manifest,
			"files":    genResp.Files,
		}
		return newState, nil
	})
}

func isSupportedDocument(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp":
		return true
	default:
		return false
	}
}

// IngestDocumentNode returns a node that ingests a document.
func IngestDocumentNode(searchServiceURL string) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		if searchServiceURL == "offline" || searchServiceURL == "" {
			log.Println("Skipping remote ingestion (search service offline)")
			return state, nil
		}

		filePath, ok := state["file_path"].(string)
		if !ok {
			return nil, fmt.Errorf("file_path not found in state")
		}
		documentContent, ok := state["document_content"].(string)
		if !ok {
			return nil, fmt.Errorf("document_content not found in state")
		}

		log.Println("Ingesting document...")

		requestBody, err := json.Marshal(map[string]string{
			"id":      filePath,
			"content": documentContent,
		})
		if err != nil {
			return nil, err
		}

		resp, err := http.Post(searchServiceURL+"/v1/documents", "application/json", bytes.NewBuffer(requestBody))
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusNoContent {
			return nil, fmt.Errorf("failed to ingest document: %s", resp.Status)
		}

		log.Println("Document ingested successfully")
		return state, nil
	})
}

// SummarizeDocumentNode returns a node that summarizes a document.
func SummarizeDocumentNode(searchServiceURL string) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		documentContent, ok := state["document_content"].(string)
		if !ok {
			return nil, fmt.Errorf("document_content not found in state")
		}

		log.Println("Summarizing document...")

		summary := simpleSummary(documentContent)

		if searchServiceURL != "offline" && searchServiceURL != "" {
			prompt := "Please summarize the following document:\n\n" + documentContent

			requestBody, err := json.Marshal(map[string]string{
				"query": prompt,
			})
			if err == nil {
				if resp, err := http.Post(searchServiceURL+"/v1/ai-search", "application/json", bytes.NewBuffer(requestBody)); err == nil {
					defer resp.Body.Close()
					if resp.StatusCode == http.StatusOK {
						var result map[string]any
						if err := json.NewDecoder(resp.Body).Decode(&result); err == nil {
							if s, ok := result["response"].(string); ok && strings.TrimSpace(s) != "" {
								summary = s
							}
						}
					} else {
						log.Printf("AI summarization service returned status %s, using heuristic summary", resp.Status)
					}
				} else {
					log.Printf("AI summarization service unavailable: %v", err)
				}
			}
		}

		newState := make(map[string]any)
		for k, v := range state {
			newState[k] = v
		}
		newState["summary"] = summary

		log.Println("Document summarized successfully")
		return newState, nil
	})
}

// RouteSummaryNode returns a node that routes a summary.
func RouteSummaryNode() stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		summary, ok := state["summary"].(string)
		if !ok {
			return nil, fmt.Errorf("summary not found in state")
		}

		log.Printf("Routing summary: %s", summary)
		// In a real application, this would route the summary to the relevant users.
		return state, nil
	})
}

func heuristicEntityExtraction(content string) map[string]any {
	peopleSet := map[string]struct{}{}
	projectsSet := map[string]struct{}{}

	namePattern := regexp.MustCompile(`\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b`)
	projectPattern := regexp.MustCompile(`(?i)project\s+[A-Z][A-Za-z0-9_-]*`)

	for _, match := range namePattern.FindAllString(content, -1) {
		peopleSet[match] = struct{}{}
	}
	for _, match := range projectPattern.FindAllString(content, -1) {
		cleaned := strings.TrimSpace(match)
		projectsSet[cleaned] = struct{}{}
	}

	people := make([]string, 0, len(peopleSet))
	for name := range peopleSet {
		people = append(people, name)
	}
	projects := make([]string, 0, len(projectsSet))
	for proj := range projectsSet {
		projects = append(projects, proj)
	}

	entities := map[string]any{
		"people":   people,
		"projects": projects,
	}

	// Provide defaults for keys expected downstream
	if _, ok := entities["dates"]; !ok {
		entities["dates"] = []string{}
	}
	if _, ok := entities["locations"]; !ok {
		entities["locations"] = []string{}
	}

	return entities
}

func simpleSummary(content string) string {
	sentences := strings.Split(content, ".")
	for _, sentence := range sentences {
		trimmed := strings.TrimSpace(sentence)
		if trimmed != "" {
			return trimmed + "."
		}
	}
	return "Summary unavailable."
}

// NewProactiveIngestionGraph creates a new graph for proactive ingestion.
func NewProactiveIngestionGraph(opts GraphOptions) (*stategraph.CompiledStateGraph, error) {
	searchServiceURL := opts.SearchServiceURL
	if searchServiceURL == "" {
		searchServiceURL = os.Getenv("SEARCH_SERVICE_URL")
		if searchServiceURL == "" {
			searchServiceURL = "offline"
		}
	}

	extractServiceURL := strings.TrimSpace(opts.ExtractHTTPURL)
	if extractServiceURL == "" {
		extractServiceURL = strings.TrimSpace(os.Getenv("EXTRACT_SERVICE_URL"))
		if extractServiceURL == "" {
			extractServiceURL = "http://extract-service:8081"
		}
	}

	extractGRPC := opts.ExtractGRPC
	if extractGRPC == nil {
		if addr := strings.TrimSpace(os.Getenv("EXTRACT_GRPC_ADDR")); addr != "" {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			client, err := extractgrpcclient.Dial(ctx, addr)
			if err != nil {
				log.Printf("extract gRPC dial failed: %v", err)
			} else {
				extractGRPC = client
			}
		}
	}

	nodes := map[string]stategraph.NodeFunc{
		"read_file":          ReadFileNode(),
		"extract_entities":   ExtractEntitiesNode(searchServiceURL, extractServiceURL, extractGRPC),
		"persist_entities":   PersistEntitiesNode(),
		"generate_training":  GenerateTrainingArtifactsNode(extractServiceURL),
		"ingest_document":    IngestDocumentNode(searchServiceURL),
		"summarize_document": SummarizeDocumentNode(searchServiceURL),
		"route_summary":      RouteSummaryNode(),
	}
	edges := []EdgeSpec{
		{From: "read_file", To: "extract_entities"},
		{From: "extract_entities", To: "persist_entities"},
		{From: "persist_entities", To: "generate_training"},
		{From: "generate_training", To: "ingest_document"},
		{From: "ingest_document", To: "summarize_document"},
		{From: "summarize_document", To: "route_summary"},
	}
	return BuildGraph("read_file", "route_summary", nodes, edges)
}
