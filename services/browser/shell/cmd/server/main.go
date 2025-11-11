package main

//go:generate npm --prefix ../../ui run build
//go:generate rm -rf ui
//go:generate mkdir -p ui/dist
//go:generate cp -R ../../ui/dist/. ui/dist

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

//go:embed ui/dist/*
var embeddedDist embed.FS

var repoRoot string

type shellConfig struct {
	LocalAIBaseURL string
	ModelsDir      string
	SGMIJSONPath   string
	SGMIEndpoint   string
	TrainingAPI    string
	DMSEndpoint    string
	FlowEndpoint   string
	RuntimeEndpoint string
}

func (s *shellServer) handleRuntimeAnalytics(w http.ResponseWriter, r *http.Request) {
	if s.cfg.RuntimeEndpoint == "" {
		http.Error(w, "runtime endpoint not configured", http.StatusBadGateway)
		return
	}

	endpoint := strings.TrimSuffix(s.cfg.RuntimeEndpoint, "/") + "/analytics/dashboard"
	payload, err := s.fetchAPIMap(r.Context(), endpoint)
	if err != nil {
		http.Error(w, fmt.Sprintf("runtime analytics error: %v", err), http.StatusBadGateway)
		return
	}

	respondJSON(w, payload)
}

type shellServer struct {
	cfg        shellConfig
	httpClient *http.Client
}

func init() {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		log.Fatal("unable to determine caller information")
	}
	repoRoot = filepath.Clean(filepath.Join(filepath.Dir(filename), "../../../../../"))
}

func main() {
	addr := flag.String("addr", ":4173", "HTTP listen address")
	flag.Parse()

	distFS, err := fs.Sub(embeddedDist, "ui/dist")
	if err != nil {
		log.Fatalf("failed to mount embedded dist: %v", err)
	}

	app := newShellServer()
	log.Printf("aModels shell server configured with LocalAI base=%q, models dir=%q, SGMI path=%q, DMS endpoint=%q, AgentFlow endpoint=%q, Runtime endpoint=%q", app.cfg.LocalAIBaseURL, app.cfg.ModelsDir, app.cfg.SGMIJSONPath, app.cfg.DMSEndpoint, app.cfg.FlowEndpoint, app.cfg.RuntimeEndpoint)

	fileServer := http.FileServer(http.FS(distFS))
	mux := http.NewServeMux()

	mux.HandleFunc("/api/localai/models", app.handleLocalAIModels)
	mux.HandleFunc("/api/training/dataset", app.handleTrainingDataset)
	mux.HandleFunc("/api/runtime/analytics/dashboard", app.handleRuntimeAnalytics)
	mux.HandleFunc("/api/sgmi/raw", app.handleSgmiRaw)

	// Proxy LocalAI chat requests to LocalAI service
	if app.cfg.LocalAIBaseURL != "" {
		localaiProxy := app.proxyHandler(app.cfg.LocalAIBaseURL, "/localai")
		mux.Handle("/localai/", localaiProxy)
		mux.Handle("/localai", localaiProxy)
	}

	dmsProxy := app.proxyHandler(app.cfg.DMSEndpoint, "/dms")
	mux.Handle("/dms/", dmsProxy)
	mux.Handle("/dms", dmsProxy)

	flowProxy := app.proxyHandler(app.cfg.FlowEndpoint, "/agentflow")
	mux.Handle("/agentflow/", flowProxy)
	mux.Handle("/agentflow", flowProxy)

	runtimeProxy := app.proxyHandler(app.cfg.RuntimeEndpoint, "/runtime")
	mux.Handle("/runtime/", runtimeProxy)
	mux.Handle("/runtime", runtimeProxy)

	// Proxy search requests to gateway service (unified search endpoints)
	searchEndpoint := strings.TrimSpace(os.Getenv("SHELL_SEARCH_ENDPOINT"))
	if searchEndpoint == "" {
		gatewayURL := strings.TrimSpace(os.Getenv("SHELL_GATEWAY_URL"))
		if gatewayURL == "" {
			gatewayURL = strings.TrimSpace(os.Getenv("GATEWAY_URL"))
		}
		if gatewayURL != "" {
			// Gateway has /search/* endpoints, so proxy to gateway directly
			searchEndpoint = gatewayURL
		} else {
			// Fallback to search-inference service
			searchEndpoint = "http://localhost:8090"
		}
	}
	if searchEndpoint != "" {
		// Proxy /search/* to gateway/search/* or search-inference service
		searchProxy := app.proxyHandler(strings.TrimSuffix(searchEndpoint, "/"), "/search")
		mux.Handle("/search/", searchProxy)
		mux.Handle("/search", searchProxy)
	}
	
	// Proxy gateway API requests (for narrative, dashboard, export endpoints)
	gatewayURL := strings.TrimSpace(os.Getenv("SHELL_GATEWAY_URL"))
	if gatewayURL == "" {
		gatewayURL = strings.TrimSpace(os.Getenv("GATEWAY_URL"))
	}
	if gatewayURL != "" {
		gatewayURL = strings.TrimSuffix(gatewayURL, "/")
		// Proxy /api/* to gateway (for backward compatibility)
		apiProxy := app.proxyHandler(gatewayURL, "/api")
		mux.Handle("/api/", apiProxy)
		// Also proxy direct gateway endpoints
		gatewayProxy := app.proxyHandler(gatewayURL, "")
		mux.Handle("/gateway/", gatewayProxy)
	}

	mux.Handle("/assets/", http.StripPrefix("/assets/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		fileServer.ServeHTTP(w, r)
	})))

	mux.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cleanPath := path.Clean(strings.TrimPrefix(r.URL.Path, "/"))

		if cleanPath == "." || cleanPath == "" || cleanPath == "index.html" {
			serveIndex(w, distFS)
			return
		}

		if fileExists(distFS, cleanPath) {
			fileServer.ServeHTTP(w, r)
			return
		}

		serveIndex(w, distFS)
	}))

	server := &http.Server{
		Addr:              *addr,
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 5 * time.Second,
	}

	// Ensure we listen on IPv4 if address is 0.0.0.0
	var listener net.Listener
	if strings.HasPrefix(*addr, "0.0.0.0:") {
		var listenErr error
		listener, listenErr = net.Listen("tcp4", *addr)
		if listenErr != nil {
			log.Fatalf("Failed to listen on IPv4: %v", listenErr)
		}
	} else {
		var listenErr error
		listener, listenErr = net.Listen("tcp", *addr)
		if listenErr != nil {
			log.Fatalf("Failed to listen: %v", listenErr)
		}
	}

	log.Printf("aModels shell server listening on http://%s", server.Addr)
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		log.Fatalf("HTTP server error: %v", err)
	}
}

func newShellServer() *shellServer {
	modelsDir := strings.TrimSpace(os.Getenv("SHELL_MODELS_DIR"))
	if modelsDir == "" {
		modelsDir = filepath.Join(repoRoot, "models")
	} else {
		modelsDir = resolveRepoPath(modelsDir)
	}

	sgmiJSON := strings.TrimSpace(os.Getenv("SHELL_SGMI_JSON"))
	if sgmiJSON == "" {
		sgmiJSON = filepath.Join(repoRoot, "data", "training", "sgmi", "json_with_changes.json")
	} else {
		sgmiJSON = resolveRepoPath(sgmiJSON)
	}

	localAIURL := strings.TrimSpace(os.Getenv("SHELL_LOCALAI_URL"))
	if localAIURL == "" {
		localAIURL = strings.TrimSpace(os.Getenv("LOCALAI_URL"))
	}
	localAIURL = strings.TrimSuffix(localAIURL, "/")

	gatewayURL := strings.TrimSpace(os.Getenv("SHELL_GATEWAY_URL"))
	if gatewayURL == "" {
		gatewayURL = strings.TrimSpace(os.Getenv("GATEWAY_URL"))
	}
	gatewayURL = strings.TrimSuffix(gatewayURL, "/")

	sgmiEndpoint := strings.TrimSpace(os.Getenv("SHELL_SGMI_ENDPOINT"))
	trainingAPI := strings.TrimSpace(os.Getenv("SHELL_TRAINING_DATA_ENDPOINT"))
	dmsEndpoint := strings.TrimSpace(os.Getenv("SHELL_DMS_ENDPOINT"))
	flowEndpoint := strings.TrimSpace(os.Getenv("SHELL_AGENTFLOW_ENDPOINT"))
	runtimeEndpoint := strings.TrimSpace(os.Getenv("SHELL_RUNTIME_ENDPOINT"))

	if sgmiEndpoint == "" && gatewayURL != "" {
		sgmiEndpoint = gatewayURL + "/shell/sgmi/raw"
	}
	if trainingAPI == "" && gatewayURL != "" {
		trainingAPI = gatewayURL + "/shell/training/dataset"
	}
	if dmsEndpoint == "" && gatewayURL != "" {
		dmsEndpoint = gatewayURL + "/dms"
	}
	if flowEndpoint == "" && gatewayURL != "" {
		flowEndpoint = gatewayURL + "/agentflow"
	}
	if runtimeEndpoint == "" && gatewayURL != "" {
		runtimeEndpoint = gatewayURL + "/runtime"
	}

	return &shellServer{
		cfg: shellConfig{
			LocalAIBaseURL: localAIURL,
			ModelsDir:      modelsDir,
			SGMIJSONPath:   sgmiJSON,
			SGMIEndpoint:   sgmiEndpoint,
			TrainingAPI:    trainingAPI,
			DMSEndpoint:    strings.TrimSuffix(dmsEndpoint, "/"),
			FlowEndpoint:   strings.TrimSuffix(flowEndpoint, "/"),
			RuntimeEndpoint: strings.TrimSuffix(runtimeEndpoint, "/"),
		},
		httpClient: &http.Client{Timeout: 15 * time.Second},
	}
}

func resolveRepoPath(candidate string) string {
	if candidate == "" {
		return ""
	}
	if filepath.IsAbs(candidate) {
		return filepath.Clean(candidate)
	}
	return filepath.Join(repoRoot, filepath.FromSlash(candidate))
}

func (s *shellServer) handleLocalAIModels(w http.ResponseWriter, r *http.Request) {
	inventory, err := s.buildLocalAIInventory(r.Context())
	if err != nil {
		http.Error(w, fmt.Sprintf("localai inventory error: %v", err), http.StatusInternalServerError)
		return
	}
	respondJSON(w, inventory)
}

func (s *shellServer) handleTrainingDataset(w http.ResponseWriter, r *http.Request) {
	if s.cfg.TrainingAPI != "" {
		if payload, err := s.fetchAPIMap(r.Context(), s.cfg.TrainingAPI); err == nil {
			respondJSON(w, payload)
			return
		} else {
			log.Printf("shell: training dataset API fallback (%s): %v", s.cfg.TrainingAPI, err)
		}
	}

	dataset, err := s.buildTrainingDataset()
	if err != nil {
		http.Error(w, fmt.Sprintf("training dataset error: %v", err), http.StatusInternalServerError)
		return
	}
	respondJSON(w, dataset)
}

func (s *shellServer) handleSgmiRaw(w http.ResponseWriter, r *http.Request) {
	if s.cfg.SGMIEndpoint != "" {
		if payload, err := s.fetchAPIMap(r.Context(), s.cfg.SGMIEndpoint); err == nil {
			respondJSON(w, payload)
			return
		} else {
			log.Printf("shell: sgmi API fallback (%s): %v", s.cfg.SGMIEndpoint, err)
		}
	}

	raw, err := s.loadSgmiRaw()
	if err != nil {
		http.Error(w, fmt.Sprintf("sgmi dataset error: %v", err), http.StatusInternalServerError)
		return
	}
	respondJSON(w, raw)
}

type localAIModelEntry struct {
	ID     string `json:"id"`
	Readme bool   `json:"readme"`
}

type localAIInventory struct {
	GeneratedAt string              `json:"generatedAt"`
	Models      []localAIModelEntry `json:"models"`
}

func (s *shellServer) buildLocalAIInventory(ctx context.Context) (localAIInventory, error) {
	readmes := s.scanModelReadmes()

	var remoteIDs []string
	if ids, err := s.fetchRemoteModelIDs(ctx); err != nil {
		log.Printf("shell: LocalAI service lookup failed, using filesystem inventory: %v", err)
	} else {
		remoteIDs = ids
	}

	idSet := make(map[string]struct{})
	for _, id := range remoteIDs {
		if id == "" {
			continue
		}
		idSet[id] = struct{}{}
	}
	for id := range readmes {
		idSet[id] = struct{}{}
	}

	if len(idSet) == 0 {
		return localAIInventory{}, errors.New("no models discovered in LocalAI service or models directory")
	}

	ids := make([]string, 0, len(idSet))
	for id := range idSet {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	models := make([]localAIModelEntry, 0, len(ids))
	for _, id := range ids {
		models = append(models, localAIModelEntry{
			ID:     id,
			Readme: readmes[id],
		})
	}

	return localAIInventory{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Models:      models,
	}, nil
}

func (s *shellServer) fetchRemoteModelIDs(ctx context.Context) ([]string, error) {
	if s.cfg.LocalAIBaseURL == "" {
		return nil, errors.New("LocalAI base URL not configured")
	}

	endpoint := s.cfg.LocalAIBaseURL + "/v1/models"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("LocalAI responded with %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}

	var ids []string
	for _, entry := range payload.Data {
		if entry.ID != "" {
			ids = append(ids, entry.ID)
		}
	}

	if len(ids) == 0 {
		return nil, errors.New("LocalAI returned no models")
	}
	return ids, nil
}

func (s *shellServer) scanModelReadmes() map[string]bool {
	results := make(map[string]bool)
	if s.cfg.ModelsDir == "" {
		return results
	}

	dirEntries, err := os.ReadDir(s.cfg.ModelsDir)
	if err != nil {
		log.Printf("shell: unable to read models dir %s: %v", s.cfg.ModelsDir, err)
		return results
	}

	for _, entry := range dirEntries {
		if !entry.IsDir() {
			continue
		}
		modelID := entry.Name()
		readmePath := filepath.Join(s.cfg.ModelsDir, modelID, "README.md")
		if _, err := os.Stat(readmePath); err == nil {
			results[modelID] = true
		} else {
			results[modelID] = false
		}
	}
	return results
}

type trainingFeature struct {
	Type string         `json:"type"`
	Data map[string]int `json:"data"`
}

type trainingDataset struct {
	Features            []trainingFeature `json:"features"`
	NodeCount           int               `json:"node_count"`
	EdgeCount           int               `json:"edge_count"`
	HasHistoricalData   bool              `json:"has_historical_data"`
	HasLearnedPatterns  bool              `json:"has_learned_patterns"`
	HasTemporalPatterns bool              `json:"has_temporal_patterns"`
	DomainFiltered      bool              `json:"domain_filtered"`
	PrivacyApplied      bool              `json:"privacy_applied"`
}

type trainingMetadata struct {
	GeneratedAt       string `json:"generated_at"`
	FeatureCount      int    `json:"feature_count"`
	NodeCount         int    `json:"node_count"`
	EdgeCount         int    `json:"edge_count"`
	HasHistoricalData bool   `json:"has_historical_data"`
	Folder            string `json:"folder"`
}

type trainingDatasetResponse struct {
	Metadata trainingMetadata `json:"metadata"`
	Dataset  trainingDataset  `json:"dataset"`
}

type sgmiSummary struct {
	JobCounts     map[string]int
	HostCounts    map[string]int
	CommandCounts map[string]int
	TriggerCount  int
	WaitCount     int
	CleanupCount  int
	JobsTotal     int
	HasTemporal   bool
	HasHistorical bool
}

func (s *shellServer) buildTrainingDataset() (trainingDatasetResponse, error) {
	raw, err := s.loadSgmiRaw()
	if err != nil {
		return trainingDatasetResponse{}, err
	}

	folderName, summary, err := summariseSgmi(raw)
	if err != nil {
		return trainingDatasetResponse{}, err
	}

	features := []trainingFeature{
		{
			Type: "job_type_distribution",
			Data: summary.JobCounts,
		},
		{
			Type: "event_dependency_counts",
			Data: map[string]int{
				"triggers":  summary.TriggerCount,
				"waits_for": summary.WaitCount,
				"cleans_up": summary.CleanupCount,
			},
		},
		{
			Type: "top_hosts",
			Data: topN(summary.HostCounts, 10),
		},
		{
			Type: "top_commands",
			Data: topN(summary.CommandCounts, 10),
		},
	}

	dataset := trainingDataset{
		Features:            features,
		NodeCount:           summary.JobsTotal,
		EdgeCount:           summary.TriggerCount + summary.WaitCount,
		HasHistoricalData:   summary.HasHistorical,
		HasLearnedPatterns:  true,
		HasTemporalPatterns: summary.HasTemporal,
		DomainFiltered:      false,
		PrivacyApplied:      false,
	}

	metadata := trainingMetadata{
		GeneratedAt:       time.Now().UTC().Format(time.RFC3339),
		FeatureCount:      len(features),
		NodeCount:         dataset.NodeCount,
		EdgeCount:         dataset.EdgeCount,
		HasHistoricalData: dataset.HasHistoricalData,
		Folder:            folderName,
	}

	return trainingDatasetResponse{Metadata: metadata, Dataset: dataset}, nil
}

func (s *shellServer) loadSgmiRaw() (map[string]any, error) {
	if s.cfg.SGMIJSONPath == "" {
		return nil, errors.New("SGMI JSON path not configured")
	}

	data, err := os.ReadFile(s.cfg.SGMIJSONPath)
	if err != nil {
		return nil, fmt.Errorf("read SGMI dataset (%s): %w", s.cfg.SGMIJSONPath, err)
	}

	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, fmt.Errorf("parse SGMI dataset: %w", err)
	}

	if len(payload) == 0 {
		return nil, errors.New("SGMI dataset is empty")
	}
	return payload, nil
}

func summariseSgmi(raw map[string]any) (string, sgmiSummary, error) {
	for folderName, anyFolder := range raw {
		folder, ok := anyFolder.(map[string]any)
		if !ok {
			continue
		}

		summary := sgmiSummary{
			JobCounts:     make(map[string]int),
			HostCounts:    make(map[string]int),
			CommandCounts: make(map[string]int),
			HasHistorical: containsHistoricalMarkers(folder),
		}

		for _, candidate := range folder {
			item, ok := candidate.(map[string]any)
			if !ok {
				continue
			}

			jobType, _ := item["Type"].(string)
			if !strings.HasPrefix(jobType, "Job") {
				continue
			}

			summary.JobsTotal++
			summary.JobCounts[jobType]++

			if host, ok := item["Host"].(string); ok && host != "" {
				summary.HostCounts[host]++
			}

			if command, ok := item["Command"].(string); ok && command != "" {
				if first := firstToken(command); first != "" {
					summary.CommandCounts[first]++
				}
			}

			summary.TriggerCount += len(extractEvents(item["eventsToAdd"]))
			summary.WaitCount += len(extractEvents(item["eventsToWaitFor"]))
			summary.CleanupCount += len(extractEvents(item["eventsToDelete"]))

			if when, ok := item["When"].(map[string]any); ok && len(when) > 0 {
				summary.HasTemporal = true
			}
		}

		if summary.JobsTotal == 0 {
			return folderName, summary, errors.New("no jobs discovered in SGMI folder")
		}
		return folderName, summary, nil
	}

	return "", sgmiSummary{}, errors.New("SGMI payload contained no folder entries")
}

func containsHistoricalMarkers(folder map[string]any) bool {
	for key, value := range folder {
		lowerKey := strings.ToLower(key)
		if !strings.Contains(lowerKey, "change") && !strings.Contains(lowerKey, "history") {
			continue
		}

		switch v := value.(type) {
		case []any:
			if len(v) > 0 {
				return true
			}
		case map[string]any:
			if len(v) > 0 {
				return true
			}
		case string:
			if strings.TrimSpace(v) != "" {
				return true
			}
		}
	}
	return false
}

func extractEvents(section any) []string {
	container, ok := section.(map[string]any)
	if !ok {
		return nil
	}

	rawEvents, ok := container["Events"].([]any)
	if !ok {
		return nil
	}

	events := make([]string, 0, len(rawEvents))
	for _, raw := range rawEvents {
		eventObj, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if event, ok := eventObj["Event"].(string); ok && event != "" {
			events = append(events, event)
		}
	}
	return events
}

func topN(counts map[string]int, n int) map[string]int {
	if len(counts) == 0 || n <= 0 {
		return map[string]int{}
	}

	type kv struct {
		Key   string
		Value int
	}

	pairs := make([]kv, 0, len(counts))
	for key, value := range counts {
		if value <= 0 {
			continue
		}
		pairs = append(pairs, kv{Key: key, Value: value})
	}

	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].Value == pairs[j].Value {
			return pairs[i].Key < pairs[j].Key
		}
		return pairs[i].Value > pairs[j].Value
	})

	if len(pairs) > n {
		pairs = pairs[:n]
	}

	result := make(map[string]int, len(pairs))
	for _, pair := range pairs {
		result[pair.Key] = pair.Value
	}
	return result
}

func firstToken(command string) string {
	fields := strings.Fields(command)
	if len(fields) == 0 {
		return ""
	}
	return fields[0]
}

func serveIndex(w http.ResponseWriter, distFS fs.FS) {
	data, err := fs.ReadFile(distFS, "index.html")
	if err != nil {
		http.Error(w, "index.html not found in embedded dist", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

func fileExists(distFS fs.FS, name string) bool {
	file, err := distFS.Open(name)
	if err != nil {
		return false
	}
	file.Close()
	return true
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}

func respondJSON(w http.ResponseWriter, payload any) {
	data, err := json.Marshal(payload)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

func (s *shellServer) fetchAPIMap(ctx context.Context, endpoint string) (map[string]any, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("%s returned %d: %s", endpoint, resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var payload map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}
	return payload, nil
}

func (s *shellServer) proxyHandler(target string, prefix string) http.Handler {
	normalizedPrefix := normalizeProxyPrefix(prefix)
	if target == "" {
		message := "proxy not configured"
		if normalizedPrefix != "" {
			message = fmt.Sprintf("%s proxy not configured", normalizedPrefix)
		}
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, message, http.StatusBadGateway)
		})
	}

	upstream, err := url.Parse(target)
	if err != nil {
		log.Printf("shell: invalid proxy target %q for prefix %s: %v", target, normalizedPrefix, err)
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "proxy target misconfigured", http.StatusBadGateway)
		})
	}

	proxy := httputil.NewSingleHostReverseProxy(upstream)

	originalDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		if normalizedPrefix != "" {
			req.URL.Path = trimProxyPath(req.URL.Path, normalizedPrefix)
			if req.URL.RawPath != "" {
				req.URL.RawPath = trimProxyPath(req.URL.RawPath, normalizedPrefix)
			} else {
				req.URL.RawPath = req.URL.Path
			}
		}
		originalDirector(req)
	}

	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("shell proxy error [%s -> %s]: %v", normalizedPrefix, target, err)
		http.Error(w, "upstream service unavailable", http.StatusBadGateway)
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxy.ServeHTTP(w, r)
	})
}

func normalizeProxyPrefix(prefix string) string {
	prefix = strings.TrimSpace(prefix)
	if prefix == "" || prefix == "/" {
		return ""
	}
	prefix = "/" + strings.Trim(prefix, "/")
	return prefix
}

func trimProxyPath(pathValue string, prefix string) string {
	if strings.HasPrefix(pathValue, prefix) {
		pathValue = strings.TrimPrefix(pathValue, prefix)
	}
	if pathValue == "" {
		return "/"
	}
	if !strings.HasPrefix(pathValue, "/") {
		pathValue = "/" + pathValue
	}
	return pathValue
}
