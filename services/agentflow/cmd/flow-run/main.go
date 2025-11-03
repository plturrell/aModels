package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentFlow/internal/catalog/flightcatalog"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentFlow/internal/langflow"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentFlow/pkg/catalog"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentFlow/pkg/telemetry"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentFlow/runner"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentFlow/store/registry"
	catalogprompt "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK/pkg/flightcatalog/prompt"

	"github.com/joho/godotenv"
)

const catalogContextKey = "agent_catalog_context"

type config struct {
	baseURL            string
	apiKey             string
	authToken          string
	flowsDir           string
	flowID             string
	flowFile           string
	projectID          string
	folderPath         string
	inputValue         string
	sessionID          string
	payload            string
	payloadFile        string
	registry           string
	probe              bool
	printReq           bool
	printResp          bool
	ensure             bool
	force              bool
	list               bool
	timeout            time.Duration
	telemetryOn        bool
	telemetryAddr      string
	telemetryPrivacy   string
	telemetryUser      string
	agentSDKFlightAddr string
}

func main() {
	loadEnvFiles()
	cfg := parseFlags()
	ctx, cancel := context.WithTimeout(context.Background(), cfg.timeout)
	defer cancel()

	clientOpts := []langflow.Option{
		langflow.WithHTTPClient(&http.Client{Timeout: cfg.timeout}),
		langflow.WithUserAgent("agentflow-cli/0.1"),
	}
	if cfg.apiKey != "" {
		clientOpts = append(clientOpts, langflow.WithAPIKey(cfg.apiKey))
	}
	if cfg.authToken != "" {
		clientOpts = append(clientOpts, langflow.WithAuthToken(cfg.authToken))
	}

	client, err := langflow.NewClient(cfg.baseURL, clientOpts...)
	if err != nil {
		fatal(err)
	}

	var telemetryClient *telemetry.Client
	if cfg.telemetryOn && strings.TrimSpace(cfg.telemetryAddr) != "" {
		dialCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		tc, telErr := telemetry.Dial(dialCtx, telemetry.Config{
			Address:      cfg.telemetryAddr,
			PrivacyLevel: cfg.telemetryPrivacy,
			UserIDHash:   cfg.telemetryUser,
		})
		cancel()
		if telErr != nil {
			fmt.Fprintf(os.Stderr, "Telemetry disabled: %v\n", telErr)
		} else {
			telemetryClient = tc
			defer telemetryClient.Close()
		}
	}

	if cfg.list {
		if err := listRemoteFlows(ctx, client); err != nil {
			fatal(err)
		}
		return
	}

	if cfg.probe {
		if err := checkConnectivity(ctx, client); err != nil {
			fatal(err)
		}
	}

	loader := catalog.NewLoader(cfg.flowsDir)
	localFlowID, spec, err := resolveFlowSpec(cfg, loader)
	if err != nil {
		fatal(err)
	}

	runReq, err := buildRunRequest(cfg)
	if err != nil {
		fatal(err)
	}

	if cfg.printReq {
		fmt.Fprintln(os.Stderr, "Run request payload:")
		if err := printAsJSON(runReq); err != nil {
			fatal(err)
		}
		return
	}
	var (
		agentCatalog    *flightcatalog.Catalog
		catalogEnriched *catalogprompt.Enrichment
	)
	if strings.TrimSpace(cfg.agentSDKFlightAddr) != "" {
		cat, err := flightcatalog.Fetch(ctx, cfg.agentSDKFlightAddr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to fetch agent catalog from %s: %v\n", cfg.agentSDKFlightAddr, err)
		} else {
			agentCatalog = &cat
			enrichment := catalogprompt.Enrich(catalogprompt.Catalog{
				Suites: cat.Suites,
				Tools:  cat.Tools,
			})
			catalogEnriched = &enrichment
			if runReq.Inputs == nil {
				runReq.Inputs = make(map[string]any)
			}
			runReq.Inputs["agent_catalog"] = cat
			runReq.Inputs["agent_tools"] = cat.Tools
			if enrichment.Summary != "" {
				runReq.Inputs["agent_catalog_summary"] = enrichment.Summary
			}
			if enrichment.Stats.SuiteCount > 0 || enrichment.Stats.UniqueToolCount > 0 {
				runReq.Inputs["agent_catalog_stats"] = enrichment.Stats
			}
			if len(enrichment.Implementations) > 0 {
				runReq.Inputs["agent_catalog_matrix"] = enrichment.Implementations
			}
			if len(enrichment.UniqueTools) > 0 {
				runReq.Inputs["agent_catalog_unique_tools"] = enrichment.UniqueTools
			}
			if len(enrichment.StandaloneTools) > 0 {
				runReq.Inputs["agent_catalog_tool_details"] = enrichment.StandaloneTools
			}
			if enrichment.Prompt != "" {
				runReq.Inputs[catalogContextKey] = enrichment.Prompt
			}
		}
	}

	r := runner.New(client, loader)
	syncOpts := runner.SyncOptions{
		Force:        cfg.force,
		ProjectID:    cfg.projectID,
		FolderPath:   cfg.folderPath,
		AgentCatalog: agentCatalog,
	}

	regPath := cfg.registry
	if strings.TrimSpace(regPath) == "" {
		regPath = registry.MappingPath
	}
	reg, err := registry.Load(regPath)
	if err != nil {
		fatal(err)
	}

	runFlowID := localFlowID
	if remote := reg.Resolve(localFlowID); remote != "" {
		syncOpts.RemoteID = remote
		syncOpts.RemoteIDs = map[string]string{localFlowID: remote}
		if !cfg.ensure {
			runFlowID = remote
		}
	}

	if cfg.ensure {
		if spec == nil {
			loaded, err := loader.Load(localFlowID)
			if err != nil {
				fatal(err)
			}
			spec = &loaded
		}
		record, err := r.SyncFlow(ctx, localFlowID, syncOpts)
		if err != nil {
			fatal(err)
		}
		if record.ID != "" {
			runFlowID = record.ID
			reg.Set(localFlowID, record.ID)
			if err := reg.Save(); err != nil {
				fatal(err)
			}
		}
		fmt.Fprintf(os.Stderr, "Synced flow %s (%s)\n", record.ID, record.Name)
	} else if runFlowID == localFlowID && spec != nil {
		// Surface a hint when the flow likely needs syncing.
		if reg.Resolve(localFlowID) == "" {
			fmt.Fprintf(os.Stderr, "Warning: no recorded Langflow ID for %s; run may fail unless the flow already exists remotely.\n", localFlowID)
		}
	}

	startTime := time.Now()
	result, err := client.RunFlow(ctx, runFlowID, runReq)
	latency := time.Since(startTime)

	if telemetryClient != nil {
		output := map[string]any{}
		if err == nil && result.Raw != nil {
			output = result.Raw
		}

		metadata := map[string]any{
			"force_sync":    cfg.force,
			"ensure_sync":   cfg.ensure,
			"project_id":    cfg.projectID,
			"folder_path":   cfg.folderPath,
			"registry_path": regPath,
		}
		if agentCatalog != nil {
			metadata["agent_catalog"] = agentCatalog.Suites
			metadata["agent_tools"] = agentCatalog.Tools
			if catalogEnriched != nil {
				if catalogEnriched.Summary != "" {
					metadata["agent_catalog_summary"] = catalogEnriched.Summary
				}
				if catalogEnriched.Stats.SuiteCount > 0 || catalogEnriched.Stats.UniqueToolCount > 0 {
					metadata["agent_catalog_stats"] = catalogEnriched.Stats
				}
				if len(catalogEnriched.Implementations) > 0 {
					metadata["agent_catalog_matrix"] = catalogEnriched.Implementations
				}
				if len(catalogEnriched.UniqueTools) > 0 {
					metadata["agent_catalog_unique_tools"] = catalogEnriched.UniqueTools
				}
				if len(catalogEnriched.StandaloneTools) > 0 {
					metadata["agent_catalog_tool_details"] = catalogEnriched.StandaloneTools
				}
				if catalogEnriched.Prompt != "" {
					metadata[catalogContextKey] = catalogEnriched.Prompt
				}
			} else if ctxText := buildCatalogContext(*agentCatalog); ctxText != "" {
				metadata[catalogContextKey] = ctxText
			}
		}
		if spec != nil {
			if spec.Name != "" {
				metadata["flow_name"] = spec.Name
			}
			if spec.Category != "" {
				metadata["flow_category"] = spec.Category
			}
			if spec.Description != "" {
				metadata["flow_description"] = spec.Description
			}
			if len(spec.Tags) > 0 {
				tags := make([]any, 0, len(spec.Tags))
				for _, tag := range spec.Tags {
					tags = append(tags, tag)
				}
				metadata["flow_tags"] = tags
			}
		}
		if syncOpts.RemoteID != "" {
			metadata["sync_remote_id"] = syncOpts.RemoteID
		}
		if len(syncOpts.RemoteIDs) > 0 {
			ids := make(map[string]any, len(syncOpts.RemoteIDs))
			for k, v := range syncOpts.RemoteIDs {
				ids[k] = v
			}
			metadata["sync_remote_ids"] = ids
		}
		if localFlowID != runFlowID {
			metadata["executed_remote_id"] = runFlowID
		}
		if cfg.timeout > 0 {
			metadata["timeout"] = cfg.timeout.String()
		}

		logErr := telemetryClient.LogFlowRun(ctx, telemetry.FlowRunRecord{
			FlowID:       runFlowID,
			LocalFlowID:  localFlowID,
			SessionID:    runReq.SessionID,
			InputValue:   runReq.InputValue,
			Inputs:       runReq.Inputs,
			Tweaks:       runReq.Tweaks,
			Stream:       runReq.Stream,
			Metadata:     metadata,
			Result:       output,
			Error:        err,
			Latency:      latency,
			StartedAt:    startTime,
			CompletedAt:  startTime.Add(latency),
			PrivacyLevel: cfg.telemetryPrivacy,
			UserIDHash:   cfg.telemetryUser,
		})
		if logErr != nil {
			fmt.Fprintf(os.Stderr, "Telemetry warning: %v\n", logErr)
		}
	}

	if err != nil {
		fatal(err)
	}

	if cfg.printResp {
		fmt.Fprintln(os.Stderr, "Langflow response payload:")
		if err := printAsJSON(result.Raw); err != nil {
			fatal(err)
		}
		return
	}

	if err := printAsJSON(result.Raw); err != nil {
		fatal(err)
	}
}

func parseFlags() config {
	cfg := config{}
	flag.StringVar(&cfg.baseURL, "base-url", envOr("LANGFLOW_URL", "http://localhost:7860"), "Langflow base URL")
	flag.StringVar(&cfg.apiKey, "api-key", os.Getenv("LANGFLOW_API_KEY"), "Langflow API key")
	flag.StringVar(&cfg.authToken, "auth-token", os.Getenv("LANGFLOW_AUTH_TOKEN"), "Bearer token for Langflow API")
	flag.StringVar(&cfg.flowsDir, "flows-dir", envOr("LANGFLOW_FLOWS_DIR", "flows"), "Directory containing flow JSON files")
	flag.StringVar(&cfg.flowID, "flow-id", "", "Identifier of the flow to run")
	flag.StringVar(&cfg.flowFile, "flow-file", "", "Path to a flow JSON file (relative to flows-dir if not absolute)")
	flag.StringVar(&cfg.projectID, "project-id", "", "Optional target project identifier when importing flows")
	flag.StringVar(&cfg.folderPath, "folder-path", "", "Optional folder path when importing flows")
	flag.StringVar(&cfg.inputValue, "input", "", "Input text for flows expecting a single input node")
	flag.StringVar(&cfg.sessionID, "session", "", "Optional session identifier")
	flag.StringVar(&cfg.payload, "payload", "", "JSON object containing named inputs for the flow")
	flag.StringVar(&cfg.payloadFile, "payload-file", "", "Path to JSON file containing named inputs for the flow")
	flag.StringVar(&cfg.registry, "registry", envOr("LANGFLOW_REGISTRY_PATH", registry.MappingPath), "Path to Langflow flow ID registry file")
	flag.BoolVar(&cfg.ensure, "ensure", envOrBool("LANGFLOW_ENSURE_SYNC", true), "Synchronise the local flow definition before running")
	flag.BoolVar(&cfg.force, "force", envOrBool("LANGFLOW_FORCE_IMPORT", true), "Force overwrite when synchronising flows")
	flag.BoolVar(&cfg.list, "list", false, "List flows available on the Langflow server")
	flag.BoolVar(&cfg.probe, "probe", envOrBool("LANGFLOW_PROBE", false), "Check Langflow connectivity (version endpoint) before running")
	flag.BoolVar(&cfg.printReq, "print-request", envOrBool("LANGFLOW_PRINT_REQUEST", false), "Print the prepared request payload and exit")
	flag.BoolVar(&cfg.printResp, "print-response", envOrBool("LANGFLOW_PRINT_RESPONSE", false), "Print the Langflow response payload to stderr")
	flag.DurationVar(&cfg.timeout, "timeout", envOrDuration("LANGFLOW_TIMEOUT", 2*time.Minute), "Timeout for individual API calls")
	flag.BoolVar(&cfg.telemetryOn, "telemetry-enable", envOrBool("POSTGRES_LANG_SERVICE_ENABLED", true), "Enable PostgresLangService telemetry logging")
	flag.StringVar(&cfg.telemetryAddr, "telemetry-addr", envOr("POSTGRES_LANG_SERVICE_ADDR", ""), "Address of the PostgresLangService gRPC endpoint (host:port)")
	flag.StringVar(&cfg.telemetryPrivacy, "telemetry-privacy", envOr("POSTGRES_LANG_SERVICE_PRIVACY", "medium"), "Privacy level recorded with telemetry entries")
	flag.StringVar(&cfg.telemetryUser, "telemetry-user", os.Getenv("POSTGRES_LANG_SERVICE_USER_ID"), "Optional hashed user identifier for telemetry records")
	flag.StringVar(&cfg.agentSDKFlightAddr, "agent-sdk-flight", envOr("AGENTSDK_FLIGHT_ADDR", ""), "Agent SDK Flight address (host:port) to enrich flows with service catalogs")
	flag.Parse()
	return cfg
}

func resolveFlowSpec(cfg config, loader *catalog.Loader) (string, *catalog.Spec, error) {
	var runID string
	var spec *catalog.Spec
	if cfg.flowFile != "" {
		path := cfg.flowFile
		if !filepath.IsAbs(path) {
			path = filepath.Join(cfg.flowsDir, path)
		}
		path = filepath.Clean(path)

		specs, err := loader.List()
		if err != nil {
			return "", nil, err
		}
		for _, candidate := range specs {
			if filepath.Clean(candidate.Path) == path {
				runID = candidate.ID
				c := candidate
				spec = &c
				break
			}
		}
		if spec == nil {
			return "", nil, fmt.Errorf("flow file %s not found in catalog %s", path, loaderRoot(loader))
		}
	}

	if cfg.flowID != "" {
		runID = cfg.flowID
		if spec == nil {
			loaded, err := loader.Load(cfg.flowID)
			if err == nil {
				spec = &loaded
			}
		}
	}

	if runID == "" {
		return "", nil, errors.New("flow-id or flow-file must be provided")
	}

	return runID, spec, nil
}

func loaderRoot(loader *catalog.Loader) string {
	if loader == nil {
		return ""
	}
	return loader.Root()
}

func loadEnvFiles() {
	candidates := []string{}
	if custom := strings.TrimSpace(os.Getenv("AGENTFLOW_ENV_FILE")); custom != "" {
		candidates = append(candidates, custom)
	}
	candidates = append(candidates,
		".env.agentflow",
		".env",
		filepath.Join("..", ".env.agentflow"),
		filepath.Join("..", ".env"),
		filepath.Join("..", "..", ".env.agentflow"),
		filepath.Join("..", "..", ".env"),
	)

	cwd, err := os.Getwd()
	if err == nil {
		candidates = append(candidates,
			filepath.Join(cwd, ".env.agentflow"),
			filepath.Join(cwd, ".env"),
		)
	}

	loaded := map[string]struct{}{}
	for _, candidate := range candidates {
		candidate = filepath.Clean(candidate)
		if candidate == "." || candidate == "" {
			continue
		}
		if _, seen := loaded[candidate]; seen {
			continue
		}
		info, err := os.Stat(candidate)
		if err != nil {
			if !errors.Is(err, os.ErrNotExist) {
				log.Printf("Warning: unable to stat env file %s: %v", candidate, err)
			}
			continue
		}
		if info.IsDir() {
			continue
		}
		if err := godotenv.Load(candidate); err != nil {
			log.Printf("Warning: failed to load env file %s: %v", candidate, err)
			continue
		}
		log.Printf("Loaded environment from %s", candidate)
		loaded[candidate] = struct{}{}
	}
}

func buildRunRequest(cfg config) (langflow.RunFlowRequest, error) {
	request := langflow.RunFlowRequest{
		InputValue: cfg.inputValue,
		SessionID:  cfg.sessionID,
	}

	payload := map[string]any{}

	if strings.TrimSpace(cfg.payload) != "" {
		if err := json.Unmarshal([]byte(cfg.payload), &payload); err != nil {
			return request, fmt.Errorf("decode payload flag: %w", err)
		}
	}

	if cfg.payloadFile != "" {
		data, err := os.ReadFile(cfg.payloadFile)
		if err != nil {
			return request, fmt.Errorf("read payload file: %w", err)
		}
		filePayload := map[string]any{}
		if err := json.Unmarshal(data, &filePayload); err != nil {
			return request, fmt.Errorf("decode payload file: %w", err)
		}
		for k, v := range filePayload {
			payload[k] = v
		}
	}

	if len(payload) > 0 {
		request.Inputs = payload
	}
	return request, nil
}

func listRemoteFlows(ctx context.Context, client *langflow.Client) error {
	flows, err := client.ListFlows(ctx)
	if err != nil {
		return err
	}
	if len(flows) == 0 {
		fmt.Println("No flows found.")
		return nil
	}
	for _, flow := range flows {
		fmt.Printf("%s\t%s\n", flow.ID, flow.Name)
	}
	return nil
}

func printAsJSON(payload any) error {
	if payload == nil {
		fmt.Println("null")
		return nil
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(data))
	return nil
}

func fatal(err error) {
	var apiErr *langflow.APIError
	if errors.As(err, &apiErr) {
		fmt.Fprintf(os.Stderr, "Langflow API error (status %d): %s\n", apiErr.StatusCode, apiErr.Error())
	} else {
		fmt.Fprintf(os.Stderr, "%s\n", err)
	}
	os.Exit(1)
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envOrBool(key string, fallback bool) bool {
	if value := os.Getenv(key); value != "" {
		switch strings.ToLower(value) {
		case "1", "true", "yes", "y":
			return true
		case "0", "false", "no", "n":
			return false
		}
	}
	return fallback
}

func envOrDuration(key string, fallback time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if parsed, err := time.ParseDuration(value); err == nil {
			return parsed
		}
	}
	return fallback
}

func buildCatalogContext(cat flightcatalog.Catalog) string {
	return catalogprompt.BuildContext(cat.Suites, cat.Tools)
}

func checkConnectivity(ctx context.Context, client *langflow.Client) error {
	version, err := client.Version(ctx)
	if err != nil {
		return fmt.Errorf("ping Langflow: %w", err)
	}
	fmt.Fprintf(os.Stderr, "Langflow version: %s\n", version)
	return nil
}
