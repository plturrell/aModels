package runner

import (
	"context"
	"fmt"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentFlow/internal/catalog/flightcatalog"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentFlow/internal/langflow"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentFlow/pkg/catalog"
	catalogprompt "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK/pkg/flightcatalog/prompt"
)

// SyncOptions controls how local flows are synchronised with a Langflow instance.
type SyncOptions struct {
	Force        bool
	ProjectID    string
	FolderPath   string
	RemoteID     string
	RemoteIDs    map[string]string
	AgentCatalog *flightcatalog.Catalog
}

// Runner coordinates catalog synchronisation and remote execution.
type Runner struct {
	client  *langflow.Client
	catalog *catalog.Loader
}

// Loader exposes the underlying flow catalog loader.
func (r *Runner) Loader() *catalog.Loader {
	return r.catalog
}

// New constructs a new runner instance.
func New(client *langflow.Client, loader *catalog.Loader) *Runner {
	return &Runner{
		client:  client,
		catalog: loader,
	}
}

// SyncFlow imports or updates a single flow in Langflow.
func (r *Runner) SyncFlow(ctx context.Context, flowID string, opts SyncOptions) (langflow.FlowRecord, error) {
	if r.catalog == nil {
		return langflow.FlowRecord{}, fmt.Errorf("catalog loader not configured")
	}
	spec, err := r.catalog.Load(flowID)
	if err != nil {
		return langflow.FlowRecord{}, err
	}
	if opts.AgentCatalog != nil {
		enrichment := catalogEnrichment(opts.AgentCatalog)
		if enrichment.Prompt != "" {
			if err := spec.InjectMetadata("agent_catalog_context", enrichment.Prompt); err != nil {
				return langflow.FlowRecord{}, fmt.Errorf("inject agent catalog context: %w", err)
			}
		}
		if err := spec.InjectMetadata("agent_catalog", opts.AgentCatalog); err != nil {
			return langflow.FlowRecord{}, fmt.Errorf("inject agent catalog metadata: %w", err)
		}
		if err := spec.InjectMetadata("agent_tools", opts.AgentCatalog.Tools); err != nil {
			return langflow.FlowRecord{}, fmt.Errorf("inject agent tools metadata: %w", err)
		}
		if enrichment.Summary != "" {
			if err := spec.InjectMetadata("agent_catalog_summary", enrichment.Summary); err != nil {
				return langflow.FlowRecord{}, fmt.Errorf("inject agent catalog summary: %w", err)
			}
		}
		if enrichment.Stats.SuiteCount > 0 || enrichment.Stats.UniqueToolCount > 0 {
			if err := spec.InjectMetadata("agent_catalog_stats", enrichment.Stats); err != nil {
				return langflow.FlowRecord{}, fmt.Errorf("inject agent catalog stats: %w", err)
			}
		}
		if len(enrichment.Implementations) > 0 {
			if err := spec.InjectMetadata("agent_catalog_matrix", enrichment.Implementations); err != nil {
				return langflow.FlowRecord{}, fmt.Errorf("inject agent catalog matrix: %w", err)
			}
		}
		if len(enrichment.UniqueTools) > 0 {
			if err := spec.InjectMetadata("agent_catalog_unique_tools", enrichment.UniqueTools); err != nil {
				return langflow.FlowRecord{}, fmt.Errorf("inject agent catalog unique tools: %w", err)
			}
		}
		if len(enrichment.StandaloneTools) > 0 {
			if err := spec.InjectMetadata("agent_catalog_tool_details", enrichment.StandaloneTools); err != nil {
				return langflow.FlowRecord{}, fmt.Errorf("inject agent catalog tool details: %w", err)
			}
		}
	}
	remoteID := opts.RemoteID
	if remoteID == "" && opts.RemoteIDs != nil {
		remoteID = opts.RemoteIDs[flowID]
	}
	req := langflow.FlowImportRequest{
		Flow:       spec.Raw,
		Force:      opts.Force,
		ProjectID:  opts.ProjectID,
		FolderPath: opts.FolderPath,
		RemoteID:   remoteID,
	}
	return r.client.ImportFlow(ctx, req)
}

// SyncAll synchronises every flow in the catalog.
func (r *Runner) SyncAll(ctx context.Context, opts SyncOptions) ([]langflow.FlowRecord, error) {
	if r.catalog == nil {
		return nil, fmt.Errorf("catalog loader not configured")
	}
	specs, err := r.catalog.List()
	if err != nil {
		return nil, err
	}
	results := make([]langflow.FlowRecord, 0, len(specs))
	var enrichment catalogprompt.Enrichment
	if opts.AgentCatalog != nil {
		enrichment = catalogEnrichment(opts.AgentCatalog)
	}
	for _, spec := range specs {
		current := spec
		if opts.AgentCatalog != nil {
			if enrichment.Prompt != "" {
				if err := current.InjectMetadata("agent_catalog_context", enrichment.Prompt); err != nil {
					return nil, fmt.Errorf("inject agent catalog context for %s: %w", spec.ID, err)
				}
			}
			if err := current.InjectMetadata("agent_catalog", opts.AgentCatalog); err != nil {
				return nil, fmt.Errorf("inject agent catalog metadata for %s: %w", spec.ID, err)
			}
			if err := current.InjectMetadata("agent_tools", opts.AgentCatalog.Tools); err != nil {
				return nil, fmt.Errorf("inject agent tools metadata for %s: %w", spec.ID, err)
			}
			if enrichment.Summary != "" {
				if err := current.InjectMetadata("agent_catalog_summary", enrichment.Summary); err != nil {
					return nil, fmt.Errorf("inject agent catalog summary for %s: %w", spec.ID, err)
				}
			}
			if enrichment.Stats.SuiteCount > 0 || enrichment.Stats.UniqueToolCount > 0 {
				if err := current.InjectMetadata("agent_catalog_stats", enrichment.Stats); err != nil {
					return nil, fmt.Errorf("inject agent catalog stats for %s: %w", spec.ID, err)
				}
			}
			if len(enrichment.Implementations) > 0 {
				if err := current.InjectMetadata("agent_catalog_matrix", enrichment.Implementations); err != nil {
					return nil, fmt.Errorf("inject agent catalog matrix for %s: %w", spec.ID, err)
				}
			}
			if len(enrichment.UniqueTools) > 0 {
				if err := current.InjectMetadata("agent_catalog_unique_tools", enrichment.UniqueTools); err != nil {
					return nil, fmt.Errorf("inject agent catalog unique tools for %s: %w", spec.ID, err)
				}
			}
			if len(enrichment.StandaloneTools) > 0 {
				if err := current.InjectMetadata("agent_catalog_tool_details", enrichment.StandaloneTools); err != nil {
					return nil, fmt.Errorf("inject agent catalog tool details for %s: %w", spec.ID, err)
				}
			}
		}
		remoteID := ""
		if opts.RemoteIDs != nil {
			remoteID = opts.RemoteIDs[spec.ID]
		}
		req := langflow.FlowImportRequest{
			Flow:       current.Raw,
			Force:      opts.Force,
			ProjectID:  opts.ProjectID,
			FolderPath: opts.FolderPath,
			RemoteID:   remoteID,
		}
		record, err := r.client.ImportFlow(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("sync flow %s: %w", spec.ID, err)
		}
		results = append(results, record)
	}
	return results, nil
}

// Run executes a flow by ID, optionally ensuring the definition is up to date first.
func (r *Runner) Run(ctx context.Context, flowID string, request langflow.RunFlowRequest, ensure bool, opts SyncOptions) (langflow.RunFlowResult, error) {
	targetID := flowID
	if ensure {
		record, err := r.SyncFlow(ctx, flowID, opts)
		if err != nil {
			return langflow.RunFlowResult{}, err
		}
		if record.ID != "" {
			targetID = record.ID
		}
	}
	return r.client.RunFlow(ctx, targetID, request)
}

// RunRequest captures the minimal inputs required to execute a flow via Runner.
type RunRequest struct {
	FlowID      string
	Inputs      map[string]any
	InputValue  string
	SessionID   string
	ChatHistory []map[string]any
	Tweaks      map[string]any
	Stream      bool
	EnsureSync  bool
	SyncOptions SyncOptions
}

// RunResult normalises the Langflow response payload.
type RunResult struct {
	Raw map[string]any
}

// Execute runs a flow with the provided inputs and returns the raw Langflow payload.
func (r *Runner) Execute(ctx context.Context, req RunRequest) (RunResult, error) {
	payload := langflow.RunFlowRequest{
		InputValue:  req.InputValue,
		Inputs:      req.Inputs,
		ChatHistory: req.ChatHistory,
		SessionID:   req.SessionID,
		Tweaks:      req.Tweaks,
		Stream:      req.Stream,
	}
	result, err := r.Run(ctx, req.FlowID, payload, req.EnsureSync, req.SyncOptions)
	if err != nil {
		return RunResult{}, err
	}
	return RunResult{Raw: result.Raw}, nil
}

func catalogContextText(cat *flightcatalog.Catalog) string {
	return catalogprompt.BuildContext(cat.Suites, cat.Tools)
}

func catalogEnrichment(cat *flightcatalog.Catalog) catalogprompt.Enrichment {
	if cat == nil {
		return catalogprompt.Enrichment{}
	}
	return catalogprompt.Enrich(catalogprompt.Catalog{
		Suites: cat.Suites,
		Tools:  cat.Tools,
	})
}
