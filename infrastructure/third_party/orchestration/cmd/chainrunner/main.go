package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	catalogprompt "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK/pkg/flightcatalog/prompt"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/catalog/flightcatalog"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"
)

const catalogContextKey = "agent_catalog_context"

type request struct {
	Prompt      string         `json:"prompt"`
	Chain       string         `json:"chain"`
	Model       string         `json:"model"`
	Temperature *float64       `json:"temperature"`
	MaxTokens   *int           `json:"max_tokens"`
	Extra       map[string]any `json:"extra"`
}

type response struct {
	Result    string         `json:"result"`
	LatencyMS float64        `json:"latency_ms"`
	Meta      map[string]any `json:"meta"`
}

type errorResponse struct {
	Error string `json:"error"`
}

func main() {
	if err := run(); err != nil {
		_ = json.NewEncoder(os.Stdout).Encode(errorResponse{Error: err.Error()})
		os.Exit(1)
	}
}

func run() error {
	var req request
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		if errors.Is(err, io.EOF) {
			return fmt.Errorf("empty request body")
		}
		return fmt.Errorf("decode request: %w", err)
	}
	if strings.TrimSpace(req.Prompt) == "" {
		return fmt.Errorf("prompt must not be empty")
	}

	ctx := context.Background()
	llm, meta := buildLLM(req)
	if addr := strings.TrimSpace(os.Getenv("AGENTSDK_FLIGHT_ADDR")); addr != "" {
		catalogCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		cat, err := flightcatalog.Fetch(catalogCtx, addr)
		cancel()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to fetch agent catalog from %s: %v\n", addr, err)
		} else {
			enrichment := catalogprompt.Enrich(catalogprompt.Catalog{
				Suites: cat.Suites,
				Tools:  cat.Tools,
			})
			contextText := enrichment.Prompt

			meta["agent_catalog"] = cat.Suites
			meta["agent_tools"] = cat.Tools
			if enrichment.Summary != "" {
				meta["agent_catalog_summary"] = enrichment.Summary
			}
			if enrichment.Stats.SuiteCount > 0 || enrichment.Stats.UniqueToolCount > 0 {
				meta["agent_catalog_stats"] = enrichment.Stats
			}
			if len(enrichment.Implementations) > 0 {
				meta["agent_catalog_matrix"] = enrichment.Implementations
			}
			if len(enrichment.UniqueTools) > 0 {
				meta["agent_catalog_unique_tools"] = enrichment.UniqueTools
			}
			if len(enrichment.StandaloneTools) > 0 {
				meta["agent_catalog_tool_details"] = enrichment.StandaloneTools
			}
			if contextText != "" {
				meta[catalogContextKey] = contextText
			}

			if req.Extra == nil {
				req.Extra = map[string]any{}
			}
			req.Extra["agent_catalog"] = cat
			req.Extra["agent_tools"] = cat.Tools
			if enrichment.Summary != "" {
				req.Extra["agent_catalog_summary"] = enrichment.Summary
			}
			if enrichment.Stats.SuiteCount > 0 || enrichment.Stats.UniqueToolCount > 0 {
				req.Extra["agent_catalog_stats"] = enrichment.Stats
			}
			if len(enrichment.Implementations) > 0 {
				req.Extra["agent_catalog_matrix"] = enrichment.Implementations
			}
			if len(enrichment.UniqueTools) > 0 {
				req.Extra["agent_catalog_unique_tools"] = enrichment.UniqueTools
			}
			if len(enrichment.StandaloneTools) > 0 {
				req.Extra["agent_catalog_tool_details"] = enrichment.StandaloneTools
			}
			if contextText != "" {
				req.Extra[catalogContextKey] = contextText
			}
		}
	}

	start := time.Now()
	var (
		result string
		err    error
	)

	switch strings.ToLower(req.Chain) {
	case "sequential_chain":
		result, err = runSequentialChain(ctx, llm, req)
	default:
		result, err = runLLMChain(ctx, llm, req)
	}
	if err != nil {
		return err
	}

	resp := response{
		Result:    result,
		LatencyMS: time.Since(start).Seconds() * 1000,
		Meta:      meta,
	}
	return json.NewEncoder(os.Stdout).Encode(resp)
}

func buildLLM(req request) (llms.Model, map[string]any) {
	meta := map[string]any{"provider": "simple"}
	useLocalAI := strings.TrimSpace(os.Getenv("LOCALAI_BASE_URL")) != "" || req.Model != ""
	if useLocalAI {
		opts := []localai.Option{}
		if base := strings.TrimSpace(os.Getenv("LOCALAI_BASE_URL")); base != "" {
			opts = append(opts, localai.WithBaseURL(base))
		}
		if req.Model != "" {
			opts = append(opts, localai.WithModel(req.Model))
		}
		if req.Temperature != nil {
			opts = append(opts, localai.WithTemperature(*req.Temperature))
		}
		if req.MaxTokens != nil {
			opts = append(opts, localai.WithMaxTokens(*req.MaxTokens))
		}
		if llm, err := localai.New(opts...); err == nil {
			meta["provider"] = "localai"
			return llm, meta
		}
	}
	return simpleLLM{}, meta
}

func runLLMChain(ctx context.Context, llm llms.Model, req request) (string, error) {
	tmpl := "You are an assistant. Respond to the following input.\n\n{{.input}}"
	if s, ok := req.Extra["prompt_template"].(string); ok && strings.TrimSpace(s) != "" {
		tmpl = s
	}
	prompt := prompts.NewPromptTemplate(tmpl, []string{"input"})
	chain := chains.NewLLMChain(llm, prompt)
	input := applyCatalogContext(req.Prompt, req.Extra)
	result, err := chains.Run(ctx, chain, input)
	if err != nil {
		return "", fmt.Errorf("run llm chain: %w", err)
	}
	return result, nil
}

func runSequentialChain(ctx context.Context, llm llms.Model, req request) (string, error) {
	summarisePrompt := prompts.NewPromptTemplate(
		"Provide a concise summary (<=3 bullets) for: {{.input}}",
		[]string{"input"},
	)
	expandPrompt := prompts.NewPromptTemplate(
		"Using the summary {{.summary}}, craft a detailed response.",
		[]string{"summary"},
	)

	first := chains.NewLLMChain(llm, summarisePrompt)
	first.OutputKey = "summary"
	second := chains.NewLLMChain(llm, expandPrompt)
	second.Prompt = expandPrompt
	second.OutputKey = "text"

	seq, err := chains.NewSimpleSequentialChain([]chains.Chain{first, second})
	if err != nil {
		return "", fmt.Errorf("build sequential chain: %w", err)
	}
	input := applyCatalogContext(req.Prompt, req.Extra)
	result, err := chains.Run(ctx, seq, input)
	if err != nil {
		return "", fmt.Errorf("run sequential chain: %w", err)
	}
	return result, nil
}

type simpleLLM struct{}

func (simpleLLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	_ = ctx
	cleaned := strings.TrimSpace(prompt)
	if cleaned == "" {
		return "No meaningful input provided.", nil
	}
	lines := strings.Split(cleaned, "\n")
	summary := strings.TrimSpace(lines[0])
	if len(summary) > 240 {
		summary = summary[:240] + "..."
	}
	return summary, nil
}

func (s simpleLLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	var builder strings.Builder
	for _, msg := range messages {
		for _, part := range msg.Parts {
			if text, ok := part.(llms.TextContent); ok {
				builder.WriteString(text.Text)
				builder.WriteByte('\n')
			}
		}
	}
	text, err := s.Call(ctx, builder.String(), options...)
	if err != nil {
		return nil, err
	}
	choice := &llms.ContentChoice{
		Content:    text,
		StopReason: "stop",
	}
	return &llms.ContentResponse{Choices: []*llms.ContentChoice{choice}}, nil
}

func applyCatalogContext(prompt string, extra map[string]any) string {
	if extra == nil {
		return prompt
	}
	if ctxText, ok := extra[catalogContextKey].(string); ok {
		trimmed := strings.TrimSpace(ctxText)
		if trimmed != "" {
			if strings.HasPrefix(prompt, trimmed) {
				return prompt
			}
			return trimmed + "\n\n" + prompt
		}
	}
	return prompt
}
