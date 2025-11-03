package localai

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"ai_benchmarks/internal/catalog/flightcatalog"
	catalogprompt "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK/pkg/flightcatalog/prompt"
)

// EnhancedInferenceEngine provides advanced inference capabilities with domain routing
type EnhancedInferenceEngine struct {
	Client        *Client
	DomainRouter  *DomainRouter
	ModelRegistry *ModelRegistry
	AgentCatalog  *flightcatalog.Catalog
	catalogView   catalogprompt.Enrichment
}

// DomainRouter routes requests to appropriate model domains
type DomainRouter struct {
	domains map[string]string // task type -> model domain
}

// ModelRegistry manages available models and their capabilities
type ModelRegistry struct {
	models map[string]*ModelCapabilities
}

// EngineOption configures the enhanced inference engine.
type EngineOption func(*EnhancedInferenceEngine)

// WithAgentCatalog enriches the engine with Agent SDK catalog data.
func WithAgentCatalog(cat *flightcatalog.Catalog) EngineOption {
	return func(engine *EnhancedInferenceEngine) {
		if cat == nil {
			return
		}
		engine.AgentCatalog = cat
		engine.catalogView = catalogprompt.Enrich(catalogprompt.Catalog{
			Suites: cat.Suites,
			Tools:  cat.Tools,
		})
		engine.DomainRouter.SyncFromCatalog(*cat)
		engine.ModelRegistry.SyncFromCatalog(*cat)
	}
}

// SyncFromCatalog updates router mappings based on the Agent SDK catalog.
func (dr *DomainRouter) SyncFromCatalog(cat flightcatalog.Catalog) {
	if dr == nil {
		return
	}
	if dr.domains == nil {
		dr.domains = map[string]string{}
	}
	for _, suite := range cat.Suites {
		target := strings.TrimSpace(suite.Implementation)
		if target == "" {
			target = suite.Name
		}
		key := strings.ToLower(suite.Name)
		dr.domains[key] = target
		for _, tool := range suite.ToolNames {
			dr.domains[strings.ToLower(tool)] = target
		}
	}
}

// SyncFromCatalog updates the registry with catalog-provided suites and tools.
func (mr *ModelRegistry) SyncFromCatalog(cat flightcatalog.Catalog) {
	if mr == nil {
		return
	}
	if mr.models == nil {
		mr.models = map[string]*ModelCapabilities{}
	}
	for _, suite := range cat.Suites {
		domain := strings.TrimSpace(suite.Implementation)
		if domain == "" {
			domain = "agent-suite"
		}
		supported := make([]string, 0, len(suite.ToolNames))
		for _, tool := range suite.ToolNames {
			supported = append(supported, strings.ToLower(tool))
		}
		mr.models[suite.Name] = &ModelCapabilities{
			Name:           suite.Name,
			Domain:         domain,
			MaxTokens:      2048,
			Temperature:    0.3,
			SupportedTasks: supported,
		}
	}
}

// ModelCapabilities describes what a model can do
type ModelCapabilities struct {
	Name           string            `json:"name"`
	Domain         string            `json:"domain"`
	MaxTokens      int               `json:"max_tokens"`
	Temperature    float64           `json:"temperature"`
	SupportedTasks []string          `json:"supported_tasks"`
	Performance    *ModelPerformance `json:"performance"`
}

// ModelPerformance tracks model performance metrics
type ModelPerformance struct {
	Accuracy    float64 `json:"accuracy"`
	Latency     float64 `json:"latency_ms"`
	Throughput  float64 `json:"throughput_tps"`
	MemoryUsage float64 `json:"memory_usage_mb"`
}

// BenchmarkResult represents the result of a benchmark inference
type BenchmarkResult struct {
	TaskType      string    `json:"task_type"`
	ModelName     string    `json:"model_name"`
	Question      string    `json:"question"`
	Answer        string    `json:"answer"`
	CorrectAnswer string    `json:"correct_answer"`
	IsCorrect     bool      `json:"is_correct"`
	Confidence    float64   `json:"confidence"`
	Latency       float64   `json:"latency_ms"`
	TokensUsed    int       `json:"tokens_used"`
	Timestamp     time.Time `json:"timestamp"`
}

// InferenceRequest represents a request for model inference
type InferenceRequest struct {
	TaskType    string                 `json:"task_type"`
	Question    string                 `json:"question"`
	Context     string                 `json:"context,omitempty"`
	Choices     []string               `json:"choices,omitempty"`
	Passage     string                 `json:"passage,omitempty"`
	Endings     []string               `json:"endings,omitempty"`
	Grid        [][]int                `json:"grid,omitempty"`
	ModelName   string                 `json:"model_name,omitempty"`
	Temperature float64                `json:"temperature,omitempty"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// InferenceResponse represents the response from model inference
type InferenceResponse struct {
	Answer     string                 `json:"answer"`
	Confidence float64                `json:"confidence"`
	ModelName  string                 `json:"model_name"`
	Latency    float64                `json:"latency_ms"`
	TokensUsed int                    `json:"tokens_used"`
	Metadata   map[string]interface{} `json:"metadata"`
}

// NewEnhancedInferenceEngine creates a new enhanced inference engine
func NewEnhancedInferenceEngine(client *Client, opts ...EngineOption) *EnhancedInferenceEngine {
	router := &DomainRouter{
		domains: map[string]string{
			"mcq":       "0x3579-VectorProcessingAgent",
			"boolq":     "0x9753-AIPreparationAgent",
			"hellaswag": "0xA1B2-DataProcessAgent",
			"arc":       "0xC3D4-StandardizationAgent",
			"trivia":    "0x5678-SQLAgent",
			"browser":   "0xBR0W-BrowserAnalysisAgent",
			"general":   "general",
		},
	}

	registry := &ModelRegistry{
		models: map[string]*ModelCapabilities{
			"0x3579-VectorProcessingAgent": {
				Name:           "Vector Processing Agent",
				Domain:         "vector",
				MaxTokens:      2000,
				Temperature:    0.3,
				SupportedTasks: []string{"mcq", "vector_ops", "similarity"},
				Performance: &ModelPerformance{
					Accuracy:    0.94,
					Latency:     150.0,
					Throughput:  10.0,
					MemoryUsage: 2048.0,
				},
			},
			"0x9753-AIPreparationAgent": {
				Name:           "AI Preparation Agent",
				Domain:         "preparation",
				MaxTokens:      2000,
				Temperature:    0.3,
				SupportedTasks: []string{"boolq", "preprocessing", "feature_engineering"},
				Performance: &ModelPerformance{
					Accuracy:    0.92,
					Latency:     120.0,
					Throughput:  12.0,
					MemoryUsage: 1536.0,
				},
			},
			"0xA1B2-DataProcessAgent": {
				Name:           "Data Process Agent",
				Domain:         "processing",
				MaxTokens:      2000,
				Temperature:    0.3,
				SupportedTasks: []string{"hellaswag", "etl", "pipeline"},
				Performance: &ModelPerformance{
					Accuracy:    0.89,
					Latency:     180.0,
					Throughput:  8.0,
					MemoryUsage: 2560.0,
				},
			},
			"0xC3D4-StandardizationAgent": {
				Name:           "Standardization Agent",
				Domain:         "standardization",
				MaxTokens:      2000,
				Temperature:    0.2,
				SupportedTasks: []string{"arc", "normalization", "validation"},
				Performance: &ModelPerformance{
					Accuracy:    0.87,
					Latency:     200.0,
					Throughput:  6.0,
					MemoryUsage: 3072.0,
				},
			},
			"0x5678-SQLAgent": {
				Name:           "SQL Agent",
				Domain:         "sql",
				MaxTokens:      2000,
				Temperature:    0.2,
				SupportedTasks: []string{"trivia", "sql_queries", "database_ops"},
				Performance: &ModelPerformance{
					Accuracy:    0.91,
					Latency:     100.0,
					Throughput:  15.0,
					MemoryUsage: 1024.0,
				},
			},
			"0xBR0W-BrowserAnalysisAgent": {
				Name:           "Browser Analysis Agent",
				Domain:         "browser",
				MaxTokens:      2000,
				Temperature:    0.3,
				SupportedTasks: []string{"browser", "web_analysis", "content_extraction"},
				Performance: &ModelPerformance{
					Accuracy:    0.93,
					Latency:     250.0,
					Throughput:  5.0,
					MemoryUsage: 4096.0,
				},
			},
			"general": {
				Name:           "General Assistant",
				Domain:         "general",
				MaxTokens:      1024,
				Temperature:    0.7,
				SupportedTasks: []string{"general", "conversation", "qa"},
				Performance: &ModelPerformance{
					Accuracy:    0.85,
					Latency:     300.0,
					Throughput:  3.0,
					MemoryUsage: 512.0,
				},
			},
		},
	}

	engine := &EnhancedInferenceEngine{
		Client:        client,
		DomainRouter:  router,
		ModelRegistry: registry,
	}
	for _, opt := range opts {
		opt(engine)
	}
	return engine
}

// Infer runs inference using the enhanced engine
func (eie *EnhancedInferenceEngine) Infer(ctx context.Context, req *InferenceRequest) (*InferenceResponse, error) {
	start := time.Now()

	// Determine model to use
	modelName := req.ModelName
	if modelName == "" {
		taskKey := strings.ToLower(req.TaskType)
		modelName = eie.DomainRouter.domains[taskKey]
		if modelName == "" {
			modelName = eie.DomainRouter.domains[req.TaskType]
		}
		if modelName == "" {
			modelName = "general"
		}
	}

	// Get model capabilities
	capabilities, exists := eie.ModelRegistry.models[modelName]
	if !exists {
		return nil, fmt.Errorf("model %s not found in registry", modelName)
	}

	// Validate task type is supported
	if !eie.isTaskSupported(capabilities, req.TaskType) {
		return nil, fmt.Errorf("task type %s not supported by model %s", req.TaskType, modelName)
	}

	// Create adapter for the specific task type
	adapter := NewBenchmarkAdapter(eie.Client)

	// Run inference based on task type
	var answer string
	var err error

	switch req.TaskType {
	case "mcq":
		answer, err = adapter.InferMCQ(modelName, req.Question, req.Choices, eie.getTemperature(req, capabilities))
	case "boolq":
		var boolAnswer bool
		boolAnswer, err = adapter.InferBool(modelName, req.Question, req.Passage, eie.getTemperature(req, capabilities))
		if err == nil {
			answer = strconv.FormatBool(boolAnswer)
		}
	case "hellaswag":
		var idx int
		idx, err = adapter.InferCompletion(modelName, req.Context, req.Endings, eie.getTemperature(req, capabilities))
		if err == nil {
			answer = strconv.Itoa(idx)
		}
	case "arc":
		answer, err = adapter.InferOpenEnded(modelName, eie.formatARCPrompt(req), eie.getTemperature(req, capabilities), eie.getMaxTokens(req, capabilities))
	case "trivia":
		answer, err = adapter.InferOpenEnded(modelName, req.Question, eie.getTemperature(req, capabilities), eie.getMaxTokens(req, capabilities))
	case "browser":
		answer, err = adapter.InferOpenEnded(modelName, eie.formatBrowserPrompt(req), eie.getTemperature(req, capabilities), eie.getMaxTokens(req, capabilities))
	default:
		answer, err = adapter.InferOpenEnded(modelName, req.Question, eie.getTemperature(req, capabilities), eie.getMaxTokens(req, capabilities))
	}

	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	latency := float64(time.Since(start).Nanoseconds()) / 1e6 // Convert to milliseconds

	// Calculate confidence based on model performance and response characteristics
	confidence := eie.calculateConfidence(capabilities, answer, req.TaskType)

	// Estimate tokens used (simplified)
	tokensUsed := eie.estimateTokens(req.Question + answer)

	metadata := map[string]interface{}{
		"task_type":     req.TaskType,
		"domain":        capabilities.Domain,
		"model_version": capabilities.Name,
		"performance":   capabilities.Performance,
	}
	if eie.AgentCatalog != nil {
		metadata["agent_catalog"] = eie.AgentCatalog.Suites
		metadata["agent_tools"] = eie.AgentCatalog.Tools
		view := eie.catalogEnrichment()
		if view.Prompt != "" {
			metadata["agent_catalog_context"] = view.Prompt
		}
		if view.Summary != "" {
			metadata["agent_catalog_summary"] = view.Summary
		}
		if view.Stats.SuiteCount > 0 || view.Stats.UniqueToolCount > 0 {
			metadata["agent_catalog_stats"] = view.Stats
		}
		if len(view.Implementations) > 0 {
			metadata["agent_catalog_matrix"] = view.Implementations
		}
		if len(view.UniqueTools) > 0 {
			metadata["agent_catalog_unique_tools"] = view.UniqueTools
		}
		if len(view.StandaloneTools) > 0 {
			metadata["agent_catalog_tool_details"] = view.StandaloneTools
		}
	}

	return &InferenceResponse{
		Answer:     answer,
		Confidence: confidence,
		ModelName:  modelName,
		Latency:    latency,
		TokensUsed: tokensUsed,
		Metadata:   metadata,
	}, nil
}

// RunBenchmark runs a benchmark task and returns the result
func (eie *EnhancedInferenceEngine) RunBenchmark(ctx context.Context, req *InferenceRequest, correctAnswer string) (*BenchmarkResult, error) {
	resp, err := eie.Infer(ctx, req)
	if err != nil {
		return nil, err
	}

	// Determine if the answer is correct
	isCorrect := eie.isAnswerCorrect(resp.Answer, correctAnswer, req.TaskType)

	return &BenchmarkResult{
		TaskType:      req.TaskType,
		ModelName:     resp.ModelName,
		Question:      req.Question,
		Answer:        resp.Answer,
		CorrectAnswer: correctAnswer,
		IsCorrect:     isCorrect,
		Confidence:    resp.Confidence,
		Latency:       resp.Latency,
		TokensUsed:    resp.TokensUsed,
		Timestamp:     time.Now(),
	}, nil
}

// GetModelCapabilities returns the capabilities of a specific model
func (eie *EnhancedInferenceEngine) GetModelCapabilities(modelName string) (*ModelCapabilities, error) {
	capabilities, exists := eie.ModelRegistry.models[modelName]
	if !exists {
		return nil, fmt.Errorf("model %s not found", modelName)
	}
	return capabilities, nil
}

// ListAvailableModels returns all available models
func (eie *EnhancedInferenceEngine) ListAvailableModels() map[string]*ModelCapabilities {
	return eie.ModelRegistry.models
}

// GetBestModelForTask returns the best model for a specific task type
func (eie *EnhancedInferenceEngine) GetBestModelForTask(taskType string) (string, error) {
	modelName, exists := eie.DomainRouter.domains[strings.ToLower(taskType)]
	if !exists {
		modelName, exists = eie.DomainRouter.domains[taskType]
	}
	if !exists {
		return "", fmt.Errorf("no model found for task type %s", taskType)
	}
	return modelName, nil
}

func (eie *EnhancedInferenceEngine) catalogEnrichment() catalogprompt.Enrichment {
	if eie == nil || eie.AgentCatalog == nil {
		return catalogprompt.Enrichment{}
	}
	if eie.catalogView.Prompt == "" &&
		len(eie.catalogView.UniqueTools) == 0 &&
		len(eie.catalogView.StandaloneTools) == 0 &&
		len(eie.catalogView.Implementations) == 0 &&
		eie.catalogView.Stats.SuiteCount == 0 &&
		eie.catalogView.Stats.UniqueToolCount == 0 {
		eie.catalogView = catalogprompt.Enrich(catalogprompt.Catalog{
			Suites: eie.AgentCatalog.Suites,
			Tools:  eie.AgentCatalog.Tools,
		})
	}
	return eie.catalogView
}

// Helper methods

// CatalogEnrichment exposes the cached Agent SDK catalog view for callers outside the package.
func (eie *EnhancedInferenceEngine) CatalogEnrichment() catalogprompt.Enrichment {
	return eie.catalogEnrichment()
}

func (eie *EnhancedInferenceEngine) isTaskSupported(capabilities *ModelCapabilities, taskType string) bool {
	for _, supportedTask := range capabilities.SupportedTasks {
		if strings.EqualFold(supportedTask, taskType) {
			return true
		}
	}
	return false
}

func (eie *EnhancedInferenceEngine) getTemperature(req *InferenceRequest, capabilities *ModelCapabilities) float64 {
	if req.Temperature > 0 {
		return req.Temperature
	}
	return capabilities.Temperature
}

func (eie *EnhancedInferenceEngine) getMaxTokens(req *InferenceRequest, capabilities *ModelCapabilities) int {
	if req.MaxTokens > 0 {
		return req.MaxTokens
	}
	return capabilities.MaxTokens
}

func (eie *EnhancedInferenceEngine) calculateConfidence(capabilities *ModelCapabilities, answer string, taskType string) float64 {
	// Base confidence on model performance
	baseConfidence := capabilities.Performance.Accuracy

	// Adjust based on answer characteristics
	if len(answer) == 0 {
		return 0.0
	}

	// For MCQ tasks, higher confidence for single letter answers
	if taskType == "mcq" && len(answer) == 1 && answer[0] >= 'A' && answer[0] <= 'D' {
		baseConfidence += 0.05
	}

	// For boolean tasks, higher confidence for clear yes/no
	if taskType == "boolq" && (strings.ToLower(answer) == "yes" || strings.ToLower(answer) == "no") {
		baseConfidence += 0.03
	}

	// Cap at 1.0
	if baseConfidence > 1.0 {
		baseConfidence = 1.0
	}

	return baseConfidence
}

func (eie *EnhancedInferenceEngine) estimateTokens(text string) int {
	// Simple token estimation (roughly 4 characters per token)
	return len(text) / 4
}

func (eie *EnhancedInferenceEngine) isAnswerCorrect(answer, correctAnswer, taskType string) bool {
	answer = strings.TrimSpace(strings.ToLower(answer))
	correctAnswer = strings.TrimSpace(strings.ToLower(correctAnswer))

	switch taskType {
	case "mcq":
		// For MCQ, compare single letters
		if len(answer) == 1 && len(correctAnswer) == 1 {
			return answer[0] == correctAnswer[0]
		}
		return answer == correctAnswer
	case "boolq":
		// For boolean questions, normalize answers
		normalizeBool := func(s string) string {
			s = strings.ToLower(s)
			if s == "yes" || s == "true" || s == "1" {
				return "yes"
			}
			if s == "no" || s == "false" || s == "0" {
				return "no"
			}
			return s
		}
		return normalizeBool(answer) == normalizeBool(correctAnswer)
	default:
		// For other tasks, direct string comparison
		return answer == correctAnswer
	}
}

func (eie *EnhancedInferenceEngine) formatARCPrompt(req *InferenceRequest) string {
	var sb strings.Builder
	sb.WriteString("Task: ")
	sb.WriteString(req.Question)
	sb.WriteString("\n\nInput Grid:\n")

	for _, row := range req.Grid {
		for _, cell := range row {
			sb.WriteString(fmt.Sprintf("%d ", cell))
		}
		sb.WriteString("\n")
	}

	sb.WriteString("\nPredict the output grid pattern.")
	return sb.String()
}

func (eie *EnhancedInferenceEngine) formatBrowserPrompt(req *InferenceRequest) string {
	var sb strings.Builder
	sb.WriteString("Browser Analysis Task: ")
	sb.WriteString(req.Question)

	if req.Context != "" {
		sb.WriteString("\n\nContext: ")
		sb.WriteString(req.Context)
	}

	if req.Passage != "" {
		sb.WriteString("\n\nContent: ")
		sb.WriteString(req.Passage)
	}

	sb.WriteString("\n\nProvide a detailed analysis:")
	return sb.String()
}
