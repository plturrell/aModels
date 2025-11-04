//go:build hana

package bootstrap

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	langintegration "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/lang_integration"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/callbacks"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/schema"
)

var _ callbacks.Handler = (*LangOperationHandler)(nil)

// LangOperationHandler implements callbacks.Handler and persists chain telemetry to HANA.
type LangOperationHandler struct {
	integration *langintegration.LangIntegration
	logFunc     func(context.Context, *langintegration.LangOperation) error
	mu          sync.Mutex
	operations  map[string]*operationState
	now         func() time.Time
}

type operationState struct {
	id       string
	start    time.Time
	input    map[string]any
	metadata SessionMetadata
}

// NewLangOperationHandler wires telemetry logging for chain executions.
func (r *Runtime) NewLangOperationHandler(schema string) (*LangOperationHandler, error) {
	if r == nil || r.HANAPool == nil {
		return nil, fmt.Errorf("hana pool not initialised")
	}

	if schema == "" {
		schema = r.hanaSchema
	}
	if schema == "" {
		schema = "AGENTICAI"
	}

	handler := &LangOperationHandler{
		integration: langintegration.NewLangIntegration(r.HANAPool.GetDB(), schema),
		logFunc:     nil,
		operations:  make(map[string]*operationState),
		now:         time.Now,
	}

	handler.logFunc = handler.integration.LogOperation

	return handler, nil
}

// HandleChainStart captures the beginning of a chain execution.
func (h *LangOperationHandler) HandleChainStart(ctx context.Context, inputs map[string]any) {
	if h == nil {
		return
	}

	key := contextKey(ctx)
	meta := SessionMetadataFromContext(ctx)

	h.mu.Lock()
	defer h.mu.Unlock()

	h.operations[key] = &operationState{
		id:       uuid.NewString(),
		start:    h.now(),
		input:    cloneMap(inputs),
		metadata: meta,
	}
}

// HandleChainEnd records successful completion of a chain.
func (h *LangOperationHandler) HandleChainEnd(ctx context.Context, outputs map[string]any) {
	h.finish(ctx, outputs, nil)
}

// HandleChainError records failed completion of a chain.
func (h *LangOperationHandler) HandleChainError(ctx context.Context, err error) {
	h.finish(ctx, nil, err)
}

func (h *LangOperationHandler) finish(ctx context.Context, outputs map[string]any, chainErr error) {
	if h == nil {
		return
	}

	key := contextKey(ctx)

	h.mu.Lock()
	state, ok := h.operations[key]
	if ok {
		delete(h.operations, key)
	}
	h.mu.Unlock()

	if !ok {
		return
	}

	completedAt := h.now()
	latency := completedAt.Sub(state.start).Milliseconds()

	libraryType := langintegration.LangChain
	if state.metadata.LibraryType != "" {
		libraryType = langintegration.LangLibraryType(state.metadata.LibraryType)
	}

	operation := state.metadata.Operation
	if operation == "" {
		operation = "chain"
	}

	status := "success"
	var errorText string
	if chainErr != nil {
		status = "error"
		errorText = chainErr.Error()
	}

	op := &langintegration.LangOperation{
		ID:           state.id,
		LibraryType:  libraryType,
		Operation:    operation,
		Input:        state.input,
		Output:       cloneMap(outputs),
		Status:       status,
		Error:        errorText,
		LatencyMs:    latency,
		CreatedAt:    state.start,
		CompletedAt:  &completedAt,
		SessionID:    state.metadata.SessionID,
		UserIDHash:   state.metadata.UserIDHash,
		PrivacyLevel: defaultPrivacy(state.metadata.PrivacyLevel),
	}

	logFn := h.logFunc
	if logFn == nil {
		logFn = h.integration.LogOperation
	}

	if err := logFn(ctx, op); err != nil {
		log.Printf("⚠️ failed to log lang operation %s: %v", state.id, err)
	}
}

func (h *LangOperationHandler) HandleLLMStart(context.Context, []string) {}

func (h *LangOperationHandler) HandleLLMGenerateContentStart(context.Context, []llms.MessageContent) {
}

func (h *LangOperationHandler) HandleLLMGenerateContentEnd(context.Context, *llms.ContentResponse) {
}

func (h *LangOperationHandler) HandleLLMError(context.Context, error) {}

func (h *LangOperationHandler) HandleToolStart(context.Context, string) {}

func (h *LangOperationHandler) HandleToolEnd(context.Context, string) {}

func (h *LangOperationHandler) HandleToolError(context.Context, error) {}

func (h *LangOperationHandler) HandleAgentAction(context.Context, schema.AgentAction) {}

func (h *LangOperationHandler) HandleAgentFinish(context.Context, schema.AgentFinish) {}

func (h *LangOperationHandler) HandleRetrieverStart(context.Context, string) {}

func (h *LangOperationHandler) HandleRetrieverEnd(context.Context, string, []schema.Document) {}

func (h *LangOperationHandler) HandleStreamingFunc(context.Context, []byte) {}

func (h *LangOperationHandler) HandleText(context.Context, string) {}

func contextKey(ctx context.Context) string {
	return fmt.Sprintf("%p", ctx)
}

func cloneMap(src map[string]any) map[string]any {
	if len(src) == 0 {
		return map[string]any{}
	}
	dst := make(map[string]any, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func defaultPrivacy(level string) string {
	if strings.TrimSpace(level) == "" {
		return "default"
	}
	return level
}
