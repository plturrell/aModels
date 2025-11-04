//go:build !hana

package bootstrap

import (
	"context"
	"fmt"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/callbacks"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/schema"
)

var _ callbacks.Handler = (*LangOperationHandler)(nil)

// LangOperationHandler is a no-op implementation used when HANA support is not compiled in.
type LangOperationHandler struct{}

// NewLangOperationHandler returns an error when the hana build tag is not enabled.
func (r *Runtime) NewLangOperationHandler(string) (*LangOperationHandler, error) {
	return nil, fmt.Errorf("hana build tag required for LangOperationHandler")
}

func (*LangOperationHandler) HandleText(context.Context, string)       {}
func (*LangOperationHandler) HandleLLMStart(context.Context, []string) {}
func (*LangOperationHandler) HandleLLMGenerateContentStart(context.Context, []llms.MessageContent) {
}
func (*LangOperationHandler) HandleLLMGenerateContentEnd(context.Context, *llms.ContentResponse) {
}
func (*LangOperationHandler) HandleLLMError(context.Context, error)                 {}
func (*LangOperationHandler) HandleChainStart(context.Context, map[string]any)      {}
func (*LangOperationHandler) HandleChainEnd(context.Context, map[string]any)        {}
func (*LangOperationHandler) HandleChainError(context.Context, error)               {}
func (*LangOperationHandler) HandleToolStart(context.Context, string)               {}
func (*LangOperationHandler) HandleToolEnd(context.Context, string)                 {}
func (*LangOperationHandler) HandleToolError(context.Context, error)                {}
func (*LangOperationHandler) HandleAgentAction(context.Context, schema.AgentAction) {}
func (*LangOperationHandler) HandleAgentFinish(context.Context, schema.AgentFinish) {}
func (*LangOperationHandler) HandleRetrieverStart(context.Context, string)          {}
func (*LangOperationHandler) HandleRetrieverEnd(context.Context, string, []schema.Document) {
}
func (*LangOperationHandler) HandleStreamingFunc(context.Context, []byte) {}
