package langchainservice

import (
	"context"
	"fmt"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/integration/langchainservice"
)

// Tool integrates the LangChain Python service as an agent tool.
type Tool struct {
	client *langchainservice.Client
	name   string
	desc   string
}

// Option configures a Tool.
type Option func(*Tool)

// WithName overrides the tool name.
func WithName(name string) Option {
	return func(t *Tool) {
		t.name = name
	}
}

// WithDescription overrides the tool description.
func WithDescription(desc string) Option {
	return func(t *Tool) {
		t.desc = desc
	}
}

// NewTool constructs a LangChain service tool.
func NewTool(client *langchainservice.Client, opts ...Option) *Tool {
	t := &Tool{
		client: client,
		name:   "langchain_retrieval",
		desc:   "Query the LangChain Python service for context-aware answers based on shared persistence backends.",
	}
	for _, opt := range opts {
		opt(t)
	}
	return t
}

// Name returns the tool name.
func (t *Tool) Name() string {
	return t.name
}

// Description returns the tool description.
func (t *Tool) Description() string {
	return t.desc
}

// Call executes the tool with the given input text.
func (t *Tool) Call(ctx context.Context, input string) (string, error) {
	resp, err := t.client.RunChain(ctx, input)
	if err != nil {
		return "", fmt.Errorf("langchain service call failed: %w", err)
	}
	if len(resp.Sources) == 0 {
		return resp.Result, nil
	}

	return fmt.Sprintf("%s\nSources: %v", resp.Result, resp.Sources), nil
}

// LangChainTool interface compatibility with agent tooling.
var _ interface {
	Name() string
	Description() string
	Call(context.Context, string) (string, error)
} = (*Tool)(nil)
