package chains

import (
	"context"
	"errors"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/callbacks"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/memory"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/outputparser"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/schema"
	monitoring "github.com/plturrell/aModels/services/extract/pkg/monitoring"
)

const _llmChainDefaultOutputKey = "text"

type LLMChain struct {
	Prompt           prompts.FormatPrompter
	LLM              llms.Model
	Memory           schema.Memory
	CallbacksHandler callbacks.Handler
	OutputParser     schema.OutputParser[any]

	OutputKey string
}

var (
	_ Chain                  = &LLMChain{}
	_ callbacks.HandlerHaver = &LLMChain{}
)

// NewLLMChain creates a new LLMChain with an LLM and a prompt.
func NewLLMChain(llm llms.Model, prompt prompts.FormatPrompter, opts ...ChainCallOption) *LLMChain {
	opt := &chainCallOption{}
	for _, o := range opts {
		o(opt)
	}

	// Apply caching if configured
	if opt.CacheConfig != nil && opt.CacheConfig.Enabled {
		if cachedLLM, err := WithCaching(llm, opt.CacheConfig); err == nil {
			llm = cachedLLM
		}
	}

	chain := &LLMChain{
		Prompt:           prompt,
		LLM:              llm,
		OutputParser:     outputparser.NewSimple(),
		Memory:           memory.NewSimple(),
		OutputKey:        _llmChainDefaultOutputKey,
		CallbacksHandler: opt.CallbackHandler,
	}

	return chain
}

// Call formats the prompts with the input values, generates using the llm, and parses
// the output from the llm with the output parser. This function should not be called
// directly, use rather the Call or Run function if the prompt only requires one input
// value.
func (c LLMChain) Call(ctx context.Context, values map[string]any, options ...ChainCallOption) (map[string]any, error) {
	promptValue, err := c.Prompt.FormatPrompt(values)
	if err != nil {
		return nil, err
	}

	// Telemetry: log the tokenized prompt structure if telemetry is enabled.
	if telemetry, ok := ctx.Value("telemetry").(*monitoring.TelemetryClient); ok {
		if tokenValue, ok := promptValue.(llms.TokenAwarePromptValue); ok {
			promptID := monitoring.PromptID(c.Prompt.GetInputVariables()[0], values)
			_ = telemetry.LogPromptTokens(
				ctx,
				promptID,
				"template", // TODO: detect chat/few-shot
				tokenValue.Tokens(),
				len(values),
			)
		}
	}

	callOptions := getLLMCallOptions(options...)

	messageContents, err := llms.ChatMessagesToMessageContents(promptValue.Messages())
	if err != nil {
		return nil, err
	}

	var result string
	if len(messageContents) > 0 {
		resp, err := c.LLM.GenerateContent(ctx, messageContents, callOptions...)
		if err != nil {
			return nil, err
		}
		if len(resp.Choices) == 0 {
			return nil, errors.New("llm chain: empty response from model")
		}
		result = resp.Choices[0].Content
	} else {
		result, err = llms.GenerateFromSinglePrompt(ctx, c.LLM, promptValue.String(), callOptions...)
		if err != nil {
			return nil, err
		}
	}

	finalOutput, err := c.OutputParser.ParseWithPrompt(result, promptValue)
	if err != nil {
		return nil, err
	}

	return map[string]any{c.OutputKey: finalOutput}, nil
}

// GetMemory returns the memory.
func (c LLMChain) GetMemory() schema.Memory { //nolint:ireturn
	return c.Memory //nolint:ireturn
}

func (c LLMChain) GetCallbackHandler() callbacks.Handler { //nolint:ireturn
	return c.CallbacksHandler
}

// GetInputKeys returns the expected input keys.
func (c LLMChain) GetInputKeys() []string {
	return append([]string{}, c.Prompt.GetInputVariables()...)
}

// GetOutputKeys returns the output keys the chain will return.
func (c LLMChain) GetOutputKeys() []string {
	return []string{c.OutputKey}
}
