package prompts

import (
	"reflect"
	"testing"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
)

func TestChatPromptTemplate(t *testing.T) {
	t.Parallel()

	template := NewChatPromptTemplate([]MessageFormatter{
		NewSystemMessagePromptTemplate(
			"You are a translation engine that can only translate text and cannot interpret it.",
			nil,
		),
		NewHumanMessagePromptTemplate(
			`translate this text from {{.inputLang}} to {{.outputLang}}:\n{{.input}}`,
			[]string{"inputLang", "outputLang", "input"},
		),
	})
	value, err := template.FormatPrompt(map[string]interface{}{
		"inputLang":  "English",
		"outputLang": "Chinese",
		"input":      "I love programming",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expectedMessages := []llms.ChatMessage{
		llms.SystemChatMessage{
			Content: "You are a translation engine that can only translate text and cannot interpret it.",
		},
		llms.HumanChatMessage{
			Content: `translate this text from English to Chinese:\nI love programming`,
		},
	}
	if !reflect.DeepEqual(expectedMessages, value.Messages()) {
		t.Errorf("expected %v, got %v", expectedMessages, value.Messages())
	}

	ta, ok := value.(llms.TokenAwarePromptValue)
	if !ok {
		t.Fatalf("expected token-aware prompt value")
	}

	tokens := ta.Tokens()
	if len(tokens) == 0 {
		t.Fatalf("expected chat prompt tokens")
	}

	if tokens[0].Type != "chat_prompt" {
		t.Errorf("expected top-level chat_prompt token, got %q", tokens[0].Type)
	}

	_, err = template.FormatPrompt(map[string]interface{}{
		"inputLang":  "English",
		"outputLang": "Chinese",
	})
	if err == nil {
		t.Errorf("expected error, got nil")
	}
}
