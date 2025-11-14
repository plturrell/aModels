package prompts

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
)

var _ llms.TokenAwarePromptValue = TokenPromptValue{}

// TokenPromptValue carries rendered prompt text, optional chat messages, and a
// tokenized representation so downstream components can reason about the
// structure without reparsing.
type TokenPromptValue struct {
	text     string
	messages []llms.ChatMessage
	tokens   []llms.Token
}

// NewTokenPromptValue creates a TokenPromptValue ensuring we always have a
// sensible string and message representation.
func NewTokenPromptValue(text string, messages []llms.ChatMessage, tokens []llms.Token) TokenPromptValue {
	clonedMessages := cloneMessages(messages)
	if len(clonedMessages) == 0 && text != "" {
		clonedMessages = []llms.ChatMessage{llms.HumanChatMessage{Content: text}}
	}

	clonedTokens := cloneTokens(tokens)

	return TokenPromptValue{
		text:     text,
		messages: clonedMessages,
		tokens:   clonedTokens,
	}
}

// NewStringTokenPromptValue wraps plain string prompts with a simple text token
// structure.
func NewStringTokenPromptValue(text string) TokenPromptValue {
	return NewTokenPromptValue(text, nil, buildTextTokens(text))
}

// NewChatTokenPromptValue wraps chat prompts with message tokens derived from
// the provided messages slice.
func NewChatTokenPromptValue(messages []llms.ChatMessage) TokenPromptValue {
	buffer, _ := llms.GetBufferString(messages, "Human", "AI")
	return NewTokenPromptValue(buffer, messages, buildChatTokens(messages))
}

func (v TokenPromptValue) String() string {
	if v.text != "" {
		return v.text
	}

	if len(v.messages) > 0 {
		if buff, err := llms.GetBufferString(v.messages, "Human", "AI"); err == nil {
			return buff
		}
	}

	return ""
}

func (v TokenPromptValue) Messages() []llms.ChatMessage {
	if len(v.messages) == 0 && v.text != "" {
		return []llms.ChatMessage{llms.HumanChatMessage{Content: v.text}}
	}
	return cloneMessages(v.messages)
}

func (v TokenPromptValue) Tokens() []llms.Token {
	return cloneTokens(v.tokens)
}

func cloneMessages(messages []llms.ChatMessage) []llms.ChatMessage {
	if len(messages) == 0 {
		return nil
	}
	cloned := make([]llms.ChatMessage, len(messages))
	copy(cloned, messages)
	return cloned
}

func cloneTokens(tokens []llms.Token) []llms.Token {
	if len(tokens) == 0 {
		return nil
	}
	cloned := make([]llms.Token, len(tokens))
	for i, token := range tokens {
		cloned[i] = llms.Token{
			Type:     token.Type,
			Value:    token.Value,
			Metadata: cloneMetadata(token.Metadata),
			Children: cloneTokens(token.Children),
		}
	}
	return cloned
}

func cloneMetadata(metadata map[string]string) map[string]string {
	if len(metadata) == 0 {
		return nil
	}
	cloned := make(map[string]string, len(metadata))
	for k, v := range metadata {
		cloned[k] = v
	}
	return cloned
}

// TokensString renders the token structure mainly for debugging purposes.
func TokensString(tokens []llms.Token) string {
	builder := &tokenStringBuilder{}
	for _, token := range tokens {
		builder.writeToken(0, token)
	}
	return builder.String()
}

type tokenStringBuilder struct {
	buf []byte
}

func (b *tokenStringBuilder) writeToken(indent int, token llms.Token) {
	b.buf = append(b.buf, []byte(fmt.Sprintf("%s[%s] %s\n", spaces(indent), token.Type, token.Value))...)
	for _, child := range token.Children {
		b.writeToken(indent+2, child)
	}
}

func (b *tokenStringBuilder) String() string {
	return string(b.buf)
}

func spaces(count int) string {
	if count <= 0 {
		return ""
	}
	return strings.Repeat(" ", count)
}

func buildTextTokens(text string) []llms.Token {
	if text == "" {
		return nil
	}
	return []llms.Token{{
		Type:  "text",
		Value: text,
		Metadata: map[string]string{
			"length": strconv.Itoa(len(text)),
		},
	}}
}

func buildTemplateTokens(format TemplateFormat, template string, resolved map[string]any, output string) []llms.Token {
	children := []llms.Token{}
	if output != "" {
		children = append(children, llms.Token{
			Type:  "rendered_text",
			Value: output,
			Metadata: map[string]string{
				"length": strconv.Itoa(len(output)),
			},
		})
	}

	if len(resolved) > 0 {
		keys := make([]string, 0, len(resolved))
		for key := range resolved {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			value := resolved[key]
			child := llms.Token{
				Type:  "variable",
				Value: key,
				Metadata: map[string]string{
					"name":  key,
					"type":  fmt.Sprintf("%T", value),
					"value": fmt.Sprint(value),
				},
				Children: []llms.Token{
					{
						Type:  fmt.Sprintf("value:%T", value),
						Value: fmt.Sprint(value),
						Metadata: map[string]string{
							"type":  fmt.Sprintf("%T", value),
							"value": fmt.Sprint(value),
						},
					},
				},
			}
			children = append(children, child)
		}
	}

	if template != "" {
		children = append(children, llms.Token{
			Type:  "template_source",
			Value: template,
			Metadata: map[string]string{
				"length": strconv.Itoa(len(template)),
			},
		})
	}

	if len(children) == 0 {
		return nil
	}

	metadata := map[string]string{
		"format":         string(format),
		"variable_count": strconv.Itoa(len(resolved)),
		"has_source":     strconv.FormatBool(template != ""),
		"has_rendered":   strconv.FormatBool(output != ""),
	}

	return []llms.Token{
		{
			Type:     "template",
			Value:    string(format),
			Metadata: metadata,
			Children: children,
		},
	}
}

func buildChatTokens(messages []llms.ChatMessage) []llms.Token {
	if len(messages) == 0 {
		return nil
	}

	msgTokens := make([]llms.Token, 0, len(messages))
	for idx, msg := range messages {
		content := msg.GetContent()
		children := []llms.Token{{
			Type:  "content",
			Value: content,
			Metadata: map[string]string{
				"length": strconv.Itoa(len(content)),
			},
		}}
		messageMetadata := map[string]string{
			"role":           string(msg.GetType()),
			"index":          strconv.Itoa(idx),
			"content_length": strconv.Itoa(len(content)),
		}
		if named, ok := msg.(interface{ GetName() string }); ok && named.GetName() != "" {
			name := named.GetName()
			children = append(children, llms.Token{
				Type:  "name",
				Value: name,
				Metadata: map[string]string{
					"value": name,
				},
			})
			messageMetadata["name"] = name
		}
		msgTokens = append(msgTokens, llms.Token{
			Type:     "message",
			Value:    string(msg.GetType()),
			Metadata: messageMetadata,
			Children: children,
		})
	}

	return []llms.Token{{
		Type:     "chat_prompt",
		Metadata: map[string]string{"message_count": strconv.Itoa(len(messages))},
		Children: msgTokens,
	}}
}

func buildFewShotPromptTokens(
	format TemplateFormat,
	prefix string,
	suffix string,
	examples []map[string]string,
	exampleStrings []string,
	assembledTemplate string,
	resolved map[string]any,
	output string,
) []llms.Token {
	templateTokens := buildTemplateTokens(format, assembledTemplate, resolved, output)
	if len(templateTokens) == 0 {
		templateTokens = []llms.Token{{Type: "template", Value: string(format)}}
	}

	pieces := []llms.Token{}
	if prefix != "" {
		pieces = append(pieces, llms.Token{
			Type:  "prefix",
			Value: prefix,
			Metadata: map[string]string{
				"length": strconv.Itoa(len(prefix)),
			},
		})
	}

	for i, example := range examples {
		exampleChildren := []llms.Token{}
		keys := make([]string, 0, len(example))
		for key := range example {
			keys = append(keys, key)
		}
		sort.Strings(keys)

		for _, key := range keys {
			value := example[key]
			exampleChildren = append(exampleChildren, llms.Token{
				Type:  "field",
				Value: fmt.Sprintf("%s=%s", key, value),
				Metadata: map[string]string{
					"key":   key,
					"value": value,
				},
			})
		}

		if i < len(exampleStrings) {
			exampleChildren = append(exampleChildren, llms.Token{
				Type:  "rendered",
				Value: exampleStrings[i],
				Metadata: map[string]string{
					"length": strconv.Itoa(len(exampleStrings[i])),
				},
			})
		}

		pieces = append(pieces, llms.Token{
			Type:     "example",
			Metadata: map[string]string{"index": strconv.Itoa(i)},
			Children: exampleChildren,
		})
	}

	if suffix != "" {
		pieces = append(pieces, llms.Token{
			Type:  "suffix",
			Value: suffix,
			Metadata: map[string]string{
				"length": strconv.Itoa(len(suffix)),
			},
		})
	}

	pieces = append(pieces, templateTokens...)

	return []llms.Token{{
		Type: "few_shot_prompt",
		Metadata: map[string]string{
			"example_count": strconv.Itoa(len(examples)),
			"format":        string(format),
			"has_prefix":    strconv.FormatBool(prefix != ""),
			"has_suffix":    strconv.FormatBool(suffix != ""),
			"output_length": strconv.Itoa(len(output)),
		},
		Children: pieces,
	}}
}
