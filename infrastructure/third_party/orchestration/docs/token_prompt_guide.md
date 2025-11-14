# Token-Aware Prompts – Developer Guide

This guide explains how to use the new **Token Oriented Object Notation** layer in the orchestration stack.

## 1. What you get

Every prompt (template, chat, few-shot) now produces:

- **String** – the final rendered text (old behavior).
- **Messages** – slice of `llms.ChatMessage` (for chat-style LLMs).
- **Tokens** – a hierarchical tree (`[]llms.Token`) that captures structure, variables, roles, and metadata **without re-parsing**.

## 2. Quick start

```go
import (
    "context"
    "fmt"

    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"
    monitoring "github.com/plturrell/aModels/services/extract/pkg/monitoring"
)

// 1. Build a prompt
pt := prompts.NewPromptTemplate(
    "Summarize {{.content}} in {{.style}} style.",
    []string{"content", "style"},
)

// 2. Create an LLMChain
chain := chains.NewLLMChain(llm, pt)

// 3. Run with telemetry
ctx := context.WithValue(context.Background(), "telemetry", telemetryClient)
out, err := chain.Call(ctx, map[string]any{
    "content": "War and Peace",
    "style":   "haiku",
})
```

## 3. Inspecting tokens

### CLI helper (one-liner)

```bash
go run ./cmd/token-inspect -template="Hello {{.name}}!" -vars='{"name":"Alice"}'
```

Example output:
```
template (go-template)
├── template_source (length=18)
├── variable "name"
│   └── value:string "Alice"
└── rendered_text (length=13)
```

### Programmatic inspection

```go
pv, _ := pt.FormatPrompt(map[string]any{"name": "Alice"})
if tv, ok := pv.(llms.TokenAwarePromptValue); ok {
    fmt.Println(prompts.TokensString(tv.Tokens()))
}
```

## 4. Telemetry payload

When telemetry is enabled (`ctx.Value("telemetry")`), every prompt emits:

```json
{
  "operation": "prompt_tokens",
  "input": {
    "prompt_id": "a1b2c3d4e5f6",
    "template_type": "template",
    "variable_count": 2,
    "tokens": [
      {
        "type": "template",
        "value": "go-template",
        "metadata": {
          "format": "go-template",
          "variable_count": "2"
        },
        "children": [...]
      }
    ]
  }
}
```

## 5. Token structure cheat-sheet

| Type            | Meaning                        | Metadata keys (sample) |
|-----------------|--------------------------------|------------------------|
| `text`          | Plain string prompt            | `length`               |
| `template`      | PromptTemplate root            | `format`, `variable_count`, `has_source`, `has_rendered` |
| `variable`      | Resolved template variable     | `name`, `type`, `value` |
| `message`       | Chat message                   | `role`, `index`, `content_length`, `name` |
| `example`       | Few-shot example               | `index`                |
| `prefix/suffix` | Few-shot prefix/suffix         | `length`               |

## 6. Debugging tips

- **Verbose mode**: set env `PROMPT_DEBUG=1` to print token trees at runtime.
- **Unit tests**: use `assertTokenTree(t, expected, actual)` helper in `prompts/token_prompt_test.go`.
- **Telemetry replay**: pipe telemetry logs into `jq '.input.tokens'` for quick inspection.

## 7. Extending tokens

If you add a new prompt type:

1. Implement `llms.TokenAwarePromptValue`.
2. Populate `Metadata` with relevant keys.
3. Update `token-inspect` CLI and this doc.

---
For questions or contributions, open an issue under `infrastructure/third_party/orchestration`.
