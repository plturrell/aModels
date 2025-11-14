package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"
)

var (
	template = flag.String("template", "", "Prompt template string")
	varsJSON = flag.String("vars", "{}", "JSON map of variables")
	chat     = flag.Bool("chat", false, "Interpret template as chat prompt")
	fewShot  = flag.Bool("few-shot", false, "Interpret template as few-shot prompt")
)

func main() {
	flag.Parse()

	if *template == "" {
		log.Fatal("-template is required")
	}

	var vars map[string]any
	if err := json.Unmarshal([]byte(*varsJSON), &vars); err != nil {
		log.Fatalf("invalid vars JSON: %v", err)
	}

	var pv prompts.PromptValue
	var err error

	switch {
	case *chat:
		pv, err = prompts.NewChatPromptTemplate([]prompts.MessageFormatter{
			prompts.NewSystemMessagePromptTemplate(*template, nil),
		}).FormatPrompt(vars)
	case *fewShot:
		pv, err = prompts.NewFewShotPrompt(
			*template,
			[]string{},
			[]map[string]string{{"example": "value"}},
		).FormatPrompt(vars)
	default:
		pv, err = prompts.NewPromptTemplate(*template, extractVars(vars)).FormatPrompt(vars)
	}

	if err != nil {
		log.Fatalf("format error: %v", err)
	}

	if tv, ok := pv.(prompts.TokenAwarePromptValue); ok {
		fmt.Println(prompts.TokensString(tv.Tokens()))
	} else {
		fmt.Println("Prompt does not expose tokens")
	}
}

func extractVars(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
