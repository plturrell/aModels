package prompts

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
)

func TestComplexTemplateTokens(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name          string
		template      string
		vars          map[string]any
		expectedNodes []string // top-level token types
		assertMeta    func(t *testing.T, tokens []llms.Token)
	}{
		{
			name:     "nested loops",
			template: `{{range .items}}Item {{.name}}: {{.value}}
{{end}}`,
			vars: map[string]any{
				"items": []map[string]string{
					{"name": "A", "value": "1"},
					{"name": "B", "value": "2"},
				},
			},
			expectedNodes: []string{"template"},
			assertMeta: func(t *testing.T, tokens []llms.Token) {
				top := tokens[0]
				if top.Metadata["variable_count"] != "1" {
					t.Errorf("expected 1 variable, got %v", top.Metadata["variable_count"])
				}
			},
		},
		{
			name:     "conditionals",
			template: `{{if .flag}}Enabled{{else}}Disabled{{end}}`,
			vars:     map[string]any{"flag": true},
			expectedNodes: []string{"template"},
			assertMeta: func(t *testing.T, tokens []llms.Token) {
				for _, c := range tokens[0].Children {
					if c.Type == "variable" && c.Metadata["name"] == "flag" {
						return
					}
				}
				t.Error("missing variable token for 'flag'")
			},
		},
		{
			name: "partials with functions",
			template: `{{.greeting}} {{.name}}`,
			vars: map[string]any{
				"greeting": "Hello",
				"name":     func() string { return "World" },
			},
			expectedNodes: []string{"template"},
			assertMeta: func(t *testing.T, tokens []llms.Token) {
				var found bool
				for _, c := range tokens[0].Children {
					if c.Type == "variable" && c.Metadata["type"] == "func() string" {
						found = true
						break
					}
				}
				if !found {
					t.Error("missing func() string variable token")
				}
			},
		},
		{
			name:     "empty template",
			template: "",
			vars:     nil,
			expectedNodes: []string{"template"},
			assertMeta: func(t *testing.T, tokens []llms.Token) {
				if tokens[0].Metadata["variable_count"] != "0" {
					t.Errorf("expected 0 variables, got %v", tokens[0].Metadata["variable_count"])
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			pt := PromptTemplate{
				Template:       tc.template,
				TemplateFormat: TemplateFormatGoTemplate,
				InputVariables: extractVars(tc.vars),
			}
			pv, err := pt.FormatPrompt(tc.vars)
			if err != nil {
				t.Fatalf("FormatPrompt error: %v", err)
			}

			tv, ok := pv.(llms.TokenAwarePromptValue)
			if !ok {
				t.Fatal("not token-aware")
			}

			tokens := tv.Tokens()
			if len(tokens) == 0 {
				t.Fatal("no tokens produced")
			}

			// Check top-level type
			if tokens[0].Type != tc.expectedNodes[0] {
				t.Errorf("expected top-level %q, got %q", tc.expectedNodes[0], tokens[0].Type)
			}

			// Run custom assertions
			if tc.assertMeta != nil {
				tc.assertMeta(t, tokens)
			}
		})
	}
}

func extractVars(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
