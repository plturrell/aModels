package prompts

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
)

// nolint: funlen
func TestFewShotPrompt_Format(t *testing.T) {
	examplePrompt := NewPromptTemplate("{{.question}}: {{.answer}}", []string{"question", "answer"})
	t.Parallel()
	testCases := []struct {
		name             string
		examplePrompt    PromptTemplate
		examples         []map[string]string
		prefix           string
		suffix           string
		input            map[string]interface{}
		partialInput     map[string]interface{}
		exampleSeparator string
		templateFormat   TemplateFormat
		validateTemplate bool
		wantErr          bool
		expected         string
	}{
		{
			"prefix only", examplePrompt,
			[]map[string]string{},
			"This is a {{.foo}} test.", "",
			map[string]interface{}{"foo": "bar"},
			nil,
			"",
			TemplateFormatGoTemplate,
			true,
			false,
			"This is a bar test.",
		},
		{
			"suffix only", examplePrompt,
			[]map[string]string{},
			"", "This is a {{.foo}} test.",
			map[string]interface{}{"foo": "bar"},
			nil,
			"",
			TemplateFormatGoTemplate,
			true,
			false,
			"This is a bar test.",
		},
		{
			"insufficient InputVariables w err",
			examplePrompt,
			[]map[string]string{},
			"",
			"This is a {{.foo}} test.",
			map[string]interface{}{"bar": "bar"},
			nil,
			"",
			TemplateFormatGoTemplate,
			true,
			true,
			`template validation failed: template execution failure: template: template:1:12: executing "template" at <.foo>: map has no entry for key "foo"`,
		},
		{
			"inputVariables neither Examples nor ExampleSelector w err",
			examplePrompt,
			nil,
			"",
			"",
			map[string]interface{}{"bar": "bar"},
			nil,
			"",
			TemplateFormatGoTemplate,
			true,
			true,
			ErrNoExample.Error(),
		},
		{
			"functionality test",
			examplePrompt,
			[]map[string]string{{"question": "foo", "answer": "bar"}, {"question": "baz", "answer": "foo"}},
			"This is a test about {{.content}}.",
			"Now you try to talk about {{.new_content}}.",
			map[string]interface{}{"content": "animals", "new_content": "party"},
			nil,
			"\n",
			TemplateFormatGoTemplate,
			true,
			false,
			"This is a test about animals.\nfoo: bar\nbaz: foo\nNow you try to talk about party.",
		},
		{
			"functionality test with partial input",
			examplePrompt,
			[]map[string]string{{"question": "foo", "answer": "bar"}, {"question": "baz", "answer": "foo"}},
			"This is a test about {{.content}}.",
			"Now you try to talk about {{.new_content}}.",
			map[string]interface{}{"content": "animals"},
			map[string]interface{}{"new_content": func() string { return "party" }},
			"\n",
			TemplateFormatGoTemplate,
			true,
			false,
			"This is a test about animals.\nfoo: bar\nbaz: foo\nNow you try to talk about party.",
		},
		{
			"invalid template w err",
			examplePrompt,
			[]map[string]string{{"question": "foo", "answer": "bar"}, {"question": "baz", "answer": "foo"}},
			"This is a test about {{.wrong_content}}.",
			"Now you try to talk about {{.new_content}}.",
			map[string]interface{}{"content": "animals"},
			map[string]interface{}{"new_content": func() string { return "party" }},
			"\n",
			TemplateFormatGoTemplate,
			true,
			true,
			"template validation failed: template execution failure: template: template:1:23: executing \"template\" at <.wrong_content>: map has no entry for key " +
				"\"wrong_content\"",
		},
		{
			"token-aware few-shot prompt test",
			examplePrompt,
			[]map[string]string{{"question": "foo", "answer": "bar"}, {"question": "baz", "answer": "foo"}},
			"This is a test about {{.content}}.",
			"Now you try to talk about {{.new_content}}.",
			map[string]interface{}{"content": "animals"},
			map[string]interface{}{"new_content": func() string { return "party" }},
			"\n",
			TemplateFormatGoTemplate,
			true,
			false,
			"This is a test about animals.\nfoo: bar\nbaz: foo\nNow you try to talk about party.",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			t.Helper()
			p, err := NewFewShotPrompt(tc.examplePrompt, tc.examples, nil, tc.prefix, tc.suffix,
				getMapKeys(tc.input), tc.partialInput, tc.exampleSeparator, tc.templateFormat, tc.validateTemplate)
			if tc.wantErr {
				checkError(t, err, tc.expected)
				return
			}
			got, err := p.Format(tc.input)
			if checkError(t, err, "") {
				return
			}
			if diff := cmp.Diff(tc.expected, got); diff != "" {
				t.Errorf("unexpected prompt output (-want +got):\n%s", diff)
			}
			if tc.name == "token-aware few-shot prompt test" {
				input := map[string]interface{}{"content": "animals", "new_content": "party"}
				if err != nil {
					t.Fatalf("unexpected error creating prompt: %v", err)
				}

				value, err := p.FormatPrompt(input)
				if err != nil {
					t.Fatalf("unexpected error formatting prompt: %v", err)
				}

				expected := "This is a test about animals.\nfoo: bar\nbaz: foo\nNow you try to talk about party."
				if diff := cmp.Diff(expected, value.String()); diff != "" {
					t.Fatalf("unexpected rendered string (-want +got):\n%s", diff)
				}

				ta, ok := value.(llms.TokenAwarePromptValue)
				if !ok {
					t.Fatalf("expected token-aware prompt value")
				}

				tokens := ta.Tokens()
				if len(tokens) != 1 {
					t.Fatalf("expected single top-level token, got %d", len(tokens))
				}

				root := tokens[0]
				if root.Type != "few_shot_prompt" {
					t.Fatalf("expected top-level token type few_shot_prompt, got %q", root.Type)
				}

				childTypes := make([]string, len(root.Children))
				for i, child := range root.Children {
					childTypes[i] = child.Type
				}

				expectedChildren := []string{"prefix", "example", "example", "suffix", "template"}
				if diff := cmp.Diff(expectedChildren, childTypes); diff != "" {
					t.Fatalf("unexpected child token types (-want +got):\n%s", diff)
				}

				if root.Children[0].Value != "This is a test about animals." {
					t.Errorf("unexpected prefix token value: %q", root.Children[0].Value)
				}

				if len(root.Children[4].Children) == 0 || root.Children[4].Children[0].Type != "rendered_text" {
					t.Errorf("expected template token to include rendered_text child")
				}
			}
		})
	}
}

func checkError(t *testing.T, err error, expected string) bool {
	t.Helper()
	if err != nil {
		if expected != "" && err.Error() != expected {
			t.Errorf("unexpected error: got %q, want %q", err.Error(), expected)
		}
		return true
	}
	return false
}
