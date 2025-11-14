package llms

// PromptValue is the interface that all prompt values must implement.
type PromptValue interface {
	String() string
	Messages() []ChatMessage
}

// TokenAwarePromptValue is implemented by prompt values that can expose their
// tokenized representation in addition to standard string/chat forms.
type TokenAwarePromptValue interface {
	PromptValue
	Tokens() []Token
}

// Token represents a typed segment of a prompt along with optional child
// tokens for hierarchical structures (for example, chat messages containing
// lexical tokens).
type Token struct {
	Type     string
	Value    string
	Metadata map[string]string
	Children []Token
}
