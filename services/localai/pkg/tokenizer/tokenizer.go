package tokenizer

// Tokenizer defines the interface for text tokenization.
type Tokenizer interface {
	// Encode converts text to token IDs
	Encode(text string) ([]int, error)

	// EncodeAsTokens converts text to token strings
	EncodeAsTokens(text string) ([]string, error)

	// Decode converts token IDs back to text
	Decode(ids []int) (string, error)

	// VocabSize returns the vocabulary size
	VocabSize() int

	// GetToken returns the token string for a given ID
	GetToken(id int) (string, error)

	// Close releases resources
	Close() error
}

// TokenizerType represents the type of tokenizer.
type TokenizerType string

const (
	TypeSentencePiece TokenizerType = "sentencepiece"
	TypeBPE           TokenizerType = "bpe"
	TypeWordPiece     TokenizerType = "wordpiece"
)

// NewTokenizer creates a new tokenizer of the specified type.
func NewTokenizer(tokType TokenizerType, modelPath string) (Tokenizer, error) {
	switch tokType {
	case TypeSentencePiece:
		return NewSentencePieceTokenizer(modelPath)
	default:
		return NewSentencePieceTokenizer(modelPath)
	}
}
