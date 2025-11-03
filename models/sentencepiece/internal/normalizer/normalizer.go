package normalizer

import (
	"strings"
	"unicode"
)

// Normalizer handles text normalization for SentencePiece.
type Normalizer struct {
	addDummyPrefix         bool
	removeExtraWhitespaces bool
	escapeWhitespaces      bool
	precompiledCharsmap    []byte
}

// Config holds normalization configuration.
type Config struct {
	AddDummyPrefix         bool
	RemoveExtraWhitespaces bool
	EscapeWhitespaces      bool
	// TODO: Add normalization rule TSV support
}

// New creates a new Normalizer with the given configuration.
func New(config *Config) *Normalizer {
	if config == nil {
		config = &Config{
			AddDummyPrefix:         true,
			RemoveExtraWhitespaces: true,
			EscapeWhitespaces:      true,
		}
	}
	return &Normalizer{
		addDummyPrefix:         config.AddDummyPrefix,
		removeExtraWhitespaces: config.RemoveExtraWhitespaces,
		escapeWhitespaces:      config.EscapeWhitespaces,
		precompiledCharsmap:    normalizationRulesData,
	}
}

// NewFromPrecompiledCharsmap creates a Normalizer from precompiled binary data.
func NewFromPrecompiledCharsmap(charsmap []byte, config *Config) *Normalizer {
	n := New(config)
	n.precompiledCharsmap = charsmap
	return n
}

// Normalize normalizes the input text according to the configuration.
func (n *Normalizer) Normalize(text string) string {
	result := text

	// Apply precompiled normalization rules if available
	if len(n.precompiledCharsmap) > 0 {
		result = n.applyNormalizationRules(result)
	}

	// Remove extra whitespaces
	if n.removeExtraWhitespaces {
		result = removeExtraWhitespace(result)
	}

	// Escape whitespaces with meta symbol
	if n.escapeWhitespaces {
		result = strings.ReplaceAll(result, " ", "▁")
	}

	// Add dummy prefix
	if n.addDummyPrefix && !strings.HasPrefix(result, "▁") {
		result = "▁" + result
	}

	return result
}

// applyNormalizationRules applies the precompiled normalization rules.
func (n *Normalizer) applyNormalizationRules(text string) string {
	// Apply Unicode normalization (NFD - Canonical Decomposition)
	// This is a simplified implementation
	// Full implementation would decode the binary rules from precompiledCharsmap
	
	var result strings.Builder
	result.Grow(len(text))
	
	for _, r := range text {
		// Apply basic normalization rules
		normalized := n.normalizeRune(r)
		result.WriteString(normalized)
	}
	
	return result.String()
}

// normalizeRune normalizes a single rune.
func (n *Normalizer) normalizeRune(r rune) string {
	// Basic normalization rules
	switch {
	case r >= 'A' && r <= 'Z':
		// Lowercase uppercase letters
		return string(r + 32)
	case r == '\t' || r == '\n' || r == '\r':
		// Normalize whitespace
		return " "
	case unicode.IsSpace(r):
		// Normalize all whitespace to space
		return " "
	default:
		return string(r)
	}
}

// removeExtraWhitespace removes leading, trailing, and duplicate internal whitespace.
func removeExtraWhitespace(s string) string {
	// Trim leading and trailing whitespace
	s = strings.TrimSpace(s)

	// Replace multiple consecutive whitespaces with single space
	var result strings.Builder
	result.Grow(len(s))
	
	prevSpace := false
	for _, r := range s {
		if unicode.IsSpace(r) {
			if !prevSpace {
				result.WriteRune(' ')
				prevSpace = true
			}
		} else {
			result.WriteRune(r)
			prevSpace = false
		}
	}

	return result.String()
}

// GetPrecompiledCharsmap returns the precompiled normalization rules.
func (n *Normalizer) GetPrecompiledCharsmap() []byte {
	return n.precompiledCharsmap
}

// TODO: Implement full Unicode normalization rule application
// The precompiledCharsmap contains binary normalization rules that need to be
// decoded and applied. This requires implementing the rule application logic
// from the upstream SentencePiece normalizer.cc file.
