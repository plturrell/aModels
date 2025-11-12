package piqa

import (
	"strings"
	"unicode"
)

// PhraseEnumerator generates all valid phrases from a context
type PhraseEnumerator struct {
	MaxPhraseLen int // maximum phrase length in tokens
	MinPhraseLen int // minimum phrase length in tokens
}

// NewPhraseEnumerator creates a new phrase enumerator
func NewPhraseEnumerator(maxLen, minLen int) *PhraseEnumerator {
	if maxLen <= 0 {
		maxLen = 7 // default from PIQA paper
	}
	if minLen <= 0 {
		minLen = 1
	}
	return &PhraseEnumerator{
		MaxPhraseLen: maxLen,
		MinPhraseLen: minLen,
	}
}

// Tokenize splits text into tokens (simple whitespace + punctuation tokenization)
func (pe *PhraseEnumerator) Tokenize(text string) []string {
	var tokens []string
	var current strings.Builder

	for _, r := range text {
		if unicode.IsSpace(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
		} else if unicode.IsPunct(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(r))
		} else {
			current.WriteRune(r)
		}
	}

	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	return tokens
}

// EnumeratePhrases generates all valid phrases from context
// Returns approximately N = contextLength * maxPhraseLen / 2 phrases
func (pe *PhraseEnumerator) EnumeratePhrases(context string) []Phrase {
	tokens := pe.Tokenize(context)
	var phrases []Phrase

	// Build character offset map for tokens
	tokenOffsets := make([]int, len(tokens))
	charPos := 0
	tokenIdx := 0

	for i := 0; i < len(context) && tokenIdx < len(tokens); {
		// Skip whitespace
		if unicode.IsSpace(rune(context[i])) {
			i++
			charPos++
			continue
		}

		// Match token
		token := tokens[tokenIdx]
		if i+len(token) <= len(context) && context[i:i+len(token)] == token {
			tokenOffsets[tokenIdx] = i
			i += len(token)
			charPos += len(token)
			tokenIdx++
		} else {
			i++
			charPos++
		}
	}

	// Enumerate all spans of length [minLen, maxLen]
	for start := 0; start < len(tokens); start++ {
		for length := pe.MinPhraseLen; length <= pe.MaxPhraseLen && start+length <= len(tokens); length++ {
			end := start + length

			// Get character offsets
			charStart := tokenOffsets[start]
			var charEnd int
			if end < len(tokens) {
				charEnd = tokenOffsets[end]
			} else {
				// Last token - find end of last token
				lastToken := tokens[end-1]
				charEnd = tokenOffsets[end-1] + len(lastToken)
			}

			// Extract phrase text
			if charEnd > len(context) {
				charEnd = len(context)
			}
			phraseText := strings.TrimSpace(context[charStart:charEnd])

			if phraseText == "" {
				continue
			}

			phrases = append(phrases, Phrase{
				Text:       phraseText,
				Start:      charStart,
				End:        charEnd,
				StartToken: start,
				EndToken:   end,
			})
		}
	}

	return phrases
}

// FilterPhrases removes invalid phrases (e.g., only punctuation, too short)
func (pe *PhraseEnumerator) FilterPhrases(phrases []Phrase) []Phrase {
	var filtered []Phrase

	for _, p := range phrases {
		// Skip if only punctuation or whitespace
		hasAlpha := false
		for _, r := range p.Text {
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				hasAlpha = true
				break
			}
		}

		if !hasAlpha {
			continue
		}

		// Skip very short phrases
		if len(p.Text) < 2 {
			continue
		}

		filtered = append(filtered, p)
	}

	return filtered
}
