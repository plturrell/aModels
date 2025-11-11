package server

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// TokenCounter handles token counting for various text inputs
type TokenCounter struct {
	// Tokenization patterns
	wordPattern        *regexp.Regexp
	numberPattern      *regexp.Regexp
	punctuationPattern *regexp.Regexp

	// Language-specific tokenizers
	tokenizers map[string]Tokenizer
}

// Tokenizer represents a language-specific tokenizer
type Tokenizer interface {
	Tokenize(text string) []string
	CountTokens(text string) int
}

// DefaultTokenizer implements basic tokenization
type DefaultTokenizer struct{}

func (dt *DefaultTokenizer) Tokenize(text string) []string {
	// Split by whitespace and punctuation
	words := strings.Fields(text)
	var tokens []string

	for _, word := range words {
		// Split punctuation from words
		tokens = append(tokens, dt.splitPunctuation(word)...)
	}

	return tokens
}

func (dt *DefaultTokenizer) CountTokens(text string) int {
	return len(dt.Tokenize(text))
}

func (dt *DefaultTokenizer) splitPunctuation(word string) []string {
	var tokens []string
	var current strings.Builder

	for _, r := range word {
		if unicode.IsPunct(r) {
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

// ChineseTokenizer implements Chinese tokenization
type ChineseTokenizer struct{}

func (ct *ChineseTokenizer) Tokenize(text string) []string {
	var tokens []string

	for _, r := range text {
		if unicode.Is(unicode.Han, r) {
			// Chinese characters are typically one token each
			tokens = append(tokens, string(r))
		} else if !unicode.IsSpace(r) {
			// Non-Chinese, non-space characters
			tokens = append(tokens, string(r))
		}
	}

	return tokens
}

func (ct *ChineseTokenizer) CountTokens(text string) int {
	return len(ct.Tokenize(text))
}

// JapaneseTokenizer implements Japanese tokenization
type JapaneseTokenizer struct{}

func (jt *JapaneseTokenizer) Tokenize(text string) []string {
	var tokens []string

	for _, r := range text {
		if unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r) || unicode.Is(unicode.Han, r) {
			// Japanese characters
			tokens = append(tokens, string(r))
		} else if !unicode.IsSpace(r) {
			// Non-Japanese, non-space characters
			tokens = append(tokens, string(r))
		}
	}

	return tokens
}

func (jt *JapaneseTokenizer) CountTokens(text string) int {
	return len(jt.Tokenize(text))
}

// NewTokenCounter creates a new token counter
func NewTokenCounter() *TokenCounter {
	tc := &TokenCounter{
		wordPattern:        regexp.MustCompile(`\b\w+\b`),
		numberPattern:      regexp.MustCompile(`\d+`),
		punctuationPattern: regexp.MustCompile(`[^\w\s]`),
		tokenizers:         make(map[string]Tokenizer),
	}

	// Register default tokenizers
	tc.tokenizers["default"] = &DefaultTokenizer{}
	tc.tokenizers["chinese"] = &ChineseTokenizer{}
	tc.tokenizers["japanese"] = &JapaneseTokenizer{}

	return tc
}

// CountTokens counts tokens in text using the specified language
func (tc *TokenCounter) CountTokens(text string, language string) int {
	tokenizer, exists := tc.tokenizers[language]
	if !exists {
		tokenizer = tc.tokenizers["default"]
	}

	return tokenizer.CountTokens(text)
}

// CountTokensAdvanced provides advanced token counting with different strategies
func (tc *TokenCounter) CountTokensAdvanced(text string, strategy string) int {
	switch strategy {
	case "character":
		return tc.countByCharacters(text)
	case "word":
		return tc.countByWords(text)
	case "subword":
		return tc.countBySubwords(text)
	case "bpe":
		return tc.countByBPE(text)
	default:
		return tc.countByWords(text)
	}
}

// countByCharacters counts tokens by characters
func (tc *TokenCounter) countByCharacters(text string) int {
	return utf8.RuneCountInString(text)
}

// countByWords counts tokens by words
func (tc *TokenCounter) countByWords(text string) int {
	words := strings.Fields(text)
	return len(words)
}

// countBySubwords counts tokens by subwords
func (tc *TokenCounter) countBySubwords(text string) int {
	// Simple subword tokenization
	words := strings.Fields(text)
	total := 0

	for _, word := range words {
		// Split into subwords (simplified)
		subwords := tc.splitIntoSubwords(word)
		total += len(subwords)
	}

	return total
}

// splitIntoSubwords splits a word into subwords
func (tc *TokenCounter) splitIntoSubwords(word string) []string {
	var subwords []string

	// Simple subword splitting
	if len(word) <= 3 {
		subwords = append(subwords, word)
	} else {
		// Split into chunks of 2-3 characters
		for i := 0; i < len(word); i += 2 {
			end := i + 2
			if end > len(word) {
				end = len(word)
			}
			subwords = append(subwords, word[i:end])
		}
	}

	return subwords
}

// countByBPE counts tokens using Byte Pair Encoding approximation
func (tc *TokenCounter) countByBPE(text string) int {
	// Simplified BPE token counting
	words := strings.Fields(text)
	total := 0

	for _, word := range words {
		// BPE typically produces 1.3x more tokens than words
		total += int(float64(len(word)) * 1.3)
	}

	return total
}

// EstimateTokens estimates tokens for different model types
func (tc *TokenCounter) EstimateTokens(text string, modelType string) int {
	switch modelType {
	case "gpt-3.5-turbo", "gpt-4":
		return tc.estimateGPTTokens(text)
	case "claude":
		return tc.estimateClaudeTokens(text)
	case "gemma", "vaultgemma":
		return tc.estimateGemmaTokens(text)
	default:
		return tc.estimateGPTTokens(text)
	}
}

// estimateGPTTokens estimates tokens for GPT models
func (tc *TokenCounter) estimateGPTTokens(text string) int {
	// GPT models typically use ~4 characters per token
	return len(text) / 4
}

// estimateClaudeTokens estimates tokens for Claude models
func (tc *TokenCounter) estimateClaudeTokens(text string) int {
	// Claude models typically use ~3.5 characters per token
	return int(float64(len(text)) / 3.5)
}

// estimateGemmaTokens estimates tokens for Gemma models
func (tc *TokenCounter) estimateGemmaTokens(text string) int {
	// Gemma models typically use ~3.8 characters per token
	return int(float64(len(text)) / 3.8)
}

// CountTokensForMessages counts tokens for chat messages
func (tc *TokenCounter) CountTokensForMessages(messages []ChatMessage) int {
	total := 0

	for _, msg := range messages {
		// Count content tokens
		total += tc.CountTokens(msg.Content, "default")

		// Count role tokens (typically 1-2 tokens)
		total += 2

		// Count tool calls if present
		if len(msg.ToolCalls) > 0 {
			for _, toolCall := range msg.ToolCalls {
				total += tc.countToolCallTokens(toolCall)
			}
		}
	}

	return total
}

// countToolCallTokens counts tokens for tool calls
func (tc *TokenCounter) countToolCallTokens(toolCall ToolCall) int {
	tokens := 0

	// Count function name
	tokens += tc.CountTokens(toolCall.Function.Name, "default")

	// Count arguments
	for key, value := range toolCall.Function.Arguments {
		tokens += tc.CountTokens(key, "default")
		tokens += tc.CountTokens(fmt.Sprintf("%v", value), "default")
	}

	return tokens
}

// CountTokensForFunction counts tokens for function definitions
func (tc *TokenCounter) CountTokensForFunction(function FunctionDefinition) int {
	tokens := 0

	// Count function name
	tokens += tc.CountTokens(function.Name, "default")

	// Count description
	tokens += tc.CountTokens(function.Description, "default")

	// Count parameters
	for key, value := range function.Parameters {
		tokens += tc.CountTokens(key, "default")
		tokens += tc.CountTokens(fmt.Sprintf("%v", value), "default")
	}

	return tokens
}

// GetTokenUsage returns detailed token usage information
func (tc *TokenCounter) GetTokenUsage(text string, modelType string) TokenUsage {
	promptTokens := tc.EstimateTokens(text, modelType)
	completionTokens := 0 // This would be set by the inference engine
	totalTokens := promptTokens + completionTokens

	return TokenUsage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      totalTokens,
	}
}

// AnalyzeTokenDistribution analyzes token distribution in text
func (tc *TokenCounter) AnalyzeTokenDistribution(text string) map[string]int {
	analysis := make(map[string]int)

	// Count different types of tokens
	words := strings.Fields(text)
	analysis["words"] = len(words)

	// Count characters
	analysis["characters"] = len(text)
	analysis["runes"] = utf8.RuneCountInString(text)

	// Count numbers
	numbers := tc.numberPattern.FindAllString(text, -1)
	analysis["numbers"] = len(numbers)

	// Count punctuation
	punctuation := tc.punctuationPattern.FindAllString(text, -1)
	analysis["punctuation"] = len(punctuation)

	// Count sentences (rough approximation)
	sentences := strings.Count(text, ".") + strings.Count(text, "!") + strings.Count(text, "?")
	analysis["sentences"] = sentences

	return analysis
}

// OptimizeForTokenLimit optimizes text to fit within token limit
func (tc *TokenCounter) OptimizeForTokenLimit(text string, maxTokens int, modelType string) string {
	currentTokens := tc.EstimateTokens(text, modelType)

	if currentTokens <= maxTokens {
		return text
	}

	// Truncate text to fit within limit
	// This is a simple truncation - in production, you'd want smarter truncation
	targetLength := int(float64(len(text)) * float64(maxTokens) / float64(currentTokens))

	if targetLength >= len(text) {
		return text
	}

	// Find a good truncation point (end of sentence)
	truncated := text[:targetLength]
	lastSentenceEnd := strings.LastIndex(truncated, ".")

	if lastSentenceEnd > targetLength/2 {
		truncated = truncated[:lastSentenceEnd+1]
	}

	return truncated
}

// GetTokenEfficiency calculates token efficiency metrics
func (tc *TokenCounter) GetTokenEfficiency(text string, modelType string) map[string]float64 {
	tokens := tc.EstimateTokens(text, modelType)
	characters := len(text)
	words := len(strings.Fields(text))

	efficiency := make(map[string]float64)

	if tokens > 0 {
		efficiency["characters_per_token"] = float64(characters) / float64(tokens)
		efficiency["words_per_token"] = float64(words) / float64(tokens)
		efficiency["token_density"] = float64(tokens) / float64(characters)
	}

	return efficiency
}

// Enhanced token counting for the VaultGemmaServer
func (s *VaultGemmaServer) CountTokensEnhanced(text string, modelType string) int {
	if s.tokenCounter == nil {
		s.tokenCounter = NewTokenCounter()
	}

	return s.tokenCounter.EstimateTokens(text, modelType)
}

// CountTokensForRequest counts tokens for a complete request
func (s *VaultGemmaServer) CountTokensForRequest(messages []ChatMessage, modelType string) int {
	if s.tokenCounter == nil {
		s.tokenCounter = NewTokenCounter()
	}

	return s.tokenCounter.CountTokensForMessages(messages)
}

// GetDetailedTokenUsage returns detailed token usage for a request
func (s *VaultGemmaServer) GetDetailedTokenUsage(text string, modelType string) map[string]interface{} {
	if s.tokenCounter == nil {
		s.tokenCounter = NewTokenCounter()
	}

	usage := s.tokenCounter.GetTokenUsage(text, modelType)
	distribution := s.tokenCounter.AnalyzeTokenDistribution(text)
	efficiency := s.tokenCounter.GetTokenEfficiency(text, modelType)

	return map[string]interface{}{
		"usage":        usage,
		"distribution": distribution,
		"efficiency":   efficiency,
		"model_type":   modelType,
	}
}
