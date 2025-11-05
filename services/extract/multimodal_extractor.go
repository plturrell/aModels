package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
)

// MultiModalExtractionResult contains results from multi-modal extraction
type MultiModalExtractionResult struct {
	OCRResult      *OCRResult           `json:"ocr_result,omitempty"`
	Embeddings     *UnifiedEmbeddings   `json:"embeddings,omitempty"`
	Classification *TableClassification `json:"classification,omitempty"`
	Tables         []ExtractedTable     `json:"tables,omitempty"`
	Text           string               `json:"text,omitempty"`
	Method         string               `json:"method"`
}

// OCRResult contains OCR extraction results
type OCRResult struct {
	Text   string           `json:"text"`
	Tables []ExtractedTable `json:"tables"`
	Method string           `json:"method"`
	Error  string           `json:"error,omitempty"`
}

// ExtractedTable represents a table extracted from OCR
type ExtractedTable struct {
	Headers     []string   `json:"headers"`
	Rows        [][]string `json:"rows"`
	RowCount    int        `json:"row_count"`
	ColumnCount int        `json:"column_count"`
}

// UnifiedEmbeddings contains embeddings from multiple models
type UnifiedEmbeddings struct {
	RelationalEmbedding []float32      `json:"relational_embedding,omitempty"`
	SemanticEmbedding   []float32      `json:"semantic_embedding,omitempty"`
	TokenizedText       *TokenizedText `json:"tokenized_text,omitempty"`
	Embeddings          map[string]any `json:"embeddings,omitempty"`
	Errors              []string       `json:"errors,omitempty"`
}

// TokenizedText contains tokenization results
type TokenizedText struct {
	Text      string `json:"text"`
	Length    int    `json:"length"`
	WordCount int    `json:"word_count"`
	Tokens    []int  `json:"tokens,omitempty"` // SentencePiece token IDs
}

// MultiModalExtractor handles unified multi-modal extraction (Phase 6)
type MultiModalExtractor struct {
	logger     *log.Logger
	enabled    bool
	ocrEnabled bool
}

// NewMultiModalExtractor creates a new multi-modal extractor
func NewMultiModalExtractor(logger *log.Logger) *MultiModalExtractor {
	return &MultiModalExtractor{
		logger:     logger,
		enabled:    os.Getenv("USE_MULTIMODAL_EXTRACTION") == "true",
		ocrEnabled: os.Getenv("USE_DEEPSEEK_OCR") == "true",
	}
}

// ExtractFromImage extracts text and tables from an image using DeepSeek-OCR
func (mme *MultiModalExtractor) ExtractFromImage(imagePath string, prompt string) (*OCRResult, error) {
	if !mme.enabled || !mme.ocrEnabled {
		return nil, fmt.Errorf("multi-modal extraction not enabled")
	}

	// Use Python script for OCR
	if prompt == "" {
		prompt = "<image>\n<|grounding|>Convert the document to markdown."
	}

	cmd := exec.Command("python3", "./scripts/unified_multimodal_extraction.py",
		"--mode", "ocr",
		"--image-path", imagePath,
		"--prompt", prompt,
	)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			mme.logger.Printf("OCR extraction failed: %v, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("OCR extraction failed: %w", err)
	}

	var ocrResult OCRResult
	if err := json.Unmarshal(output, &ocrResult); err != nil {
		return nil, fmt.Errorf("failed to parse OCR result: %w", err)
	}

	return &ocrResult, nil
}

// ExtractFromImageBase64 extracts from base64-encoded image
func (mme *MultiModalExtractor) ExtractFromImageBase64(imageBase64 string, prompt string) (*OCRResult, error) {
	if !mme.enabled || !mme.ocrEnabled {
		return nil, fmt.Errorf("multi-modal extraction not enabled")
	}

	cmd := exec.Command("python3", "./scripts/unified_multimodal_extraction.py",
		"--mode", "ocr",
		"--image-base64", imageBase64,
		"--prompt", prompt,
	)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			mme.logger.Printf("OCR extraction failed: %v, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("OCR extraction failed: %w", err)
	}

	var ocrResult OCRResult
	if err := json.Unmarshal(output, &ocrResult); err != nil {
		return nil, fmt.Errorf("failed to parse OCR result: %w", err)
	}

	return &ocrResult, nil
}

// GenerateUnifiedEmbeddings generates embeddings using all available models
func (mme *MultiModalExtractor) GenerateUnifiedEmbeddings(
	text string,
	imagePath string,
	tableName string,
	columns []map[string]any,
	tables []ExtractedTable,
) (*UnifiedEmbeddings, error) {
	if !mme.enabled {
		return nil, fmt.Errorf("multi-modal extraction not enabled")
	}

	columnsJSON, _ := json.Marshal(columns)

	args := []string{
		"./scripts/unified_multimodal_extraction.py",
		"--mode", "embed",
	}
	if text != "" {
		args = append(args, "--text", text)
	}
	if imagePath != "" {
		args = append(args, "--image-path", imagePath)
	}
	if tableName != "" {
		args = append(args, "--table-name", tableName)
		args = append(args, "--columns", string(columnsJSON))
	}

	cmd := exec.Command("python3", args...)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			mme.logger.Printf("Unified embedding generation failed: %v, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("embedding generation failed: %w", err)
	}

	var embeddings UnifiedEmbeddings
	if err := json.Unmarshal(output, &embeddings); err != nil {
		return nil, fmt.Errorf("failed to parse embeddings: %w", err)
	}

	return &embeddings, nil
}

// ExtractUnified performs full unified extraction pipeline
func (mme *MultiModalExtractor) ExtractUnified(
	imagePath string,
	imageBase64 string,
	tableName string,
	columns []map[string]any,
	text string,
	prompt string,
	trainingDataPath string,
) (*MultiModalExtractionResult, error) {
	if !mme.enabled {
		return nil, fmt.Errorf("multi-modal extraction not enabled")
	}

	columnsJSON, _ := json.Marshal(columns)

	args := []string{
		"./scripts/unified_multimodal_extraction.py",
		"--mode", "unified",
	}
	if imagePath != "" {
		args = append(args, "--image-path", imagePath)
	}
	if imageBase64 != "" {
		args = append(args, "--image-base64", imageBase64)
	}
	if tableName != "" {
		args = append(args, "--table-name", tableName)
		args = append(args, "--columns", string(columnsJSON))
	}
	if text != "" {
		args = append(args, "--text", text)
	}
	if prompt != "" {
		args = append(args, "--prompt", prompt)
	}
	if trainingDataPath != "" {
		args = append(args, "--training-data", trainingDataPath)
	}

	cmd := exec.Command("python3", args...)

	output, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			mme.logger.Printf("Unified extraction failed: %v, stderr: %s", err, string(exitErr.Stderr))
		}
		return nil, fmt.Errorf("unified extraction failed: %w", err)
	}

	var result MultiModalExtractionResult
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse extraction result: %w", err)
	}

	return &result, nil
}

// TokenizeWithSentencePiece tokenizes text using SentencePiece (Go implementation)
func (mme *MultiModalExtractor) TokenizeWithSentencePiece(text string, modelPath string) (*TokenizedText, error) {
	if !mme.enabled {
		return nil, fmt.Errorf("multi-modal extraction not enabled")
	}

	// Use SentencePiece Go binary
	cmd := exec.Command("./models/sentencepiece/spm_encode", "--model", modelPath)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}

	go func() {
		defer stdin.Close()
		io.WriteString(stdin, text)
	}()

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("sentencepiece tokenization failed: %w", err)
	}

	// Parse token IDs from output
	tokenIDs := []int{}
	for _, line := range strings.Split(string(output), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Parse token IDs (format depends on spm_encode output)
		parts := strings.Fields(line)
		for _, part := range parts {
			var id int
			if _, err := fmt.Sscanf(part, "%d", &id); err == nil {
				tokenIDs = append(tokenIDs, id)
			}
		}
	}

	return &TokenizedText{
		Text:      text,
		Length:    len(text),
		WordCount: len(strings.Fields(text)),
		Tokens:    tokenIDs,
	}, nil
}

// ConvertExtractedTableToNodes converts OCR-extracted tables to knowledge graph nodes
func (mme *MultiModalExtractor) ConvertExtractedTableToNodes(
	table ExtractedTable,
	tableName string,
	sourceID string,
) ([]Node, []Edge) {
	nodes := []Node{}
	edges := []Edge{}

	// Create table node
	tableNode := Node{
		ID:    fmt.Sprintf("table:%s", tableName),
		Type:  "table",
		Label: tableName,
		Props: map[string]any{
			"source":            "ocr",
			"source_id":         sourceID,
			"row_count":         table.RowCount,
			"column_count":      table.ColumnCount,
			"extraction_method": "deepseek-ocr",
		},
	}
	nodes = append(nodes, tableNode)

	// Create column nodes
	for i, header := range table.Headers {
		colNode := Node{
			ID:    fmt.Sprintf("column:%s:%s", tableName, header),
			Type:  "column",
			Label: header,
			Props: map[string]any{
				"table_name":        tableName,
				"column_index":      i,
				"source":            "ocr",
				"source_id":         sourceID,
				"extraction_method": "deepseek-ocr",
			},
		}
		nodes = append(nodes, colNode)

		// Create HAS_COLUMN edge
		edge := Edge{
			SourceID: tableNode.ID,
			TargetID: colNode.ID,
			Label:    "HAS_COLUMN",
			Props: map[string]any{
				"source":       "ocr",
				"column_index": i,
			},
		}
		edges = append(edges, edge)
	}

	return nodes, edges
}
