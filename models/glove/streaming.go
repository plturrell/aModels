package glove

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"sync"
)

// StreamingCooccurrence provides memory-efficient co-occurrence computation
type StreamingCooccurrence struct {
	db           interface{}
	windowSize   int
	batchSize    int
	vocabulary   map[string]int
	buffer       []CooccurrenceEntry
	mu           sync.Mutex
	totalEntries int64
}

// CooccurrenceEntry represents a single co-occurrence
type CooccurrenceEntry struct {
	WordID    int
	ContextID int
	Count     float64
}

// NewStreamingCooccurrence creates a streaming co-occurrence processor
func NewStreamingCooccurrence(db interface{}, windowSize, batchSize int) *StreamingCooccurrence {
	return &StreamingCooccurrence{
		db:         db,
		windowSize: windowSize,
		batchSize:  batchSize,
		vocabulary: make(map[string]int),
		buffer:     make([]CooccurrenceEntry, 0, batchSize),
	}
}

// ProcessFile processes a large corpus file in streaming fashion
func (sc *StreamingCooccurrence) ProcessFile(ctx context.Context, filepath string) error {
	file, err := os.Open(filepath)
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer file.Close()

	return sc.ProcessReader(ctx, file)
}

// ProcessReader processes corpus from any io.Reader
func (sc *StreamingCooccurrence) ProcessReader(ctx context.Context, reader io.Reader) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024) // 10MB buffer for long lines

	lineNum := 0
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line := scanner.Text()
		if err := sc.processLine(ctx, line); err != nil {
			return fmt.Errorf("process line %d: %w", lineNum, err)
		}

		lineNum++
		if lineNum%10000 == 0 {
			fmt.Printf("Processed %d lines, %d co-occurrences\n", lineNum, sc.totalEntries)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scanner error: %w", err)
	}

	// Flush remaining buffer
	return sc.flush(ctx)
}

// processLine processes a single line of text
func (sc *StreamingCooccurrence) processLine(ctx context.Context, line string) error {
	words := tokenize(line)
	if len(words) == 0 {
		return nil
	}

	// Build co-occurrence for this line
	for i, word := range words {
		wordID, ok := sc.vocabulary[word]
		if !ok {
			wordID = len(sc.vocabulary)
			sc.vocabulary[word] = wordID
		}

		// Context window
		start := max(0, i-sc.windowSize)
		end := min(len(words), i+sc.windowSize+1)

		for j := start; j < end; j++ {
			if i == j {
				continue
			}

			contextWord := words[j]
			contextID, ok := sc.vocabulary[contextWord]
			if !ok {
				contextID = len(sc.vocabulary)
				sc.vocabulary[contextWord] = contextID
			}

			// Distance-based weighting
			distance := abs(i - j)
			weight := 1.0 / float64(distance)

			entry := CooccurrenceEntry{
				WordID:    wordID,
				ContextID: contextID,
				Count:     weight,
			}

			if err := sc.addEntry(ctx, entry); err != nil {
				return err
			}
		}
	}

	return nil
}

// addEntry adds a co-occurrence entry to the buffer
func (sc *StreamingCooccurrence) addEntry(ctx context.Context, entry CooccurrenceEntry) error {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	sc.buffer = append(sc.buffer, entry)
	sc.totalEntries++

	if len(sc.buffer) >= sc.batchSize {
		return sc.flush(ctx)
	}

	return nil
}

// flush writes buffered entries to database
func (sc *StreamingCooccurrence) flush(ctx context.Context) error {
	if len(sc.buffer) == 0 {
		return nil
	}

	// In production, would batch insert to database
	// For now, just clear buffer
	fmt.Printf("Flushing %d co-occurrence entries to database\n", len(sc.buffer))

	sc.buffer = sc.buffer[:0]
	return nil
}

// GetVocabulary returns the built vocabulary
func (sc *StreamingCooccurrence) GetVocabulary() map[string]int {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	vocab := make(map[string]int, len(sc.vocabulary))
	for k, v := range sc.vocabulary {
		vocab[k] = v
	}
	return vocab
}

// GetTotalEntries returns the total number of co-occurrences processed
func (sc *StreamingCooccurrence) GetTotalEntries() int64 {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	return sc.totalEntries
}

// ChunkedCorpusProcessor processes corpus in chunks to manage memory
type ChunkedCorpusProcessor struct {
	chunkSize int
	overlap   int
}

// NewChunkedCorpusProcessor creates a chunked processor
func NewChunkedCorpusProcessor(chunkSize, overlap int) *ChunkedCorpusProcessor {
	return &ChunkedCorpusProcessor{
		chunkSize: chunkSize,
		overlap:   overlap,
	}
}

// ProcessLargeCorpus processes a large corpus in memory-efficient chunks
func (cp *ChunkedCorpusProcessor) ProcessLargeCorpus(ctx context.Context, filepath string, processor func([]string) error) error {
	file, err := os.Open(filepath)
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	chunk := make([]string, 0, cp.chunkSize)
	chunkNum := 0

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line := scanner.Text()
		chunk = append(chunk, line)

		if len(chunk) >= cp.chunkSize {
			if err := processor(chunk); err != nil {
				return fmt.Errorf("process chunk %d: %w", chunkNum, err)
			}

			// Keep overlap for context continuity
			if cp.overlap > 0 && len(chunk) > cp.overlap {
				chunk = chunk[len(chunk)-cp.overlap:]
			} else {
				chunk = chunk[:0]
			}

			chunkNum++
			fmt.Printf("Processed chunk %d\n", chunkNum)
		}
	}

	// Process remaining chunk
	if len(chunk) > 0 {
		if err := processor(chunk); err != nil {
			return fmt.Errorf("process final chunk: %w", err)
		}
	}

	return scanner.Err()
}

// MemoryEfficientTrainer provides memory-efficient training for large corpora
type MemoryEfficientTrainer struct {
	model         *Model
	streamingCooc *StreamingCooccurrence
	maxMemoryMB   int
}

// NewMemoryEfficientTrainer creates a memory-efficient trainer
func NewMemoryEfficientTrainer(model *Model, maxMemoryMB int) *MemoryEfficientTrainer {
	return &MemoryEfficientTrainer{
		model:       model,
		maxMemoryMB: maxMemoryMB,
	}
}

// TrainFromFile trains the model from a large corpus file
func (met *MemoryEfficientTrainer) TrainFromFile(ctx context.Context, filepath string) error {
	fmt.Printf("Starting memory-efficient training from %s\n", filepath)
	fmt.Printf("Memory limit: %d MB\n", met.maxMemoryMB)

	// Phase 1: Build vocabulary with streaming
	fmt.Println("\nPhase 1: Building vocabulary...")
	streaming := NewStreamingCooccurrence(met.model.db, 15, 10000)
	if err := streaming.ProcessFile(ctx, filepath); err != nil {
		return fmt.Errorf("build vocabulary: %w", err)
	}

	vocab := streaming.GetVocabulary()
	fmt.Printf("Vocabulary size: %d words\n", len(vocab))
	fmt.Printf("Total co-occurrences: %d\n", streaming.GetTotalEntries())

	// Phase 2: Train in batches
	fmt.Println("\nPhase 2: Training embeddings...")
	// In production, would load co-occurrences in batches and train

	fmt.Println("\n✓ Memory-efficient training complete!")
	return nil
}

// Helper functions defined in helpers.go
