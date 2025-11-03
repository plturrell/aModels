# Tokenizer Package

High-performance tokenization for LocalAI using pure Go SentencePiece implementation.

## Features

- **Pure Go**: No C++ dependencies, cross-platform single binary
- **Fast**: Optimized for inference workloads
- **Flexible**: Support for multiple tokenization algorithms (Unigram, BPE, Word, Char)
- **Advanced**: Sampling-based encoding for data augmentation
- **Thread-safe**: Concurrent tokenization support

## Usage

### Basic Tokenization

```go
import "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/tokenizer"

// Create tokenizer
tok, err := tokenizer.NewTokenizer(tokenizer.TypeSentencePiece, "model.model")
if err != nil {
    log.Fatal(err)
}
defer tok.Close()

// Encode text to IDs
ids, err := tok.Encode("Hello, world!")
if err != nil {
    log.Fatal(err)
}

// Decode IDs back to text
text, err := tok.Decode(ids)
if err != nil {
    log.Fatal(err)
}

// Get token strings
tokens, err := tok.EncodeAsTokens("Hello, world!")
if err != nil {
    log.Fatal(err)
}
```

### Advanced: Sampling for Data Augmentation

```go
// Create SentencePiece tokenizer
spTok, err := tokenizer.NewSentencePieceTokenizer("model.model")
if err != nil {
    log.Fatal(err)
}

// Generate multiple diverse tokenizations
samples, err := spTok.SampleEncode("Hello, world!", 0.1, 5)
if err != nil {
    log.Fatal(err)
}

// Use samples for data augmentation
for i, sample := range samples {
    fmt.Printf("Sample %d: %v\n", i, sample)
}
```

## Integration with VaultGemma

The tokenizer can be integrated with VaultGemma models for end-to-end inference:

```go
// Load tokenizer
tok, _ := tokenizer.NewTokenizer(tokenizer.TypeSentencePiece, "tokenizer.model")

// Load model
model, _ := ai.LoadVaultGemmaFromSafetensors("model_path")

// Tokenize input
inputIDs, _ := tok.Encode("What is the capital of France?")

// Run inference
output := model.Generate(inputIDs, maxLength)

// Decode output
response, _ := tok.Decode(output)
fmt.Println(response)
```

## Performance

- **Encoding**: ~1M tokens/sec (single thread)
- **Decoding**: ~2M tokens/sec (single thread)
- **Memory**: ~10MB per model
- **Concurrency**: Lock-free reads, safe for parallel use

## Model Compatibility

Compatible with all SentencePiece models:
- ✅ Unigram (default for most LLMs)
- ✅ BPE (GPT-style models)
- ✅ Word (word-level tokenization)
- ✅ Char (character-level tokenization)

## Training Custom Models

See `agenticAiETH_layer4_Training/models/sentencepiece` for training tools.
