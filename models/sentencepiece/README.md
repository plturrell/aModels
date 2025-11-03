# SentencePiece Go

Pure Go implementation of SentencePiece, providing full feature parity with the upstream C++ implementation.

## Project Structure

```
sentencepiece_go/
├── go.mod                    # Go module definition
├── README.md                 # This file
├── internal/
│   ├── proto/               # Generated protobuf bindings
│   ├── normalizer/          # Unicode normalization logic and tables
│   └── util/                # Shared utilities
├── model/                   # Model interface and implementations
│   ├── interface.go         # Core Model interface
│   ├── unigram.go          # Unigram model (TODO)
│   ├── bpe.go              # BPE model (TODO)
│   ├── word.go             # Word model (TODO)
│   └── char.go             # Char model (TODO)
├── processor/               # Main SentencePieceProcessor
│   └── processor.go         # Processor implementation
├── trainer/                 # Training interfaces and implementations
│   ├── interface.go         # Trainer interface with stubs
│   ├── unigram/            # Unigram trainer (TODO)
│   ├── bpe/                # BPE trainer (TODO)
│   ├── word/               # Word trainer (TODO)
│   └── char/               # Char trainer (TODO)
└── cmd/                     # Command-line tools
    ├── spm_train/          # Training CLI (TODO)
    ├── spm_encode/         # Encoding CLI (TODO)
    ├── spm_decode/         # Decoding CLI (TODO)
    ├── spm_export_vocab/   # Vocab export CLI (TODO)
    ├── spm_normalize/      # Normalization CLI (TODO)
    └── gen_normdata/       # Normalization data generator (TODO)
```

## Implementation Status

### Phase 1: Foundation (Completed)
- [x] Module scaffolding
- [x] Core interfaces (Model, Trainer, Processor)
- [x] Protobuf generation tooling (`cmd/gen_proto`)
- [x] Normalization data codegen (`cmd/gen_normdata`)
- [x] Generated 974KB of normalization rules from upstream
- [x] Generated protobuf bindings (sentencepiece.pb.go, sentencepiece_model.pb.go)

### Phase 2: Runtime (Completed)
- [x] Model loading from protobuf
- [x] Unigram model inference (Viterbi algorithm, lattice-based tokenization)
- [x] BPE model inference (merge-based tokenization)
- [x] Word model inference (word boundary-based tokenization)
- [x] Char model inference (character-level tokenization)
- [x] Normalization runtime with rule application (Unicode normalization, whitespace handling)
- [x] HANA DB integration for model persistence

### Phase 3: Training (Completed)
- [x] Unigram trainer (EM algorithm, vocabulary pruning, Viterbi segmentation)
- [x] BPE trainer (pair frequency counting, iterative merging)
- [x] Word trainer (word frequency counting, vocabulary building)
- [x] Char trainer (character frequency counting, vocabulary building)
- [x] Model building from trained vocabulary (ModelProto construction)
- [x] End-to-end training pipeline (data loading → training → model building)
- [x] HANA DB storage for trained models (save/load/list/delete)

### Phase 4: CLI & Testing (Completed)
- [x] Command-line tools (spm_train, spm_encode, spm_decode, spm_export_vocab)
- [x] Integration tests (end-to-end testing for all 4 model types)
- [x] Model comparison and validation tests
- [x] Advanced sampling algorithms (Unigram sampling, BPE dropout, N-best decoding)
- [x] Vocabulary export utilities (vocab, syms, json, tsv formats)
- [x] Entropy calculation for tokenization uncertainty
- [x] Production-ready with comprehensive test coverage

## Development

### Prerequisites
- Go 1.24+
- protoc (for proto generation)
- protoc-gen-go (for Go proto bindings)

### Code Generation

Generate protobuf bindings and normalization data:
```bash
make generate
# or individually:
make proto     # Generate protobuf bindings
make normdata  # Generate normalization data
```

### Building
```bash
make build
# or directly:
go build ./...
```

### Testing
```bash
make test
# or directly:
go test ./...
```

## Usage Examples

### Basic Encoding/Decoding
```bash
# Train a model
./spm_train --input=data.txt --model_prefix=m --vocab_size=8000

# Encode text
echo "Hello world" | ./spm_encode --model=m.model

# Decode tokens
echo "123 456 789" | ./spm_decode --model=m.model
```

### Advanced Sampling
```bash
# Sample multiple tokenizations (Unigram only)
echo "Hello world" | ./spm_encode --model=m.model --nbest_size=10 --alpha=0.1

# BPE dropout sampling
echo "Hello world" | ./spm_encode --model=bpe.model --alpha=0.1
```

### Vocabulary Export
```bash
# Export vocabulary with scores
./spm_export_vocab --model=m.model --output=vocab.txt --output_format=vocab

# Export as symbol table (piece -> ID)
./spm_export_vocab --model=m.model --output=syms.txt --output_format=syms

# Export as JSON
./spm_export_vocab --model=m.model --output=vocab.json --output_format=json

# Export as TSV with header
./spm_export_vocab --model=m.model --output=vocab.tsv --output_format=tsv
```

### Programmatic Usage
```go
import (
    "context"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece_go/processor"
)

// Load model
proc := processor.New()
proc.Load("model.model")

// Basic encoding
ids, _ := proc.Encode(context.Background(), "Hello world")

// Sampling with configuration
config := processor.SamplingConfig{
    Alpha:              0.1,
    NumSamples:         5,
    WithoutReplacement: true,
    IncludeBest:        true,
}
samples, _ := proc.SampleEncode(context.Background(), "Hello world", config)

// Calculate entropy
entropy, _ := proc.CalculateEntropy(context.Background(), "Hello world", 0.1)

// Export vocabulary
for i := 0; i < proc.VocabSize(); i++ {
    piece, score := proc.GetPieceAndScore(i)
    fmt.Printf("%s\t%f\n", piece, score)
}
```

## Estimated Timeline

- **Phase 1 (Foundation)**: 2-3 weeks
- **Phase 2 (Runtime)**: 6-8 weeks
- **Phase 3 (Training)**: 10-12 weeks
- **Phase 4 (CLI & Testing)**: 4-6 weeks

**Total**: 6-12 months for full feature parity with comprehensive testing.

## References

- Upstream SentencePiece: `../sentencepiece/`
- Original paper: [SentencePiece: A simple and language independent approach to subword tokenization](https://arxiv.org/abs/1808.06226)
