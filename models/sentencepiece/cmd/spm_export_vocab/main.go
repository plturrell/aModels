package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/processor"
)

func main() {
	// Command-line flags
	var (
		modelFile    = flag.String("model", "", "Model file path")
		outputFile   = flag.String("output", "", "Output file (default: stdout)")
		outputFormat = flag.String("output_format", "vocab", "Output format (vocab, syms)")
	)

	flag.Parse()

	if *modelFile == "" {
		fmt.Fprintf(os.Stderr, "Error: --model is required\n")
		flag.Usage()
		os.Exit(1)
	}

	// Load model
	proc := processor.New()
	if err := proc.Load(*modelFile); err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}

	// Setup output
	var out *os.File
	if *outputFile == "" {
		out = os.Stdout
	} else {
		var err error
		out, err = os.Create(*outputFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating output file: %v\n", err)
			os.Exit(1)
		}
		defer out.Close()
	}

	writer := bufio.NewWriter(out)
	defer writer.Flush()

	// Export vocabulary
	if err := exportVocab(proc, writer, *outputFormat); err != nil {
		fmt.Fprintf(os.Stderr, "Error exporting vocabulary: %v\n", err)
		os.Exit(1)
	}
}

func exportVocab(proc *processor.Processor, writer *bufio.Writer, format string) error {
	vocabSize := proc.VocabSize()

	switch format {
	case "vocab":
		// Output: piece \t score
		for i := 0; i < vocabSize; i++ {
			piece, score := proc.GetPieceAndScore(i)
			fmt.Fprintf(writer, "%s\t%f\n", piece, score)
		}

	case "syms":
		// Output: piece \t index
		for i := 0; i < vocabSize; i++ {
			piece, _ := proc.GetPieceAndScore(i)
			fmt.Fprintf(writer, "%s\t%d\n", piece, i)
		}

	case "json":
		// Output: JSON format
		fmt.Fprintf(writer, "[\n")
		for i := 0; i < vocabSize; i++ {
			piece, score := proc.GetPieceAndScore(i)
			if i > 0 {
				fmt.Fprintf(writer, ",\n")
			}
			fmt.Fprintf(writer, "  {\"id\": %d, \"piece\": %q, \"score\": %f}", i, piece, score)
		}
		fmt.Fprintf(writer, "\n]\n")

	case "tsv":
		// Output: TSV with header
		fmt.Fprintf(writer, "id\tpiece\tscore\n")
		for i := 0; i < vocabSize; i++ {
			piece, score := proc.GetPieceAndScore(i)
			fmt.Fprintf(writer, "%d\t%s\t%f\n", i, piece, score)
		}

	default:
		return fmt.Errorf("unsupported output format: %s (use: vocab, syms, json, tsv)", format)
	}

	return nil
}
