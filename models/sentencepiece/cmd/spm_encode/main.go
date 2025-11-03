package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/processor"
)

func main() {
	// Command-line flags
	var (
		modelFile  = flag.String("model", "", "Model file path")
		outputFile = flag.String("output", "", "Output file (default: stdout)")
		outputType = flag.String("output_format", "id", "Output format (id, piece)")
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

	// Read from stdin and encode
	scanner := bufio.NewScanner(os.Stdin)
	ctx := context.Background()

	for scanner.Scan() {
		line := scanner.Text()
		
		if *outputType == "piece" {
			pieces, err := proc.EncodeAsPieces(ctx, line)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error encoding: %v\n", err)
				continue
			}
			fmt.Fprintln(out, strings.Join(pieces, " "))
		} else {
			ids, err := proc.Encode(ctx, line)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error encoding: %v\n", err)
				continue
			}
			
			// Convert to strings
			idStrs := make([]string, len(ids))
			for i, id := range ids {
				idStrs[i] = fmt.Sprintf("%d", id)
			}
			fmt.Fprintln(out, strings.Join(idStrs, " "))
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		os.Exit(1)
	}
}
