package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/processor"
)

func main() {
	// Command-line flags
	var (
		modelFile  = flag.String("model", "", "Model file path")
		outputFile = flag.String("output", "", "Output file (default: stdout)")
		inputType  = flag.String("input_format", "id", "Input format (id, piece)")
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

	// Read from stdin and decode
	scanner := bufio.NewScanner(os.Stdin)
	ctx := context.Background()

	for scanner.Scan() {
		line := scanner.Text()
		
		if *inputType == "piece" {
			// Decode from pieces (not commonly used)
			pieces := strings.Fields(line)
			// TODO: Implement piece-to-id conversion
			fmt.Fprintln(out, strings.Join(pieces, ""))
		} else {
			// Parse IDs
			idStrs := strings.Fields(line)
			ids := make([]int, 0, len(idStrs))
			for _, idStr := range idStrs {
				id, err := strconv.Atoi(idStr)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Error parsing ID '%s': %v\n", idStr, err)
					continue
				}
				ids = append(ids, id)
			}
			
			text, err := proc.Decode(ctx, ids)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error decoding: %v\n", err)
				continue
			}
			
			// Remove meta symbols
			text = strings.ReplaceAll(text, "▁", " ")
			text = strings.TrimSpace(text)
			
			fmt.Fprintln(out, text)
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		os.Exit(1)
	}
}
