package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	// Get the current working directory
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting current directory: %v\n", err)
		os.Exit(1)
	}

	// Source proto files from upstream sentencepiece (sibling directory)
	// From sentencepiece/ go to ../sentencepiece/src/
	protoDir := filepath.Join(cwd, "..", "sentencepiece", "src")
	protoFiles := []string{
		filepath.Join(protoDir, "sentencepiece.proto"),
		filepath.Join(protoDir, "sentencepiece_model.proto"),
	}

	// Output directory for generated Go files
	outDir := filepath.Join(cwd, "internal", "proto")

	// Ensure output directory exists
	if err := os.MkdirAll(outDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output directory: %v\n", err)
		os.Exit(1)
	}

	// Check if protoc is available
	if _, err := exec.LookPath("protoc"); err != nil {
		fmt.Fprintf(os.Stderr, "Error: protoc not found in PATH\n")
		fmt.Fprintf(os.Stderr, "Please install protoc: https://grpc.io/docs/protoc-installation/\n")
		os.Exit(1)
	}

	// Check if protoc-gen-go is available
	if _, err := exec.LookPath("protoc-gen-go"); err != nil {
		fmt.Fprintf(os.Stderr, "Error: protoc-gen-go not found in PATH\n")
		fmt.Fprintf(os.Stderr, "Please install: go install google.golang.org/protobuf/cmd/protoc-gen-go@latest\n")
		os.Exit(1)
	}

	// Generate Go code for each proto file
	for _, protoFile := range protoFiles {
		fmt.Printf("Generating Go code for %s...\n", filepath.Base(protoFile))

		// Build protoc command
		// Use M option to specify Go package mapping since upstream protos don't have go_package
		cmd := exec.Command("protoc",
			fmt.Sprintf("--proto_path=%s", protoDir),
			fmt.Sprintf("--go_out=%s", outDir),
			"--go_opt=paths=source_relative",
			fmt.Sprintf("--go_opt=Msentencepiece.proto=github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"),
			fmt.Sprintf("--go_opt=Msentencepiece_model.proto=github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/models/sentencepiece/internal/proto"),
			protoFile,
		)

		// Run protoc
		output, err := cmd.CombinedOutput()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error running protoc: %v\n", err)
			fmt.Fprintf(os.Stderr, "Output: %s\n", string(output))
			os.Exit(1)
		}

		fmt.Printf("  ✓ Generated %s\n", filepath.Base(protoFile))
	}

	fmt.Println("\nProto generation complete!")
	fmt.Printf("Generated files in: %s\n", outDir)
}
