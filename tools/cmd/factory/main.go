package main

import (
	"ai_benchmarks/scripts/factory"
	"ai_benchmarks/scripts/factory/connectors"
	"ai_benchmarks/scripts/factory/mappers"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
)

// JsonFormatter writes tasks to a JSON file.
type JsonFormatter struct{}

func (f *JsonFormatter) Write(tasks []factory.BenchmarkTask, outputPath string) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ") // Pretty print
	return encoder.Encode(tasks)
}

func main() {
	// --- Command Line Flags ---
	inputFile := flag.String("in", "", "Input file path or URL")
	outputFile := flag.String("out", "", "Output file path for the generated benchmark")
	connectorType := flag.String("connector", "csv", "Connector to use (e.g., 'csv', 'pdf', 'api')")
	mapperType := flag.String("mapper", "boolq", "Mapper to use (e.g., 'boolq', 'arc', 'hellaswag')")
	flag.Parse()

	if *inputFile == "" || *outputFile == "" {
		fmt.Println("Usage: go run main.go --in <input_file> --out <output_file> --connector <type> --mapper <type>")
		flag.PrintDefaults()
		return
	}

	// --- Factory Initialization ---
	var conn factory.Connector
	switch *connectorType {
	case "csv":
		conn = &connectors.CsvConnector{}
	default:
		log.Fatalf("Unsupported connector type: %s", *connectorType)
	}

	var mapper factory.Mapper
	switch *mapperType {
	case "boolq":
		mapper = &mappers.BoolQMapper{}
	case "hellaswag":
		mapper = &mappers.HellaSwagMapper{}
	case "piqa":
		mapper = &mappers.PIQAMapper{}
	case "socialiqa":
		mapper = &mappers.SocialIQAMapper{}
	case "triviaqa":
		mapper = &mappers.TriviaQAMapper{}
	case "arc":
		mapper = mappers.NewARCMapper(42)
	default:
		log.Fatalf("Unsupported mapper type: %s", *mapperType)
	}

	formatter := &JsonFormatter{}

	// --- ETL Pipeline Execution ---
	fmt.Printf("Starting data factory pipeline...\n")
	fmt.Printf("  Connector: %s\n  Mapper: %s\n  Input: %s\n", *connectorType, *mapperType, *inputFile)

	// 1. Extract
	sourceData, err := conn.Connect(*inputFile)
	if err != nil {
		log.Fatalf("Error during extraction: %v", err)
	}
	fmt.Printf("  Extracted %d source records.\n", len(sourceData))

	// 2. Transform
	var allTasks []factory.BenchmarkTask
	for _, data := range sourceData {
		tasks, err := mapper.Map(data)
		if err != nil {
			log.Printf("Warning: failed to map a record: %v", err)
			continue
		}
		allTasks = append(allTasks, tasks...)
	}
	fmt.Printf("  Mapped to %d benchmark tasks.\n", len(allTasks))

	// 3. Load
	if err := formatter.Write(allTasks, *outputFile); err != nil {
		log.Fatalf("Error during formatting/loading: %v", err)
	}

	fmt.Printf("Successfully generated benchmark file: %s\n", *outputFile)
}
