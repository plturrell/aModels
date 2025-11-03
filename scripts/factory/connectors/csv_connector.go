package connectors

import (
	"ai_benchmarks/scripts/factory"
	"encoding/csv"
	"os"
)

// CsvConnector reads data from a CSV file.
type CsvConnector struct{}

func (c *CsvConnector) Connect(path string) ([]factory.SourceData, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	if len(records) < 2 { // Must have header + at least one data row
		return []factory.SourceData{}, nil
	}

	header := records[0]
	var sources []factory.SourceData

	for _, record := range records[1:] {
		rowMap := make(map[string]string)
		for i, value := range record {
			rowMap[header[i]] = value
		}
		sources = append(sources, factory.SourceData{
			Type:    "table_row",
			Content: rowMap,
			Meta:    map[string]string{"source_file": path},
		})
	}

	return sources, nil
}
