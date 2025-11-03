package factory

// SourceData represents the raw, unstructured data from a connector.
type SourceData struct {
	Type    string // e.g., "text", "table", "json"
	Content interface{}
	Meta    map[string]string // e.g., file name, URL
}

// BenchmarkTask represents a single item for a benchmark (e.g., one question).
type BenchmarkTask interface{}

// Connector is the interface for extracting data from a source.
type Connector interface {
	Connect(pathOrURL string) ([]SourceData, error)
}

// Mapper is the interface for transforming source data into benchmark tasks.
type Mapper interface {
	Map(data SourceData) ([]BenchmarkTask, error)
}

// Formatter writes the generated tasks to disk.
type Formatter interface {
	Write(tasks []BenchmarkTask, outputPath string) error
}
