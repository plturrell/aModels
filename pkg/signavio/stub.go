package signavio

import (
    "context"
    "errors"
    "fmt"
    "path/filepath"
    "strings"
    "time"
)

// UploadRequest describes the parameters required to push data into SAP Signavio.
type UploadRequest struct {
    // Dataset identifies the logical target inside the Signavio workspace (for example, subject slug).
    Dataset string
    // FilePath points to the prepared CSV, TSV, or XES artifact on disk.
    FilePath string
    // SchemaPath optionally references an Avro schema when using the ingestion API.
    SchemaPath string
    // PrimaryKeys lists the column names that constitute the ingestion table primary key.
    PrimaryKeys []string
}

// ODataQuery defines the filter used when retrieving analytical results from SAP Signavio.
type ODataQuery struct {
    ViewName string
    // SelectFields restricts which columns are requested. Leave empty for all fields.
    SelectFields []string
    // Filter is an optional OData filter expression.
    Filter string
    // OrderBy is an optional OData order expression.
    OrderBy string
    // Top limits the number of rows returned per page (Signavio is fixed at 10k, but we expose for completeness).
    Top int
}

// StubClient imitates the behaviour of a Signavio API client without making any network calls.
// It is intended for local and manual testing workflows until the real integration is wired up.
type StubClient struct {
    // Name identifies the stub instance for logging or debugging purposes.
    Name string
}

// NewStubClient constructs a StubClient with the provided name.
func NewStubClient(name string) *StubClient {
    if name == "" {
        name = "signavio-stub"
    }
    return &StubClient{Name: name}
}

// Upload simulates sending a file to Signavio and returns a deterministic stub identifier.
func (c *StubClient) Upload(ctx context.Context, req UploadRequest) (string, error) {
    if strings.TrimSpace(req.FilePath) == "" {
        return "", errors.New("missing file path for stub upload")
    }

    dataset := strings.TrimSpace(req.Dataset)
    if dataset == "" {
        dataset = "default-dataset"
    }

    select {
    case <-ctx.Done():
        return "", ctx.Err()
    case <-time.After(25 * time.Millisecond):
        // Simulate a small amount of processing latency.
    }

    slug := filepath.Base(req.FilePath)
    return fmt.Sprintf("stub://signavio/uploads/%s/%s", dataset, slug), nil
}

// FetchOData simulates retrieving an analytical dataset from Signavio by returning a canned identifier.
func (c *StubClient) FetchOData(ctx context.Context, query ODataQuery) (string, error) {
    if strings.TrimSpace(query.ViewName) == "" {
        return "", errors.New("missing OData view name for stub fetch")
    }

    select {
    case <-ctx.Done():
        return "", ctx.Err()
    case <-time.After(15 * time.Millisecond):
        // Simulate remote call latency.
    }

    return fmt.Sprintf("stub://signavio/odata/%s?fields=%d", query.ViewName, len(query.SelectFields)), nil
}
