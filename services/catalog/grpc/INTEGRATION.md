# Murex API Integration Guide

This guide explains how to integrate the break detection service into Murex's API using both REST and gRPC.

## Service Architecture

The break detection service runs as a standalone service that exposes:
- **REST API** on port 8084 (default)
- **gRPC API** on port 8085 (default)

Both APIs provide the same functionality, allowing Murex to choose the integration method that best fits their architecture.

## Quick Start

### 1. Generate Proto Code

Before using gRPC, generate the Go code from the proto file:

```bash
cd services/catalog
make proto
```

Or manually:

```bash
cd services/catalog/grpc
./generate_proto.sh
```

### 2. Start the Service

The service starts both HTTP and gRPC servers automatically:

```bash
export CATALOG_DATABASE_URL="postgres://user:pass@localhost/catalog"
export GRPC_PORT=8085
export PORT=8084
./bin/catalog
```

## Integration Options

### Option 1: gRPC Integration (Recommended for Murex)

gRPC provides better performance, type safety, and is ideal for service-to-service communication.

#### Add to Murex's go.mod

```go
require (
    google.golang.org/grpc v1.76.0
    google.golang.org/protobuf v1.34.2
    github.com/plturrell/aModels/services/catalog/gen/breakdetectionpb v0.0.0
)

replace github.com/plturrell/aModels/services/catalog/gen/breakdetectionpb => /path/to/aModels/services/catalog/gen/breakdetectionpb
```

#### Client Code Example

```go
package main

import (
    "context"
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
    breakdetectionpb "github.com/plturrell/aModels/services/catalog/gen/breakdetectionpb"
)

type BreakDetectionClient struct {
    client breakdetectionpb.BreakDetectionServiceClient
    conn   *grpc.ClientConn
}

func NewBreakDetectionClient(addr string) (*BreakDetectionClient, error) {
    conn, err := grpc.Dial(
        addr,
        grpc.WithTransportCredentials(insecure.NewCredentials()),
        grpc.WithTimeout(30*time.Second),
    )
    if err != nil {
        return nil, err
    }

    return &BreakDetectionClient{
        client: breakdetectionpb.NewBreakDetectionServiceClient(conn),
        conn:   conn,
    }, nil
}

func (c *BreakDetectionClient) Close() error {
    return c.conn.Close()
}

// DetectBreaks is called during Murex version migration
func (c *BreakDetectionClient) DetectBreaks(ctx context.Context, systemName, baselineID, detectionType string) (*breakdetectionpb.DetectBreaksResponse, error) {
    req := &breakdetectionpb.DetectBreaksRequest{
        SystemName:    systemName,
        BaselineId:    baselineID,
        DetectionType: detectionType,
    }
    return c.client.DetectBreaks(ctx, req)
}
```

#### Murex Workflow Integration

```go
// In Murex version migration workflow
func (w *MurexMigrationWorkflow) RunBreakDetection(ctx context.Context) error {
    client, err := NewBreakDetectionClient("break-detection-service:8085")
    if err != nil {
        return err
    }
    defer client.Close()

    // Step 1: Create baseline
    baseline, err := w.createBaseline(ctx, client)
    if err != nil {
        return err
    }

    // Step 2: After new version ingestion, detect breaks
    result, err := client.DetectBreaks(ctx, "sap_fioneer", baseline.BaselineId, "finance")
    if err != nil {
        return err
    }

    // Step 3: Process breaks
    return w.processBreaks(ctx, result.Breaks)
}
```

### Option 2: REST API Integration

REST API is simpler to integrate and can be called from any language.

#### Example: HTTP Client

```go
type BreakDetectionRESTClient struct {
    baseURL string
    httpClient *http.Client
}

func (c *BreakDetectionRESTClient) DetectBreaks(ctx context.Context, req *DetectBreaksRequest) (*DetectBreaksResponse, error) {
    url := fmt.Sprintf("%s/catalog/break-detection/detect", c.baseURL)
    
    jsonData, err := json.Marshal(req)
    if err != nil {
        return nil, err
    }

    httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonData))
    if err != nil {
        return nil, err
    }
    httpReq.Header.Set("Content-Type", "application/json")

    resp, err := c.httpClient.Do(httpReq)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result DetectBreaksResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }

    return &result, nil
}
```

#### Example: cURL

```bash
# Create baseline
curl -X POST http://break-detection-service:8084/catalog/break-detection/baselines \
  -H "Content-Type: application/json" \
  -d '{
    "system_name": "sap_fioneer",
    "version": "v1.0.0",
    "snapshot_type": "full",
    "snapshot_data": {...}
  }'

# Detect breaks
curl -X POST http://break-detection-service:8084/catalog/break-detection/detect \
  -H "Content-Type: application/json" \
  -d '{
    "system_name": "sap_fioneer",
    "baseline_id": "baseline-sap-fioneer-v1.0.0",
    "detection_type": "finance"
  }'
```

## Murex Version Migration Workflow

The break detection service is designed to integrate into Murex's version migration workflow:

1. **Before Migration**: Create baseline snapshot
2. **After Migration**: Detect breaks in finance, capital, liquidity, regulatory systems
3. **Process Results**: Use AI-enhanced break information for remediation

### Complete Example

```go
func MurexVersionMigration(ctx context.Context) error {
    // Connect to break detection service
    client, err := NewBreakDetectionClient("break-detection-service:8085")
    if err != nil {
        return err
    }
    defer client.Close()

    // 1. Create baseline before migration
    baselineReq := &breakdetectionpb.CreateBaselineRequest{
        SystemName: "sap_fioneer",
        Version: "v1.0.0",
        SnapshotType: "full",
        SnapshotData: getSAPFioneerSnapshot(),
        CreatedBy: "murex-workflow",
    }
    baseline, err := client.CreateBaseline(ctx, baselineReq)
    if err != nil {
        return err
    }

    // 2. Perform version migration (your existing logic)
    if err := performMurexMigration(ctx); err != nil {
        return err
    }

    // 3. Detect breaks in all systems
    systems := []struct{
        name string
        detectionType string
    }{
        {"sap_fioneer", "finance"},
        {"bcrs", "capital"},
        {"rco", "liquidity"},
        {"axiomsl", "regulatory"},
    }

    for _, sys := range systems {
        result, err := client.DetectBreaks(ctx, sys.name, baseline.Baseline.BaselineId, sys.detectionType)
        if err != nil {
            log.Printf("Warning: Break detection failed for %s: %v", sys.name, err)
            continue
        }

        if result.TotalBreaksDetected > 0 {
            log.Printf("Detected %d breaks in %s", result.TotalBreaksDetected, sys.name)
            processBreaks(ctx, result.Breaks)
        }
    }

    return nil
}
```

## API Endpoints

### gRPC Endpoints

- `DetectBreaks` - Detect breaks in a system
- `CreateBaseline` - Create baseline snapshot
- `GetBaseline` - Get baseline by ID
- `ListBaselines` - List baselines for a system
- `ListBreaks` - List breaks for a system
- `GetBreak` - Get break by ID
- `HealthCheck` - Health check

### REST Endpoints

- `POST /catalog/break-detection/detect` - Detect breaks
- `POST /catalog/break-detection/baselines` - Create baseline
- `GET /catalog/break-detection/baselines/{id}` - Get baseline
- `GET /catalog/break-detection/baselines?system={name}` - List baselines
- `GET /catalog/break-detection/breaks?system={name}&status={status}` - List breaks
- `GET /catalog/break-detection/breaks/{id}` - Get break

## Configuration

Environment variables:

- `GRPC_PORT` - gRPC server port (default: 8085)
- `PORT` - HTTP server port (default: 8084)
- `CATALOG_DATABASE_URL` - PostgreSQL connection string
- `DEEP_RESEARCH_URL` - Deep Research service URL
- `LOCALAI_URL` - LocalAI service URL
- `EXTRACT_SERVICE_URL` - Extract service URL for search
- `SAP_FIONEER_URL` - SAP Fioneer API URL
- `BCRS_URL` - BCRS API URL
- `RCO_URL` - RCO API URL
- `AXIOMSL_URL` - AxiomSL API URL

## Deployment

### Docker

```dockerfile
FROM golang:1.23 AS builder
WORKDIR /app
COPY . .
RUN make proto && make build

FROM alpine:latest
RUN apk --no-cache add ca-certificates
COPY --from=builder /app/bin/catalog /catalog
EXPOSE 8084 8085
CMD ["/catalog"]
```

### Kubernetes

```yaml
apiVersion: v1
kind: Service
metadata:
  name: break-detection-service
spec:
  ports:
    - name: http
      port: 8084
      targetPort: 8084
    - name: grpc
      port: 8085
      targetPort: 8085
  selector:
    app: break-detection
```

## Error Handling

The service returns gRPC status codes:

- `codes.InvalidArgument` - Invalid request parameters
- `codes.NotFound` - Baseline or break not found
- `codes.Internal` - Internal server error
- `codes.Unimplemented` - Feature not yet implemented

## Performance

- gRPC provides better performance for service-to-service calls
- Supports streaming (can be added for real-time break notifications)
- Type-safe with proto definitions
- Automatic code generation for multiple languages

