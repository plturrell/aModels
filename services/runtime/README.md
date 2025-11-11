# Runtime Analytics Service

Standalone service that aggregates analytics from the catalog and exposes them as dashboard data.

## Quick start

```bash
# Start the runtime server
RUNTIME_ADDR=:8098 \
CATALOG_URL=http://localhost:8084 \
go run ./cmd/server
```

```bash
# Run unit tests
go test ./...
```

## Integration checklist

1. Launch catalog analytics service (or mock) reachable at `CATALOG_URL`.
2. Start the runtime server using the quick-start command above.
3. Configure the shell backend:

   ```bash
   SHELL_RUNTIME_ENDPOINT=http://localhost:8098 \
   go run ./services/browser/shell/cmd/server
   ```

4. From the browser shell UI or via curl, hit `http://localhost:4173/api/runtime/analytics/dashboard` and confirm a JSON response with `stats` and `templates`.
5. (Optional) Directly hit `http://localhost:8098/analytics/dashboard` to compare payloads.

## Environment variables

| Variable              | Default                    | Description                        |
|-----------------------|----------------------------|------------------------------------|
| `RUNTIME_ADDR`        | `:8098`                    | HTTP listen address                |
| `CATALOG_URL`         | `http://localhost:8084`    | Catalog service base URL           |
| `TRAINING_SERVICE_URL`| `http://localhost:8001`    | Training service base URL          |
| `SEARCH_SERVICE_URL`  | `http://localhost:8000`    | Search service base URL            |

## Endpoints

### Standard Endpoints
- `GET /analytics/dashboard` – aggregated dashboard data
- `GET /analytics/ws` – WebSocket connection for real-time updates
- `GET /healthz` – health check

### Unified Analytics API (v1)
- `GET /api/v1/analytics` – Get analytics data for a specific service or all services
- `POST /api/v1/analytics` – Get analytics with filters and time range
- `GET /api/v1/analytics/system` – System-wide analytics overview
- `GET /api/v1/analytics/docs` – API documentation

### Dashboard Management
- `POST /dashboard/create` – Create a new custom dashboard
- `GET /dashboard/get` – Get a dashboard by ID
- `GET /dashboard/list` – List all dashboards
- `PUT /dashboard/update` – Update a dashboard
- `DELETE /dashboard/delete` – Delete a dashboard
- `POST /dashboard/share` – Share a dashboard with other users
- `GET /dashboard/versions` – Get version history for a dashboard
- `GET /dashboard/export` – Export a dashboard as JSON
- `POST /dashboard/import` – Import a dashboard from JSON

## Integration with shell

Configure the shell backend to proxy the runtime service:

```bash
SHELL_RUNTIME_ENDPOINT=http://localhost:8098 \
go run ./services/browser/shell/cmd/server
```

The shell will expose `/api/runtime/analytics/dashboard` which forwards to the runtime service.

## Testing

```bash
# Unit tests
go test ./...

# Integration tests
go test -tags=integration ./...

# Test coverage
go test -cover ./...
```

## Features

### Service Integration
- **Training Service**: Fetches training metrics (active experiments, completed runs, accuracy)
- **Search Service**: Fetches search analytics (query counts, latency, cache hit rates)
- **Catalog Service**: Fetches dashboard statistics and templates

### Resilience
- **Caching**: 30-second TTL cache for service responses
- **Retry Logic**: Automatic retry with exponential backoff (up to 3 attempts)
- **Graceful Degradation**: Returns default metrics if services are unavailable
- **Stale Cache**: Returns stale cached data if service is down

### Observability
- **Metrics Endpoint**: `GET /metrics` - Request count, error count, latency, error rate
- **Structured Logging**: All requests logged with method, path, status, latency
- **Health Monitoring**: System-wide health status via `/api/v1/analytics/system`

## API Documentation

See [ANALYTICS_API_DOCUMENTATION.md](../ANALYTICS_API_DOCUMENTATION.md) for complete API documentation.
