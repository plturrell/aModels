# Docker Deployment Guide

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+

### Launch All Services

```bash
# Start all services (PostgreSQL, gRPC service, FastAPI gateway)
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps
```

Services will be available at:
- **gRPC Service**: `localhost:50055`
- **FastAPI Gateway**: `http://localhost:8000`
- **PostgreSQL**: `localhost:5432`
- **Arrow Flight**: `localhost:8825`

### Test the Deployment

```bash
# Health check via gateway
curl http://localhost:8000/health

# List operations
curl http://localhost:8000/operations

# Log an operation
curl -X POST http://localhost:8000/operations \
  -H "Content-Type: application/json" \
  -d '{
    "library_type": "langchain",
    "operation": "execute",
    "status": "success",
    "latency_ms": 150
  }'
```

### Shutdown

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Building Images

### Build gRPC Service

```bash
docker build -t postgres-lang-service:latest .
```

### Build FastAPI Gateway

```bash
docker build -t postgres-gateway:latest ./gateway
```

### Build Both

```bash
docker-compose build
```

## Production Deployment

### Using Docker Only (No Compose)

#### 1. Start PostgreSQL

```bash
docker run -d \
  --name postgres-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=changeme \
  -e POSTGRES_DB=lang_ops \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:16-alpine
```

#### 2. Run Migrations

```bash
docker run --rm \
  --network container:postgres-db \
  -e POSTGRES_DSN="postgres://postgres:changeme@localhost:5432/lang_ops?sslmode=disable" \
  -v $(pwd)/migrations:/migrations \
  postgres:16-alpine \
  psql -h localhost -U postgres -d lang_ops -f /migrations/001_init.sql
```

#### 3. Start gRPC Service

```bash
docker run -d \
  --name postgres-service \
  --network container:postgres-db \
  -e POSTGRES_DSN="postgres://postgres:changeme@localhost:5432/lang_ops?sslmode=disable" \
  -e GRPC_PORT=50055 \
  -e SERVICE_VERSION=1.0.0 \
  -p 50055:50055 \
  -p 8825:8825 \
  postgres-lang-service:latest
```

#### 4. Start FastAPI Gateway

```bash
docker run -d \
  --name postgres-gateway \
  --network container:postgres-service \
  -e POSTGRES_LANG_SERVICE_ADDR="localhost:50055" \
  -e POSTGRES_LANG_DB_DSN="postgres://postgres:changeme@localhost:5432/lang_ops?sslmode=disable" \
  -e POSTGRES_DB_ALLOW_MUTATIONS=false \
  -p 8000:8000 \
  postgres-gateway:latest
```

## Kubernetes Deployment

### Example Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-lang-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: postgres-lang-service
  template:
    metadata:
      labels:
        app: postgres-lang-service
    spec:
      containers:
      - name: service
        image: postgres-lang-service:1.0.0
        ports:
        - containerPort: 50055
          name: grpc
        - containerPort: 8825
          name: flight
        env:
        - name: POSTGRES_DSN
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: dsn
        - name: GRPC_PORT
          value: "50055"
        - name: SERVICE_VERSION
          value: "1.0.0"
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          exec:
            command:
            - /app/healthcheck
            - -addr=localhost:50055
            - -timeout=2s
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - /app/healthcheck
            - -addr=localhost:50055
            - -timeout=2s
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-lang-service
spec:
  selector:
    app: postgres-lang-service
  ports:
  - name: grpc
    port: 50055
    targetPort: 50055
  - name: flight
    port: 8825
    targetPort: 8825
  type: ClusterIP
```

### Gateway Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-gateway
spec:
  replicas: 2
  selector:
    matchLabels:
      app: postgres-gateway
  template:
    metadata:
      labels:
        app: postgres-gateway
    spec:
      containers:
      - name: gateway
        image: postgres-gateway:1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: POSTGRES_LANG_SERVICE_ADDR
          value: "postgres-lang-service:50055"
        - name: POSTGRES_LANG_DB_DSN
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: dsn
        - name: POSTGRES_DB_ALLOW_MUTATIONS
          value: "false"
        resources:
          requests:
            memory: "64Mi"
            cpu: "50m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-gateway
spec:
  selector:
    app: postgres-gateway
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Environment Variables

### gRPC Service

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_DSN` | *required* | PostgreSQL connection string |
| `GRPC_PORT` | `50055` | gRPC server port |
| `FLIGHT_ADDR` | `:8825` | Arrow Flight server address |
| `FLIGHT_MAX_ROWS` | `200` | Max rows per Flight request |
| `SERVICE_VERSION` | `0.1.0` | Service version |
| `POSTGRES_MAX_OPEN_CONN` | `20` | Max open connections |
| `POSTGRES_MAX_IDLE_CONN` | `10` | Max idle connections |
| `POSTGRES_CONN_MAX_LIFETIME_MINUTES` | `30` | Connection max lifetime |
| `SHUTDOWN_GRACE_PERIOD_SECONDS` | `15` | Graceful shutdown timeout |

### FastAPI Gateway

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_LANG_SERVICE_ADDR` | `localhost:50055` | gRPC service address |
| `POSTGRES_LANG_DB_DSN` | *optional* | Direct DB access (admin endpoints) |
| `POSTGRES_DB_ALLOW_MUTATIONS` | `false` | Allow write SQL queries |
| `POSTGRES_DB_DEFAULT_LIMIT` | `100` | Default query result limit |
| `POSTGRES_LANG_GATEWAY_CORS` | `*` | CORS allowed origins |
| `SERVICE_VERSION` | `0.1.0` | Service version |
| `FASTAPI_HOST` | `0.0.0.0` | FastAPI bind host |
| `FASTAPI_PORT` | `8000` | FastAPI bind port |
| `FASTAPI_RELOAD` | `false` | Enable auto-reload (dev only) |

## Security Considerations

### Production Checklist

- [ ] Change default PostgreSQL credentials
- [ ] Use SSL/TLS for database connections (`sslmode=require`)
- [ ] Disable database admin endpoints (`unset POSTGRES_LANG_DB_DSN`)
- [ ] Configure CORS origins restrictively
- [ ] Use secrets management (K8s secrets, Vault)
- [ ] Enable authentication (see SECURITY.md)
- [ ] Run as non-root user (already configured)
- [ ] Set resource limits
- [ ] Enable network policies
- [ ] Use private container registries

### Secrets Management

#### Docker Secrets

```bash
# Create secrets
echo "postgres://user:pass@host/db" | docker secret create postgres_dsn -

# Use in service
docker service create \
  --name postgres-service \
  --secret postgres_dsn \
  --env POSTGRES_DSN_FILE=/run/secrets/postgres_dsn \
  postgres-lang-service:latest
```

#### Kubernetes Secrets

```bash
# Create secret
kubectl create secret generic postgres-credentials \
  --from-literal=dsn="postgres://user:pass@host/db?sslmode=require"

# Reference in deployment (see K8s manifests above)
```

## Monitoring

### Prometheus Metrics (Future Enhancement)

```yaml
# Add to docker-compose.yml
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
```

### Health Checks

```bash
# gRPC service
grpcurl -plaintext localhost:50055 agentic.layer4.postgres.v1.PostgresLangService/HealthCheck

# Gateway
curl http://localhost:8000/health

# PostgreSQL
docker exec postgres-lang-db pg_isready -U postgres
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs postgres-service

# Verify database connectivity
docker exec postgres-service /app/healthcheck -addr=localhost:50055 -timeout=2s

# Check environment variables
docker exec postgres-service env | grep POSTGRES
```

### Database Connection Issues

```bash
# Test database connection
docker exec postgres-lang-db psql -U postgres -c "SELECT 1"

# Check network connectivity
docker exec postgres-service ping postgres

# Verify DSN format
# Correct: postgres://user:pass@host:5432/db?sslmode=disable
```

### Gateway Can't Connect to gRPC

```bash
# Verify gRPC service is running
docker-compose ps postgres-service

# Test gRPC connectivity
docker exec gateway grpcurl -plaintext postgres-service:50055 list

# Check service discovery
docker exec gateway nslookup postgres-service
```

## Performance Tuning

### Database Connection Pool

```bash
# Increase connections for high load
export POSTGRES_MAX_OPEN_CONN=50
export POSTGRES_MAX_IDLE_CONN=25
```

### Resource Limits

```yaml
# In docker-compose.yml
services:
  postgres-service:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

## Backup and Restore

### Backup Database

```bash
docker exec postgres-lang-db pg_dump -U postgres lang_ops > backup.sql
```

### Restore Database

```bash
docker exec -i postgres-lang-db psql -U postgres lang_ops < backup.sql
```

## Development Mode

```bash
# Start with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Mount local code
# Create docker-compose.dev.yml:
version: '3.8'
services:
  postgres-service:
    build:
      context: .
      target: builder
    volumes:
      - .:/app
    command: go run ./cmd/server
  
  gateway:
    volumes:
      - ./gateway:/app/gateway
    environment:
      FASTAPI_RELOAD: "true"
```
