# aModels Services Startup Guide

Complete guide for starting and managing all aModels services with proper dependency handling, health checks, and orchestration.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Startup Modes](#startup-modes)
- [Service Profiles](#service-profiles)
- [Management Commands](#management-commands)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## Quick Start

### Option 1: Using Make (Recommended)

```bash
# Start development environment (Docker)
make -f Makefile.services quick-start

# Start minimal environment
make -f Makefile.services minimal

# Start full stack with all services
make -f Makefile.services full

# Check service status
make -f Makefile.services status

# Run health checks
make -f Makefile.services health
```

### Option 2: Using Docker Compose

```bash
# Start all services
./scripts/docker-manager.sh start full

# Start specific service groups
./scripts/docker-manager.sh start infrastructure
./scripts/docker-manager.sh start core
./scripts/docker-manager.sh start app

# View logs
./scripts/docker-manager.sh logs
```

### Option 3: Native Startup

```bash
# Start with native mode (no Docker for application services)
MODE=native PROFILE=development ./scripts/start-system.sh start

# Stop all services
./scripts/start-system.sh stop
```

## Architecture Overview

### Service Layers

The system is organized in four layers with clear dependencies:

```
┌─────────────────────────────────────────┐
│     Application Services                │
│  Extract, Graph, Runtime, Orchestration │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│     Core Services                       │
│  LocalAI, Catalog, Transformers         │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│     Infrastructure                      │
│  Redis, PostgreSQL, Neo4j, Elasticsearch│
└─────────────────────────────────────────┘
```

### Service Dependencies

```yaml
Infrastructure (no dependencies):
  - Redis: 6379
  - PostgreSQL: 5432
  - Neo4j: 7474, 7687
  - Elasticsearch: 9200, 9300

Core Services:
  - LocalAI: 8081 (requires: Redis, PostgreSQL)
  - Transformers: 9090 (optional GPU service)
  - Catalog: 8084 (requires: Neo4j, Redis, PostgreSQL)

Application Services:
  - Extract: 8083 (requires: Neo4j, PostgreSQL, Catalog)
  - Graph: 8080 (requires: Neo4j, LocalAI)
  - Search: 8090 (requires: Elasticsearch, LocalAI)
  - DeepAgents: 9004 (requires: PostgreSQL, Redis, LocalAI)
  - Runtime: 8098 (requires: Catalog)
  - Orchestration: 8085 (standalone)
  - Training: 8087 (requires: PostgreSQL, Redis, LocalAI)
  - DMS: 8096 (requires: PostgreSQL, Redis, Neo4j, Extract, Catalog)
  - Regulatory Audit: 8099 (requires: Neo4j, LocalAI)
```

## Startup Modes

### Docker Mode (Default)

All services run in Docker containers with GPU support.

```bash
MODE=docker ./scripts/start-system.sh start
# or
make -f Makefile.services docker-start
```

**Advantages:**
- Isolated environments
- GPU support
- Consistent across machines
- Easy cleanup

**Requirements:**
- Docker & Docker Compose
- GPU drivers (for GPU services)

### Native Mode

Application services run natively, infrastructure in Docker.

```bash
MODE=native ./scripts/start-system.sh start
```

**Advantages:**
- Faster development iteration
- Direct log access
- Easier debugging

**Requirements:**
- Go 1.21+
- Python 3.11+
- Docker (for infrastructure)

### Hybrid Mode

Mix of Docker and native services.

```bash
MODE=hybrid ./scripts/start-system.sh start
```

## Service Profiles

### Minimal Profile

Infrastructure + Core services only. Best for development of individual services.

```bash
PROFILE=minimal ./scripts/start-system.sh start
```

**Services:**
- Redis
- PostgreSQL
- Neo4j
- LocalAI
- Catalog

**Use Cases:**
- Testing catalog integration
- Developing new services
- Minimal resource usage

### Development Profile (Default)

Full development stack without GPU-intensive services.

```bash
PROFILE=development ./scripts/start-system.sh start
```

**Services:**
- All Minimal services
- Elasticsearch
- Extract
- Search Inference
- Runtime
- Orchestration

**Use Cases:**
- Full-stack development
- Integration testing
- Local development

### Full Profile

Complete system with all services including GPU-intensive ones.

```bash
PROFILE=full ./scripts/start-system.sh start
```

**Services:**
- All Development services
- Transformers (GPU)
- Graph (GPU)
- DeepAgents
- Training
- DMS
- PostgreSQL Lang
- Regulatory Audit

**Use Cases:**
- Production testing
- Performance benchmarking
- End-to-end testing

## Management Commands

### Starting Services

```bash
# Quick start with Make
make -f Makefile.services quick-start

# Start specific profile
PROFILE=minimal ./scripts/start-system.sh start

# Start specific services only
./scripts/docker-manager.sh start redis postgres neo4j

# Start service groups
./scripts/docker-manager.sh start infrastructure
```

### Stopping Services

```bash
# Stop all services
make -f Makefile.services stop

# Stop specific services
./scripts/docker-manager.sh stop localai catalog

# Graceful shutdown
./scripts/start-system.sh stop
```

### Restarting Services

```bash
# Restart all
make -f Makefile.services restart

# Restart specific service
./scripts/docker-manager.sh restart catalog
```

### Viewing Logs

```bash
# Tail all logs
make -f Makefile.services logs

# Follow specific service logs
./scripts/docker-manager.sh logs catalog

# Live tail
FOLLOW=true ./scripts/docker-manager.sh logs localai

# Native service logs
tail -f logs/startup/*.log
```

### Health Checks

```bash
# Run comprehensive health check
make -f Makefile.services health

# Or directly
./scripts/health-check.sh
```

**Health Check Output:**
```
═══════════════════════════════════════════
aModels System Health Check
═══════════════════════════════════════════

✓ Redis (port 6379)
✓ PostgreSQL (port 5432)
✓ Neo4j HTTP
✓ Neo4j Bolt (port 7687)
✓ Elasticsearch
✓ LocalAI
✓ Catalog
✓ Extract
...
```

### Service Status

```bash
# View status
make -f Makefile.services status

# Or
./scripts/start-system.sh status
```

## Troubleshooting

### Common Issues

#### 1. Port Already in Use

```bash
# Check what's using a port
lsof -i :8084

# Kill process
kill $(lsof -t -i :8084)
```

#### 2. Service Won't Start

```bash
# Check logs
./scripts/docker-manager.sh logs SERVICE_NAME

# Or for native services
cat logs/startup/SERVICE_NAME.log

# Check dependencies
./scripts/health-check.sh
```

#### 3. Database Connection Issues

```bash
# Verify PostgreSQL is ready
pg_isready -h localhost -p 5432 -U postgres

# Check Neo4j
curl http://localhost:7474

# Verify Redis
redis-cli ping
```

#### 4. Docker Issues

```bash
# Rebuild containers
make -f Makefile.services docker-build

# Clean and restart
make -f Makefile.services docker-clean
make -f Makefile.services docker-start

# View Docker logs
docker-compose -f infrastructure/docker/brev/docker-compose.yml logs
```

#### 5. Permissions Issues

```bash
# Fix script permissions
make -f Makefile.services fix-permissions

# Or manually
chmod +x scripts/*.sh scripts/lib/*.sh
```

### Debug Mode

```bash
# Get debug information
make -f Makefile.services debug

# Shows:
# - Docker version
# - Running containers
# - Service health
# - System resources
```

### Service-Specific Troubleshooting

#### LocalAI Not Loading Models

```bash
# Check models directory
ls -la models/

# Verify model configuration
./scripts/docker-manager.sh exec localai ls /models

# Check LocalAI logs
./scripts/docker-manager.sh logs localai
```

#### Neo4j Authentication Failure

Default credentials:
- Username: `neo4j`
- Password: `amodels123`

```bash
# Reset Neo4j (WARNING: deletes data)
docker-compose -f infrastructure/docker/brev/docker-compose.yml stop neo4j
docker volume rm amodels_neo4jdata
docker-compose -f infrastructure/docker/brev/docker-compose.yml up -d neo4j
```

#### Catalog Service Won't Connect

```bash
# Verify all dependencies are running
./scripts/health-check.sh

# Check environment variables
./scripts/docker-manager.sh exec catalog env | grep NEO4J

# Restart with fresh config
./scripts/docker-manager.sh restart catalog
```

## Advanced Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Infrastructure
REDIS_URL=redis://localhost:6379/0
POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/amodels?sslmode=disable
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=amodels123

# Services
LOCALAI_URL=http://localhost:8081
CATALOG_URL=http://localhost:8084
EXTRACT_SERVICE_URL=http://localhost:8083

# Feature Flags
USE_MULTIMODAL_EXTRACTION=true
DEEPAGENTS_ENABLED=true
USE_GNN_PATTERNS=false
```

### Custom Profiles

Edit `config/services.yaml` to create custom profiles:

```yaml
profiles:
  my_custom:
    description: "Custom development setup"
    services: [redis, postgres, neo4j, localai, catalog, extract]
```

Then use:

```bash
PROFILE=my_custom ./scripts/start-system.sh start
```

### Resource Limits

Edit `infrastructure/docker/brev/docker-compose.yml`:

```yaml
services:
  localai:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          memory: 4G
```

### Startup Order Customization

Modify `scripts/start-system.sh` to change startup order:

```bash
start_all_services() {
    # Add your custom order
    start_redis
    start_postgres
    start_custom_service
    start_neo4j
    # ...
}
```

## Monitoring & Observability

### Service URLs

Once started, access services at:

- **Catalog**: http://localhost:8084
- **LocalAI**: http://localhost:8081/v1/models
- **Neo4j Browser**: http://localhost:7474
- **Elasticsearch**: http://localhost:9200
- **Runtime Analytics**: http://localhost:8098/analytics/dashboard
- **Regulatory Audit**: http://localhost:8099
- **Extract**: http://localhost:8083
- **Orchestration**: http://localhost:8085

### Metrics

```bash
# Runtime metrics
curl http://localhost:8098/metrics

# Service-specific metrics
./scripts/docker-manager.sh exec catalog curl localhost:8084/metrics
```

### Log Aggregation

All logs are centralized in:
- Docker logs: `docker-compose logs`
- Native logs: `logs/startup/*.log`
- Application logs: `logs/*.log`

## Production Deployment

### Best Practices

1. **Use Docker Mode**: More reliable and consistent
2. **Enable Health Checks**: Monitor service health continuously
3. **Set Resource Limits**: Prevent resource exhaustion
4. **Use Persistent Volumes**: Ensure data persistence
5. **Enable Authentication**: Secure all services
6. **Monitor Logs**: Set up log aggregation (ELK, Loki)
7. **Backup Databases**: Regular backups of PostgreSQL and Neo4j

### Production Checklist

- [ ] All services have health checks
- [ ] Resource limits configured
- [ ] Persistent volumes mounted
- [ ] Authentication enabled
- [ ] TLS/SSL configured
- [ ] Monitoring set up
- [ ] Backup strategy defined
- [ ] Disaster recovery plan
- [ ] Load testing completed

## Getting Help

### Documentation

- Main README: `README.md`
- Service-specific docs: `services/*/README.md`
- Docker setup: `infrastructure/docker/README.md`
- Regulatory framework: `services/regulatory/README.md`

### Support

```bash
# List available commands
make -f Makefile.services help

# View service list
make -f Makefile.services list-services

# Debug information
make -f Makefile.services debug
```

### Common Commands Reference

```bash
# Start
make -f Makefile.services quick-start

# Stop
make -f Makefile.services stop

# Status
make -f Makefile.services status

# Health
make -f Makefile.services health

# Logs
make -f Makefile.services logs

# Clean
make -f Makefile.services clean

# Reset
make -f Makefile.services reset
```

## Next Steps

1. **Choose your profile**: Start with `minimal` or `development`
2. **Start services**: Use Make commands or scripts
3. **Verify health**: Run health checks
4. **Access services**: Use provided URLs
5. **Monitor**: Check logs and metrics
6. **Develop**: Build your application

For production deployment, see the [Production Deployment](#production-deployment) section.
