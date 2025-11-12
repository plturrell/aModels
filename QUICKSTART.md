# aModels Quick Start Guide

Get the entire aModels system up and running in minutes.

## Prerequisites

- **Docker** & **Docker Compose** (required)
- **GPU Drivers** (optional, for GPU-accelerated services)
- **8GB RAM minimum** (16GB recommended)
- **20GB free disk space**

## 🚀 Fastest Start (3 commands)

```bash
# 1. Make scripts executable
chmod +x scripts/*.sh scripts/lib/*.sh

# 2. Start the system
make -f Makefile.services quick-start

# 3. Check everything is healthy
make -f Makefile.services health
```

That's it! Your system is now running.

## 📊 What Just Started?

The `quick-start` command starts the **development profile**:

| Service | Port | URL |
|---------|------|-----|
| **Catalog** | 8084 | http://localhost:8084 |
| **LocalAI** | 8081 | http://localhost:8081/v1/models |
| **Extract** | 8083 | http://localhost:8083/health |
| **Runtime** | 8098 | http://localhost:8098/analytics/dashboard |
| **Orchestration** | 8085 | http://localhost:8085/healthz |
| Neo4j Browser | 7474 | http://localhost:7474 |
| Elasticsearch | 9200 | http://localhost:9200 |
| PostgreSQL | 5432 | `psql -h localhost -U postgres -d amodels` |
| Redis | 6379 | `redis-cli -h localhost` |

## 🎯 Common Tasks

### View Service Status

```bash
make -f Makefile.services status
```

### View Logs

```bash
# All logs
make -f Makefile.services logs

# Specific service
./scripts/docker-manager.sh logs catalog

# Follow logs live
FOLLOW=true ./scripts/docker-manager.sh logs localai
```

### Stop Services

```bash
make -f Makefile.services stop
```

### Restart Services

```bash
make -f Makefile.services restart
```

## 🔧 Different Startup Profiles

### Minimal (Fastest)

Just infrastructure + core services for lightweight development:

```bash
make -f Makefile.services minimal
```

**Services:** Redis, PostgreSQL, Neo4j, LocalAI, Catalog

### Full Stack

Everything including GPU services and advanced features:

```bash
make -f Makefile.services full
```

**Adds:** Graph, Transformers, DeepAgents, Training, DMS, Regulatory Audit

### Docker Only

Start specific service groups:

```bash
# Infrastructure only
./scripts/docker-manager.sh start infrastructure

# Add core services
./scripts/docker-manager.sh start core

# Add application services
./scripts/docker-manager.sh start app
```

## 🩺 Troubleshooting

### Service Won't Start?

```bash
# Check what's wrong
make -f Makefile.services health

# View detailed logs
./scripts/docker-manager.sh logs SERVICE_NAME

# Rebuild if needed
make -f Makefile.services docker-build
```

### Port Already in Use?

```bash
# Find what's using the port
lsof -i :8084

# Kill it
kill $(lsof -t -i :8084)
```

### Docker Issues?

```bash
# Get debug info
make -f Makefile.services debug

# Clean everything and restart
make -f Makefile.services docker-clean
make -f Makefile.services docker-start
```

### Fresh Start Needed?

```bash
# Complete reset (⚠️ deletes all data!)
make -f Makefile.services reset
make -f Makefile.services quick-start
```

## 📖 Next Steps

### For Developers

1. **Read the full guide**: `docs/SERVICES_STARTUP.md`
2. **Explore services**: Check `services/*/README.md`
3. **Build Go services**: `make -f Makefile.services build-go`
4. **Run tests**: `make -f Makefile.services test-services`

### For Production

1. Configure environment variables in `.env`
2. Review security settings
3. Set up monitoring
4. Configure backups
5. See `docs/SERVICES_STARTUP.md` - Production section

## 🆘 Getting Help

### View All Commands

```bash
make -f Makefile.services help
```

### List All Services

```bash
make -f Makefile.services list-services
```

### Check System Health

```bash
./scripts/health-check.sh
```

## 🎓 Examples

### Start for Development

```bash
# Start development stack
make -f Makefile.services quick-start

# Open in browser
open http://localhost:8084  # Catalog
open http://localhost:7474  # Neo4j
open http://localhost:8098/analytics/dashboard  # Analytics
```

### Start for Testing

```bash
# Minimal stack for fast iteration
make -f Makefile.services minimal

# Run your tests
go test ./services/catalog/...

# Clean up
make -f Makefile.services stop
```

### Start Individual Services

```bash
# Start infrastructure
./scripts/docker-manager.sh start infrastructure

# Wait for it to be ready
sleep 30

# Start specific application service
./scripts/docker-manager.sh start catalog

# Check it's healthy
curl http://localhost:8084/health
```

### Database Access

```bash
# PostgreSQL
make -f Makefile.services db-shell

# Neo4j
make -f Makefile.services neo4j-shell

# Redis
make -f Makefile.services redis-cli
```

## 🔥 Power User Tips

### Custom Profiles

Create your own in `config/services.yaml`:

```yaml
profiles:
  my_dev:
    services: [redis, postgres, localai, catalog, extract]
```

Use it:

```bash
PROFILE=my_dev ./scripts/start-system.sh start
```

### Native Mode (No Docker for Apps)

Faster iteration for development:

```bash
MODE=native PROFILE=development ./scripts/start-system.sh start
```

### Service-Specific Commands

```bash
# Build specific Go service
cd services/catalog && go build

# Run locally
./catalog

# Or use Make targets
make -f Makefile.services start-catalog
make -f Makefile.services logs-catalog
```

## 📝 Configuration Files

| File | Purpose |
|------|---------|
| `config/services.yaml` | Service registry and profiles |
| `infrastructure/docker/brev/docker-compose.yml` | Docker service definitions |
| `Makefile.services` | Management commands |
| `.env` | Environment variables (create from `.env.example`) |

## ⚡ Performance Tips

1. **Use SSD**: Significantly faster for databases
2. **Allocate RAM**: Give Docker 8GB+ RAM
3. **GPU Support**: Enable for LocalAI/Transformers
4. **Minimal Profile**: Use for development to save resources
5. **Prune Docker**: Regularly run `docker system prune`

## 🎉 You're Ready!

Your aModels system is now running. Start exploring:

- Browse the catalog: http://localhost:8084
- Query LocalAI: http://localhost:8081/v1/models
- View Neo4j graph: http://localhost:7474
- Check analytics: http://localhost:8098/analytics/dashboard

**Happy coding! 🚀**

For detailed documentation, see `docs/SERVICES_STARTUP.md`
