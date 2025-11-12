# aModels Scripts Directory

Organized collection of scripts for managing, testing, and maintaining the aModels system.

## 📁 Directory Structure

```
scripts/
├── system/          # System startup, health checks, infrastructure
├── data/            # Data management, SGMI operations
├── quality/         # Quality checks, metrics, benchmarks
├── testing/         # Integration and service tests
├── signavio/        # Signavio test data generation
├── dev-tools/       # Development utilities and visualization
├── templates/       # Configuration templates
├── lib/             # Shared utility functions
└── README.md        # This file
```

## 📚 Directory Guide

### 🔧 system/

**System startup, management, and infrastructure scripts**

| Script | Purpose |
|--------|----------|
| `start-system.sh` | **Main orchestrator** - Start/stop all services with dependency management |
| `health-check.sh` | Comprehensive health checks for all services |
| `docker-manager.sh` | Simplified Docker Compose operations |
| `diagnose_docker_networking.sh` | Diagnose Docker network issues |
| `setup_all_tunnels.sh` | Set up SSH tunnels for remote services |
| `setup_neo4j_tunnel.sh` | Set up Neo4j SSH tunnel |
| `generate-secrets.sh` | Generate secure secrets for services |

**Quick Start:**
```bash
# Start full system
cd system
./start-system.sh start

# Check health
./health-check.sh

# Manage Docker services
./docker-manager.sh start infrastructure
```

### 📊 data/

**Data management, SGMI operations, and ETL scripts**

| Script | Purpose |
|--------|----------|
| `load_sgmi_data.sh` | Load SGMI data into Neo4j |
| `clear_and_reload_sgmi.sh` | Clear and reload all SGMI data |
| `explore_sgmi_data.sh` | Explore SGMI data in Neo4j |
| `explore_data_flows.sh` | Explore data flow patterns |
| `export_sgmi_for_training.sh` | Export SGMI data for ML training |
| `run_sgmi_from_container.sh` | Run SGMI operations in container |
| `generate_training_from_postgres.sh` | Generate training data from PostgreSQL |
| `fix_and_reconcile.sh` | Fix data inconsistencies |
| `fix_orphan_columns.sh` | Fix orphaned column references |
| `enrich_missing_properties.sh` | Enrich missing data properties |
| `reconcile_graph_to_postgres.sh` | Reconcile graph and PostgreSQL data |

**Common Tasks:**
```bash
cd data

# Load SGMI data
./load_sgmi_data.sh

# Export for training
./export_sgmi_for_training.sh

# Fix data issues
./fix_and_reconcile.sh
```

### ✅ quality/

**Quality checks, metrics, and benchmarking**

| Script | Purpose |
|--------|----------|
| `check_all_quality.sh` | Run all quality checks |
| `check_orphans.sh` | Check for orphaned entities |
| `run_quality_metrics.sh` | Calculate quality metrics |
| `quality_metrics_report.sh` | Generate quality report |
| `benchmark_improvements.sh` | Benchmark performance improvements |

**Usage:**
```bash
cd quality

# Run all checks
./check_all_quality.sh

# Generate quality report
./quality_metrics_report.sh > report.txt

# Benchmark
./benchmark_improvements.sh
```

### 🧪 testing/

**Integration tests and service validation**

| Script | Purpose |
|--------|----------|
| `test_all_services.sh` | Test all service endpoints |
| `test_improvements.sh` | Test performance improvements |
| `test_sgmi_end_to_end.sh` | End-to-end SGMI pipeline test |
| `validate_test_data.sh` | Validate test data integrity |
| `test_deepagents_localai.py` | Test DeepAgents + LocalAI integration |
| `test_localai_integration.py` | Test LocalAI integration |

**Running Tests:**
```bash
cd testing

# Test all services
./test_all_services.sh

# End-to-end test
./test_sgmi_end_to_end.sh

# Python integration tests
python test_localai_integration.py
```

### 🏭 signavio/

**Signavio test data generation and validation**

| Script | Purpose |
|--------|----------|
| `demo_signavio_generator.sh` | Generate demo Signavio data |
| `generate_signavio_testdata.sh` | Generate test datasets |
| `signavio_test_generator.py` | Python test data generator |
| `validate_signavio_testdata.go` | Validate generated test data |

**Generating Test Data:**
```bash
cd signavio

# Generate demo data
./demo_signavio_generator.sh

# Generate with Python
python signavio_test_generator.py --count 100

# Validate
go run validate_signavio_testdata.go
```

### 🛠️ dev-tools/

**Development utilities and visualization**

| Script | Purpose |
|--------|----------|
| `validate_sgmi_data_flow.py` | Validate SGMI data flows |
| `visualize_graph.py` | Visualize graph structure |

**Development Tasks:**
```bash
cd dev-tools

# Validate data flows
python validate_sgmi_data_flow.py

# Visualize graph
python visualize_graph.py --output graph.png
```

### 📋 templates/

**Configuration templates**

| File | Purpose |
|------|----------|
| `signavio_config.template.json` | Signavio configuration template |

### 📚 lib/

**Shared utility functions**

| File | Purpose |
|------|----------|
| `service-utils.sh` | Reusable service management functions |

**Available Functions:**
- `check_http_health(url, timeout)` - Check HTTP endpoint health
- `check_grpc_health(host, port)` - Check gRPC health
- `check_postgres_health(host, port, user)` - PostgreSQL health
- `check_redis_health(host, port)` - Redis health
- `check_neo4j_health(url)` - Neo4j health
- `get_service_pid(name)` - Get service PID
- `is_service_running(name)` - Check if service is running
- `stop_service(name)` - Stop a service
- `build_go_service(path, output)` - Build Go service
- `install_python_deps(path, venv)` - Install Python dependencies

**Usage:**
```bash
source lib/service-utils.sh
check_http_health "http://localhost:8084/health"
```

## Usage Patterns

### Development Workflow

```bash
# 1. Start minimal stack
cd scripts/system
./start-system.sh start --profile=minimal

# 2. Develop your service
cd ../../services/myservice
go build && ./myservice

# 3. Check health
cd ../../scripts/system
./health-check.sh

# 4. Run quality checks
cd ../quality
./check_all_quality.sh

# 5. Stop when done
cd ../system
./start-system.sh stop
```

### Testing Workflow

```bash
# 1. Start infrastructure
cd scripts/system
./docker-manager.sh start infrastructure

# 2. Run integration tests
cd ../testing
./test_all_services.sh

# 3. Run specific tests
python test_localai_integration.py

# 4. Clean up
cd ../system
./docker-manager.sh stop
```

### Data Operations Workflow

```bash
# 1. Load SGMI data
cd scripts/data
./load_sgmi_data.sh

# 2. Check quality
cd ../quality
./run_quality_metrics.sh

# 3. Fix any issues
cd ../data
./fix_and_reconcile.sh

# 4. Export for training
./export_sgmi_for_training.sh
```

### Production Deployment

```bash
# 1. Start full stack with Docker
cd scripts/system
MODE=docker PROFILE=full ./start-system.sh start

# 2. Monitor health
watch -n 30 ./health-check.sh

# 3. View logs
./docker-manager.sh logs

# 4. Run quality monitoring
cd ../quality
watch -n 300 ./quality_metrics_report.sh
```

## Environment Variables

### System Scripts (system/)

```bash
# start-system.sh
PROFILE=minimal|development|full|docker
MODE=native|docker|hybrid

# docker-manager.sh
FOLLOW=true  # For following logs
```

### Data Scripts (data/)

```bash
# Default Neo4j connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=amodels123
```

## Configuration Files

Scripts read configuration from:

- `config/services.yaml` - Service registry and profiles
- `.env` - Environment variables
- `infrastructure/docker/brev/docker-compose.yml` - Docker services

## Output Directories

Scripts create the following directories:

- `logs/startup/` - Native service logs
- `.pids/` - Process ID files for native services

## Error Handling

All scripts follow these patterns:

1. **Exit on error**: `set -euo pipefail`
2. **Colored output**: Success (green), warnings (yellow), errors (red)
3. **Non-zero exit codes** on failure
4. **Detailed error messages** with context

## Integration with Make

Use via Makefile for convenience:

```bash
make -f Makefile.services quick-start
make -f Makefile.services health
make -f Makefile.services logs
```

## Quick Reference

### Most Common Commands

```bash
# Start the system
cd scripts/system && ./start-system.sh start

# Check health
cd scripts/system && ./health-check.sh

# Load data
cd scripts/data && ./load_sgmi_data.sh

# Run tests
cd scripts/testing && ./test_all_services.sh

# Quality checks
cd scripts/quality && ./check_all_quality.sh
```

### Navigating the Scripts

```bash
# From project root
cd scripts/system     # System management
cd scripts/data       # Data operations
cd scripts/quality    # Quality checks
cd scripts/testing    # Integration tests
cd scripts/signavio   # Test data generation
cd scripts/dev-tools  # Development utilities
```

## Troubleshooting

### Scripts Won't Run

```bash
# Fix permissions for all scripts
chmod +x scripts/system/*.sh
chmod +x scripts/data/*.sh
chmod +x scripts/quality/*.sh
chmod +x scripts/testing/*.sh
chmod +x scripts/signavio/*.sh
chmod +x scripts/lib/*.sh

# Or fix all at once
find scripts -name '*.sh' -exec chmod +x {} \;
```

### Port Conflicts

```bash
# Check what's using a port
lsof -i :8084

# Kill the process
kill $(lsof -t -i :8084)
```

### Docker Issues

```bash
# Check Docker is running
docker ps

# View Docker logs
docker-compose -f infrastructure/docker/brev/docker-compose.yml logs
```

### Health Checks Failing

```bash
# Run detailed health check
./health-check.sh

# Check service-specific logs
./docker-manager.sh logs SERVICE_NAME
```

## Development

### Adding a New Service

1. Add service definition to `config/services.yaml`
2. Add startup function to `start-system.sh`:
   ```bash
   start_myservice() {
       log_step "Starting My Service"
       # Implementation
   }
   ```
3. Add to appropriate profile in `config/services.yaml`
4. Test: `./start-system.sh start`

### Adding Health Checks

Edit `health-check.sh`:

```bash
declare -A SERVICES=(
    # ...
    ["My Service"]="http:http://localhost:9999/health"
)
```

## Best Practices

1. **Always run health checks** after starting services
2. **Use profiles** to minimize resource usage
3. **Check logs** when debugging issues
4. **Clean up** regularly with `./start-system.sh clean`
5. **Use Make commands** for convenience
6. **Monitor resources** especially in Docker mode

## Script Dependencies

```
system/
├── start-system.sh → lib/service-utils.sh, config/services.yaml
├── health-check.sh → (standalone)
└── docker-manager.sh → infrastructure/docker/brev/docker-compose.yml

data/
└── *.sh → Neo4j, PostgreSQL

quality/
└── *.sh → Neo4j, Extract service

testing/
├── *.sh → All services
└── *.py → Python 3.11+, pytest

signavio/
├── *.sh → Neo4j, Extract service
├── *.py → Python 3.11+
└── *.go → Go 1.21+

dev-tools/
└── *.py → Python 3.11+, matplotlib, networkx
```

## Performance Considerations

- **Native mode**: Faster startup, less resource usage
- **Docker mode**: Isolated, consistent, GPU support
- **Minimal profile**: Use for development
- **Health checks**: Run after startup, not during

## Security Notes

- Default credentials in scripts are for **development only**
- Change passwords in production
- Use environment variables for secrets
- Don't commit `.env` files

## Migration Guide

### Old → New Paths

**System scripts moved to `system/`:**
- `scripts/start-system.sh` → `scripts/system/start-system.sh`
- `scripts/health-check.sh` → `scripts/system/health-check.sh`
- `scripts/docker-manager.sh` → `scripts/system/docker-manager.sh`

**Data scripts moved to `data/`:**
- `scripts/load_sgmi_data.sh` → `scripts/data/load_sgmi_data.sh`
- `scripts/export_sgmi_for_training.sh` → `scripts/data/export_sgmi_for_training.sh`

**Quality scripts moved to `quality/`:**
- `scripts/check_all_quality.sh` → `scripts/quality/check_all_quality.sh`
- `scripts/quality_metrics_report.sh` → `scripts/quality/quality_metrics_report.sh`

**Testing scripts moved to `testing/`:**
- `scripts/test_all_services.sh` → `scripts/testing/test_all_services.sh`
- `scripts/test_sgmi_end_to_end.sh` → `scripts/testing/test_sgmi_end_to_end.sh`

### Update Your Scripts

If you have scripts that call these, update the paths:

```bash
# Old
./scripts/start-system.sh start

# New
./scripts/system/start-system.sh start

# Or use from Makefile (paths already updated)
make -f Makefile.services quick-start
```

## Additional Resources

- **Full documentation**: `../docs/SERVICES_STARTUP.md`
- **Quick start**: `../QUICKSTART.md`
- **Service configs**: `../config/services.yaml`
- **Docker setup**: `../infrastructure/docker/brev/docker-compose.yml`
- **Makefile**: `../Makefile.services` (uses updated paths)

## Support

For issues or questions:
1. Check script help: `cd scripts/system && ./start-system.sh --help`
2. Review logs: `logs/startup/`
3. Run diagnostics: `cd scripts/system && ./health-check.sh`
4. See troubleshooting: `../docs/SERVICES_STARTUP.md`
5. Check this README: `scripts/README.md`
