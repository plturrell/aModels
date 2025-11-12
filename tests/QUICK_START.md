# Tests Quick Start

Get started with the reorganized test structure in 60 seconds.

## TL;DR

```bash
# Run all tests
make test-all

# Or
cd tests/scripts && ./run_all_tests.sh --check
```

## Directory Layout

```
tests/
├── integration/        # Service & workflow integration tests
├── performance/        # Load, stress, performance tests
├── benchmarks/         # Model evaluation benchmarks
├── domain/            # Domain-specific functionality tests
├── e2e/               # End-to-end system tests
├── fixtures/          # Test data & helpers
└── scripts/           # Test execution scripts
```

## Most Common Commands

### Run Everything
```bash
make test-all           # All tests with health check
```

### Run by Category
```bash
make test-integration   # Integration tests only
make test-performance   # Performance tests only
make test-benchmarks    # Model benchmarks only
```

### Run Specific Test
```bash
# Integration test
cd tests/integration/services
python3 test_localai_integration.py

# Performance test
cd tests/performance
python3 test_load.py

# Benchmark
cd tests/benchmarks/hellaswag
go test -v .
```

## Before Running Tests

### Check Services
```bash
cd tests/scripts
./check_services.sh
```

### Start Services (if needed)
```bash
docker-compose -f infrastructure/docker/brev/docker-compose.yml up -d
```

## Test Categories at a Glance

| Category | Location | Count | Purpose |
|----------|----------|-------|---------|
| Integration - Services | `integration/services/` | 6 | Service-to-service |
| Integration - Workflows | `integration/workflows/` | 5 | Multi-step workflows |
| Performance | `performance/` | 6 | Load & stress tests |
| Domain | `domain/` | 5 | Domain-specific |
| Benchmarks | `benchmarks/` | 9 | Model evaluation |
| E2E | `e2e/` | 1 | Full system tests |

## Environment Setup

### Python
```bash
cd tests
pip install -r requirements.txt
```

### Go
```bash
cd tests
go mod download
```

## Common Scenarios

### 1. Testing a New Feature
```bash
# 1. Write your test in appropriate category
cd tests/integration/services
vim test_new_feature.py

# 2. Run it
python3 test_new_feature.py

# 3. Run all integration tests to ensure nothing broke
make test-integration
```

### 2. Before Committing
```bash
# Quick smoke test
cd tests/scripts
./run_smoke_tests.sh

# Full test suite
make test-all
```

### 3. Performance Testing
```bash
# Run specific performance test
cd tests/performance
python3 test_load.py --users 100

# Run all performance tests
make test-performance
```

### 4. Running Benchmarks
```bash
# All benchmarks
make test-benchmarks

# Specific benchmark
cd tests/benchmarks/hellaswag
go test -v .
```

## Troubleshooting

### Services Not Running
```bash
# Check status
docker ps

# Start services
docker-compose -f infrastructure/docker/brev/docker-compose.yml up -d

# Wait a bit, then check health
cd tests/scripts && ./check_services.sh
```

### Import Errors (Python)
```python
# Use this pattern
from tests.fixtures.test_helpers import helper_function
from tests.fixtures.test_data_fixtures import sample_data
```

### Test Not Found
```bash
# Find any test
find tests/ -name "*search_term*"
```

## Documentation Links

- **Main Guide**: `/tests/README.md` - Full documentation
- **Integration**: `/tests/integration/README.md` - Integration testing
- **Migration**: `/tests/MIGRATION_GUIDE.md` - For migrating from old structure
- **Audit**: `/TESTING_AUDIT.md` - Why we reorganized

## New to Testing Here?

**Read these in order**:
1. This file (you're here!)
2. `/tests/README.md` - Main testing guide
3. Category-specific README (e.g., `/tests/integration/README.md`)

## Quick Tips

✅ **DO**:
- Check services before testing: `./check_services.sh`
- Use Makefile targets: `make test-integration`
- Put tests in correct category
- Clean up test data after tests

❌ **DON'T**:
- Run tests without checking services
- Put all tests in one file
- Hard-code test data
- Skip cleaning up resources

## Help

Can't find something? Try:

```bash
# Search for test files
find tests/ -name "*keyword*"

# Check main README
cat tests/README.md

# Validate migration
cd tests/scripts && ./validate_migration.sh

# List all available Makefile targets
make help
```

---

**Still stuck?** See `/tests/README.md` for comprehensive documentation.
