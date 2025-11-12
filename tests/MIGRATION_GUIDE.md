# Test Migration Guide

Quick reference for developers working with the reorganized test structure.

## What Changed

### Old Structure → New Structure

| Old Location | New Location | Reason |
|-------------|--------------|---------|
| `/testing/test_*.py` | `/tests/<category>/*.py` | Organized by test type |
| `/scripts/testing/*.sh` | `/tests/scripts/*.sh` | Consolidated test scripts |
| `/testing/benchmarks/` | `/tests/benchmarks/` | Clearer location |

## Quick Migration Reference

### Finding Tests

#### Old Way
```bash
# Where is the LocalAI test?
ls testing/test_*.py | grep localai
# testing/test_localai_integration_suite.py

# Where are the test scripts?
ls scripts/testing/
```

#### New Way
```bash
# Integration tests
ls tests/integration/services/
# test_localai_integration.py

# Test scripts
ls tests/scripts/
```

### Running Tests

#### Old Commands → New Commands

| Old | New | Notes |
|-----|-----|-------|
| `./testing/01_test_*.py` | `cd tests/integration/services && python3 test_*.py` | By category |
| `./scripts/testing/run_all_tests.sh` | `cd tests/scripts && ./run_all_tests.sh` | Same script, new location |
| `./scripts/testing/00_check_services.sh` | `cd tests/scripts && ./check_services.sh` | Renamed |
| N/A | `make test-integration` | New Makefile target |
| N/A | `make test-performance` | New Makefile target |

## Test Categories

### Integration Tests (`tests/integration/`)
**What**: Tests that verify service-to-service interactions
**When to use**: Testing API integrations, service communication
**Examples**:
```bash
# Old
testing/test_localai_integration_suite.py

# New
tests/integration/services/test_localai_integration.py
```

### Performance Tests (`tests/performance/`)
**What**: Load, stress, and performance tests
**When to use**: Testing system performance under load
**Examples**:
```bash
# Old
testing/test_load.py

# New
tests/performance/test_load.py
```

### Domain Tests (`tests/domain/`)
**What**: Domain-specific functionality tests
**When to use**: Testing domain detection, filtering, etc.
**Examples**:
```bash
# Old
testing/test_domain_detection.py

# New
tests/domain/test_domain_detection.py
```

### Benchmarks (`tests/benchmarks/`)
**What**: Model evaluation benchmarks
**When to use**: Measuring AI/ML model performance
**Examples**:
```bash
# Old
testing/benchmarks/hellaswag/

# New
tests/benchmarks/hellaswag/
```

## Common Tasks

### Adding a New Test

#### Old Way
```bash
# Add to /testing directory
cd testing
touch test_new_feature.py
```

#### New Way
```bash
# Choose appropriate category
cd tests/integration/services  # For integration tests
touch test_new_feature.py

# Or
cd tests/performance  # For performance tests
touch test_new_feature.py
```

### Running All Tests

#### Old Way
```bash
cd testing
./run_all_tests.sh  # If it existed
```

#### New Way
```bash
# Option 1: Direct script
cd tests/scripts
./run_all_tests.sh

# Option 2: Makefile
make test-all

# Option 3: Category-specific
make test-integration
make test-performance
```

### Running Specific Test

#### Old Way
```bash
cd testing
python3 test_localai_integration_suite.py
```

#### New Way
```bash
cd tests/integration/services
python3 test_localai_integration.py
```

## Import Paths

### Python Imports

#### Old Imports
```python
# From anywhere
from testing.test_helpers import helper_function
from testing.test_data_fixtures import sample_data
```

#### New Imports
```python
# From anywhere
from tests.fixtures.test_helpers import helper_function
from tests.fixtures.test_data_fixtures import sample_data
```

### Go Imports

#### Old Imports
```go
// Rarely used from testing directory
```

#### New Imports
```go
// Integration tests use standard go test conventions
// Located in tests/integration/services/
```

## CI/CD Updates

### GitHub Actions

#### Old
```yaml
- name: Run tests
  run: ./testing/run_all_tests.sh
```

#### New
```yaml
- name: Run tests
  run: |
    cd tests/scripts
    ./run_all_tests.sh
    
# Or use Makefile
- name: Run tests
  run: make test-all
```

### GitLab CI

#### Old
```yaml
test:
  script:
    - ./scripts/testing/run_all_tests.sh
```

#### New
```yaml
test:
  script:
    - cd tests/scripts && ./run_all_tests.sh
    
# Or use categories
integration-tests:
  script:
    - make test-integration
    
performance-tests:
  script:
    - make test-performance
```

## Environment Variables

No changes to environment variables, they work the same:

```bash
export LOCALAI_URL="http://localhost:8081"
export EXTRACT_SERVICE_URL="http://localhost:19080"
# ... etc
```

## Troubleshooting

### "Test not found"

**Problem**: Can't find a test file in old location

**Solution**: Use the category-based structure
```bash
# Search for the test
find tests/ -name "*test_name*"

# Check specific categories
ls tests/integration/services/
ls tests/performance/
ls tests/domain/
```

### "Import error"

**Problem**: Python imports failing

**Solution**: Update import paths
```python
# Old
from testing.test_helpers import foo

# New
from tests.fixtures.test_helpers import foo
```

### "Script not found"

**Problem**: Test script not in `/scripts/testing`

**Solution**: Check `/tests/scripts`
```bash
# Old
./scripts/testing/run_all_tests.sh

# New
cd tests/scripts
./run_all_tests.sh

# Or use Makefile
make test-all
```

## FAQ

### Q: Where did my test file go?

**A**: Check the test category:
- Integration → `tests/integration/`
- Performance → `tests/performance/`
- Domain → `tests/domain/`
- Benchmarks → `tests/benchmarks/`

Use `find` to locate:
```bash
find tests/ -name "*your_test*"
```

### Q: Can I still run tests the old way?

**A**: The old directories still exist temporarily, but use the new structure going forward. They will be removed once validation is complete.

### Q: Do I need to update my service tests?

**A**: No! Service unit tests (`*_test.go` co-located with code) stay in place. This reorganization is for project-level integration, performance, and benchmark tests.

### Q: How do I know which category for my test?

**A**: Use this decision tree:

```
Is it testing a single function/unit?
  → Yes: Co-locate with source (services/*/package_test.go)
  → No: Continue

Is it testing multiple services together?
  → Yes: tests/integration/services/

Is it testing a complete workflow?
  → Yes: tests/integration/workflows/

Is it testing performance/load?
  → Yes: tests/performance/

Is it a model benchmark?
  → Yes: tests/benchmarks/

Is it domain-specific functionality?
  → Yes: tests/domain/
```

### Q: What about test data?

**A**: Test data and fixtures are in `tests/fixtures/`:
- `test_data/` - Static test data files
- `test_helpers.py` - Helper functions
- `test_data_fixtures.py` - Data generators

## Makefile Quick Reference

```bash
# All tests
make test              # Unit + Integration
make test-all          # Everything with health check

# By category
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-e2e         # End-to-end tests only
make test-performance  # Performance tests only
make test-benchmarks   # Benchmarks only
```

## Script Quick Reference

```bash
cd tests/scripts

# Main runners
./run_all_tests.sh              # All tests
./run_all_tests.sh --check      # With service health check
./run_all_tests.sh --host       # Use localhost URLs
./run_all_tests.sh --strict     # Exit on first failure

# Category-specific
./run_integration_tests.sh      # Integration only
./run_performance_tests.sh      # Performance only
./run_benchmarks.sh             # Benchmarks only
./run_unit_tests.sh             # Unit tests only
./run_e2e_tests.sh              # E2E only

# Utilities
./check_services.sh             # Health check
./validate_migration.sh         # Validate migration
./setup_test_database.sh        # Setup test DB
```

## Getting Help

1. **Main Guide**: `/tests/README.md`
2. **Integration Guide**: `/tests/integration/README.md`
3. **Audit Report**: `/TESTING_AUDIT.md`
4. **Completion Report**: `/TESTING_REORGANIZATION_COMPLETE.md`
5. **This Guide**: `/tests/MIGRATION_GUIDE.md`

## Summary

- ✅ Tests organized by type in `/tests`
- ✅ Old structure preserved temporarily
- ✅ New Makefile targets available
- ✅ Scripts in `/tests/scripts`
- ✅ Comprehensive documentation
- ✅ Validation scripts available

**When in doubt**: Check `/tests/README.md` or use `find tests/ -name "*your_test*"`
