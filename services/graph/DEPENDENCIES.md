# Dependency Management Guide

## Overview

This document explains the dependency management strategy for the graph service, which is part of the aModels mono-repository.

## Architecture

The graph service uses **Go modules** with **replace directives** for local mono-repo development. This approach:
- ✅ Enables fast local iteration without publishing modules
- ✅ Maintains clean dependency boundaries between services
- ✅ Supports both mono-repo and standalone builds
- ✅ Works seamlessly with Docker multi-stage builds

---

## Dependency Categories

### 1. Core External Dependencies

**Purpose**: Standard third-party Go modules from public repositories

```go
require (
    github.com/apache/arrow-go/v18 v18.4.1      // High-performance data transfer
    github.com/neo4j/neo4j-go-driver/v5 v5.28.4 // Knowledge graph database
    google.golang.org/grpc v1.76.0              // gRPC communication
    github.com/redis/go-redis/v9 v9.16.0        // Redis client for checkpointing
    // ... see go.mod for complete list
)
```

**Management**: Standard `go get` and `go mod tidy`

### 2. Internal aModels Service Dependencies

**Purpose**: Other services within the aModels mono-repo

```go
require (
    github.com/plturrell/aModels/services/catalog v0.0.0
    github.com/plturrell/aModels/services/extract v0.0.0
    github.com/plturrell/aModels/services/orchestration v0.0.0
    github.com/plturrell/aModels/services/postgres v0.0.0
    github.com/plturrell/aModels/services/shared v0.0.0
)
```

**Management**: Use replace directives (see below)

### 3. Third-Party Forks

**Purpose**: Customized versions of third-party libraries

```go
require (
    github.com/SAP/go-hdb v1.14.9  // SAP HANA database driver (forked)
)
```

**Management**: Replace directive points to `../../infrastructure/third_party/go-hdb`

---

## Replace Directives

Replace directives map module paths to local filesystem locations **during development**.

### Current Replace Directives

```go
// Self-reference for internal imports
replace github.com/plturrell/aModels/services/graph => .

// Internal aModels services (sibling directories)
replace github.com/plturrell/aModels/services/catalog => ../catalog
replace github.com/plturrell/aModels/services/extract => ../extract
replace github.com/plturrell/aModels/services/orchestration => ../orchestration
replace github.com/plturrell/aModels/services/postgres => ../postgres
replace github.com/plturrell/aModels/services/shared => ../shared

// Third-party forks in infrastructure/third_party
replace github.com/SAP/go-hdb => ../../infrastructure/third_party/go-hdb
```

### Why Replace Directives?

1. **Local Development**: Edit code across services without publishing
2. **Atomic Changes**: Modify interfaces and implementations together
3. **Fast Iteration**: No need to version/tag/publish for every change
4. **Testing**: Test cross-service changes before committing

---

## Build Scenarios

### Scenario 1: Local Development (Mono-Repo)

**Setup**: Use `go.work` (workspace mode)

```bash
# In /home/aModels root
go work init
go work use ./services/graph
go work use ./services/catalog
go work use ./services/extract
go work use ./services/orchestration
go work use ./services/postgres
go work use ./services/shared
```

**Benefits**:
- Single `go.work` file manages all modules
- IDE autocomplete works across services
- `go build` in any service resolves local modules
- `go.mod` files remain clean

### Scenario 2: Docker Build (Mono-Repo)

**Setup**: Copy entire mono-repo, use replace directives

```dockerfile
# Copy entire repo to preserve relative paths
COPY . /workspace/src
WORKDIR /workspace/src/services/graph

# go.mod replace directives work automatically
RUN go mod download
RUN go build ./cmd/graph-server
```

**Benefits**:
- Clean Dockerfile (no sed hacks)
- Reproducible builds
- Works with Docker layer caching

### Scenario 3: Standalone External Build (Future)

**Setup**: Publish modules to private proxy or use Git tags

```bash
# Option A: Private Go module proxy
GOPROXY=https://proxy.company.com go get github.com/plturrell/aModels/services/graph@v1.2.3

# Option B: Git tags
go get github.com/plturrell/aModels/services/graph@v1.2.3
```

**Requirements**:
1. Remove replace directives from go.mod
2. Tag releases: `git tag services/graph/v1.2.3`
3. Configure `GOPRIVATE=github.com/plturrell/aModels`

---

## Common Operations

### Adding a New Dependency

#### External Dependency
```bash
cd /home/aModels/services/graph
go get github.com/new/package@v1.0.0
go mod tidy
```

#### Internal Service Dependency
```bash
cd /home/aModels/services/graph

# 1. Add to go.mod require block
go mod edit -require=github.com/plturrell/aModels/services/newservice@v0.0.0

# 2. Add replace directive
go mod edit -replace=github.com/plturrell/aModels/services/newservice=../newservice

# 3. Tidy
go mod tidy
```

### Updating Dependencies

```bash
# Update all dependencies to latest compatible versions
go get -u ./...
go mod tidy

# Update specific dependency
go get github.com/neo4j/neo4j-go-driver/v5@latest
go mod tidy
```

### Verifying Dependencies

```bash
# Check for missing/unused dependencies
go mod verify

# Download all dependencies
go mod download

# Show dependency graph
go mod graph | grep -v indirect
```

---

## Troubleshooting

### Issue: "module not found" during build

**Cause**: Replace directive path is incorrect

**Fix**:
```bash
# Verify relative paths are correct
cd /home/aModels/services/graph
ls -la ../catalog  # Should exist
ls -la ../extract  # Should exist

# Rebuild go.mod if needed
go mod tidy
```

### Issue: Version conflicts between services

**Cause**: Different services require incompatible versions

**Fix**:
```bash
# Use go.work to unify versions across workspace
cd /home/aModels
cat > go.work <<EOF
go 1.23

use (
    ./services/graph
    ./services/catalog
    ./services/extract
    // ... other services
)
EOF

# Sync versions
go work sync
```

### Issue: Docker build fails with "module not found"

**Cause**: COPY command doesn't include all required directories

**Fix**:
```dockerfile
# Must copy entire mono-repo root, not just services/graph
COPY . /workspace/src
WORKDIR /workspace/src/services/graph
```

### Issue: IDE doesn't recognize internal imports

**Cause**: Missing `go.work` file

**Fix**:
```bash
# Create workspace file
cd /home/aModels
go work init
go work use ./services/graph
go work use ./services/catalog
# ... add all services

# Restart IDE
```

---

## Migration Path to Published Modules

When ready to publish modules externally:

### Step 1: Version Modules
```bash
# Tag each service independently
git tag services/graph/v1.0.0
git tag services/catalog/v1.0.0
git tag services/extract/v1.0.0
git push --tags
```

### Step 2: Remove Replace Directives
```bash
# Edit go.mod - remove all replace directives
# Keep only external dependencies

go mod tidy
```

### Step 3: Update Dependents
```bash
# In services that depend on graph
cd /home/aModels/services/other-service
go get github.com/plturrell/aModels/services/graph@v1.0.0
```

### Step 4: Configure GOPRIVATE
```bash
# In environment or ~/.gitconfig
export GOPRIVATE=github.com/plturrell/aModels
go env -w GOPRIVATE=github.com/plturrell/aModels
```

---

## Best Practices

### ✅ DO

- **Keep go.mod clean**: Group dependencies logically with comments
- **Use go.work locally**: Simplifies multi-module development
- **Document replace directives**: Explain why each one exists
- **Run `go mod tidy`**: After any dependency change
- **Verify builds**: Test Docker builds regularly
- **Pin versions**: Use specific versions for stability

### ❌ DON'T

- **Don't commit go.work**: Add to .gitignore (per-developer)
- **Don't use absolute paths**: Replace directives must be relative
- **Don't manually edit go.sum**: Let Go tooling manage it
- **Don't mix GOPATH mode**: Always use module mode
- **Don't nest replace chains**: Keep replacements direct

---

## Dependency Tree

```
github.com/plturrell/aModels/services/graph
├── External Dependencies
│   ├── github.com/apache/arrow-go/v18 (data transfer)
│   ├── github.com/neo4j/neo4j-go-driver/v5 (knowledge graph)
│   ├── google.golang.org/grpc (communication)
│   ├── github.com/redis/go-redis/v9 (checkpointing)
│   └── ... (see go.mod)
├── Internal aModels Services
│   ├── services/catalog (ISO 11179 registry)
│   ├── services/extract (entity extraction)
│   ├── services/orchestration (workflow orchestration)
│   ├── services/postgres (database operations)
│   └── services/shared (common utilities)
└── Third-Party Forks
    └── infrastructure/third_party/go-hdb (SAP HANA driver)
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build Graph Service
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      # No special setup needed - replace directives work automatically
      - uses: actions/setup-go@v4
        with:
          go-version: '1.23'
      
      - name: Build
        working-directory: services/graph
        run: |
          go mod download
          go mod verify
          go build ./cmd/graph-server
      
      - name: Test
        working-directory: services/graph
        run: go test ./...
```

---

## Resources

- [Go Modules Reference](https://go.dev/ref/mod)
- [Go Workspaces](https://go.dev/doc/tutorial/workspaces)
- [Replace Directive Documentation](https://go.dev/ref/mod#go-mod-file-replace)
- [Mono-Repo Best Practices](https://github.com/golang/go/wiki/Modules#when-should-i-use-the-replace-directive)

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-11-10 | Initial dependency cleanup and documentation | System |
| 2025-11-10 | Removed sed hacks from Dockerfile | System |
| 2025-11-10 | Clarified replace directive strategy | System |

---

## Support

For dependency issues:
1. Check this document first
2. Verify replace directive paths
3. Run `go mod verify` and `go mod tidy`
4. Review Docker build logs
5. Check go.work configuration (local dev)
