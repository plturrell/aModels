# Legacy Code

This directory contains legacy and read-only code that is maintained for compatibility but not actively developed.

## Contents

### `stage3/`
Legacy search and graph services from Layer 4 Stage 3:
- `search/` - Java-based search microservices (26K+ files)
- `graph/` - GPU-accelerated graph components (Go)

**Status:** Read-only exports - maintain for reference only
**Purpose:** Historical reference and deployment compatibility

### `internal/` (if present)
Shared internal packages (may be moved elsewhere)

### `search/` (if present)
Legacy search service code

## Usage

These directories are **not part of the active development** workflow. They are kept for:
- Historical reference
- Deployment compatibility
- Migration purposes

Do **not** modify code in these directories. If you need functionality from legacy code:
1. Extract the needed functionality
2. Reimplement in the appropriate service directory
3. Update dependencies accordingly

## Migration

If you need to migrate functionality from legacy code:
1. Identify the functionality needed
2. Create a new service or add to existing service
3. Document the migration in the service README
4. Update references throughout the codebase

