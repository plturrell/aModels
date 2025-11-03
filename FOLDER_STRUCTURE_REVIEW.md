# aModels Repository Structure & Naming Review

**Date:** 2024  
**Overall Rating:** 7.5/10

## Executive Summary

The `aModels` repository has a generally well-organized structure with clear separation of concerns. However, there are several inconsistencies in naming conventions and some organizational improvements that could enhance maintainability and discoverability.

---

## Strengths ✅

### 1. Clear Service Separation (9/10)
- **Microservices well-organized**: `agentflow/`, `extract/`, `postgres/`, `hana/`, `gateway/`, `localai/`
- Each service has its own `README.md`, `Dockerfile`, and build scripts
- Clear boundaries between services

### 2. Third-Party Management (10/10)
- **Excellent**: `third_party/` with Git submodules properly documented
- Clear README explaining each submodule
- Good separation from project code

### 3. Training Data Organization (8/10)
- `training/` directory with `sgmi/` subdirectory
- Clear documentation of training data purpose
- Well-structured pipeline metamodel organization

### 4. Docker Organization (9/10)
- `docker/` directory with compose files
- Clear separation of base (`compose.yml`) and GPU (`compose.gpu.yml`)
- Brev-specific configs in subdirectory

### 5. Documentation Structure (8/10)
- `docs/` directory with model-specific documentation
- README files in most major directories
- Good use of markdown files

---

## Issues & Recommendations ⚠️

### 1. **Naming Convention Inconsistencies** (Rating: 6/10)

#### Problems:
- **Mixed case**: `agentflow/`, `extract/`, `postgres/` vs `BUILD.md` (uppercase)
- **Inconsistent service naming**: Some use kebab-case (`agentflow`), others use single words (`extract`, `postgres`)
- **Directory with spaces**: `HIVE DDLS/` (should be `hive-ddl/` or `hive_ddl/`)
- **Mixed case files**: `JSON_with_changes.json` vs `go.mod` (lowercase)

#### Recommendations:
```bash
# Standardize on kebab-case for directories
agentflow/          → ✅ Keep
extract/            → ✅ Keep (acceptable)
postgres/           → ✅ Keep (acceptable)
HIVE DDLS/          → ❌ Rename to: hive-ddl/ or hive_ddl/
SGMI_Scripts/       → ❌ Rename to: sgmi-scripts/
SGMI-controlm/       → ❌ Rename to: sgmi-controlm/

# Standardize file naming
JSON_with_changes.json → ❌ Rename to: json_with_changes.json or json-with-changes.json
BUILD.md              → ❌ Rename to: build.md or BUILDING.md
```

### 2. **Top-Level Clutter** (Rating: 6/10)

#### Problems:
- **Binary files at root**: `aibench`, `arcagi_service`, `benchmark-server` (should be in `bin/` or removed)
- **Mixed concerns**: Root has both services and binaries

#### Recommendations:
```
# Move binaries
aibench           → bin/aibench
arcagi_service    → bin/arcagi_service
benchmark-server  → bin/benchmark-server

# Or if they're build artifacts, add to .gitignore
```

### 3. **Inconsistent Service Structure** (Rating: 7/10)

#### Problems:
- **Different patterns**: Some services have `cmd/`, others have `main.go` at root
- **Inconsistent internal organization**: `extract/` has files at root, `agentflow/` has `internal/`, `pkg/`

#### Current patterns:
```
extract/
  ├── main.go          # Root level
  ├── ddl.go
  └── cmd/             # Only has one entry point

agentflow/
  ├── internal/        # Internal packages
  ├── pkg/            # Public packages
  └── cmd/            # Entry points

postgres/
  ├── cmd/            # Entry points
  ├── internal/       # Internal packages
  └── pkg/           # Public packages
```

#### Recommendations:
- **Standardize Go service structure**:
```
services/
  ├── extract/
  │   ├── cmd/
  │   │   └── extract-service/
  │   │       └── main.go
  │   ├── internal/
  │   ├── pkg/
  │   └── README.md
  ├── postgres/
  ├── hana/
  └── agentflow/
```

### 4. **Data Directory Organization** (Rating: 7/10)

#### Problems:
- Mixed naming: `ARC-AGI/`, `ARC-AGI-2/`, `GSM-Symbolic/` (uppercase with hyphens)
- Inconsistent: `boolean-questions/` (lowercase), `SocialIQ/` (mixed case)

#### Recommendations:
```bash
# Standardize to lowercase-kebab-case
ARC-AGI/          → data/arc-agi/
ARC-AGI-2/        → data/arc-agi-2/
GSM-Symbolic/     → data/gsm-symbolic/
SocialIQ/         → data/social-iq/
boolean-questions/ → ✅ Keep (good example)
```

### 5. **Documentation Naming** (Rating: 7/10)

#### Problems:
- Mixed case: `BUILD.md` vs `README.md` (lowercase)
- Inconsistent: `docs/RELATIONAL_TRANSFORMER.md` (uppercase) vs `docs/metrics_table.md` (lowercase)

#### Recommendations:
```bash
# Standardize docs to lowercase-kebab-case
docs/RELATIONAL_TRANSFORMER.md → docs/relational-transformer.md
docs/INFERENCE_ENHANCEMENTS.md → docs/inference-enhancements.md
docs/MLOPS_GUIDE.md            → docs/mlops-guide.md
docs/metrics_table.md          → docs/metrics-table.md
BUILD.md                       → docs/building.md or BUILDING.md
```

### 6. **Stage3 Directory** (Rating: 5/10)

#### Problems:
- **Unclear naming**: `stage3/` suggests temporary staging, but contains permanent code
- **Mixed languages**: Java (search) and Go (graph) in same parent
- **Large Java codebase** (26K+ files) might be better as submodule

#### Recommendations:
```bash
# Rename to be more descriptive
stage3/          → services/legacy/ or services/search-graph/

# Or split into separate submodules if they're independent
```

### 7. **Web Directory Duplication** (Rating: 6/10)

#### Problems:
- Two `web/` directories: root `web/` and `training/web/`
- Unclear purpose of root `web/`

#### Recommendations:
```bash
web/              → Clarify purpose or move to services/web/
training/web/     → ✅ Keep (training-specific UI)
```

### 8. **Scripts Organization** (Rating: 8/10)

#### Problems:
- Good overall structure, but some scripts are service-specific
- `scripts/factory/` and `scripts/datagen/` could be better organized

#### Recommendations:
```bash
scripts/
  ├── build/          # Build scripts
  ├── training/       # Training scripts
  ├── deployment/     # Deployment scripts
  └── utils/          # Utility scripts
```

---

## Proposed Standard Structure

```
aModels/
├── .github/              # GitHub workflows
├── bin/                 # Compiled binaries (gitignored or in releases)
├── cmd/                 # Root-level command tools
│   ├── aibench/
│   ├── arcagi_service/
│   └── benchmark-server/
├── configs/             # Configuration files
├── data/                # Training/evaluation data (standardized naming)
│   ├── arc-agi/
│   ├── arc-agi-2/
│   ├── gsm-symbolic/
│   └── social-iq/
├── docs/                # Documentation (lowercase-kebab-case)
│   ├── relational-transformer.md
│   ├── inference-enhancements.md
│   └── mlops-guide.md
├── docker/              # Docker configurations
│   ├── compose.yml
│   ├── compose.gpu.yml
│   └── brev/
├── models/              # Model metadata (weights in releases)
├── scripts/             # Build and utility scripts
│   ├── build/
│   ├── training/
│   └── deployment/
├── services/            # Microservices (standardized structure)
│   ├── agentflow/
│   ├── extract/
│   ├── gateway/
│   ├── hana/
│   ├── localai/
│   ├── postgres/
│   └── search-graph/    # Renamed from stage3
├── third_party/         # Git submodules
├── tools/               # Helper tools
├── training/            # Training data and configs
│   ├── sgmi/
│   │   ├── hive-ddl/    # Renamed from HIVE DDLS
│   │   ├── sgmi-scripts/ # Renamed from SGMI_Scripts
│   │   ├── sgmi-controlm/ # Renamed from SGMI-controlm
│   │   └── json-with-changes.json
│   └── web/
├── tests/               # Integration tests
└── README.md
```

---

## Priority Fixes

### High Priority 🔴
1. **Remove spaces from directory names**: `HIVE DDLS/` → `hive-ddl/`
2. **Standardize service structure**: Move all services under `services/` or document pattern
3. **Move binaries**: `aibench`, `arcagi_service`, `benchmark-server` to `bin/` or gitignore

### Medium Priority 🟡
4. **Standardize data directory names**: Uppercase → lowercase-kebab-case
5. **Standardize documentation naming**: All docs to lowercase-kebab-case
6. **Clarify `stage3/` purpose**: Rename or document as permanent legacy code

### Low Priority 🟢
7. **Consolidate web directories**: Document purpose of root `web/`
8. **Organize scripts subdirectories**: Group by purpose

---

## Naming Convention Standard

### Recommended Standard:
- **Directories**: `lowercase-kebab-case` (e.g., `hive-ddl/`, `sgmi-scripts/`)
- **Files**: `lowercase_snake_case` for code, `lowercase-kebab-case.md` for docs
- **Go packages**: `lowercase` (single word) or `lowercasesnakecase` (no separators)
- **Services**: `lowercase` (single word) or `kebab-case` for multi-word

### Exceptions:
- **README.md**: Always uppercase (standard)
- **LICENSE**: Always uppercase (standard)
- **Go files**: Follow Go conventions (already good)
- **Third-party submodules**: Keep original names

---

## Scoring Breakdown

| Category | Score | Notes |
|----------|-------|-------|
| Overall Structure | 8/10 | Good separation of concerns |
| Naming Consistency | 6/10 | Mixed conventions need standardization |
| Documentation | 8/10 | Good coverage, inconsistent naming |
| Service Organization | 7/10 | Clear but inconsistent patterns |
| Data Organization | 7/10 | Functional but naming issues |
| Third-Party Management | 10/10 | Excellent |
| Docker Organization | 9/10 | Very good |
| **Overall** | **7.5/10** | **Good foundation, needs polish** |

---

## Conclusion

The `aModels` repository has a solid foundation with clear service boundaries and good documentation. The main improvements needed are:

1. **Standardize naming conventions** (especially directories with spaces and mixed case)
2. **Clarify service structure patterns** (document or standardize)
3. **Clean up root directory** (move binaries or gitignore them)
4. **Consistent documentation naming** (lowercase-kebab-case)

These changes would improve maintainability, discoverability, and onboarding for new contributors.

