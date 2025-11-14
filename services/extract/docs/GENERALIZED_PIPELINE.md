# Generalized Code-to-Knowledge Graph Pipeline

## Overview

The Extract service now supports a generalized code-to-knowledge graph conversion pipeline that works with any codebase, not just SGMI. The pipeline integrates with:

- **Glean Catalog**: Real-time export via `RealTimeGleanExporter`
- **Postgres**: Schema replication via `schema/replication.go` (glean_nodes, glean_edges tables)
- **Petri Nets**: Workflow conversion from Control-M and other formats
- **CWM/Local AI**: Enhanced semantic analysis via Code World Model

## Features

### 1. Multi-Source Support
- **File-based**: Process files from filesystem
- **Git Repositories**: Clone and process code from Gitea, GitHub, GitLab, or any Git host
- **Mixed Sources**: Combine files and Git repositories in a single pipeline

### 2. AI Enhancement
- **CWM Integration**: Uses Code World Model for semantic code analysis
- **Relationship Discovery**: AI discovers implicit relationships between code entities
- **Documentation Generation**: Automatic code documentation
- **Fallback**: Gracefully falls back to `phi-3.5-mini` if CWM unavailable

### 3. Generalized Configuration
- **YAML/JSON Config**: Project-specific configuration files
- **Parser Selection**: Enable/disable parsers per project
- **AI Configuration**: Configure AI model and tasks per project

## Usage

### Basic Usage with Config File

```bash
./scripts/pipelines/run_code_to_kg.sh project-config.yaml
```

### Configuration File Format

```yaml
project:
  id: "my-codebase"
  name: "My Application"
  system_id: "backend"

sources:
  files:
    - "/path/to/code/*.sql"
    - "/path/to/code/*.hql"
  
  git_repositories:
    - url: "https://gitea.example.com/user/repo.git"
      type: "gitea"
      branch: "main"
      auth:
        type: "token"
        token: "${GITEA_TOKEN}"
      file_patterns:
        - "**/*.sql"
        - "**/*.hql"

parsers:
  - type: "ddl"
    enabled: true
  - type: "sql"
    enabled: true

ai:
  enabled: true
  model: "cwm"
  localai_url: "http://localai:8080"
```

### API Usage

```bash
curl -X POST http://localhost:8083/knowledge-graph \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "my-project",
    "system_id": "backend",
    "git_repositories": [
      {
        "url": "https://gitea.example.com/user/repo.git",
        "branch": "main",
        "file_patterns": ["**/*.sql", "**/*.hql"]
      }
    ],
    "ai_enabled": true,
    "ai_model": "cwm"
  }'
```

## Integration Points

### Glean Catalog
- **Automatic**: Real-time export happens automatically when graphs are saved
- **Configuration**: Set `GLEAN_REALTIME_ENABLE=true` and `GLEAN_DB_NAME`
- **Location**: `services/extract/pkg/persistence/glean_realtime.go`

### Postgres
- **Automatic**: Schema replication happens automatically
- **Tables**: `glean_nodes` and `glean_edges`
- **Location**: `services/extract/pkg/schema/replication.go`

### Petri Nets
- **Automatic**: Control-M workflows are converted to Petri nets
- **Storage**: Stored in catalog and knowledge graph
- **Location**: `services/extract/pkg/workflow/petri_net.go`

## Backward Compatibility

The SGMI pipeline continues to work:
- `run_sgmi_etl_automated.sh` now tries to use generalized pipeline if config exists
- Falls back to SGMI-specific builder if config not found
- All existing SGMI scripts remain functional

## Files Created

### New Packages
- `pkg/pipeline/`: Pipeline configuration and orchestration
- `pkg/ai/`: CWM client wrapper
- `pkg/git/`: Git repository processing

### New Scripts
- `scripts/pipelines/code_view_builder.py`: Generalized view builder
- `scripts/pipelines/run_code_to_kg.sh`: Universal pipeline script
- `scripts/pipelines/example-config.yaml`: Example configuration
- `scripts/pipelines/sgmi-config.yaml`: SGMI example configuration

### Modified Files
- `cmd/extract/main.go`: Added Git repository support, AI enhancement, removed hard-coded SGMI IDs
- `scripts/pipelines/run_sgmi_etl_automated.sh`: Updated to use generalized pipeline when available

## Testing

1. **Test with SGMI**: Verify backward compatibility
2. **Test with Git**: Clone a repository and process code
3. **Test AI**: Verify CWM enhancement works
4. **Test Integrations**: Verify Glean and Postgres exports

## Next Steps

- Add support for more Git hosts (Bitbucket, etc.)
- Extend Petri net converter for more workflow formats
- Add Python/Go code parsers
- Enhance AI relationship discovery

