# File System to Gitea Storage

## Overview

The Extract service now supports storing code from the **file system** into Gitea repositories, in addition to storing code from Git repositories.

## Features

- ✅ Extract files from file system (JSON, DDL, Control-M, Signavio files)
- ✅ Store extracted files in Gitea repository
- ✅ Create file nodes in knowledge graph
- ✅ Link files to Gitea repository
- ✅ Automatic file type detection
- ✅ Size limits and security scanning

## Supported File Sources

The system extracts files from these file system sources:

1. **JSON Tables** (`json_tables`): JSON schema files
2. **Hive DDLs** (`hive_ddls`): DDL files (if provided as file paths)
3. **Control-M Files** (`control_m_files`): Control-M XML files
4. **Signavio Files** (`signavio_files`): Signavio BPMN files

## Usage

### Basic Configuration

```json
{
  "project_id": "my-project",
  "system_id": "backend",
  "json_tables": [
    "/path/to/schema1.json",
    "/path/to/schema2.json"
  ],
  "hive_ddls": [
    "/path/to/tables.hql"
  ],
  "control_m_files": [
    "/path/to/workflow.xml"
  ],
  "gitea_storage": {
    "enabled": true,
    "gitea_url": "https://gitea.example.com",
    "gitea_token": "${GITEA_TOKEN}",
    "owner": "extract-service",
    "repo_name": "my-project-extracted-code",
    "base_path": "filesystem/",
    "auto_create": true
  }
}
```

### File System Files Storage

When `gitea_storage.enabled` is `true`, all file system files are:

1. **Extracted**: Files are read from the file system
2. **Scanned**: Content is scanned for secrets/PII
3. **Stored in Gitea**: Files are committed to the Gitea repository
4. **Stored in Knowledge Graph**: File nodes are created with raw content

### Repository Structure

Files are stored in Gitea with the following structure:

```
<base_path>/
├── schema1.json
├── schema2.json
├── tables.hql
└── workflow.xml
```

Default `base_path` is `filesystem/` for file system sources.

### Combined Sources

You can combine file system and Git repository sources:

```json
{
  "project_id": "my-project",
  "json_tables": ["/path/to/file.json"],
  "git_repositories": [
    {
      "url": "https://github.com/user/repo.git",
      "file_patterns": ["**/*.sql"]
    }
  ],
  "gitea_storage": {
    "enabled": true,
    "repo_name": "my-project-extracted-code"
  }
}
```

Files from both sources will be stored in the same Gitea repository:
- File system files: `filesystem/`
- Git repository files: `git-repos/` (or custom base path)

## File Node Properties

Each file system file node includes:

- `content`: Raw code (for files <10KB)
- `content_ref`: Reference for large files
- `content_preview`: Preview of large files
- `content_hash`: SHA256 hash
- `has_secrets`: Boolean flag
- `risk_level`: Security risk assessment
- `security_findings`: Detailed findings
- `source`: "filesystem"
- `path`: File path
- `size`, `extension`, `last_modified`: File metadata

## Knowledge Graph Structure

```
Project
├── GiteaRepository (extracted-code)
│   ├── File (schema1.json) [STORED_IN]
│   ├── File (schema2.json) [STORED_IN]
│   └── File (tables.hql) [STORED_IN]
└── FileSystem (filesystem:project-id)
    ├── File (schema1.json) [CONTAINS]
    ├── File (schema2.json) [CONTAINS]
    └── File (tables.hql) [CONTAINS]
```

## Implementation Details

### File System Extractor

**Location**: `pkg/git/filesystem_extractor.go`

- Extracts files from file system paths
- Handles both single files and directories
- Applies size limits (10MB per file, 100MB total)
- Detects text vs binary files
- Calculates content hashes

### Integration Point

**Location**: `cmd/extract/main.go` (lines 1104-1225)

- Collects files from all file system sources
- Stores in Gitea if configured
- Creates file nodes in knowledge graph
- Links files to Gitea repository

## Environment Variables

```bash
# Gitea Configuration
GITEA_URL=https://gitea.example.com
GITEA_TOKEN=your-token-here

# File Size Limits (optional)
MAX_FILE_SIZE=10485760      # 10MB
MAX_TOTAL_SIZE=104857600    # 100MB
```

## Example Request

```bash
curl -X POST http://localhost:8083/knowledge-graph \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "test-project",
    "system_id": "backend",
    "json_tables": [
      "/data/schemas/customer.json",
      "/data/schemas/order.json"
    ],
    "hive_ddls": [
      "/data/ddl/tables.hql"
    ],
    "gitea_storage": {
      "enabled": true,
      "gitea_url": "https://gitea.example.com",
      "gitea_token": "gitea_token_here",
      "owner": "extract-service",
      "repo_name": "test-project-extracted-code",
      "base_path": "filesystem/",
      "auto_create": true
    }
  }'
```

## Benefits

1. **Version Control**: All extracted code is versioned in Gitea
2. **Traceability**: Full history of code changes
3. **Collaboration**: Team can review and collaborate on extracted code
4. **Backup**: Code is stored in Git repository for backup
5. **Integration**: Works with existing Git workflows

## Notes

- File system files are stored with just the filename (not full path) to avoid path conflicts
- Files from directories maintain relative paths
- Large files (>10KB) are stored with references, not inline content
- Secret scanning is performed on all files before storage

