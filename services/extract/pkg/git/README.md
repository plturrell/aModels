# Gitea Client Package

This package provides a comprehensive client for interacting with Gitea repositories, including repository management, file operations, and branch/commit queries.

## Features

- **Repository Management**: Create, list, get, and delete repositories
- **File Operations**: List files, get file content, create/update files
- **Branch & Commit Queries**: List branches and commits with pagination
- **Security**: Input validation, path sanitization, secure token handling
- **Reliability**: Automatic retry logic with exponential backoff
- **Pagination**: Support for paginated list operations
- **Caching**: Optional response caching for GET requests

## Usage

### Basic Setup

```go
import "github.com/plturrell/aModels/services/extract/pkg/git"

// Create a new Gitea client
client := git.NewGiteaClient("https://gitea.example.com", "your-token-here")
```

### List Repositories

```go
ctx := context.Background()

// List all repositories for the authenticated user
repos, pagination, err := client.ListRepositories(ctx, "", nil)
if err != nil {
    log.Fatal(err)
}

// List repositories for a specific owner with pagination
pagination := &git.PaginationOptions{
    Page:  1,
    Limit: 20,
}
repos, resultPagination, err := client.ListRepositories(ctx, "owner-name", pagination)
```

### Create Repository

```go
req := git.CreateRepositoryRequest{
    Name:        "my-repo",
    Description: "My repository description",
    Private:     false,
    AutoInit:    true,
    Readme:      "default",
}

repo, err := client.CreateRepository(ctx, "", req)
if err != nil {
    log.Fatal(err)
}
```

### File Operations

```go
// List files in a directory
files, pagination, err := client.ListFiles(ctx, "owner", "repo", "path/to/dir", "main", nil)

// Get file content
content, err := client.GetFileContent(ctx, "owner", "repo", "path/to/file.txt", "main")

// Create or update a file
err := client.CreateOrUpdateFile(ctx, "owner", "repo", "path/to/file.txt", 
    "file content", "commit message", "main")
```

### Branches and Commits

```go
// List branches
branches, err := client.ListBranches(ctx, "owner", "repo")

// List commits with pagination
pagination := &git.PaginationOptions{Page: 1, Limit: 30}
commits, resultPagination, err := client.ListCommits(ctx, "owner", "repo", "main", 0, pagination)
```

## Security

All input is validated and sanitized:
- Repository names: Alphanumeric, dots, hyphens, underscores (max 100 chars)
- Owner names: Alphanumeric, dots, hyphens, underscores (max 100 chars)
- Branch names: Alphanumeric, forward slashes, dots, hyphens, underscores (max 255 chars)
- File paths: Validated to prevent path traversal attacks

## Error Handling

The client uses structured error responses with error codes:

```go
// Example error response
{
    "error": "Invalid repository name",
    "code": "INVALID_REPO_NAME",
    "details": "Repository name must contain only alphanumeric characters..."
}
```

## Retry Logic

All HTTP requests automatically retry on transient failures:
- Network errors
- Timeout errors
- 5xx server errors
- 429 rate limit errors

Retry configuration:
- Max retries: 3
- Initial delay: 100ms
- Max delay: 5s
- Exponential backoff

## Pagination

Pagination is supported for list operations:

```go
type PaginationOptions struct {
    Page  int // 1-based page number
    Limit int // Items per page (typically 1-100)
}
```

When pagination is used, the response includes pagination metadata:
- `data`: Array of items
- `page`: Current page number
- `limit`: Items per page
- `has_more`: Whether more pages are available

## HTTP Headers

For security, Gitea credentials should be passed via HTTP headers:

```
X-Gitea-URL: https://gitea.example.com
X-Gitea-Token: your-token-here
```

Query parameters are supported for backward compatibility but are deprecated.

## Examples

See the handler implementations in `cmd/extract/gitea_handlers.go` for complete examples of using the Gitea client in HTTP handlers.

