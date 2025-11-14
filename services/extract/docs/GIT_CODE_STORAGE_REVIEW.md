# Review: Raw Code Storage from Self-Hosted Git Repositories

## Executive Summary

**Overall Rating: 6.5/10**

The current implementation successfully extracts and parses code from Git repositories, but **does not store raw code content** in the knowledge graph. Only parsed entities (tables, columns, etc.) are stored, which limits traceability and code retrieval capabilities.

## Current Implementation Analysis

### Process Flow

```
1. Clone Repository → Temp Directory
2. Extract Files (matching patterns)
3. Read File Content → Memory
4. Parse Files → Entities (tables, columns, etc.)
5. Store Parsed Entities → Knowledge Graph
6. Link Entities → Repository Node
7. Cleanup Temp Directory
```

### What's Working Well ✅

1. **Repository Cloning** (8/10)
   - ✅ Supports multiple Git hosts (Gitea, GitHub, GitLab, generic)
   - ✅ Handles authentication (token, SSH, basic)
   - ✅ Branch/tag/commit selection
   - ⚠️ Uses shallow clone (`--depth 1`) - good for performance but loses history

2. **File Extraction** (7/10)
   - ✅ Pattern-based file filtering
   - ✅ Recursive directory traversal
   - ✅ Skips hidden directories (`.git`, etc.)
   - ⚠️ Basic glob matching (could be improved)

3. **Metadata Extraction** (7/10)
   - ✅ Repository metadata (URL, branch, commit, author)
   - ✅ Commit information
   - ✅ File count
   - ⚠️ Missing file-level metadata in graph

4. **Integration** (8/10)
   - ✅ Works with existing parsers (DDL, SQL, JSON)
   - ✅ Links to repository node
   - ✅ Integrates with Glean/Postgres automatically

### Critical Issues ❌

#### 1. **Raw Code Not Stored** (Rating: 3/10)

**Problem:**
- File content is read into memory, parsed, then **discarded**
- Only parsed entities (tables, columns) are stored in knowledge graph
- **No way to retrieve original code** after processing

**Impact:**
- Cannot view original source code from knowledge graph
- Cannot trace parsed entities back to source files
- Loses context for AI analysis
- No code review or audit trail

**Current Code:**
```go
// In cmd/extract/main.go:1134-1198
// File content is used for parsing but NOT stored
fileNodes, fileEdges, _, err := s.extractSchemaFromJSON(tmpFile.Name())
// Content is lost after parsing
```

#### 2. **No File Nodes Created** (Rating: 4/10)

**Problem:**
- Files are not represented as nodes in the knowledge graph
- Only repository node and parsed entities exist
- No file-level metadata (path, size, last modified, etc.)

**Impact:**
- Cannot query for files
- Cannot track file changes over time
- No file-to-entity relationships visible

#### 3. **Memory Concerns** (Rating: 5/10)

**Problem:**
- All file content loaded into memory simultaneously
- No size limits or chunking
- Large repositories could cause OOM errors

**Impact:**
- Risk of memory exhaustion
- No handling for very large files (>100MB)
- No streaming or lazy loading

#### 4. **No Versioning/Tracking** (Rating: 4/10)

**Problem:**
- Only stores current commit
- No history of changes
- No diff tracking
- Cannot see code evolution

**Impact:**
- Cannot track code changes over time
- No audit trail
- Limited for compliance/security requirements

#### 5. **No Code Deduplication** (Rating: 5/10)

**Problem:**
- Same code in multiple files stored multiple times
- No content hashing or deduplication
- Wastes storage space

**Impact:**
- Duplicate storage in knowledge graph
- Increased storage costs
- Harder to find duplicate code

## Detailed Component Ratings

### RepositoryClient (`pkg/git/repository_client.go`)

**Rating: 7/10**

**Strengths:**
- Clean interface
- Proper cleanup handling
- Authentication support

**Weaknesses:**
- Shallow clone only (no history)
- No retry logic for network failures
- No timeout configuration
- Token injection into URL is insecure (should use credential helper)

**Recommendations:**
```go
// Add retry logic
func (c *RepositoryClient) CloneWithRetry(ctx context.Context, url string, branch string, auth Auth, maxRetries int) (*CloneResult, error)

// Add full clone option
args := []string{"clone"}
if shallow {
    args = append(args, "--depth", "1")
}

// Use Git credential helper instead of URL injection
// Set GIT_ASKPASS or use credential.helper
```

### CodeExtractor (`pkg/git/code_extractor.go`)

**Rating: 6/10**

**Strengths:**
- Simple and straightforward
- Pattern matching works for basic cases

**Weaknesses:**
- Basic glob matching (doesn't support `**/*.sql` properly)
- No file size limits
- No content filtering (binary files, etc.)
- Loads all files into memory

**Recommendations:**
```go
// Add file size limit
const MaxFileSize = 10 * 1024 * 1024 // 10MB

// Add content type detection
func isTextFile(content []byte) bool {
    // Check for binary content
}

// Add streaming for large files
func (e *CodeExtractor) ExtractFilesStreaming(repoPath string, patterns []string, callback func(ExtractedFile) error) error
```

### Pipeline Integration (`cmd/extract/main.go`)

**Rating: 5/10**

**Strengths:**
- Integrates with existing parsers
- Creates repository nodes
- Links entities to repository

**Weaknesses:**
- **Does not create file nodes**
- **Does not store file content**
- Only processes `.hql`, `.sql`, `.json` files
- No support for other file types (Python, Go, etc.)

**Current Missing:**
```go
// Should create file nodes like this:
fileNode := graph.Node{
    ID:    fmt.Sprintf("file:%s:%s", repoID, file.Path),
    Type:  "File",
    Label: filepath.Base(file.Path),
    Props: map[string]interface{}{
        "path":          file.Path,
        "content":       file.Content,  // MISSING!
        "size":          file.Size,
        "last_modified": file.LastModified,
        "repository_id": repoID,
        "commit":        meta.Commit,
    },
}
```

## Recommendations

### Priority 1: Store Raw Code Content

**Action:** Create file nodes with raw code content

```go
// In cmd/extract/main.go, after extracting files:
for _, file := range extractedFiles {
    // Create file node
    fileNodeID := fmt.Sprintf("file:%s:%s", repoNodes[0].ID, file.Path)
    fileNode := graph.Node{
        ID:    fileNodeID,
        Type:  "File",
        Label: filepath.Base(file.Path),
        Props: map[string]interface{}{
            "path":          file.Path,
            "content":       file.Content,  // Store raw content
            "size":          file.Size,
            "last_modified": file.LastModified,
            "extension":     filepath.Ext(file.Path),
            "repository_id": repoNodes[0].ID,
            "commit":        repoMeta.Commit,
            "source":        "git",
        },
    }
    nodes = append(nodes, fileNode)
    
    // Link file to repository
    edges = append(edges, graph.Edge{
        SourceID: repoNodes[0].ID,
        TargetID: fileNodeID,
        Label:    "CONTAINS",
    })
    
    // Then parse and link parsed entities to file
    // ...
}
```

**Storage Considerations:**
- Neo4j: Store in node properties (works for files < 32KB)
- Large files: Use external storage (S3, file system) and store reference
- Glean: Store in Glean facts
- Postgres: Store in `glean_nodes.properties_json` (JSONB supports large text)

### Priority 2: Add File Size Limits

**Action:** Prevent memory issues with large files

```go
const (
    MaxFileSize      = 10 * 1024 * 1024 // 10MB
    MaxTotalSize     = 100 * 1024 * 1024 // 100MB per repo
)

func (e *CodeExtractor) ExtractFiles(repoPath string, patterns []string) ([]ExtractedFile, error) {
    var totalSize int64
    // ...
    if info.Size() > MaxFileSize {
        // Skip or truncate
        continue
    }
    if totalSize > MaxTotalSize {
        return files, fmt.Errorf("total size exceeds limit")
    }
}
```

### Priority 3: Improve Pattern Matching

**Action:** Use proper glob library

```go
import "github.com/gobwas/glob"

func matchesPattern(path string, patterns []string) bool {
    for _, pattern := range patterns {
        g := glob.MustCompile(pattern)
        if g.Match(path) {
            return true
        }
    }
    return false
}
```

### Priority 4: Add File Type Support

**Action:** Support more file types (Python, Go, YAML, etc.)

```go
// Add file type detection
func detectFileType(path string, content []byte) string {
    ext := strings.ToLower(filepath.Ext(path))
    // Add more types
    switch ext {
    case ".py": return "python"
    case ".go": return "go"
    case ".yaml", ".yml": return "yaml"
    // ...
    }
}

// Route to appropriate parser
switch fileType {
case "python":
    // Parse Python code
case "go":
    // Parse Go code
}
```

### Priority 5: Add Versioning Support

**Action:** Store commit history and diffs

```go
// Store file versions
type FileVersion struct {
    Commit    string
    Content   string
    Timestamp time.Time
    Author    string
}

// Create version nodes
versionNode := graph.Node{
    ID: fmt.Sprintf("file_version:%s:%s", fileNodeID, commit),
    Type: "FileVersion",
    Props: map[string]interface{}{
        "commit": commit,
        "content": content,
        "timestamp": timestamp,
    },
}
```

## Storage Architecture Recommendations

### Option 1: Store in Neo4j Properties (Small Files)

**Pros:**
- Simple
- Queryable via Cypher
- Integrated with existing graph

**Cons:**
- Neo4j property size limit (~32KB)
- Not suitable for large files

**Use Case:** Files < 10KB

### Option 2: External Storage + Reference (Large Files)

**Pros:**
- Handles files of any size
- Reduces Neo4j storage
- Can use object storage (S3, etc.)

**Cons:**
- Additional storage system
- More complex queries

**Implementation:**
```go
// Store reference in Neo4j
fileNode.Props["content_ref"] = fmt.Sprintf("s3://bucket/files/%s", fileHash)
fileNode.Props["content_size"] = file.Size

// Store actual content in S3/object storage
```

### Option 3: Hybrid Approach (Recommended)

**Pros:**
- Best of both worlds
- Small files in Neo4j, large files external

**Implementation:**
```go
const InlineContentLimit = 10 * 1024 // 10KB

if len(file.Content) < InlineContentLimit {
    fileNode.Props["content"] = file.Content
} else {
    // Store in external storage
    ref, err := storeInExternalStorage(file.Content)
    fileNode.Props["content_ref"] = ref
    fileNode.Props["content_size"] = len(file.Content)
}
```

## Integration with Glean/Postgres

### Current State
- ✅ Repository metadata flows to Glean/Postgres
- ✅ Parsed entities flow to Glean/Postgres
- ❌ **File nodes and raw content do NOT flow**

### Recommendations

1. **Glean Export:**
   - Add `File` predicate to Glean schema
   - Export file nodes with content
   - Use Glean's fact storage for code content

2. **Postgres Storage:**
   - Store file content in `glean_nodes.properties_json`
   - Use JSONB for efficient storage and querying
   - Add indexes on file paths and extensions

## Security Considerations

### Current Issues

1. **Token in URL** (Line 49 in `repository_client.go`)
   - ⚠️ Token visible in process list
   - ⚠️ Token in logs potentially

2. **No Content Scanning**
   - ⚠️ Could store sensitive data (passwords, keys)
   - ⚠️ No PII detection

### Recommendations

1. **Use Git Credential Helper:**
```go
// Instead of URL injection
os.Setenv("GIT_ASKPASS", "git-credential-helper")
os.Setenv("GIT_CREDENTIAL", fmt.Sprintf("username=%s\npassword=%s", username, password))
```

2. **Add Content Scanning:**
```go
func scanForSecrets(content string) []string {
    // Scan for API keys, passwords, etc.
    // Return list of potential secrets
}
```

3. **Add Access Control:**
   - Restrict which repositories can be cloned
   - Validate repository URLs
   - Rate limiting

## Performance Considerations

### Current Bottlenecks

1. **Sequential Processing**
   - Files processed one at a time
   - No parallelization

2. **Memory Usage**
   - All files loaded into memory
   - No streaming

3. **Network**
   - No caching of cloned repositories
   - Re-clones on every run

### Recommendations

1. **Parallel Processing:**
```go
// Process files in parallel
var wg sync.WaitGroup
sem := make(chan struct{}, 10) // Limit concurrency

for _, file := range files {
    wg.Add(1)
    go func(f ExtractedFile) {
        defer wg.Done()
        sem <- struct{}{}
        defer func() { <-sem }()
        // Process file
    }(file)
}
wg.Wait()
```

2. **Repository Caching:**
```go
// Cache cloned repositories
cacheKey := fmt.Sprintf("%s:%s", url, branch)
if cached, exists := cache.Get(cacheKey); exists {
    // Use cached version
}
```

3. **Streaming for Large Files:**
```go
// Stream file content instead of loading all at once
func (e *CodeExtractor) ExtractFilesStreaming(repoPath string, patterns []string, callback func(ExtractedFile) error) error
```

## Summary Ratings

| Component | Rating | Status |
|-----------|--------|--------|
| Repository Cloning | 7/10 | ✅ Good |
| File Extraction | 6/10 | ⚠️ Needs improvement |
| Metadata Extraction | 7/10 | ✅ Good |
| **Raw Code Storage** | **3/10** | ❌ **Critical Issue** |
| File Node Creation | 4/10 | ❌ Missing |
| Versioning | 4/10 | ❌ Missing |
| Memory Management | 5/10 | ⚠️ Needs limits |
| Pattern Matching | 6/10 | ⚠️ Basic |
| Security | 5/10 | ⚠️ Needs improvement |
| Performance | 6/10 | ⚠️ Could be optimized |
| **Overall** | **6.5/10** | ⚠️ **Functional but incomplete** |

## Action Items (Priority Order)

1. **🔴 Critical:** Store raw code content in file nodes
2. **🔴 Critical:** Create file nodes in knowledge graph
3. **🟡 High:** Add file size limits
4. **🟡 High:** Improve pattern matching
5. **🟡 High:** Add more file type support
6. **🟢 Medium:** Add versioning support
7. **🟢 Medium:** Improve security (credential handling)
8. **🟢 Medium:** Add parallel processing
9. **🟢 Low:** Add repository caching
10. **🟢 Low:** Add content scanning for secrets

## Conclusion

The current implementation successfully extracts and parses code from Git repositories, but **fails to store raw code content**, which is a critical limitation. The system can parse code and create knowledge graph entities, but cannot retrieve or reference the original source code.

**Key Recommendation:** Implement file node creation with raw code content storage as the highest priority. This will enable:
- Code traceability
- AI-enhanced analysis
- Code review capabilities
- Audit trails
- Better integration with Glean/Postgres

