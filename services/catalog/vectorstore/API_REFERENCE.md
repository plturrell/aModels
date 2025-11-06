# HANA Cloud Vector Store API Reference

## Endpoints

### 1. Store Public Information

**POST** `/vectorstore/store`

Store public information in HANA Cloud vector store.

**Request Body:**
```json
{
  "type": "break_pattern",
  "system": "murex",
  "category": "finance",
  "title": "Reconciliation Break Pattern",
  "content": "Break description...",
  "metadata": {
    "frequency": 10,
    "resolution": "Check ETL pipeline"
  },
  "tags": ["reconciliation", "finance"],
  "is_public": true,
  "generate_embedding": true
}
```

**Response:**
```json
{
  "success": true,
  "id": "pattern-123",
  "message": "Information stored successfully"
}
```

---

### 2. Search Public Information (Semantic)

**POST** `/vectorstore/search`

Semantic search using vector similarity.

**Request Body:**
```json
{
  "query": "reconciliation break in finance system",
  "type": "break_pattern",
  "system": "general",
  "limit": 10,
  "threshold": 0.7
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": "pattern-123",
      "type": "break_pattern",
      "system": "sap_fioneer",
      "title": "Reconciliation Break Pattern",
      "content": "...",
      "metadata": {...},
      "tags": ["reconciliation", "finance"],
      "is_public": true,
      "created_at": "2025-11-06T10:00:00Z"
    }
  ],
  "count": 1
}
```

---

### 3. Get Information by ID

**GET** `/vectorstore/{id}`

Retrieve specific information by ID.

**Example:**
```bash
GET /vectorstore/pattern-123
```

**Response:**
```json
{
  "id": "pattern-123",
  "type": "break_pattern",
  "system": "sap_fioneer",
  "category": "finance",
  "title": "Reconciliation Break Pattern",
  "content": "Full content...",
  "metadata": {...},
  "tags": ["reconciliation", "finance"],
  "is_public": true,
  "created_at": "2025-11-06T10:00:00Z",
  "updated_at": "2025-11-06T10:00:00Z"
}
```

---

### 4. List Public Information

**GET** `/vectorstore`

List public information with filtering and pagination.

**Query Parameters:**

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `type` | string | Filter by type | `break_pattern`, `regulatory_rule`, `best_practice`, `knowledge_base` |
| `system` | string | Filter by system | `murex`, `sap_fioneer`, `bcrs`, `rco`, `axiomsl`, `general` |
| `category` | string | Filter by category | `finance`, `capital`, `liquidity`, `regulatory` |
| `tags` | string | Filter by tags (comma-separated) | `reconciliation,finance` |
| `is_public` | boolean | Filter by public status | `true`, `false`, `1`, `0` |
| `limit` | integer | Max results (default: 100) | `10`, `50`, `100` |
| `offset` | integer | Pagination offset (default: 0) | `0`, `10`, `20` |
| `order_by` | string | Order by field | `created_at`, `updated_at`, `title` |
| `order_desc` | boolean | Order descending | `true`, `false`, `1`, `0` |

**Examples:**

```bash
# List all break patterns for Murex
GET /vectorstore?type=break_pattern&system=murex&limit=20

# List all public regulatory rules
GET /vectorstore?type=regulatory_rule&is_public=true

# List best practices, ordered by creation date
GET /vectorstore?type=best_practice&order_by=created_at&order_desc=true

# List all public information for finance category
GET /vectorstore?category=finance&is_public=true&limit=50

# Paginated listing
GET /vectorstore?limit=10&offset=0   # First page
GET /vectorstore?limit=10&offset=10  # Second page

# Filter by tags
GET /vectorstore?tags=reconciliation,finance&is_public=true
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "id": "pattern-123",
      "type": "break_pattern",
      "system": "murex",
      "category": "finance",
      "title": "Reconciliation Break Pattern",
      "content": "Content...",
      "metadata": {...},
      "tags": ["reconciliation", "finance"],
      "is_public": true,
      "created_at": "2025-11-06T10:00:00Z",
      "updated_at": "2025-11-06T10:00:00Z"
    }
  ],
  "count": 1,
  "limit": 100,
  "offset": 0
}
```

---

## Information Types

### Break Pattern (`break_pattern`)
- Historical break patterns
- System-specific or general
- Includes resolution and prevention

### Regulatory Rule (`regulatory_rule`)
- Regulatory requirements (Basel III, IFRS 9, etc.)
- Compliance guidelines
- Public information for all systems

### Best Practice (`best_practice`)
- Best practices for break detection
- System-specific or general
- Application guidelines

### Knowledge Base (`knowledge_base`)
- General knowledge entries
- System documentation
- Custom information

## System Values

- `murex` - Murex-specific information
- `sap_fioneer` - SAP Fioneer-specific information
- `bcrs` - BCRS-specific information
- `rco` - RCO-specific information
- `axiomsl` - AxiomSL-specific information
- `general` - Available to all systems

## Error Responses

All endpoints return errors in the following format:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Error message",
    "details": {...}
  }
}
```

**HTTP Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - Invalid request
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

## Authentication

Currently, authentication is optional. For production, implement authentication middleware:

```go
// In main.go
mux.HandleFunc("/vectorstore/", authMiddleware.Middleware(
    vectorStoreHandler.HandleListPublicInformation,
))
```

## Rate Limiting

Rate limiting can be enabled via middleware:

```go
mux.HandleFunc("/vectorstore/", rateLimiter.Middleware(
    vectorStoreHandler.HandleListPublicInformation,
))
```

## Examples

### Example 1: List All Public Break Patterns

```bash
curl "http://localhost:8084/vectorstore?type=break_pattern&is_public=true&limit=50"
```

### Example 2: Get Murex-Specific Knowledge

```bash
curl "http://localhost:8084/vectorstore?system=murex&is_public=true"
```

### Example 3: Search Regulatory Rules

```bash
curl -X POST http://localhost:8084/vectorstore/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Basel III capital requirements",
    "type": "regulatory_rule",
    "limit": 10
  }'
```

### Example 4: Get Specific Information

```bash
curl "http://localhost:8084/vectorstore/pattern-123"
```

### Example 5: Paginated List

```bash
# First page
curl "http://localhost:8084/vectorstore?limit=10&offset=0"

# Second page
curl "http://localhost:8084/vectorstore?limit=10&offset=10"
```

