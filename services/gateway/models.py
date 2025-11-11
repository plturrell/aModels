"""
Pydantic models for request/response validation.
Provides type safety and automatic validation for API endpoints.
"""
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


# ============================================================================
# Search Models
# ============================================================================

class SearchSource(str, Enum):
    """Available search sources."""
    INFERENCE = "inference"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    CATALOG = "catalog"
    PERPLEXITY = "perplexity"


class UnifiedSearchRequest(BaseModel):
    """Request model for unified search."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    sources: List[SearchSource] = Field(
        default=[SearchSource.INFERENCE, SearchSource.KNOWLEDGE_GRAPH, SearchSource.CATALOG],
        description="Sources to search"
    )
    use_perplexity: bool = Field(False, description="Include Perplexity AI web search")
    
    @validator('query')
    def query_not_empty(cls, v):
        """Validate query is not just whitespace."""
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace')
        return v.strip()


class SearchResult(BaseModel):
    """Individual search result."""
    source: str = Field(..., description="Source of the result")
    id: str = Field(..., description="Result identifier")
    content: str = Field(..., description="Result content")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    citations: Optional[List[str]] = Field(None, description="Citations for web results")


class SearchMetadata(BaseModel):
    """Metadata about search execution."""
    total_sources: int = Field(..., ge=0)
    sources_successful: int = Field(..., ge=0)
    sources_failed: int = Field(..., ge=0)
    query_time_ms: Optional[float] = Field(None, description="Query execution time")


class UnifiedSearchResponse(BaseModel):
    """Response model for unified search."""
    query: str
    combined_results: List[SearchResult]
    sources: Dict[str, Any]
    metadata: SearchMetadata


# ============================================================================
# Processing Models (Perplexity, DMS, Relational, Murex)
# ============================================================================

class ProcessingStatus(str, Enum):
    """Processing job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessRequest(BaseModel):
    """Generic processing request."""
    documents: Optional[List[str]] = Field(None, description="Documents to process")
    tables: Optional[List[str]] = Field(None, description="Tables to process")
    query: Optional[str] = Field(None, description="Query or prompt")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")


class ProcessResponse(BaseModel):
    """Generic processing response."""
    request_id: str = Field(..., description="Unique request identifier")
    status: ProcessingStatus = Field(..., description="Processing status")
    message: Optional[str] = Field(None, description="Status message")
    status_url: str = Field(..., description="URL to check status")
    results_url: str = Field(..., description="URL to fetch results")


class StatusResponse(BaseModel):
    """Status check response."""
    request_id: str
    status: ProcessingStatus
    progress: Optional[float] = Field(None, ge=0.0, le=100.0, description="Progress percentage")
    message: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


# ============================================================================
# History and Pagination Models
# ============================================================================

class HistoryRequest(BaseModel):
    """Request model for history queries."""
    limit: int = Field(50, ge=1, le=100, description="Number of items to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")
    status: Optional[ProcessingStatus] = Field(None, description="Filter by status")
    query: Optional[str] = Field(None, description="Search query filter")


class HistoryItem(BaseModel):
    """Individual history item."""
    request_id: str
    status: ProcessingStatus
    created_at: float
    query: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HistoryResponse(BaseModel):
    """History query response."""
    items: List[HistoryItem]
    total: int = Field(..., ge=0)
    limit: int
    offset: int
    has_more: bool


# ============================================================================
# Batch Processing Models
# ============================================================================

class BatchRequest(BaseModel):
    """Batch processing request."""
    queries: List[str] = Field(..., min_items=1, max_items=100, description="Batch queries")
    options: Dict[str, Any] = Field(default_factory=dict, description="Batch options")


class BatchJobStatus(BaseModel):
    """Status of individual batch job."""
    index: int
    query: str
    status: ProcessingStatus
    result_url: Optional[str] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    """Batch processing response."""
    batch_id: str
    total_jobs: int
    jobs: List[BatchJobStatus]
    status: ProcessingStatus


# ============================================================================
# Catalog Models
# ============================================================================

class DataElementRequest(BaseModel):
    """Request to create/update data element."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    data_type: Optional[str] = Field(None)
    domain: Optional[str] = Field(None)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataProductRequest(BaseModel):
    """Request to build data product."""
    topic: str = Field(..., min_length=1, description="Data product topic")
    customer_need: str = Field(..., min_length=1, description="Customer requirement")
    options: Dict[str, Any] = Field(default_factory=dict)


class SemanticSearchRequest(BaseModel):
    """Semantic search request."""
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")


# ============================================================================
# Knowledge Graph Models
# ============================================================================

class GraphQueryRequest(BaseModel):
    """Knowledge graph query request."""
    query: str = Field(..., description="Graph query (Cypher, SPARQL, etc)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Query parameters")
    limit: int = Field(100, ge=1, le=1000)


class GraphRelationship(BaseModel):
    """Graph relationship model."""
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphQueryResponse(BaseModel):
    """Knowledge graph query response."""
    nodes: List[Dict[str, Any]]
    relationships: List[GraphRelationship]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Deep Research Models
# ============================================================================

class ResearchRequest(BaseModel):
    """Deep research request."""
    query: str = Field(..., min_length=1, max_length=1000)
    context: Dict[str, Any] = Field(default_factory=dict, description="Research context")
    tools: List[str] = Field(
        default_factory=lambda: ["sparql_query", "catalog_search"],
        description="Research tools to use"
    )
    depth: Literal["quick", "standard", "comprehensive"] = Field(
        "standard",
        description="Research depth"
    )


# ============================================================================
# Health and Admin Models
# ============================================================================

class ServiceHealth(BaseModel):
    """Individual service health status."""
    name: str
    status: Literal["ok", "degraded", "error", "unknown"]
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    circuit_breaker_state: Optional[Literal["closed", "half_open", "open"]] = None


class HealthResponse(BaseModel):
    """Overall health response."""
    gateway: str = "ok"
    version: str
    timestamp: float
    services: Dict[str, ServiceHealth]
    circuit_breakers: Optional[Dict[str, str]] = None


class CircuitBreakerStats(BaseModel):
    """Circuit breaker statistics."""
    name: str
    state: Literal["closed", "half_open", "open"]
    failures: int
    successes: int
    total_requests: int
    total_failures: int
    total_successes: int
    success_rate: float
    opened_at: Optional[float] = None
    last_failure_time: Optional[float] = None


class RateLimitInfo(BaseModel):
    """Rate limit information."""
    allowed: bool
    tokens_remaining: int
    retry_after: int = Field(0, description="Seconds to wait before retry")


# ============================================================================
# Error Models
# ============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""
    error: str
    message: str
    correlation_id: Optional[str] = None
    timestamp: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class ValidationErrorDetail(BaseModel):
    """Validation error details."""
    field: str
    message: str
    type: str


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    error: str = "Validation Error"
    details: List[ValidationErrorDetail]


# ============================================================================
# Generic Response Models
# ============================================================================

class GenericResponse(BaseModel):
    """Generic success response."""
    status: str = "success"
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class ExportRequest(BaseModel):
    """Export request."""
    format: Literal["json", "csv", "excel", "pdf"] = Field("json", description="Export format")
    options: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Redis/Cache Models
# ============================================================================

class CacheStats(BaseModel):
    """Cache statistics."""
    total_keys: int
    hit_rate: float = Field(..., ge=0.0, le=1.0)
    miss_rate: float = Field(..., ge=0.0, le=1.0)
    total_hits: int
    total_misses: int
    memory_usage_mb: Optional[float] = None


class RedisSetRequest(BaseModel):
    """Redis set request."""
    key: str = Field(..., min_length=1, max_length=200)
    value: str = Field(..., max_length=10000)
    ex: Optional[int] = Field(None, ge=1, le=86400, description="Expiration in seconds")


class RedisGetRequest(BaseModel):
    """Redis get request."""
    key: str = Field(..., min_length=1, max_length=200)
