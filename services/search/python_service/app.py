import json
import logging
import os
import re
import secrets
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Path, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator

try:  # Optional dependency: Elasticsearch Python client
    from elasticsearch import Elasticsearch
except ImportError:  # pragma: no cover - optional dependency missing
    Elasticsearch = None  # type: ignore

try:  # Optional dependency: Redis client
    import redis
except ImportError:  # pragma: no cover - optional dependency missing
    redis = None  # type: ignore

try:  # Optional dependency: SAP HANA client
    from hdbcli import dbapi as hdbapi
except ImportError:  # pragma: no cover - optional dependency missing
    hdbapi = None  # type: ignore


def sanitize_query(query: str) -> str:
    """Sanitize and validate search query input."""
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    
    # Strip whitespace
    query = query.strip()
    
    # Check length
    if len(query) < MIN_QUERY_LENGTH:
        raise ValueError(f"Query must be at least {MIN_QUERY_LENGTH} character(s)")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query must not exceed {MAX_QUERY_LENGTH} characters")
    
    # Remove control characters except newlines and tabs
    query = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', query)
    
    return query


def validate_document_id(doc_id: str) -> str:
    """Validate document ID format and length."""
    if not doc_id or not isinstance(doc_id, str):
        raise ValueError("Document ID must be a non-empty string")
    
    doc_id = doc_id.strip()
    
    if len(doc_id) > MAX_DOCUMENT_ID_LENGTH:
        raise ValueError(f"Document ID must not exceed {MAX_DOCUMENT_ID_LENGTH} characters")
    
    # Allow alphanumeric, hyphens, underscores, and dots
    if not re.match(r'^[a-zA-Z0-9._-]+$', doc_id):
        raise ValueError("Document ID can only contain alphanumeric characters, dots, hyphens, and underscores")
    
    return doc_id


def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize metadata dictionary."""
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary")
    
    if len(metadata) > MAX_METADATA_KEYS:
        raise ValueError(f"Metadata must not exceed {MAX_METADATA_KEYS} keys")
    
    sanitized = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("Metadata keys must be strings")
        if len(key) > MAX_DOCUMENT_ID_LENGTH:
            raise ValueError(f"Metadata key '{key}' exceeds maximum length")
        
        # Convert value to string and validate length
        str_value = str(value)
        if len(str_value) > MAX_METADATA_VALUE_LENGTH:
            raise ValueError(f"Metadata value for '{key}' exceeds maximum length")
        
        sanitized[key] = value
    
    return sanitized


class SearchDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        return validate_document_id(v)
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("Content must be a non-empty string")
        if len(v) > MAX_DOCUMENT_CONTENT_LENGTH:
            raise ValueError(f"Content must not exceed {MAX_DOCUMENT_CONTENT_LENGTH} characters")
        return v.strip()
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        return validate_metadata(v)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=200)
    filters: Dict[str, str] = Field(default_factory=dict)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)
    date_from: Optional[str] = None  # ISO format date string
    date_to: Optional[str] = None    # ISO format date string
    source: Optional[str] = None     # Filter by source
    doc_type: Optional[str] = None   # Filter by document type
    facets: List[str] = Field(default_factory=list)  # Faceted search fields
    include_explanation: bool = Field(default=False)  # Include ranking explanation
    include_highlighting: bool = Field(default=True)  # Highlight query terms
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        return sanitize_query(v)
    
    @field_validator('filters')
    @classmethod
    def validate_filters(cls, v: Dict[str, str]) -> Dict[str, str]:
        if not isinstance(v, dict):
            raise ValueError("Filters must be a dictionary")
        if len(v) > MAX_METADATA_KEYS:
            raise ValueError(f"Filters must not exceed {MAX_METADATA_KEYS} keys")
        # Validate filter values are strings and within length limits
        sanitized = {}
        for key, value in v.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Filter keys and values must be strings")
            if len(value) > MAX_METADATA_VALUE_LENGTH:
                raise ValueError(f"Filter value for '{key}' exceeds maximum length")
            sanitized[key.strip()] = value.strip()
        return sanitized
    
    @field_validator('date_from', 'date_to')
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
        return v


class RankingExplanation(BaseModel):
    score: float
    factors: Dict[str, float] = Field(default_factory=dict)  # e.g., {"semantic_similarity": 0.85, "keyword_match": 0.12}
    explanation: str = ""  # Human-readable explanation


class SearchResult(BaseModel):
    id: str
    content: str
    similarity: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    highlighted_content: Optional[str] = None  # Content with query terms highlighted
    explanation: Optional[RankingExplanation] = None  # Ranking explanation
    preview: Optional[str] = None  # Short preview snippet


class FacetValue(BaseModel):
    value: str
    count: int


class Facet(BaseModel):
    field: str
    values: List[FacetValue]


class SearchResponse(BaseModel):
    backend: str
    results: List[SearchResult]
    total: int = 0  # Total number of results
    page: int = 1
    page_size: int = 10
    total_pages: int = 0
    facets: List[Facet] = Field(default_factory=list)  # Faceted search results
    query_time_ms: float = 0.0  # Query execution time


class HealthResponse(BaseModel):
    mode: str
    go_search: bool
    elasticsearch: bool
    redis: bool
    hana: bool


def validate_conversation_id(conv_id: Optional[str]) -> Optional[str]:
    """Validate conversation ID format (UUID)."""
    if conv_id is None:
        return None
    
    if not isinstance(conv_id, str):
        raise ValueError("Conversation ID must be a string")
    
    conv_id = conv_id.strip()
    
    # Validate UUID format
    try:
        uuid.UUID(conv_id)
    except ValueError:
        raise ValueError("Conversation ID must be a valid UUID")
    
    return conv_id


class AISearchRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    max_sources: int = Field(default=5, ge=1, le=10)
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        return sanitize_query(v)
    
    @field_validator('conversation_id')
    @classmethod
    def validate_conversation_id(cls, v: Optional[str]) -> Optional[str]:
        return validate_conversation_id(v)


class AISearchResponse(BaseModel):
    response: str
    sources: List[SearchResult]
    conversation_id: str
    follow_up_suggestions: List[str] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: float
    sources: List[SearchResult] = Field(default_factory=list)


class Conversation(BaseModel):
    id: str
    messages: List[ConversationMessage]
    created_at: float
    last_accessed: float


logger = logging.getLogger(__name__)

# Constants
DEFAULT_CACHE_TTL = 120  # seconds
CONVERSATION_EXPIRY_TIME = 3600  # 1 hour in seconds
SOURCE_CONTENT_TRUNCATE_LENGTH = 500  # characters
CONVERSATION_HISTORY_CONTEXT_MESSAGES = 3  # number of previous messages to include
LOCALAI_TIMEOUT = 30.0  # seconds
LOCALAI_MAX_TOKENS = 1000
FOLLOW_UP_SUGGESTIONS_COUNT = 3

# Input validation constants
MAX_QUERY_LENGTH = 1000  # characters
MAX_DOCUMENT_ID_LENGTH = 255  # characters
MAX_DOCUMENT_CONTENT_LENGTH = 1_000_000  # characters (1MB)
MIN_QUERY_LENGTH = 1  # characters
MAX_METADATA_KEYS = 50  # maximum number of metadata keys
MAX_METADATA_VALUE_LENGTH = 1000  # characters per metadata value


class GoSearchClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=10.0)

    def ping(self) -> bool:
        try:
            response = self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def search(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        response = self._client.post(f"{self.base_url}/v1/search", json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        data = response.json()
        return data.get("results", [])

    def add_document(self, doc: SearchDocument) -> None:
        response = self._client.post(
            f"{self.base_url}/v1/documents",
            json={"id": doc.id, "content": doc.content},
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

    def embed(self, text: str) -> Optional[List[float]]:
        try:
            response = self._client.post(
                f"{self.base_url}/v1/embed",
                json={"text": text},
            )
        except httpx.HTTPError as exc:
            logger.warning("Go search embed request failed: %s", exc)
            return None
        if response.status_code != 200:
            logger.warning("Go search embed returned status %s", response.status_code)
            return None
        data = response.json()
        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            return None
        return embedding


class ElasticConnector:
    def __init__(self, urls: List[str], api_key: Optional[str], username: Optional[str], password: Optional[str]):
        if Elasticsearch is None or not urls:
            self.client = None
            return
        auth: Dict[str, Any] = {}
        if api_key:
            auth["api_key"] = api_key
        elif username and password:
            auth["basic_auth"] = (username, password)
        self.client = Elasticsearch(urls, **auth)

    @property
    def available(self) -> bool:
        if self.client is None:
            return False
        try:
            return bool(self.client.ping())
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.debug(f"Elasticsearch connection check failed: {e}")
            return False
        except Exception as e:  # pragma: no cover - unexpected error
            logger.warning(f"Unexpected error checking Elasticsearch availability: {e}")
            return False

    def search(self, index: str, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.client is None:
            raise RuntimeError("elasticsearch client not available")
        response = self.client.search(index=index, body=body)
        return response.get("hits", {}).get("hits", [])

    def index_document(self, index: str, doc_id: str, document: Dict[str, Any]) -> None:
        if self.client is None:
            raise RuntimeError("elasticsearch client not available")
        self.client.index(index=index, id=doc_id, document=document, refresh="wait_for")


class RedisConnector:
    def __init__(self, url: Optional[str]) -> None:
        if redis is None or not url:
            self.client = None
        else:
            self.client = redis.Redis.from_url(url, decode_responses=True)

    @property
    def available(self) -> bool:
        if self.client is None:
            return False
        try:
            return bool(self.client.ping())
        except (ConnectionError, TimeoutError, OSError) as e:
            logger.debug(f"Redis connection check failed: {e}")
            return False
        except Exception as e:  # pragma: no cover - unexpected error
            logger.warning(f"Unexpected error checking Redis availability: {e}")
            return False

    def cache_search(self, key: str, payload: Dict[str, Any], ttl: int = DEFAULT_CACHE_TTL) -> None:
        if self.client is None:
            return
        try:
            self.client.setex(key, ttl, json.dumps(payload))
        except (redis.RedisError, json.JSONEncodeError, TypeError) as e:
            logger.debug(f"Redis cache write failed: {e}")
        except Exception as e:  # pragma: no cover - unexpected error
            logger.warning(f"Unexpected error caching search result: {e}")

    def get_cached_search(self, key: str) -> Optional[Dict[str, Any]]:
        if self.client is None:
            return None
        raw = self.client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:  # pragma: no cover - corrupted cache
            return None


class HANAConnector:
    def __init__(self, dsn: Optional[str]) -> None:
        self._conn = None
        if dsn and hdbapi is not None:
            try:
                self._conn = hdbapi.connect(dsn=dsn)
            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(f"HANA connection failed: {e}")
                self._conn = None
            except Exception as e:  # pragma: no cover - unexpected error
                logger.warning(f"Unexpected error connecting to HANA: {e}")
                self._conn = None

    @property
    def available(self) -> bool:
        return self._conn is not None


class InMemoryStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._docs: Dict[str, Dict[str, Any]] = {}

    def add(self, doc: SearchDocument, embedding: Optional[List[float]] = None) -> None:
        with self._lock:
            stored = {
                "id": doc.id,
                "content": doc.content,
                "metadata": dict(doc.metadata),
            }
            if embedding is not None:
                stored["embedding"] = embedding
            self._docs[doc.id] = stored

    def all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._docs.values())

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._docs.get(doc_id)


class ConversationManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._conversations: Dict[str, Conversation] = {}
        self._expiry_time = CONVERSATION_EXPIRY_TIME

    def create_conversation(self) -> str:
        conversation_id = str(uuid.uuid4())
        with self._lock:
            self._conversations[conversation_id] = Conversation(
                id=conversation_id,
                messages=[],
                created_at=time.time(),
                last_accessed=time.time()
            )
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv:
                conv.last_accessed = time.time()
            return conv

    def add_message(self, conversation_id: str, message: ConversationMessage) -> None:
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv:
                conv.messages.append(message)
                conv.last_accessed = time.time()

    def cleanup_expired(self) -> None:
        """Remove expired conversations. Thread-safe."""
        current_time = time.time()
        expired_ids = []
        
        # First pass: identify expired conversations while holding read lock
        with self._lock:
            for conv_id, conv in self._conversations.items():
                if current_time - conv.last_accessed > self._expiry_time:
                    expired_ids.append(conv_id)
        
        # Second pass: remove expired conversations (write lock)
        if expired_ids:
            with self._lock:
                # Re-check in case conversations were accessed between passes
                current_time = time.time()
                for conv_id in expired_ids[:]:  # Copy list to avoid modification during iteration
                    conv = self._conversations.get(conv_id)
                    if conv and current_time - conv.last_accessed > self._expiry_time:
                        del self._conversations[conv_id]


class AISearchOrchestrator:
    def __init__(self, search_orchestrator: "SearchOrchestrator") -> None:
        self.search_orchestrator = search_orchestrator
        self.conversation_manager = ConversationManager()
        self.localai_url = os.getenv("LOCALAI_BASE_URL", "http://localhost:8080")
        self.localai_model = os.getenv("LOCALAI_MODEL", "mistral-7b-instruct")

    def _build_prompt(self, query: str, sources: List[SearchResult], conversation_history: List[ConversationMessage] = None) -> str:
        prompt = "You are an AI assistant helping users search corporate knowledge. "
        prompt += "Use the following sources to answer the question. Cite sources using [1], [2], etc.\n\n"
        
        if sources:
            prompt += "Sources:\n"
            for i, source in enumerate(sources, 1):
                prompt += f"[{i}] {source.content[:SOURCE_CONTENT_TRUNCATE_LENGTH]}...\n"
            prompt += "\n"
        
        if conversation_history:
            prompt += "Previous conversation:\n"
            for msg in conversation_history[-CONVERSATION_HISTORY_CONTEXT_MESSAGES:]:
                prompt += f"{msg.role}: {msg.content}\n"
            prompt += "\n"
        
        prompt += f"Question: {query}\n\nAnswer:"
        return prompt

    def _stream_llm_response(self, prompt: str) -> str:
        """Stream response from LocalAI"""
        try:
            client = httpx.Client(timeout=LOCALAI_TIMEOUT)
            response = client.post(
                f"{self.localai_url}/v1/chat/completions",
                json={
                    "model": self.localai_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "temperature": 0.7,
                    "max_tokens": LOCALAI_MAX_TOKENS
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"LocalAI error: {response.status_code} - {response.text}")
                return "I'm sorry, I couldn't generate a response at this time."
            
            full_response = ""
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                full_response += delta["content"]
                    except json.JSONDecodeError:
                        continue
            
            return full_response
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            logger.error(f"Network error streaming LLM response: {e}")
            return "I'm sorry, I couldn't connect to the AI service. Please try again."
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error streaming LLM response: {e.response.status_code} - {e.response.text}")
            return "I'm sorry, the AI service returned an error. Please try again."
        except Exception as e:
            logger.error(f"Unexpected error streaming LLM response: {e}")
            return "I'm sorry, I encountered an error while generating a response."

    def _generate_follow_up_suggestions(self, query: str, response: str) -> List[str]:
        """Generate follow-up question suggestions"""
        suggestions = [
            "Can you provide more details about this?",
            "What are the key requirements?",
            "How does this relate to other policies?",
            "What are the next steps?",
            "Are there any exceptions to this rule?"
        ]
        return suggestions[:FOLLOW_UP_SUGGESTIONS_COUNT]

    def search_with_ai(self, request: AISearchRequest) -> AISearchResponse:
        # Get or create conversation
        conversation_id = request.conversation_id or self.conversation_manager.create_conversation()
        conversation = self.conversation_manager.get_conversation(conversation_id)
        
        if not conversation:
            conversation_id = self.conversation_manager.create_conversation()
            conversation = self.conversation_manager.get_conversation(conversation_id)

        # Search for relevant documents
        search_request = SearchRequest(
            query=request.query,
            top_k=request.max_sources
        )
        search_response = self.search_orchestrator.search(search_request)
        
        # Build prompt with sources and conversation history
        prompt = self._build_prompt(
            request.query,
            search_response.results,
            conversation.messages[-2:] if len(conversation.messages) > 0 else None
        )
        
        # Generate AI response
        ai_response = self._stream_llm_response(prompt)
        
        # Generate follow-up suggestions
        follow_ups = self._generate_follow_up_suggestions(request.query, ai_response)
        
        # Add messages to conversation
        user_message = ConversationMessage(
            role="user",
            content=request.query,
            timestamp=time.time(),
            sources=[]
        )
        assistant_message = ConversationMessage(
            role="assistant",
            content=ai_response,
            timestamp=time.time(),
            sources=search_response.results
        )
        
        self.conversation_manager.add_message(conversation_id, user_message)
        self.conversation_manager.add_message(conversation_id, assistant_message)
        
        return AISearchResponse(
            response=ai_response,
            sources=search_response.results,
            conversation_id=conversation_id,
            follow_up_suggestions=follow_ups
        )

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self.conversation_manager.get_conversation(conversation_id)


class SearchHistory:
    """Manages search history for users"""
    def __init__(self, max_history: int = 100):
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.lock = threading.Lock()
    
    def add(self, query: str, filters: Dict[str, str], results_count: int) -> None:
        with self.lock:
            entry = {
                "query": query,
                "filters": filters,
                "results_count": results_count,
                "timestamp": time.time()
            }
            self.history.insert(0, entry)
            if len(self.history) > self.max_history:
                self.history = self.history[:self.max_history]
    
    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self.lock:
            return self.history[:limit]
    
    def get_suggestions(self, prefix: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on prefix"""
        with self.lock:
            suggestions = set()
            prefix_lower = prefix.lower()
            for entry in self.history:
                query = entry["query"].lower()
                if query.startswith(prefix_lower) and query != prefix_lower:
                    suggestions.add(entry["query"])
                if len(suggestions) >= limit:
                    break
            return list(suggestions)[:limit]


class SearchOrchestrator:
    def __init__(self) -> None:
        go_url = os.getenv("GO_SEARCH_URL")
        self.go_client = GoSearchClient(go_url) if go_url else None
        self.search_history = SearchHistory()

        es_urls = [u.strip() for u in os.getenv("ELASTICSEARCH_URLS", "").split(",") if u.strip()]
        self.es_index = os.getenv("ELASTICSEARCH_INDEX", "agenticaieth-docs")
        self.elasticsearch = ElasticConnector(
            urls=es_urls,
            api_key=os.getenv("ELASTICSEARCH_API_KEY"),
            username=os.getenv("ELASTICSEARCH_USERNAME"),
            password=os.getenv("ELASTICSEARCH_PASSWORD"),
        )

        if not self.elasticsearch.available:
            logger.warning("Elasticsearch canonical backend unavailable; falling back to in-memory search")

        redis_url = os.getenv("REDIS_URL")
        self.redis = RedisConnector(redis_url)

        hana_dsn = os.getenv("HANA_DSN")
        self.hana = HANAConnector(hana_dsn)

        self.in_memory = InMemoryStore()

    def _ping_go(self) -> bool:
        return self.go_client.ping() if self.go_client else False

    @property
    def mode(self) -> str:
        if self.elasticsearch.available:
            return "canonical-elasticsearch"
        if self._ping_go():
            return "go-search"
        return "in-memory"

    def health(self) -> HealthResponse:
        return HealthResponse(
            mode=self.mode,
            go_search=self._ping_go(),
            elasticsearch=self.elasticsearch.available,
            redis=self.redis.available,
            hana=self.hana.available,
        )

    def add_document(self, doc: SearchDocument) -> None:
        embedding = self._embed_text(doc.content)

        body = {
            "content": doc.content,
            "metadata": dict(doc.metadata),
        }
        if embedding is not None:
            body["embedding"] = embedding

        if self.elasticsearch.available:
            logger.debug("Indexing document %s into Elasticsearch canonical store", doc.id)
            self.elasticsearch.index_document(self.es_index, doc.id, body)
        else:
            logger.warning("Canonical Elasticsearch unavailable; storing document %s in in-memory fallback", doc.id)
            self.in_memory.add(doc, embedding)

        if self._ping_go():
            try:
                self.go_client.add_document(doc)
            except HTTPException as exc:
                logger.warning("Secondary Go search ingest failed for %s: %s", doc.id, exc.detail)

    def search(self, request: SearchRequest) -> SearchResponse:
        # Apply date range filters if provided
        enhanced_filters = dict(request.filters)
        if request.date_from or request.date_to:
            date_filter = {}
            if request.date_from:
                date_filter["gte"] = request.date_from
            if request.date_to:
                date_filter["lte"] = request.date_to
            enhanced_filters["date"] = date_filter
        
        # Apply source filter
        if request.source:
            enhanced_filters["source"] = request.source
        
        # Apply document type filter
        if request.doc_type:
            enhanced_filters["type"] = request.doc_type
        
        # Create modified request with enhanced filters
        modified_request = SearchRequest(
            query=request.query,
            top_k=request.top_k * request.page_size,  # Get more results for pagination
            filters=enhanced_filters,
            page=request.page,
            page_size=request.page_size
        )
        
        cache_key = None
        if self.redis.available:
            cache_key = f"search::{request.query}::{json.dumps(enhanced_filters, sort_keys=True)}::{request.top_k}::{request.page}"
            cached = self.redis.get_cached_search(cache_key)
            if cached is not None:
                results = [SearchResult(**item) for item in cached.get("results", [])]
                return SearchResponse(backend=cached.get("backend", "cache"), results=results)

        query_embedding = self._embed_text(request.query)

        if self.elasticsearch.available:
            response = self._search_elasticsearch(modified_request, query_embedding)
        else:
            response = self._search_in_memory(modified_request, query_embedding)

        # Record search in history
        self.search_history.add(request.query, enhanced_filters, len(response.results))

        if cache_key and response.results:
            cached_payload = {
                "backend": response.backend,
                "results": [result.dict() for result in response.results],
            }
            self.redis.cache_search(cache_key, cached_payload)

        return response
    
    def highlight_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Highlight query terms in search results"""
        query_terms = query.lower().split()
        for result in results:
            content = result.content
            highlighted = content
            for term in query_terms:
                if term:
                    # Case-insensitive highlighting
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    highlighted = pattern.sub(lambda m: f"<mark>{m.group()}</mark>", highlighted)
            result.highlighted_content = highlighted
        return results
    
    def generate_preview(self, content: str, query: str, max_length: int = 200) -> str:
        """Generate a preview snippet around query terms"""
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        # Find first occurrence of any query term
        best_pos = -1
        for term in query_terms:
            if term:
                pos = content_lower.find(term)
                if pos != -1 and (best_pos == -1 or pos < best_pos):
                    best_pos = pos
        
        if best_pos == -1:
            # No match found, return beginning
            return content[:max_length] + ("..." if len(content) > max_length else "")
        
        # Extract snippet around match
        start = max(0, best_pos - max_length // 2)
        end = min(len(content), start + max_length)
        
        preview = content[start:end]
        if start > 0:
            preview = "..." + preview
        if end < len(content):
            preview = preview + "..."
        
        return preview
    
    def explain_rankings(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Generate ranking explanations for results"""
        query_terms = query.lower().split()
        for result in results:
            factors = {}
            explanation_parts = []
            
            # Semantic similarity factor
            if result.similarity > 0.8:
                factors["semantic_similarity"] = result.similarity
                explanation_parts.append(f"High semantic similarity ({result.similarity:.2f})")
            elif result.similarity > 0.5:
                factors["semantic_similarity"] = result.similarity
                explanation_parts.append(f"Moderate semantic similarity ({result.similarity:.2f})")
            
            # Keyword match factor
            content_lower = result.content.lower()
            keyword_matches = sum(1 for term in query_terms if term in content_lower)
            if keyword_matches > 0:
                keyword_score = keyword_matches / len(query_terms)
                factors["keyword_match"] = keyword_score
                explanation_parts.append(f"Matches {keyword_matches} query term(s)")
            
            # Metadata relevance
            if result.metadata:
                metadata_score = 0.1  # Small boost for having metadata
                factors["metadata_presence"] = metadata_score
                explanation_parts.append("Contains metadata")
            
            explanation = RankingExplanation(
                score=result.similarity,
                factors=factors,
                explanation="; ".join(explanation_parts) if explanation_parts else "Relevance score based on content matching"
            )
            result.explanation = explanation
        
        return results
    
    def calculate_facets(self, results: List[SearchResult], facet_fields: List[str]) -> List[Facet]:
        """Calculate facet counts for specified fields"""
        facets = {}
        for field in facet_fields:
            facet_values = defaultdict(int)
            for result in results:
                value = result.metadata.get(field, "Unknown")
                if isinstance(value, (str, int, float)):
                    facet_values[str(value)] += 1
                elif isinstance(value, list):
                    for v in value:
                        facet_values[str(v)] += 1
            
            facet_values_list = [
                FacetValue(value=k, count=v)
                for k, v in sorted(facet_values.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
            facets[field] = Facet(field=field, values=facet_values_list)
        
        return list(facets.values())

    def _embed_text(self, text: str) -> Optional[List[float]]:
        if not text.strip() or self.go_client is None:
            return None
        embedding = self.go_client.embed(text)
        if embedding is None:
            logger.debug("Embedding unavailable from Go service; falling back to text search")
        return embedding

    def _search_elasticsearch(self, request: SearchRequest, query_embedding: Optional[List[float]]) -> SearchResponse:
        backend_label = "elasticsearch-text"
        body: Dict[str, Any]
        if query_embedding:
            backend_label = "elasticsearch-knn"
            body = {
                "size": request.top_k,
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": request.top_k,
                    "num_candidates": max(request.top_k * 4, 50),
                },
            }
            if request.filters:
                body["query"] = {
                    "bool": {
                        "filter": [{"term": {k: v}} for k, v in request.filters.items()],
                    }
                }
        else:
            body = {
                "size": request.top_k,
                "query": {
                    "bool": {
                        "must": {"match": {"content": request.query}},
                    }
                },
            }
            if request.filters:
                body["query"]["bool"]["filter"] = [{"term": {k: v}} for k, v in request.filters.items()]

        hits = self.elasticsearch.search(self.es_index, body)
        results: List[SearchResult] = []
        for hit in hits:
            source = hit.get("_source", {})
            results.append(
                SearchResult(
                    id=hit.get("_id", ""),
                    content=source.get("content", ""),
                    similarity=float(hit.get("_score", 0.0)),
                    metadata=source.get("metadata", {}),
                )
            )
        return SearchResponse(backend=backend_label, results=results)

    def _search_in_memory(self, request: SearchRequest, query_embedding: Optional[List[float]]) -> SearchResponse:
        docs = self.in_memory.all()
        ranked: List[SearchResult] = []
        needle = request.query.lower()
        for doc in docs:
            metadata = doc.get("metadata", {})
            if not metadata_matches_filters(metadata, request.filters):
                continue
            content = doc.get("content", "")
            embedding = doc.get("embedding") if isinstance(doc.get("embedding"), list) else None
            if query_embedding is not None and embedding is not None:
                similarity = cosine_similarity(query_embedding, embedding)
            else:
                similarity = float(content.lower().count(needle))
                if needle in content.lower():
                    similarity += 1.0
            ranked.append(
                SearchResult(
                    id=doc.get("id", ""),
                    content=content,
                    similarity=similarity,
                    metadata=metadata,
                )
            )
        ranked.sort(key=lambda item: item.similarity, reverse=True)
        return SearchResponse(backend="in-memory", results=ranked[: request.top_k])


orchestrator = SearchOrchestrator()
ai_orchestrator = AISearchOrchestrator(orchestrator)
app = FastAPI(title="AgenticAiETH Search Gateway", version="0.1.0")

# Authentication setup
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
BEARER_SCHEME = HTTPBearer(auto_error=False)

# Rate limiting setup
class RateLimiter:
    """Simple in-memory rate limiter. For production, use Redis-based rate limiting."""
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
        self.lock = threading.Lock()
        self.cleanup_interval = 300  # Clean up old entries every 5 minutes
        self.last_cleanup = time.time()
    
    def is_allowed(self, key: str) -> tuple[bool, Optional[int]]:
        """Check if request is allowed. Returns (allowed, retry_after_seconds)."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)
        
        with self.lock:
            # Cleanup old entries periodically
            if time.time() - self.last_cleanup > self.cleanup_interval:
                self._cleanup(cutoff)
                self.last_cleanup = time.time()
            
            # Get requests in the last minute
            recent_requests = [req_time for req_time in self.requests[key] if req_time > cutoff]
            
            if len(recent_requests) >= self.requests_per_minute:
                # Calculate retry after
                oldest_request = min(recent_requests)
                retry_after = int((oldest_request + timedelta(minutes=1) - now).total_seconds()) + 1
                return False, retry_after
            
            # Add current request
            recent_requests.append(now)
            self.requests[key] = recent_requests
            return True, None
    
    def _cleanup(self, cutoff: datetime):
        """Remove old entries."""
        keys_to_remove = []
        for key, requests in self.requests.items():
            recent = [req for req in requests if req > cutoff]
            if recent:
                self.requests[key] = recent
            else:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.requests[key]

# Initialize rate limiter
rate_limiter = RateLimiter(requests_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")))


def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Try to get from X-Forwarded-For header (if behind proxy)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP (original client)
        return forwarded_for.split(",")[0].strip()
    
    # Fall back to direct client IP
    if request.client:
        return request.client.host
    
    return "unknown"


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(API_KEY_HEADER),
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(BEARER_SCHEME),
) -> str:
    """Verify API key from header or Bearer token."""
    # Check if authentication is enabled
    auth_enabled = os.getenv("AUTH_ENABLED", "false").lower() == "true"
    if not auth_enabled:
        return "anonymous"
    
    # Get API key from header or Bearer token
    provided_key = api_key
    if not provided_key and bearer:
        provided_key = bearer.credentials
    
    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header or Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get valid API keys from environment
    valid_keys_str = os.getenv("API_KEYS", "")
    if not valid_keys_str:
        # If no keys configured, allow all (for development)
        logger.warning("API_KEYS not configured - allowing all requests")
        return "anonymous"
    
    valid_keys = [key.strip() for key in valid_keys_str.split(",") if key.strip()]
    
    # Use constant-time comparison to prevent timing attacks
    for valid_key in valid_keys:
        if secrets.compare_digest(provided_key, valid_key):
            return "authenticated"
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key",
    )


async def check_rate_limit(request: Request) -> None:
    """Check rate limit for the request."""
    rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    if not rate_limit_enabled:
        return
    
    # Skip rate limiting for health endpoint
    if request.url.path == "/health":
        return
    
    client_id = get_client_identifier(request)
    allowed, retry_after = rate_limiter.is_allowed(client_id)
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


@app.get("/health", response_model=HealthResponse)
def read_health() -> HealthResponse:
    return orchestrator.health()


@app.post("/v1/documents", status_code=204)
def add_document(doc: SearchDocument) -> None:
    orchestrator.add_document(doc)


@app.post("/v1/search", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    import time
    start_time = time.time()
    
    response = orchestrator.search(request)
    
    # Calculate pagination
    total = len(response.results)
    page = request.page
    page_size = request.page_size
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0
    
    # Apply pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_results = response.results[start_idx:end_idx]
    
    # Add highlighting if requested
    if request.include_highlighting:
        paginated_results = orchestrator.highlight_results(paginated_results, request.query)
    
    # Add preview snippets
    for result in paginated_results:
        if not result.preview:
            result.preview = orchestrator.generate_preview(result.content, request.query)
    
    # Add ranking explanations if requested
    if request.include_explanation:
        paginated_results = orchestrator.explain_rankings(paginated_results, request.query)
    
    # Calculate facets if requested
    facets = []
    if request.facets:
        facets = orchestrator.calculate_facets(response.results, request.facets)
    
    query_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return SearchResponse(
        backend=response.backend,
        results=paginated_results,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        facets=facets,
        query_time_ms=query_time
    )


@app.get("/v1/search/history")
def get_search_history(limit: int = 10) -> Dict[str, Any]:
    """Get recent search history"""
    history = orchestrator.search_history.get_recent(limit)
    return {"history": history, "count": len(history)}


@app.get("/v1/search/suggestions")
def get_search_suggestions(prefix: str, limit: int = 5) -> Dict[str, Any]:
    """Get search suggestions based on prefix"""
    suggestions = orchestrator.search_history.get_suggestions(prefix, limit)
    return {"suggestions": suggestions, "count": len(suggestions)}


@app.post("/v1/search/feedback")
def submit_search_feedback(
    result_id: str,
    query: str,
    helpful: bool,
    comment: Optional[str] = None
) -> Dict[str, str]:
    """Submit feedback on search results (thumbs up/down)"""
    # Store feedback (could be stored in Redis or database)
    feedback_entry = {
        "result_id": result_id,
        "query": query,
        "helpful": helpful,
        "comment": comment,
        "timestamp": time.time()
    }
    
    # In a real implementation, this would be stored persistently
    logger.info(f"Search feedback received: {feedback_entry}")
    
    return {"status": "success", "message": "Feedback recorded"}


@app.get("/v1/search/analytics")
def get_search_analytics(
    days: int = 7,
    limit: int = 20
) -> Dict[str, Any]:
    """Get search analytics (trends, popular queries, etc.)"""
    history = orchestrator.search_history.get_recent(limit * 10)  # Get more for analysis
    
    # Calculate popular queries
    query_counts = defaultdict(int)
    for entry in history:
        query_counts[entry["query"]] += 1
    
    popular_queries = [
        {"query": q, "count": c}
        for q, c in sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
    ]
    
    # Calculate average results count
    avg_results = sum(entry["results_count"] for entry in history) / len(history) if history else 0
    
    return {
        "popular_queries": popular_queries,
        "total_searches": len(history),
        "average_results_per_query": avg_results,
        "period_days": days
    }


@app.post("/v1/search/export")
def export_search_results(
    request: SearchRequest,
    format: str = "json"  # json or csv
) -> StreamingResponse:
    """Export search results in various formats"""
    response = orchestrator.search(request)
    
    if format.lower() == "csv":
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(["ID", "Content", "Similarity", "Metadata"])
        
        # Write data
        for result in response.results:
            metadata_str = json.dumps(result.metadata)
            writer.writerow([result.id, result.content, result.similarity, metadata_str])
        
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=search_results_{int(time.time())}.csv"}
        )
    else:  # JSON
        results_data = [result.dict() for result in response.results]
        json_data = json.dumps(results_data, indent=2)
        return StreamingResponse(
            iter([json_data]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=search_results_{int(time.time())}.json"}
        )


@app.post("/v1/ai-search", response_model=AISearchResponse)
def ai_search(request: AISearchRequest) -> AISearchResponse:
    """AI-powered conversational search with source citations"""
    return ai_orchestrator.search_with_ai(request)


@app.get("/v1/conversation/{conversation_id}", response_model=Conversation, dependencies=[Depends(check_rate_limit)])
def get_conversation(
    conversation_id: str = Path(..., description="Conversation UUID"),
    api_key: str = Depends(verify_api_key),
) -> Conversation:
    """Get conversation history"""
    # Validate conversation ID format
    try:
        conversation_id = validate_conversation_id(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    conversation = ai_orchestrator.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.get("/v1/sources/{doc_id}", dependencies=[Depends(check_rate_limit)])
def get_source_document(
    doc_id: str = Path(..., description="Document ID"),
    api_key: str = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get full source document by ID"""
    # Validate document ID format
    try:
        doc_id = validate_document_id(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Try to get from Elasticsearch first
    if orchestrator.elasticsearch.available:
        try:
            response = orchestrator.elasticsearch.client.get(
                index=orchestrator.es_index,
                id=doc_id
            )
            return response["_source"]
        except Exception as e:
            # Check if it's a not found error (404)
            if hasattr(e, 'status_code') and e.status_code == 404:
                logger.debug(f"Document {doc_id} not found in Elasticsearch")
            else:
                logger.warning(f"Failed to get document {doc_id} from Elasticsearch: {e}")
    
    # Fallback to in-memory store
    doc = orchestrator.in_memory.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def metadata_matches_filters(metadata: Dict[str, Any], filters: Dict[str, str]) -> bool:
    if not filters:
        return True
    for raw_key, raw_value in filters.items():
        key = raw_key.strip()
        expected = raw_value.strip()
        if not key or not expected:
            continue
        if key.startswith("metadata."):
            field = key.partition(".")[2]
            actual = str(metadata.get(field, "")).strip()
        else:
            actual = str(metadata.get(key, "")).strip()
        if actual != expected:
            return False
    return True


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
