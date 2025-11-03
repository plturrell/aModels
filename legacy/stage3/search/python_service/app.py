import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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


class SearchDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=200)
    filters: Dict[str, str] = Field(default_factory=dict)


class SearchResult(BaseModel):
    id: str
    content: str
    similarity: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    backend: str
    results: List[SearchResult]


class HealthResponse(BaseModel):
    mode: str
    go_search: bool
    elasticsearch: bool
    redis: bool
    hana: bool


class AISearchRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    max_sources: int = Field(default=5, ge=1, le=10)


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
        except Exception:  # pragma: no cover - network failure
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
        except Exception:  # pragma: no cover - network failure
            return False

    def cache_search(self, key: str, payload: Dict[str, Any], ttl: int = 120) -> None:
        if self.client is None:
            return
        try:
            self.client.setex(key, ttl, json.dumps(payload))
        except Exception:  # pragma: no cover - serialization failure
            pass

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
            except Exception:
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
        self._expiry_time = 3600  # 1 hour

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
        current_time = time.time()
        with self._lock:
            expired_ids = [
                conv_id for conv_id, conv in self._conversations.items()
                if current_time - conv.last_accessed > self._expiry_time
            ]
            for conv_id in expired_ids:
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
                prompt += f"[{i}] {source.content[:500]}...\n"
            prompt += "\n"
        
        if conversation_history:
            prompt += "Previous conversation:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages for context
                prompt += f"{msg.role}: {msg.content}\n"
            prompt += "\n"
        
        prompt += f"Question: {query}\n\nAnswer:"
        return prompt

    def _stream_llm_response(self, prompt: str) -> str:
        """Stream response from LocalAI"""
        try:
            client = httpx.Client(timeout=30.0)
            response = client.post(
                f"{self.localai_url}/v1/chat/completions",
                json={
                    "model": self.localai_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "temperature": 0.7,
                    "max_tokens": 1000
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
        except Exception as e:
            logger.error(f"Error streaming LLM response: {e}")
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
        return suggestions[:3]  # Return top 3 suggestions

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


class SearchOrchestrator:
    def __init__(self) -> None:
        go_url = os.getenv("GO_SEARCH_URL")
        self.go_client = GoSearchClient(go_url) if go_url else None

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
        cache_key = None
        if self.redis.available:
            cache_key = f"search::{request.query}::{json.dumps(request.filters, sort_keys=True)}::{request.top_k}"
            cached = self.redis.get_cached_search(cache_key)
            if cached is not None:
                results = [SearchResult(**item) for item in cached.get("results", [])]
                return SearchResponse(backend=cached.get("backend", "cache"), results=results)

        query_embedding = self._embed_text(request.query)

        if self.elasticsearch.available:
            response = self._search_elasticsearch(request, query_embedding)
        else:
            response = self._search_in_memory(request, query_embedding)

        if cache_key and response.results:
            cached_payload = {
                "backend": response.backend,
                "results": [result.dict() for result in response.results],
            }
            self.redis.cache_search(cache_key, cached_payload)

        return response

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


@app.get("/health", response_model=HealthResponse)
def read_health() -> HealthResponse:
    return orchestrator.health()


@app.post("/v1/documents", status_code=204)
def add_document(doc: SearchDocument) -> None:
    orchestrator.add_document(doc)


@app.post("/v1/search", response_model=SearchResponse)
def search_documents(request: SearchRequest) -> SearchResponse:
    return orchestrator.search(request)


@app.post("/v1/ai-search", response_model=AISearchResponse)
def ai_search(request: AISearchRequest) -> AISearchResponse:
    """AI-powered conversational search with source citations"""
    return ai_orchestrator.search_with_ai(request)


@app.get("/v1/conversation/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str) -> Conversation:
    """Get conversation history"""
    conversation = ai_orchestrator.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.get("/v1/sources/{doc_id}")
def get_source_document(doc_id: str) -> Dict[str, Any]:
    """Get full source document by ID"""
    # Try to get from Elasticsearch first
    if orchestrator.elasticsearch.available:
        try:
            response = orchestrator.elasticsearch.client.get(
                index=orchestrator.es_index,
                id=doc_id
            )
            return response["_source"]
        except Exception as e:
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
