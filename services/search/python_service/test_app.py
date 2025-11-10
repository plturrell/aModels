"""
Unit tests for the search service application.
"""
import pytest
import time
import uuid
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from app import (
    app,
    SearchDocument,
    SearchRequest,
    AISearchRequest,
    SearchOrchestrator,
    ConversationManager,
    InMemoryStore,
    sanitize_query,
    validate_document_id,
    validate_metadata,
    validate_conversation_id,
    cosine_similarity,
    metadata_matches_filters,
    MAX_QUERY_LENGTH,
    MIN_QUERY_LENGTH,
    MAX_DOCUMENT_ID_LENGTH,
    MAX_DOCUMENT_CONTENT_LENGTH,
)


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return SearchDocument(
        id="test-doc-1",
        content="This is a test document about vacation policies.",
        metadata={"category": "HR", "type": "policy"}
    )


@pytest.fixture
def in_memory_store():
    """Create an in-memory store for testing."""
    return InMemoryStore()


@pytest.fixture
def conversation_manager():
    """Create a conversation manager for testing."""
    return ConversationManager()


class TestInputValidation:
    """Test input validation and sanitization functions."""
    
    def test_sanitize_query_valid(self):
        """Test sanitizing a valid query."""
        query = "vacation policy"
        result = sanitize_query(query)
        assert result == "vacation policy"
    
    def test_sanitize_query_strips_whitespace(self):
        """Test that query whitespace is stripped."""
        query = "  vacation policy  "
        result = sanitize_query(query)
        assert result == "vacation policy"
    
    def test_sanitize_query_removes_control_chars(self):
        """Test that control characters are removed."""
        query = "vacation\x00policy\x01test"
        result = sanitize_query(query)
        assert "\x00" not in result
        assert "\x01" not in result
    
    def test_sanitize_query_too_short(self):
        """Test that empty query raises error."""
        with pytest.raises(ValueError, match="at least"):
            sanitize_query("")
    
    def test_sanitize_query_too_long(self):
        """Test that query exceeding max length raises error."""
        long_query = "a" * (MAX_QUERY_LENGTH + 1)
        with pytest.raises(ValueError, match="exceed"):
            sanitize_query(long_query)
    
    def test_validate_document_id_valid(self):
        """Test validating a valid document ID."""
        doc_id = "test-doc-123"
        result = validate_document_id(doc_id)
        assert result == "test-doc-123"
    
    def test_validate_document_id_invalid_characters(self):
        """Test that invalid characters in ID raise error."""
        with pytest.raises(ValueError, match="alphanumeric"):
            validate_document_id("test@doc#123")
    
    def test_validate_document_id_too_long(self):
        """Test that ID exceeding max length raises error."""
        long_id = "a" * (MAX_DOCUMENT_ID_LENGTH + 1)
        with pytest.raises(ValueError, match="exceed"):
            validate_document_id(long_id)
    
    def test_validate_metadata_valid(self):
        """Test validating valid metadata."""
        metadata = {"key1": "value1", "key2": "value2"}
        result = validate_metadata(metadata)
        assert result == metadata
    
    def test_validate_metadata_too_many_keys(self):
        """Test that too many metadata keys raise error."""
        metadata = {f"key{i}": f"value{i}" for i in range(51)}
        with pytest.raises(ValueError, match="exceed"):
            validate_metadata(metadata)
    
    def test_validate_conversation_id_valid_uuid(self):
        """Test validating a valid UUID conversation ID."""
        conv_id = str(uuid.uuid4())
        result = validate_conversation_id(conv_id)
        assert result == conv_id
    
    def test_validate_conversation_id_invalid_uuid(self):
        """Test that invalid UUID raises error."""
        with pytest.raises(ValueError, match="valid UUID"):
            validate_conversation_id("not-a-uuid")
    
    def test_validate_conversation_id_none(self):
        """Test that None conversation ID is allowed."""
        result = validate_conversation_id(None)
        assert result is None


class TestSearchDocument:
    """Test SearchDocument model validation."""
    
    def test_valid_document(self, sample_document):
        """Test that a valid document passes validation."""
        assert sample_document.id == "test-doc-1"
        assert sample_document.content == "This is a test document about vacation policies."
    
    def test_document_id_validation(self):
        """Test that invalid document ID raises error."""
        with pytest.raises(ValueError):
            SearchDocument(
                id="invalid@id",
                content="Test content"
            )
    
    def test_document_content_too_long(self):
        """Test that content exceeding max length raises error."""
        long_content = "a" * (MAX_DOCUMENT_CONTENT_LENGTH + 1)
        with pytest.raises(ValueError, match="exceed"):
            SearchDocument(
                id="test-1",
                content=long_content
            )
    
    def test_document_metadata_validation(self):
        """Test that invalid metadata raises error."""
        too_many_keys = {f"key{i}": f"value{i}" for i in range(51)}
        with pytest.raises(ValueError, match="exceed"):
            SearchDocument(
                id="test-1",
                content="Test content",
                metadata=too_many_keys
            )


class TestSearchRequest:
    """Test SearchRequest model validation."""
    
    def test_valid_search_request(self):
        """Test that a valid search request passes validation."""
        request = SearchRequest(query="vacation policy", top_k=10)
        assert request.query == "vacation policy"
        assert request.top_k == 10
    
    def test_search_request_query_validation(self):
        """Test that invalid query raises error."""
        with pytest.raises(ValueError):
            SearchRequest(query="", top_k=10)
    
    def test_search_request_top_k_bounds(self):
        """Test that top_k is within bounds."""
        # Should default to 10 if not specified
        request = SearchRequest(query="test")
        assert request.top_k == 10
        
        # Should raise error if out of bounds
        with pytest.raises(ValueError):
            SearchRequest(query="test", top_k=0)
        
        with pytest.raises(ValueError):
            SearchRequest(query="test", top_k=201)


class TestInMemoryStore:
    """Test InMemoryStore functionality."""
    
    def test_add_document(self, in_memory_store, sample_document):
        """Test adding a document to the store."""
        in_memory_store.add(sample_document)
        doc = in_memory_store.get(sample_document.id)
        assert doc is not None
        assert doc["id"] == sample_document.id
        assert doc["content"] == sample_document.content
    
    def test_get_nonexistent_document(self, in_memory_store):
        """Test getting a document that doesn't exist."""
        doc = in_memory_store.get("nonexistent")
        assert doc is None
    
    def test_add_document_with_embedding(self, in_memory_store, sample_document):
        """Test adding a document with embedding."""
        embedding = [0.1, 0.2, 0.3]
        in_memory_store.add(sample_document, embedding)
        doc = in_memory_store.get(sample_document.id)
        assert doc["embedding"] == embedding
    
    def test_get_all_documents(self, in_memory_store, sample_document):
        """Test getting all documents."""
        in_memory_store.add(sample_document)
        docs = in_memory_store.all()
        assert len(docs) == 1
        assert docs[0]["id"] == sample_document.id


class TestConversationManager:
    """Test ConversationManager functionality."""
    
    def test_create_conversation(self, conversation_manager):
        """Test creating a new conversation."""
        conv_id = conversation_manager.create_conversation()
        assert conv_id is not None
        assert isinstance(conv_id, str)
        
        conv = conversation_manager.get_conversation(conv_id)
        assert conv is not None
        assert conv.id == conv_id
        assert len(conv.messages) == 0
    
    def test_get_nonexistent_conversation(self, conversation_manager):
        """Test getting a conversation that doesn't exist."""
        conv = conversation_manager.get_conversation("nonexistent")
        assert conv is None
    
    def test_add_message(self, conversation_manager):
        """Test adding a message to a conversation."""
        conv_id = conversation_manager.create_conversation()
        
        from app import ConversationMessage
        message = ConversationMessage(
            role="user",
            content="Test message",
            timestamp=time.time()
        )
        
        conversation_manager.add_message(conv_id, message)
        conv = conversation_manager.get_conversation(conv_id)
        assert len(conv.messages) == 1
        assert conv.messages[0].content == "Test message"
    
    def test_cleanup_expired(self, conversation_manager):
        """Test cleaning up expired conversations."""
        # Create a conversation
        conv_id = conversation_manager.create_conversation()
        conv = conversation_manager.get_conversation(conv_id)
        
        # Manually set last_accessed to expired time
        conv.last_accessed = time.time() - 4000  # 4000 seconds ago
        
        # Cleanup should remove it
        conversation_manager.cleanup_expired()
        assert conversation_manager.get_conversation(conv_id) is None


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        similarity = cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 0.001  # Should be 1.0 (identical vectors)
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity for orthogonal vectors."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        similarity = cosine_similarity(a, b)
        assert abs(similarity - 0.0) < 0.001  # Should be 0.0 (orthogonal)
    
    def test_cosine_similarity_different_lengths(self):
        """Test cosine similarity with different length vectors."""
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        similarity = cosine_similarity(a, b)
        assert similarity == 0.0  # Should return 0.0 for mismatched lengths
    
    def test_metadata_matches_filters(self):
        """Test metadata filter matching."""
        metadata = {"category": "HR", "type": "policy"}
        
        # Should match
        assert metadata_matches_filters(metadata, {"category": "HR"}) is True
        assert metadata_matches_filters(metadata, {"type": "policy"}) is True
        
        # Should not match
        assert metadata_matches_filters(metadata, {"category": "IT"}) is False
        assert metadata_matches_filters(metadata, {"status": "active"}) is False
    
    def test_metadata_matches_filters_empty(self):
        """Test that empty filters match all."""
        metadata = {"category": "HR"}
        assert metadata_matches_filters(metadata, {}) is True


class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "mode" in data
        assert "go_search" in data
        assert "elasticsearch" in data
    
    def test_add_document_endpoint(self, client, sample_document):
        """Test adding a document via API."""
        response = client.post("/v1/documents", json=sample_document.dict())
        assert response.status_code == 204
    
    def test_add_document_invalid_id(self, client):
        """Test adding document with invalid ID."""
        doc = {
            "id": "invalid@id",
            "content": "Test content"
        }
        response = client.post("/v1/documents", json=doc)
        assert response.status_code == 422  # Validation error
    
    def test_search_endpoint(self, client):
        """Test search endpoint."""
        request = {"query": "test query", "top_k": 5}
        response = client.post("/v1/search", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "backend" in data
    
    def test_search_endpoint_invalid_query(self, client):
        """Test search with invalid query."""
        request = {"query": "", "top_k": 5}
        response = client.post("/v1/search", json=request)
        assert response.status_code == 422  # Validation error
    
    def test_get_conversation_invalid_uuid(self, client):
        """Test getting conversation with invalid UUID."""
        response = client.get("/v1/conversation/invalid-uuid")
        assert response.status_code == 400
    
    def test_get_source_document_invalid_id(self, client):
        """Test getting source document with invalid ID."""
        response = client.get("/v1/sources/invalid@id")
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

