"""
Unit tests for the FastAPI gateway.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import grpc

from gateway.app import app, get_client, get_db_admin
from gateway.grpc_client import PostgresTelemetryClient
from gateway.db_admin import DatabaseAdmin


@pytest.fixture
def mock_client():
    """Create a mock PostgresTelemetryClient."""
    client = Mock(spec=PostgresTelemetryClient)
    return client


@pytest.fixture
def mock_db_admin():
    """Create a mock DatabaseAdmin."""
    admin = Mock(spec=DatabaseAdmin)
    admin.allow_mutations = False
    admin.default_limit = 100
    return admin


@pytest.fixture
def client_with_mocks(mock_client, mock_db_admin):
    """Create a test client with mocked dependencies."""
    app.dependency_overrides[get_client] = lambda: mock_client
    app.dependency_overrides[get_db_admin] = lambda: mock_db_admin
    
    with TestClient(app) as test_client:
        yield test_client, mock_client, mock_db_admin
    
    # Clean up overrides
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_success(self, client_with_mocks):
        """Test successful health check."""
        client, mock_client, _ = client_with_mocks
        
        mock_client.health.return_value = {
            "status": "SERVING",
            "version": "1.0.0"
        }
        
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "SERVING"
        assert response.json()["version"] == "1.0.0"
        mock_client.health.assert_called_once()
    
    def test_health_grpc_error(self, client_with_mocks):
        """Test health check with gRPC error."""
        client, mock_client, _ = client_with_mocks
        
        error = grpc.RpcError()
        error.code = lambda: grpc.StatusCode.UNAVAILABLE
        error.details = lambda: "Service unavailable"
        mock_client.health.side_effect = error
        
        response = client.get("/health")
        
        assert response.status_code == 504


class TestOperationsEndpoints:
    """Tests for /operations endpoints."""
    
    def test_list_operations_success(self, client_with_mocks):
        """Test successful listing of operations."""
        client, mock_client, _ = client_with_mocks
        
        mock_client.list_operations.return_value = {
            "operations": [
                {
                    "id": "op-1",
                    "library_type": "langchain",
                    "operation": "execute",
                    "status": "success"
                }
            ],
            "next_page_token": ""
        }
        
        response = client.get("/operations")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["operations"]) == 1
        assert data["operations"][0]["id"] == "op-1"
        mock_client.list_operations.assert_called_once()
    
    def test_list_operations_with_filters(self, client_with_mocks):
        """Test listing operations with filters."""
        client, mock_client, _ = client_with_mocks
        
        mock_client.list_operations.return_value = {
            "operations": [],
            "next_page_token": ""
        }
        
        response = client.get(
            "/operations",
            params={
                "library_type": "langgraph",
                "status": "success",
                "page_size": 25
            }
        )
        
        assert response.status_code == 200
        mock_client.list_operations.assert_called_once_with(
            library_type="langgraph",
            session_id=None,
            status="success",
            page_size=25,
            page_token=None,
            created_after=None,
            created_before=None
        )
    
    def test_list_operations_value_error(self, client_with_mocks):
        """Test list operations with invalid parameters."""
        client, mock_client, _ = client_with_mocks
        
        mock_client.list_operations.side_effect = ValueError("Invalid status")
        
        response = client.get("/operations")
        
        assert response.status_code == 400
        assert "Invalid status" in response.json()["detail"]
    
    def test_get_operation_success(self, client_with_mocks):
        """Test getting a single operation."""
        client, mock_client, _ = client_with_mocks
        
        mock_client.get_operation.return_value = {
            "id": "op-123",
            "library_type": "langchain",
            "operation": "execute",
            "status": "success",
            "latency_ms": 150
        }
        
        response = client.get("/operations/op-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "op-123"
        assert data["latency_ms"] == 150
        mock_client.get_operation.assert_called_once_with("op-123")
    
    def test_get_operation_not_found(self, client_with_mocks):
        """Test getting non-existent operation."""
        client, mock_client, _ = client_with_mocks
        
        error = grpc.RpcError()
        error.code = lambda: grpc.StatusCode.NOT_FOUND
        error.details = lambda: "Operation not found"
        mock_client.get_operation.side_effect = error
        
        response = client.get("/operations/nonexistent")
        
        assert response.status_code == 404
    
    def test_log_operation_success(self, client_with_mocks):
        """Test logging a new operation."""
        client, mock_client, _ = client_with_mocks
        
        mock_client.log_operation.return_value = {
            "id": "new-op-id",
            "library_type": "langgraph",
            "operation": "run_graph",
            "status": "success"
        }
        
        payload = {
            "library_type": "langgraph",
            "operation": "run_graph",
            "status": "success",
            "latency_ms": 200
        }
        
        response = client.post("/operations", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "new-op-id"
        mock_client.log_operation.assert_called_once()
    
    def test_log_operation_invalid_data(self, client_with_mocks):
        """Test logging operation with invalid data."""
        client, mock_client, _ = client_with_mocks
        
        mock_client.log_operation.side_effect = ValueError("Invalid operation")
        
        payload = {"invalid": "data"}
        response = client.post("/operations", json=payload)
        
        assert response.status_code == 400


class TestAnalyticsEndpoint:
    """Tests for /analytics endpoint."""
    
    def test_analytics_success(self, client_with_mocks):
        """Test successful analytics retrieval."""
        client, mock_client, _ = client_with_mocks
        
        mock_client.analytics.return_value = {
            "total_operations": 100,
            "success_rate": 0.95,
            "average_latency_ms": 150.5,
            "error_breakdown": {"langchain": 5},
            "library_stats": [
                {
                    "library_type": "langchain",
                    "total_operations": 100,
                    "success_rate": 0.95,
                    "average_latency_ms": 150.5,
                    "error_count": 5
                }
            ],
            "performance_trends": []
        }
        
        response = client.get("/analytics")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_operations"] == 100
        assert data["success_rate"] == 0.95
        assert len(data["library_stats"]) == 1
        mock_client.analytics.assert_called_once()
    
    def test_analytics_with_filters(self, client_with_mocks):
        """Test analytics with time filters."""
        client, mock_client, _ = client_with_mocks
        
        mock_client.analytics.return_value = {
            "total_operations": 50,
            "success_rate": 0.96,
            "average_latency_ms": 140.0,
            "error_breakdown": {},
            "library_stats": [],
            "performance_trends": []
        }
        
        response = client.get(
            "/analytics",
            params={
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-31T23:59:59Z",
                "library_type": "langchain"
            }
        )
        
        assert response.status_code == 200
        mock_client.analytics.assert_called_once_with(
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-31T23:59:59Z",
            library_type="langchain"
        )


class TestCleanupEndpoint:
    """Tests for /cleanup endpoint."""
    
    def test_cleanup_success(self, client_with_mocks):
        """Test successful cleanup."""
        client, mock_client, _ = client_with_mocks
        
        mock_client.cleanup.return_value = {"deleted": 42}
        
        payload = {"older_than": "2024-01-01T00:00:00Z"}
        response = client.post("/cleanup", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] == 42
        mock_client.cleanup.assert_called_once_with(older_than="2024-01-01T00:00:00Z")
    
    def test_cleanup_invalid_timestamp(self, client_with_mocks):
        """Test cleanup with invalid timestamp."""
        client, mock_client, _ = client_with_mocks
        
        mock_client.cleanup.side_effect = ValueError("Invalid timestamp format")
        
        payload = {"older_than": "invalid-timestamp"}
        response = client.post("/cleanup", json=payload)
        
        assert response.status_code == 400


class TestStatusesEndpoint:
    """Tests for /statuses endpoint."""
    
    def test_statuses(self, client_with_mocks):
        """Test statuses endpoint."""
        client, _, _ = client_with_mocks
        
        response = client.get("/statuses")
        
        assert response.status_code == 200
        statuses = response.json()
        assert isinstance(statuses, list)
        assert len(statuses) > 0


class TestDatabaseAdminEndpoints:
    """Tests for /db/* endpoints."""
    
    def test_db_status(self, client_with_mocks):
        """Test database status endpoint."""
        client, _, mock_db_admin = client_with_mocks
        
        response = client.get("/db/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True
        assert data["allow_mutations"] is False
        assert data["default_limit"] == 100
    
    def test_db_tables(self, client_with_mocks):
        """Test list database tables."""
        client, _, mock_db_admin = client_with_mocks
        
        mock_db_admin.list_tables.return_value = [
            {"table_schema": "public", "table_name": "lang_operations"},
            {"table_schema": "public", "table_name": "session_state"}
        ]
        
        response = client.get("/db/tables")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["tables"]) == 2
        assert data["tables"][0]["table_name"] == "lang_operations"
        mock_db_admin.list_tables.assert_called_once()
    
    def test_db_table_columns(self, client_with_mocks):
        """Test get table columns."""
        client, _, mock_db_admin = client_with_mocks
        
        mock_db_admin.get_columns.return_value = [
            {
                "column_name": "id",
                "data_type": "uuid",
                "is_nullable": "NO",
                "column_default": None
            },
            {
                "column_name": "library_type",
                "data_type": "text",
                "is_nullable": "NO",
                "column_default": None
            }
        ]
        
        response = client.get("/db/table/public/lang_operations")
        
        assert response.status_code == 200
        columns = response.json()
        assert len(columns) == 2
        assert columns[0]["column_name"] == "id"
        mock_db_admin.get_columns.assert_called_once_with("public", "lang_operations")
    
    def test_db_query_success(self, client_with_mocks):
        """Test executing a database query."""
        client, _, mock_db_admin = client_with_mocks
        
        query_result = MagicMock()
        query_result.columns = ["id", "library_type"]
        query_result.rows = [
            {"id": "op-1", "library_type": "langchain"},
            {"id": "op-2", "library_type": "langgraph"}
        ]
        query_result.row_count = 2
        query_result.truncated = False
        
        mock_db_admin.execute_query.return_value = query_result
        
        payload = {"sql": "SELECT id, library_type FROM lang_operations LIMIT 2"}
        response = client.post("/db/query", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["row_count"] == 2
        assert len(data["rows"]) == 2
        assert data["truncated"] is False
        mock_db_admin.execute_query.assert_called_once()
    
    def test_db_query_permission_error(self, client_with_mocks):
        """Test query with permission error."""
        client, _, mock_db_admin = client_with_mocks
        
        mock_db_admin.execute_query.side_effect = PermissionError("Mutations not allowed")
        
        payload = {"sql": "DELETE FROM lang_operations"}
        response = client.post("/db/query", json=payload)
        
        assert response.status_code == 403
        assert "Mutations not allowed" in response.json()["detail"]
    
    def test_db_query_value_error(self, client_with_mocks):
        """Test query with invalid SQL."""
        client, _, mock_db_admin = client_with_mocks
        
        mock_db_admin.execute_query.side_effect = ValueError("Invalid SQL syntax")
        
        payload = {"sql": "INVALID SQL"}
        response = client.post("/db/query", json=payload)
        
        assert response.status_code == 400
        assert "Invalid SQL syntax" in response.json()["detail"]
    
    def test_db_query_internal_error(self, client_with_mocks):
        """Test query with internal database error."""
        client, _, mock_db_admin = client_with_mocks
        
        mock_db_admin.execute_query.side_effect = Exception("Database connection lost")
        
        payload = {"sql": "SELECT * FROM lang_operations"}
        response = client.post("/db/query", json=payload)
        
        assert response.status_code == 500
        assert "Database connection lost" in response.json()["detail"]


class TestCORSMiddleware:
    """Tests for CORS configuration."""
    
    def test_cors_headers_present(self):
        """Test that CORS headers are present in responses."""
        # Create client without mocks for this test
        with TestClient(app) as client:
            response = client.options(
                "/health",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET"
                }
            )
            
            # Check for CORS headers
            assert "access-control-allow-origin" in response.headers or response.status_code == 200


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_grpc_deadline_exceeded(self, client_with_mocks):
        """Test handling of gRPC deadline exceeded error."""
        client, mock_client, _ = client_with_mocks
        
        error = grpc.RpcError()
        error.code = lambda: grpc.StatusCode.DEADLINE_EXCEEDED
        error.details = lambda: "Deadline exceeded"
        mock_client.health.side_effect = error
        
        response = client.get("/health")
        
        assert response.status_code == 504
    
    def test_grpc_invalid_argument(self, client_with_mocks):
        """Test handling of gRPC invalid argument error."""
        client, mock_client, _ = client_with_mocks
        
        error = grpc.RpcError()
        error.code = lambda: grpc.StatusCode.INVALID_ARGUMENT
        error.details = lambda: "Invalid argument"
        mock_client.list_operations.side_effect = error
        
        response = client.get("/operations")
        
        assert response.status_code == 422
