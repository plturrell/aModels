#!/usr/bin/env python3
"""
Week 1: Test Data Fixtures

Creates test data for Week 1 testing:
- Sample domain configurations
- Sample knowledge graphs
- Sample training data
- Sample queries
"""

import os
import json
from typing import Dict, List, Any
from pathlib import Path

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)


def create_sample_domain_configs() -> Dict[str, Any]:
    """Create sample domain configurations for testing."""
    return {
        "domains": {
            "test-financial": {
                "name": "Test Financial Domain",
                "agent_id": "test-0x1111",
                "layer": "business",
                "team": "FinanceTeam",
                "backend_type": "hf-transformers",
                "model_name": "phi-3.5-mini",
                "transformers_config": {
                    "endpoint": "http://transformers-service:9090/v1/chat/completions",
                    "model_name": "phi-3.5-mini",
                    "timeout_seconds": 60
                },
                "keywords": ["financial", "amount", "price", "cost", "revenue", "payment", "transaction", "account", "balance"],
                "tags": ["financial", "business", "money"],
                "max_tokens": 512,
                "temperature": 0.3
            },
            "test-customer": {
                "name": "Test Customer Domain",
                "agent_id": "test-0x2222",
                "layer": "application",
                "team": "CustomerTeam",
                "backend_type": "hf-transformers",
                "model_name": "phi-3.5-mini",
                "transformers_config": {
                    "endpoint": "http://transformers-service:9090/v1/chat/completions",
                    "model_name": "phi-3.5-mini",
                    "timeout_seconds": 60
                },
                "keywords": ["customer", "client", "user", "person", "contact", "email", "phone", "address"],
                "tags": ["customer", "application", "user"],
                "max_tokens": 256,
                "temperature": 0.5
            },
            "test-product": {
                "name": "Test Product Domain",
                "agent_id": "test-0x3333",
                "layer": "data",
                "team": "ProductTeam",
                "backend_type": "hf-transformers",
                "model_name": "phi-3.5-mini",
                "transformers_config": {
                    "endpoint": "http://transformers-service:9090/v1/chat/completions",
                    "model_name": "phi-3.5-mini",
                    "timeout_seconds": 60
                },
                "keywords": ["product", "item", "sku", "catalog", "inventory", "stock", "quantity"],
                "tags": ["product", "data", "inventory"],
                "max_tokens": 256,
                "temperature": 0.4
            }
        },
        "default_domain": "test-financial"
    }


def create_sample_knowledge_graph() -> Dict[str, Any]:
    """Create sample knowledge graph for testing."""
    return {
        "nodes": [
            {
                "id": "node_1",
                "label": "customers",
                "type": "table",
                "props": {
                    "schema": "public",
                    "database": "test_db",
                    "domain": "test-customer",
                    "agent_id": "test-0x2222"
                }
            },
            {
                "id": "node_2",
                "label": "customer_id",
                "type": "column",
                "props": {
                    "table": "customers",
                    "type": "bigint",
                    "domain": "test-customer",
                    "agent_id": "test-0x2222"
                }
            },
            {
                "id": "node_3",
                "label": "orders",
                "type": "table",
                "props": {
                    "schema": "public",
                    "database": "test_db",
                    "domain": "test-financial",
                    "agent_id": "test-0x1111"
                }
            },
            {
                "id": "node_4",
                "label": "order_amount",
                "type": "column",
                "props": {
                    "table": "orders",
                    "type": "decimal",
                    "domain": "test-financial",
                    "agent_id": "test-0x1111"
                }
            }
        ],
        "edges": [
            {
                "source_id": "node_2",
                "target_id": "node_4",
                "type": "RELATES_TO",
                "props": {
                    "domain": "test-financial",
                    "agent_id": "test-0x1111"
                }
            }
        ],
        "metrics": {
            "total_nodes": 4,
            "total_edges": 1,
            "domains_detected": ["test-customer", "test-financial"]
        }
    }


def create_sample_training_data() -> Dict[str, Any]:
    """Create sample training data for testing."""
    return {
        "domain_id": "test-financial",
        "training_samples": [
            {
                "text": "customer payment transaction amount",
                "domain": "test-financial",
                "features": {
                    "has_payment": 1,
                    "has_amount": 1,
                    "has_transaction": 1
                }
            },
            {
                "text": "user contact email address",
                "domain": "test-customer",
                "features": {
                    "has_user": 1,
                    "has_contact": 1,
                    "has_email": 1
                }
            },
            {
                "text": "product inventory stock quantity",
                "domain": "test-product",
                "features": {
                    "has_product": 1,
                    "has_inventory": 1,
                    "has_quantity": 1
                }
            }
        ],
        "metadata": {
            "total_samples": 3,
            "domains": ["test-financial", "test-customer", "test-product"],
            "created_at": "2025-01-01T00:00:00Z"
        }
    }


def create_sample_queries() -> List[Dict[str, Any]]:
    """Create sample queries for testing."""
    return [
        {
            "query": "What is the total revenue from customer payments?",
            "expected_domain": "test-financial",
            "keywords": ["revenue", "customer", "payments"]
        },
        {
            "query": "Find all customer contact information",
            "expected_domain": "test-customer",
            "keywords": ["customer", "contact", "information"]
        },
        {
            "query": "Show product inventory levels",
            "expected_domain": "test-product",
            "keywords": ["product", "inventory", "levels"]
        }
    ]


def create_sample_metrics() -> Dict[str, Any]:
    """Create sample metrics for testing."""
    return {
        "domain_id": "test-financial",
        "performance": {
            "latest": {
                "accuracy": 0.87,
                "latency_ms": 120,
                "tokens_per_second": 45.2,
                "throughput": 100
            },
            "average": {
                "accuracy": 0.85,
                "latency_ms": 125,
                "tokens_per_second": 44.0
            },
            "history": [
                {
                    "version": "v1",
                    "metrics": {
                        "accuracy": 0.80,
                        "latency_ms": 150
                    },
                    "updated_at": "2025-01-01T00:00:00Z"
                },
                {
                    "version": "v2",
                    "metrics": {
                        "accuracy": 0.85,
                        "latency_ms": 130
                    },
                    "updated_at": "2025-01-02T00:00:00Z"
                },
                {
                    "version": "v3",
                    "metrics": {
                        "accuracy": 0.87,
                        "latency_ms": 120
                    },
                    "updated_at": "2025-01-03T00:00:00Z"
                }
            ]
        },
        "usage": {
            "total_requests": 1000,
            "requests_per_day": 100
        },
        "quality": {
            "completeness": 0.95,
            "consistency": 0.92
        }
    }


def save_test_data():
    """Save all test data to files."""
    print("Creating test data fixtures...")
    
    # Domain configs
    domain_configs = create_sample_domain_configs()
    with open(TEST_DATA_DIR / "domain_configs.json", "w") as f:
        json.dump(domain_configs, f, indent=2)
    print(f"✅ Created {TEST_DATA_DIR / 'domain_configs.json'}")
    
    # Knowledge graph
    kg = create_sample_knowledge_graph()
    with open(TEST_DATA_DIR / "knowledge_graph.json", "w") as f:
        json.dump(kg, f, indent=2)
    print(f"✅ Created {TEST_DATA_DIR / 'knowledge_graph.json'}")
    
    # Training data
    training_data = create_sample_training_data()
    with open(TEST_DATA_DIR / "training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)
    print(f"✅ Created {TEST_DATA_DIR / 'training_data.json'}")
    
    # Queries
    queries = create_sample_queries()
    with open(TEST_DATA_DIR / "queries.json", "w") as f:
        json.dump(queries, f, indent=2)
    print(f"✅ Created {TEST_DATA_DIR / 'queries.json'}")
    
    # Metrics
    metrics = create_sample_metrics()
    with open(TEST_DATA_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Created {TEST_DATA_DIR / 'metrics.json'}")
    
    print(f"\n✅ All test data fixtures created in {TEST_DATA_DIR}")


if __name__ == "__main__":
    save_test_data()

