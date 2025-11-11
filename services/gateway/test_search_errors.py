#!/usr/bin/env python3
"""
Test script for enhanced error messages in unified search endpoint.

This script tests the /search/unified endpoint with services unavailable
to verify that error messages are clear and actionable.
"""
import asyncio
import httpx
import json
from typing import Dict, Any


async def test_unified_search_error_messages():
    """Test unified search endpoint with services unavailable."""
    gateway_url = "http://localhost:8000"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("=" * 80)
        print("Testing Unified Search Error Messages")
        print("=" * 80)
        print()
        
        # Test 1: Search with all services unavailable
        print("Test 1: Search with all services unavailable")
        print("-" * 80)
        try:
            response = await client.post(
                f"{gateway_url}/search/unified",
                json={
                    "query": "test query",
                    "top_k": 10,
                    "sources": ["inference", "knowledge_graph", "catalog"]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Request succeeded (status: {response.status_code})")
                print()
                print("Response structure:")
                print(json.dumps(data, indent=2))
                print()
                
                # Check error messages
                sources = data.get("sources", {})
                metadata = data.get("metadata", {})
                
                print("Error Analysis:")
                print(f"  Sources queried: {metadata.get('sources_queried', 0)}")
                print(f"  Sources successful: {metadata.get('sources_successful', 0)}")
                print(f"  Sources failed: {metadata.get('sources_failed', 0)}")
                print()
                
                for source_name, source_data in sources.items():
                    if isinstance(source_data, dict) and "error" in source_data:
                        error_info = source_data.get("error", "")
                        error_type = source_data.get("type", "unknown")
                        error_url = source_data.get("url", "")
                        
                        print(f"  {source_name}:")
                        print(f"    Error: {error_info}")
                        print(f"    Type: {error_type}")
                        print(f"    URL: {error_url}")
                        
                        # Verify error message quality
                        if "Connection refused" in error_info or "Service may not be running" in error_info:
                            print(f"    ✓ Error message is clear and actionable")
                        if error_type in ["connection_error", "timeout", "unknown_error"]:
                            print(f"    ✓ Error type is properly classified")
                        if error_url:
                            print(f"    ✓ Service URL is included")
                        print()
            else:
                print(f"✗ Request failed (status: {response.status_code})")
                print(f"  Response: {response.text}")
        except Exception as e:
            print(f"✗ Request exception: {e}")
        
        print()
        
        # Test 2: Check health endpoint
        print("Test 2: Check service health endpoint")
        print("-" * 80)
        try:
            response = await client.get(f"{gateway_url}/healthz")
            
            if response.status_code == 200:
                health = response.json()
                print(f"✓ Health check succeeded (status: {response.status_code})")
                print()
                print("Service Health Status:")
                for service, status in health.items():
                    status_str = status if isinstance(status, str) else str(status)
                    icon = "✓" if status_str == "ok" else "✗"
                    print(f"  {icon} {service}: {status_str}")
            else:
                print(f"✗ Health check failed (status: {response.status_code})")
                print(f"  Response: {response.text}")
        except Exception as e:
            print(f"✗ Health check exception: {e}")
        
        print()
        
        # Test 3: Test with retry logic (simulate transient failure)
        print("Test 3: Verify retry logic is working")
        print("-" * 80)
        print("Note: This test requires services to be temporarily unavailable")
        print("      and then become available during retry attempts.")
        print("      Manual testing recommended.")
        print()
        
        print("=" * 80)
        print("Test Summary")
        print("=" * 80)
        print("✓ Error messages should include:")
        print("  - Clear description of the problem")
        print("  - Service URL")
        print("  - Error type (connection_error, timeout, unknown_error)")
        print("  - Actionable guidance (e.g., 'Service may not be running')")
        print()
        print("✓ Retry logic should:")
        print("  - Attempt 3 times (initial + 2 retries)")
        print("  - Use exponential backoff")
        print("  - Only retry on transient errors (ConnectError, TimeoutException)")


if __name__ == "__main__":
    asyncio.run(test_unified_search_error_messages())

