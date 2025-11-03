#!/usr/bin/env python3
"""
Test script to validate the AI search functionality end-to-end.
"""

import json
import time
import requests
from typing import Dict, Any


def test_search_gateway_health(base_url: str = "http://localhost:8081") -> bool:
    """Test if search gateway is healthy"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Search Gateway Health: {health_data}")
            return True
        else:
            print(f"❌ Search Gateway unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Search Gateway not reachable: {e}")
        return False


def test_document_indexing(base_url: str = "http://localhost:8081") -> bool:
    """Test document indexing"""
    try:
        # Test document
        test_doc = {
            "id": "test_doc_1",
            "content": "This is a test document about company vacation policies. Employees get 20 days of vacation per year.",
            "metadata": {
                "category": "HR",
                "type": "policy",
                "test": True
            }
        }
        
        response = requests.post(
            f"{base_url}/v1/documents",
            json=test_doc,
            timeout=10
        )
        
        if response.status_code == 204:
            print("✅ Document indexing successful")
            return True
        else:
            print(f"❌ Document indexing failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Document indexing error: {e}")
        return False


def test_basic_search(base_url: str = "http://localhost:8081") -> bool:
    """Test basic search functionality"""
    try:
        search_request = {
            "query": "vacation policy",
            "top_k": 3
        }
        
        response = requests.post(
            f"{base_url}/v1/search",
            json=search_request,
            timeout=10
        )
        
        if response.status_code == 200:
            search_data = response.json()
            print(f"✅ Basic search successful: {len(search_data.get('results', []))} results")
            return True
        else:
            print(f"❌ Basic search failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Basic search error: {e}")
        return False


def test_ai_search(base_url: str = "http://localhost:8081") -> bool:
    """Test AI search functionality"""
    try:
        ai_search_request = {
            "query": "How many vacation days do employees get?",
            "max_sources": 3
        }
        
        response = requests.post(
            f"{base_url}/v1/ai-search",
            json=ai_search_request,
            timeout=30
        )
        
        if response.status_code == 200:
            ai_data = response.json()
            print(f"✅ AI search successful")
            print(f"   Response: {ai_data.get('response', '')[:100]}...")
            print(f"   Sources: {len(ai_data.get('sources', []))}")
            print(f"   Conversation ID: {ai_data.get('conversation_id', '')}")
            return True
        else:
            print(f"❌ AI search failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ AI search error: {e}")
        return False


def test_conversation_context(base_url: str = "http://localhost:8081") -> bool:
    """Test conversation context with follow-up questions"""
    try:
        # First question
        first_request = {
            "query": "What is the vacation policy?",
            "max_sources": 3
        }
        
        response1 = requests.post(
            f"{base_url}/v1/ai-search",
            json=first_request,
            timeout=30
        )
        
        if response1.status_code != 200:
            print(f"❌ First question failed: {response1.status_code}")
            return False
        
        ai_data1 = response1.json()
        conversation_id = ai_data1.get('conversation_id')
        
        if not conversation_id:
            print("❌ No conversation ID returned")
            return False
        
        # Follow-up question
        follow_up_request = {
            "query": "How do I request vacation time?",
            "conversation_id": conversation_id,
            "max_sources": 3
        }
        
        response2 = requests.post(
            f"{base_url}/v1/ai-search",
            json=follow_up_request,
            timeout=30
        )
        
        if response2.status_code == 200:
            ai_data2 = response2.json()
            print(f"✅ Follow-up question successful")
            print(f"   Response: {ai_data2.get('response', '')[:100]}...")
            return True
        else:
            print(f"❌ Follow-up question failed: {response2.status_code} - {response2.text}")
            return False
            
    except Exception as e:
        print(f"❌ Conversation context error: {e}")
        return False


def test_browser_proxy(base_url: str = "http://localhost:9222") -> bool:
    """Test browser proxy endpoints"""
    try:
        # Test AI search through browser proxy
        search_request = {
            "query": "What are the expense reimbursement guidelines?",
            "max_sources": 3
        }
        
        response = requests.post(
            f"{base_url}/api/v1/ai-search",
            json=search_request,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Browser proxy AI search successful")
            return True
        else:
            print(f"❌ Browser proxy AI search failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Browser proxy error: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing AI Search System")
    print("=" * 50)
    
    tests = [
        ("Search Gateway Health", test_search_gateway_health),
        ("Document Indexing", test_document_indexing),
        ("Basic Search", test_basic_search),
        ("AI Search", test_ai_search),
        ("Conversation Context", test_conversation_context),
        ("Browser Proxy", test_browser_proxy),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The AI search system is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the logs above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
