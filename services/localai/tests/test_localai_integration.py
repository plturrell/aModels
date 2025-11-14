#!/usr/bin/env python3
"""
LocalAI Integration Testing Framework

Comprehensive test suite to verify LocalAI integration with all models:
- Service health and connectivity
- Model accessibility (GGUF, SafeTensors, HF-Transformers)
- Domain routing accuracy
- Backend integration
- End-to-end query processing
- Performance metrics
"""

import json
import os
import sys
import time
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback

# Configuration - Auto-detect Docker environment
def detect_docker_environment():
    """Detect if running in Docker and configure service URLs accordingly"""
    # Check if we're in Docker by looking for /.dockerenv or container hostname
    in_docker = os.path.exists("/.dockerenv") or os.getenv("container") == "docker"
    
    if in_docker:
        # Running in Docker - use Docker service names
        return {
            "localai": "http://localhost:8080",  # Same container
            "transformers": "http://transformers-service:9090",  # Docker DNS
            "model_server": "http://model-server:8088"  # Docker DNS
        }
    else:
        # Running on host - use localhost with default ports
        return {
            "localai": "http://localhost:8081",
            "transformers": "http://localhost:9090",
            "model_server": "http://localhost:8088"
        }

# Auto-configure service URLs
_docker_config = detect_docker_environment()
LOCALAI_SERVICE_URL = os.getenv("LOCALAI_SERVICE_URL", _docker_config["localai"])
TRANSFORMERS_SERVICE_URL = os.getenv("TRANSFORMERS_SERVICE_URL", _docker_config["transformers"])
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", _docker_config["model_server"])
TEST_TIMEOUT = int(os.getenv("TEST_TIMEOUT", "120"))
RESULTS_DIR = os.getenv("RESULTS_DIR", "/tmp/localai_test_results")

@dataclass
class ServiceHealthResult:
    """Result of a service health check"""
    service_name: str
    url: str
    healthy: bool
    response_time_ms: float
    status_code: Optional[int] = None
    error: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class ModelAccessResult:
    """Result of model accessibility test"""
    model_name: str
    model_type: str  # "gguf", "safetensors", "hf-transformers"
    accessible: bool
    load_time_ms: Optional[float] = None
    error: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class RoutingTestResult:
    """Result of domain routing test"""
    query: str
    expected_domain: str
    actual_domain: Optional[str] = None
    routed_correctly: bool = False
    confidence: Optional[float] = None
    response_time_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class QueryTestResult:
    """Result of end-to-end query test"""
    query: str
    domain: str
    backend_type: str
    success: bool
    response_text: str
    response_time_ms: float
    token_count: Optional[int] = None
    quality_score: Optional[float] = None
    error: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class PerformanceMetrics:
    """Performance metrics for a test"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float

class LocalAITestFramework:
    """Comprehensive test framework for LocalAI integration"""
    
    def __init__(self):
        self.health_results: List[ServiceHealthResult] = []
        self.model_results: List[ModelAccessResult] = []
        self.routing_results: List[RoutingTestResult] = []
        self.query_results: List[QueryTestResult] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        
    def test_service_health(self) -> List[ServiceHealthResult]:
        """Test health of all services"""
        print("\n" + "="*80)
        print("SERVICE HEALTH TESTS")
        print("="*80)
        print(f"Environment: {'Docker' if os.path.exists('/.dockerenv') else 'Host'}")
        print(f"LocalAI URL: {LOCALAI_SERVICE_URL}")
        print(f"Transformers URL: {TRANSFORMERS_SERVICE_URL}")
        print(f"Model-Server URL: {MODEL_SERVER_URL}")
        
        services = [
            ("LocalAI", f"{LOCALAI_SERVICE_URL}/healthz"),
            ("Transformers-Service", f"{TRANSFORMERS_SERVICE_URL}/health"),
            ("Model-Server", f"{MODEL_SERVER_URL}/health"),
        ]
        
        results = []
        for service_name, url in services:
            print(f"\nTesting {service_name} at {url}...")
            start_time = time.time()
            
            try:
                response = requests.get(url, timeout=10)
                response_time = (time.time() - start_time) * 1000
                
                healthy = response.status_code == 200
                error = None if healthy else f"Status {response.status_code}"
                
                result = ServiceHealthResult(
                    service_name=service_name,
                    url=url,
                    healthy=healthy,
                    response_time_ms=response_time,
                    status_code=response.status_code,
                    error=error
                )
                
                status_icon = "✅" if healthy else "❌"
                print(f"  {status_icon} {service_name}: {'HEALTHY' if healthy else 'UNHEALTHY'}")
                if error:
                    print(f"     Error: {error}")
                
            except requests.exceptions.RequestException as e:
                response_time = (time.time() - start_time) * 1000
                result = ServiceHealthResult(
                    service_name=service_name,
                    url=url,
                    healthy=False,
                    response_time_ms=response_time,
                    error=str(e)
                )
                print(f"  ❌ {service_name}: ERROR - {e}")
            
            results.append(result)
            self.health_results.append(result)
        
        return results
    
    def test_model_accessibility(self) -> List[ModelAccessResult]:
        """Test accessibility of all model types"""
        print("\n" + "="*80)
        print("MODEL ACCESSIBILITY TESTS")
        print("="*80)
        
        results = []
        
        # Test HF-Transformers models via transformers-service
        print("\nTesting HF-Transformers models via transformers-service...")
        hf_models = ["phi-3.5-mini", "granite-4.0-h-micro", "granite-4.0"]
        
        for model_name in hf_models:
            print(f"  Testing {model_name}...")
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{TRANSFORMERS_SERVICE_URL}/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5,
                        "temperature": 0.1
                    },
                    timeout=TEST_TIMEOUT
                )
                
                load_time = (time.time() - start_time) * 1000
                accessible = response.status_code == 200
                error = None if accessible else f"Status {response.status_code}: {response.text[:200]}"
                
                result = ModelAccessResult(
                    model_name=model_name,
                    model_type="hf-transformers",
                    accessible=accessible,
                    load_time_ms=load_time,
                    error=error
                )
                
                status_icon = "✅" if accessible else "❌"
                print(f"    {status_icon} {model_name}: {'ACCESSIBLE' if accessible else 'NOT ACCESSIBLE'}")
                if error:
                    print(f"       Error: {error[:100]}")
                
            except Exception as e:
                load_time = (time.time() - start_time) * 1000
                result = ModelAccessResult(
                    model_name=model_name,
                    model_type="hf-transformers",
                    accessible=False,
                    load_time_ms=load_time,
                    error=str(e)
                )
                print(f"    ❌ {model_name}: ERROR - {e}")
            
            results.append(result)
            self.model_results.append(result)
        
        # Test GGUF models via LocalAI (if accessible)
        print("\nTesting GGUF models via LocalAI...")
        gguf_models = ["gemma-2b-it", "gemma-7b-it"]
        
        for model_name in gguf_models:
            print(f"  Testing {model_name}...")
            start_time = time.time()
            
            try:
                # Try to list models or check if model is available
                response = requests.get(
                    f"{LOCALAI_SERVICE_URL}/v1/models",
                    timeout=10
                )
                
                load_time = (time.time() - start_time) * 1000
                # Check if model is in the list
                if response.status_code == 200:
                    models_data = response.json()
                    model_list = models_data.get("data", [])
                    model_ids = [m.get("id", "") for m in model_list]
                    accessible = any(model_name in m_id for m_id in model_ids)
                    error = None if accessible else "Model not found in LocalAI models list"
                else:
                    accessible = False
                    error = f"Status {response.status_code}"
                
                result = ModelAccessResult(
                    model_name=model_name,
                    model_type="gguf",
                    accessible=accessible,
                    load_time_ms=load_time,
                    error=error
                )
                
                status_icon = "✅" if accessible else "❌"
                print(f"    {status_icon} {model_name}: {'ACCESSIBLE' if accessible else 'NOT ACCESSIBLE'}")
                if error:
                    print(f"       Error: {error}")
                
            except Exception as e:
                load_time = (time.time() - start_time) * 1000
                result = ModelAccessResult(
                    model_name=model_name,
                    model_type="gguf",
                    accessible=False,
                    load_time_ms=load_time,
                    error=str(e)
                )
                print(f"    ❌ {model_name}: ERROR - {e}")
            
            results.append(result)
            self.model_results.append(result)
        
        return results
    
    def test_domain_routing(self, test_queries: List[Dict[str, str]]) -> List[RoutingTestResult]:
        """Test domain routing accuracy"""
        print("\n" + "="*80)
        print("DOMAIN ROUTING TESTS")
        print("="*80)
        
        results = []
        
        for test_case in test_queries:
            query = test_case.get("query", "")
            expected_domain = test_case.get("expected_domain", "")
            
            print(f"\nTesting query: '{query[:60]}...'")
            print(f"  Expected domain: {expected_domain}")
            
            start_time = time.time()
            
            try:
                # Send query to LocalAI
                response = requests.post(
                    f"{LOCALAI_SERVICE_URL}/v1/chat/completions",
                    json={
                        "model": "auto",  # Let LocalAI auto-detect domain
                        "messages": [{"role": "user", "content": query}],
                        "max_tokens": 50,
                        "temperature": 0.7
                    },
                    timeout=TEST_TIMEOUT
                )
                
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    # Try to extract domain from response metadata
                    # This depends on LocalAI's response format
                    actual_domain = None
                    if "model" in data:
                        actual_domain = data["model"]
                    
                    # Check if routing was correct
                    routed_correctly = expected_domain.lower() in actual_domain.lower() if actual_domain else False
                    
                    result = RoutingTestResult(
                        query=query,
                        expected_domain=expected_domain,
                        actual_domain=actual_domain,
                        routed_correctly=routed_correctly,
                        response_time_ms=response_time
                    )
                    
                    status_icon = "✅" if routed_correctly else "⚠️"
                    print(f"  {status_icon} Routed to: {actual_domain or 'unknown'}")
                    if not routed_correctly:
                        print(f"     Expected: {expected_domain}")
                
                else:
                    result = RoutingTestResult(
                        query=query,
                        expected_domain=expected_domain,
                        routed_correctly=False,
                        response_time_ms=response_time,
                        error=f"Status {response.status_code}: {response.text[:200]}"
                    )
                    print(f"  ❌ Error: {result.error[:100]}")
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                result = RoutingTestResult(
                    query=query,
                    expected_domain=expected_domain,
                    routed_correctly=False,
                    response_time_ms=response_time,
                    error=str(e)
                )
                print(f"  ❌ Exception: {e}")
            
            results.append(result)
            self.routing_results.append(result)
        
        return results
    
    def test_backend_integration(self, domains: List[Dict]) -> List[QueryTestResult]:
        """Test integration with different backends"""
        print("\n" + "="*80)
        print("BACKEND INTEGRATION TESTS")
        print("="*80)
        
        results = []
        
        # Test HF-Transformers backend domains
        hf_domains = [d for d in domains if d.get("backend_type") == "hf-transformers"]
        print(f"\nTesting {len(hf_domains)} HF-Transformers domains...")
        
        for domain in hf_domains[:10]:  # Test first 10 to avoid timeout
            domain_id = domain.get("agent_id", "")
            domain_name = domain.get("name", "")
            model_name = domain.get("model_name", "")
            
            print(f"\n  Testing domain: {domain_name} ({domain_id})")
            print(f"    Model: {model_name}")
            
            test_query = f"Say hello, I am testing {domain_name}"
            
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{LOCALAI_SERVICE_URL}/v1/chat/completions",
                    json={
                        "model": domain_id,
                        "messages": [{"role": "user", "content": test_query}],
                        "max_tokens": 20,
                        "temperature": 0.1
                    },
                    timeout=TEST_TIMEOUT
                )
                
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    response_text = choices[0].get("message", {}).get("content", "") if choices else ""
                    usage = data.get("usage", {})
                    token_count = usage.get("total_tokens", 0)
                    
                    quality_score = self._calculate_quality_score(response_text, test_query)
                    success = len(response_text) > 0 and quality_score > 0.3
                    
                    result = QueryTestResult(
                        query=test_query,
                        domain=domain_id,
                        backend_type="hf-transformers",
                        success=success,
                        response_text=response_text[:200],
                        response_time_ms=response_time,
                        token_count=token_count,
                        quality_score=quality_score
                    )
                    
                    status_icon = "✅" if success else "❌"
                    print(f"    {status_icon} Response: '{response_text[:50]}...'")
                    print(f"       Quality: {quality_score:.2f}, Tokens: {token_count}, Time: {response_time:.1f}ms")
                
                else:
                    result = QueryTestResult(
                        query=test_query,
                        domain=domain_id,
                        backend_type="hf-transformers",
                        success=False,
                        response_text="",
                        response_time_ms=response_time,
                        error=f"Status {response.status_code}: {response.text[:200]}"
                    )
                    print(f"    ❌ Error: {result.error[:100]}")
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                result = QueryTestResult(
                    query=test_query,
                    domain=domain_id,
                    backend_type="hf-transformers",
                    success=False,
                    response_text="",
                    response_time_ms=response_time,
                    error=str(e)
                )
                print(f"    ❌ Exception: {e}")
            
            results.append(result)
            self.query_results.append(result)
        
        return results
    
    def test_end_to_end(self, test_queries: List[Dict[str, str]]) -> List[QueryTestResult]:
        """Test end-to-end query processing"""
        print("\n" + "="*80)
        print("END-TO-END QUERY TESTS")
        print("="*80)
        
        results = []
        
        for test_case in test_queries:
            query = test_case.get("query", "")
            domain = test_case.get("domain", "auto")
            
            print(f"\nTesting query: '{query[:60]}...'")
            
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{LOCALAI_SERVICE_URL}/v1/chat/completions",
                    json={
                        "model": domain,
                        "messages": [{"role": "user", "content": query}],
                        "max_tokens": 100,
                        "temperature": 0.7
                    },
                    timeout=TEST_TIMEOUT
                )
                
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    response_text = choices[0].get("message", {}).get("content", "") if choices else ""
                    usage = data.get("usage", {})
                    token_count = usage.get("total_tokens", 0)
                    model_used = data.get("model", "unknown")
                    
                    quality_score = self._calculate_quality_score(response_text, query)
                    success = len(response_text) > 0 and quality_score > 0.3
                    
                    result = QueryTestResult(
                        query=query,
                        domain=model_used,
                        backend_type="unknown",
                        success=success,
                        response_text=response_text[:500],
                        response_time_ms=response_time,
                        token_count=token_count,
                        quality_score=quality_score
                    )
                    
                    status_icon = "✅" if success else "❌"
                    print(f"  {status_icon} Model: {model_used}")
                    print(f"     Response: '{response_text[:80]}...'")
                    print(f"     Quality: {quality_score:.2f}, Tokens: {token_count}, Time: {response_time:.1f}ms")
                
                else:
                    result = QueryTestResult(
                        query=query,
                        domain=domain,
                        backend_type="unknown",
                        success=False,
                        response_text="",
                        response_time_ms=response_time,
                        error=f"Status {response.status_code}: {response.text[:200]}"
                    )
                    print(f"  ❌ Error: {result.error[:100]}")
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                result = QueryTestResult(
                    query=query,
                    domain=domain,
                    backend_type="unknown",
                    success=False,
                    response_text="",
                    response_time_ms=response_time,
                    error=str(e)
                )
                print(f"  ❌ Exception: {e}")
            
            results.append(result)
            self.query_results.append(result)
        
        return results
    
    def _calculate_quality_score(self, response_text: str, query: str) -> float:
        """Calculate quality score for response (0.0 to 1.0)"""
        if not response_text or len(response_text.strip()) == 0:
            return 0.0
        
        score = 0.0
        
        # Length check
        length = len(response_text)
        if 10 <= length <= 2000:
            score += 0.3
        elif length > 0:
            score += 0.1
        
        # Non-empty check
        if response_text.strip():
            score += 0.2
        
        # Error indicators
        error_indicators = ["error", "exception", "failed", "not found", "404", "500"]
        has_errors = any(indicator in response_text.lower() for indicator in error_indicators)
        if not has_errors:
            score += 0.3
        
        # Relevance
        query_words = set(query.lower().split())
        response_words = set(response_text.lower().split())
        if len(query_words & response_words) > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def calculate_performance_metrics(self) -> List[PerformanceMetrics]:
        """Calculate performance metrics from test results"""
        metrics = []
        
        # Calculate metrics for query tests
        if self.query_results:
            response_times = [r.response_time_ms for r in self.query_results if r.success]
            if response_times:
                response_times.sort()
                total = len(response_times)
                successful = len([r for r in self.query_results if r.success])
                
                metric = PerformanceMetrics(
                    test_name="Query Processing",
                    total_requests=len(self.query_results),
                    successful_requests=successful,
                    failed_requests=len(self.query_results) - successful,
                    average_response_time_ms=sum(response_times) / len(response_times),
                    min_response_time_ms=min(response_times),
                    max_response_time_ms=max(response_times),
                    p95_response_time_ms=response_times[int(len(response_times) * 0.95)] if len(response_times) > 0 else 0,
                    p99_response_time_ms=response_times[int(len(response_times) * 0.99)] if len(response_times) > 0 else 0
                )
                metrics.append(metric)
        
        self.performance_metrics = metrics
        return metrics
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("="*80)
        report.append("LOCALAI INTEGRATION TEST REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Service Health Summary
        report.append("SERVICE HEALTH SUMMARY")
        report.append("-"*80)
        healthy_count = sum(1 for r in self.health_results if r.healthy)
        report.append(f"Healthy Services: {healthy_count}/{len(self.health_results)}")
        for result in self.health_results:
            status_icon = "✅" if result.healthy else "❌"
            report.append(f"{status_icon} {result.service_name}: {result.response_time_ms:.1f}ms")
            if result.error:
                report.append(f"   Error: {result.error}")
        report.append("")
        
        # Model Accessibility Summary
        report.append("MODEL ACCESSIBILITY SUMMARY")
        report.append("-"*80)
        accessible_count = sum(1 for r in self.model_results if r.accessible)
        report.append(f"Accessible Models: {accessible_count}/{len(self.model_results)}")
        for result in self.model_results:
            status_icon = "✅" if result.accessible else "❌"
            report.append(f"{status_icon} {result.model_name} ({result.model_type}): {result.load_time_ms:.1f}ms")
            if result.error:
                report.append(f"   Error: {result.error[:100]}")
        report.append("")
        
        # Routing Accuracy Summary
        if self.routing_results:
            report.append("ROUTING ACCURACY SUMMARY")
            report.append("-"*80)
            correct_count = sum(1 for r in self.routing_results if r.routed_correctly)
            accuracy = (correct_count / len(self.routing_results) * 100) if self.routing_results else 0
            report.append(f"Routing Accuracy: {accuracy:.1f}% ({correct_count}/{len(self.routing_results)})")
            report.append("")
        
        # Query Test Summary
        if self.query_results:
            report.append("QUERY TEST SUMMARY")
            report.append("-"*80)
            successful_count = sum(1 for r in self.query_results if r.success)
            report.append(f"Successful Queries: {successful_count}/{len(self.query_results)}")
            if self.query_results:
                avg_quality = sum(r.quality_score for r in self.query_results if r.quality_score) / len([r for r in self.query_results if r.quality_score])
                report.append(f"Average Quality Score: {avg_quality:.2f}")
            report.append("")
        
        # Performance Metrics
        if self.performance_metrics:
            report.append("PERFORMANCE METRICS")
            report.append("-"*80)
            for metric in self.performance_metrics:
                report.append(f"{metric.test_name}:")
                report.append(f"  Total Requests: {metric.total_requests}")
                report.append(f"  Successful: {metric.successful_requests}")
                report.append(f"  Failed: {metric.failed_requests}")
                report.append(f"  Avg Response Time: {metric.average_response_time_ms:.1f}ms")
                report.append(f"  Min: {metric.min_response_time_ms:.1f}ms")
                report.append(f"  Max: {metric.max_response_time_ms:.1f}ms")
                report.append(f"  P95: {metric.p95_response_time_ms:.1f}ms")
                report.append(f"  P99: {metric.p99_response_time_ms:.1f}ms")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, output_dir: str = RESULTS_DIR):
        """Save test results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "localai_url": LOCALAI_SERVICE_URL,
            "transformers_url": TRANSFORMERS_SERVICE_URL,
            "model_server_url": MODEL_SERVER_URL,
            "health_results": [asdict(r) for r in self.health_results],
            "model_results": [asdict(r) for r in self.model_results],
            "routing_results": [asdict(r) for r in self.routing_results],
            "query_results": [asdict(r) for r in self.query_results],
            "performance_metrics": [asdict(m) for m in self.performance_metrics]
        }
        
        json_path = os.path.join(output_dir, f"localai_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save text report
        report = self.generate_report()
        report_path = os.path.join(output_dir, f"localai_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✅ Results saved to:")
        print(f"   JSON: {json_path}")
        print(f"   Report: {report_path}")
        
        return json_path, report_path


def load_test_queries() -> List[Dict[str, str]]:
    """Load test queries from file or use defaults"""
    # Try multiple possible paths for Docker and host environments
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "test_queries.json"),
        "/tmp/test_queries.json",  # Docker copied location
        "/workspace/services/localai/tests/test_queries.json",  # Docker workspace
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_queries.json"),
    ]
    
    for queries_file in possible_paths:
        if os.path.exists(queries_file):
            try:
                with open(queries_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    
    # Default test queries if file not found
    return [
        {"query": "Write a SQL query to select all users", "expected_domain": "SQL", "domain": "auto"},
        {"query": "What is a ledger entry?", "expected_domain": "Finance", "domain": "auto"},
        {"query": "Write a Python function to sort a list", "expected_domain": "Code", "domain": "auto"},
        {"query": "What is artificial intelligence?", "expected_domain": "General", "domain": "auto"},
    ]


def load_domains_config() -> List[Dict]:
    """Load domains configuration"""
    # Try multiple possible paths for Docker and host environments
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "../config/domains.json"),
        "/config/domains.json",  # Docker mounted location
        "/workspace/services/localai/config/domains.json",  # Docker workspace
        "/tmp/config/domains.json",  # Docker copied location
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/domains.json"),
    ]
    
    for domains_file in possible_paths:
        if os.path.exists(domains_file):
            try:
                with open(domains_file, 'r') as f:
                    config = json.load(f)
                    return list(config.get("domains", {}).values())
            except (json.JSONDecodeError, IOError):
                continue
    
    return []


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("LOCALAI INTEGRATION TESTING FRAMEWORK")
    print("="*80)
    
    # Pre-flight checks
    print("\nPre-flight checks:")
    print("-"*80)
    
    # Check Python version
    import sys
    print(f"Python version: {sys.version.split()[0]}")
    
    # Check required modules
    try:
        import requests
        print("✅ requests module available")
    except ImportError:
        print("❌ requests module not available - install with: pip install requests")
        sys.exit(1)
    
    # Check Docker environment
    in_docker = os.path.exists("/.dockerenv")
    print(f"Environment: {'Docker container' if in_docker else 'Host machine'}")
    
    # Check service URLs
    print(f"\nService URLs:")
    print(f"  LocalAI: {LOCALAI_SERVICE_URL}")
    print(f"  Transformers: {TRANSFORMERS_SERVICE_URL}")
    print(f"  Model-Server: {MODEL_SERVER_URL}")
    print(f"  Test Timeout: {TEST_TIMEOUT}s")
    
    # Quick connectivity check
    print(f"\nQuick connectivity check:")
    try:
        import requests
        r = requests.get(f"{LOCALAI_SERVICE_URL}/healthz", timeout=5)
        print(f"  ✅ LocalAI reachable: {r.status_code}")
    except Exception as e:
        print(f"  ⚠️  LocalAI not reachable: {e}")
    
    try:
        r = requests.get(f"{TRANSFORMERS_SERVICE_URL}/health", timeout=5)
        print(f"  ✅ Transformers-Service reachable: {r.status_code}")
    except Exception as e:
        print(f"  ⚠️  Transformers-Service not reachable: {e}")
    
    try:
        r = requests.get(f"{MODEL_SERVER_URL}/health", timeout=5)
        print(f"  ✅ Model-Server reachable: {r.status_code}")
    except Exception as e:
        print(f"  ⚠️  Model-Server not reachable: {e}")
    
    print("")
    
    framework = LocalAITestFramework()
    
    try:
        # Step 1: Test service health
        framework.test_service_health()
        
        # Step 2: Test model accessibility
        framework.test_model_accessibility()
        
        # Step 3: Test domain routing
        test_queries = load_test_queries()
        framework.test_domain_routing(test_queries)
        
        # Step 4: Test backend integration
        domains = load_domains_config()
        framework.test_backend_integration(domains)
        
        # Step 5: Test end-to-end
        framework.test_end_to_end(test_queries)
        
        # Step 6: Calculate performance metrics
        framework.calculate_performance_metrics()
        
        # Generate and print report
        report = framework.generate_report()
        print("\n" + report)
        
        # Save results
        framework.save_results()
        
        # Exit code based on results
        all_healthy = all(r.healthy for r in framework.health_results)
        models_accessible = all(r.accessible for r in framework.model_results)
        queries_successful = all(r.success for r in framework.query_results) if framework.query_results else False
        
        if all_healthy and models_accessible and queries_successful:
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print("\n⚠️  Some tests failed - see report for details")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

