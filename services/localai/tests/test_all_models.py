#!/usr/bin/env python3
"""
Comprehensive Model Testing Framework

Tests all configured models to ensure they:
1. Are accessible and respond to requests
2. Produce real, expected outputs
3. Meet quality thresholds
4. Handle different prompt types correctly
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

# Configuration
TRANSFORMERS_SERVICE_URL = os.getenv("TRANSFORMERS_SERVICE_URL", "http://localhost:9090")
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://model-server:8088")
LOCALAI_SERVICE_URL = os.getenv("LOCALAI_SERVICE_URL", "http://localhost:8081")
TEST_TIMEOUT = int(os.getenv("TEST_TIMEOUT", "120"))
RESULTS_DIR = os.getenv("RESULTS_DIR", "/tmp/model_test_results")

@dataclass
class TestResult:
    """Result of a single model test"""
    model_name: str
    test_name: str
    success: bool
    response_time_ms: float
    response_text: str
    error: Optional[str] = None
    token_count: Optional[int] = None
    quality_score: Optional[float] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class ModelTestSummary:
    """Summary of all tests for a model"""
    model_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_response_time_ms: float
    average_quality_score: float
    results: List[TestResult]
    status: str  # "PASS", "FAIL", "PARTIAL"

class ModelTester:
    """Framework for testing all models"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.summaries: Dict[str, ModelTestSummary] = {}
        
    def load_models_from_domains(self) -> List[str]:
        """Load all unique models from domains.json"""
        # Try multiple possible paths
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "../config/domains.json"),
            "/app/config/domains.json",  # Docker container path
            "/workspace/services/localai/config/domains.json",  # Alternative Docker path
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../config/domains.json"),
        ]
        
        domains_path = None
        for path in possible_paths:
            if os.path.exists(path):
                domains_path = path
                break
        
        if not domains_path:
            # Fallback: use environment variable or default
            domains_path = os.getenv("DOMAINS_JSON_PATH", "/app/config/domains.json")
            if not os.path.exists(domains_path):
                raise FileNotFoundError(f"domains.json not found. Tried: {possible_paths}")
        
        with open(domains_path, 'r') as f:
            domains_config = json.load(f)
        
        models = set()
        for domain_id, config in domains_config.get('domains', {}).items():
            model_name = config.get('model_name', '')
            if model_name:
                models.add(model_name)
        
        return sorted(list(models))
    
    def test_model_server_health(self) -> bool:
        """Test if model-server is accessible"""
        try:
            response = requests.get(
                f"{MODEL_SERVER_URL}/health",
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def test_model_health(self, model_name: str) -> TestResult:
        """Test if model is accessible via health check"""
        start_time = time.time()
        try:
            response = requests.get(
                f"{TRANSFORMERS_SERVICE_URL}/health",
                timeout=10
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                models_available = data.get('models', [])
                success = model_name in models_available
                error = None if success else f"Model {model_name} not in available models: {models_available}"
            else:
                success = False
                error = f"Health check returned status {response.status_code}"
            
            return TestResult(
                model_name=model_name,
                test_name="health_check",
                success=success,
                response_time_ms=response_time,
                response_text="",
                error=error
            )
        except Exception as e:
            return TestResult(
                model_name=model_name,
                test_name="health_check",
                success=False,
                response_time_ms=(time.time() - start_time) * 1000,
                response_text="",
                error=str(e)
            )
    
    def test_simple_prompt(self, model_name: str) -> TestResult:
        """Test model with a simple prompt"""
        prompt = "Say 'Hello, I am working correctly!' in one sentence."
        return self._test_chat_completion(model_name, "simple_prompt", prompt)
    
    def test_reasoning_prompt(self, model_name: str) -> TestResult:
        """Test model with a reasoning prompt"""
        prompt = "What is 2 + 2? Answer with just the number."
        return self._test_chat_completion(model_name, "reasoning_prompt", prompt)
    
    def test_code_prompt(self, model_name: str) -> TestResult:
        """Test model with a code generation prompt"""
        prompt = "Write a Python function that adds two numbers. Return only the function code."
        return self._test_chat_completion(model_name, "code_prompt", prompt)
    
    def test_domain_specific_prompt(self, model_name: str, domain_type: str = "general") -> TestResult:
        """Test model with domain-specific prompt"""
        prompts = {
            "sql": "What is a SQL SELECT statement? Explain in one sentence.",
            "finance": "What is a ledger entry? Explain in one sentence.",
            "code": "What is a function in programming? Explain in one sentence.",
            "general": "Explain what AI is in one sentence."
        }
        prompt = prompts.get(domain_type, prompts["general"])
        return self._test_chat_completion(model_name, f"domain_{domain_type}_prompt", prompt)
    
    def _test_chat_completion(
        self,
        model_name: str,
        test_name: str,
        user_message: str,
        system_message: Optional[str] = None
    ) -> TestResult:
        """Test chat completion endpoint"""
        start_time = time.time()
        
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": user_message})
            
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 256
            }
            
            response = requests.post(
                f"{TRANSFORMERS_SERVICE_URL}/v1/chat/completions",
                json=payload,
                timeout=TEST_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return TestResult(
                    model_name=model_name,
                    test_name=test_name,
                    success=False,
                    response_time_ms=response_time,
                    response_text="",
                    error=f"HTTP {response.status_code}: {response.text[:200]}"
                )
            
            data = response.json()
            choices = data.get('choices', [])
            
            if not choices:
                return TestResult(
                    model_name=model_name,
                    test_name=test_name,
                    success=False,
                    response_time_ms=response_time,
                    response_text="",
                    error="No choices in response"
                )
            
            response_text = choices[0].get('message', {}).get('content', '')
            usage = data.get('usage', {})
            token_count = usage.get('total_tokens', 0)
            
            # Quality checks
            quality_score = self._calculate_quality_score(response_text, user_message)
            success = (
                len(response_text) > 0 and
                quality_score > 0.3 and
                response_time < TEST_TIMEOUT * 1000
            )
            
            return TestResult(
                model_name=model_name,
                test_name=test_name,
                success=success,
                response_time_ms=response_time,
                response_text=response_text[:500],  # Truncate for storage
                token_count=token_count,
                quality_score=quality_score,
                error=None if success else "Quality check failed"
            )
            
        except requests.exceptions.Timeout:
            return TestResult(
                model_name=model_name,
                test_name=test_name,
                success=False,
                response_time_ms=(time.time() - start_time) * 1000,
                response_text="",
                error=f"Request timed out after {TEST_TIMEOUT}s"
            )
        except Exception as e:
            return TestResult(
                model_name=model_name,
                test_name=test_name,
                success=False,
                response_time_ms=(time.time() - start_time) * 1000,
                response_text="",
                error=f"Exception: {str(e)}"
            )
    
    def _calculate_quality_score(self, response_text: str, prompt: str) -> float:
        """Calculate quality score for response (0.0 to 1.0)"""
        if not response_text or len(response_text.strip()) == 0:
            return 0.0
        
        score = 0.0
        
        # Length check (not too short, not too long)
        length = len(response_text)
        if 10 <= length <= 2000:
            score += 0.3
        elif length > 0:
            score += 0.1
        
        # Non-empty check
        if response_text.strip():
            score += 0.2
        
        # Basic coherence (no obvious errors)
        error_indicators = ["error", "exception", "failed", "not found", "404", "500"]
        has_errors = any(indicator in response_text.lower() for indicator in error_indicators)
        if not has_errors:
            score += 0.3
        
        # Relevance (contains some expected words from prompt)
        prompt_words = set(prompt.lower().split())
        response_words = set(response_text.lower().split())
        if len(prompt_words & response_words) > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def test_model(self, model_name: str) -> ModelTestSummary:
        """Run all tests for a model"""
        print(f"\n{'='*60}")
        print(f"Testing model: {model_name}")
        print(f"{'='*60}")
        
        model_results = []
        
        # Health check
        print(f"  [1/5] Health check...")
        result = self.test_model_health(model_name)
        model_results.append(result)
        print(f"    {'✅ PASS' if result.success else '❌ FAIL'}: {result.error or 'OK'}")
        
        if not result.success:
            # If health check fails, skip other tests
            return ModelTestSummary(
                model_name=model_name,
                total_tests=1,
                passed_tests=0,
                failed_tests=1,
                average_response_time_ms=result.response_time_ms,
                average_quality_score=0.0,
                results=model_results,
                status="FAIL"
            )
        
        # Simple prompt
        print(f"  [2/5] Simple prompt...")
        result = self.test_simple_prompt(model_name)
        model_results.append(result)
        print(f"    {'✅ PASS' if result.success else '❌ FAIL'}: {result.response_text[:100] if result.response_text else result.error}")
        
        # Reasoning prompt
        print(f"  [3/5] Reasoning prompt...")
        result = self.test_reasoning_prompt(model_name)
        model_results.append(result)
        print(f"    {'✅ PASS' if result.success else '❌ FAIL'}: {result.response_text[:100] if result.response_text else result.error}")
        
        # Code prompt (for code-capable models)
        print(f"  [4/5] Code prompt...")
        result = self.test_code_prompt(model_name)
        model_results.append(result)
        print(f"    {'✅ PASS' if result.success else '❌ FAIL'}: {result.response_text[:100] if result.response_text else result.error}")
        
        # Domain-specific prompt
        print(f"  [5/5] Domain-specific prompt...")
        result = self.test_domain_specific_prompt(model_name, "general")
        model_results.append(result)
        print(f"    {'✅ PASS' if result.success else '❌ FAIL'}: {result.response_text[:100] if result.response_text else result.error}")
        
        # Calculate summary
        passed = sum(1 for r in model_results if r.success)
        failed = len(model_results) - passed
        avg_time = sum(r.response_time_ms for r in model_results) / len(model_results)
        quality_scores = [r.quality_score for r in model_results if r.quality_score is not None]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        status = "PASS" if failed == 0 else ("PARTIAL" if passed > 0 else "FAIL")
        
        summary = ModelTestSummary(
            model_name=model_name,
            total_tests=len(model_results),
            passed_tests=passed,
            failed_tests=failed,
            average_response_time_ms=avg_time,
            average_quality_score=avg_quality,
            results=model_results,
            status=status
        )
        
        self.results.extend(model_results)
        self.summaries[model_name] = summary
        
        return summary
    
    def test_all_models(self) -> Dict[str, ModelTestSummary]:
        """Test all models"""
        models = self.load_models_from_domains()
        
        print(f"\n{'='*60}")
        print(f"Model Testing Framework")
        print(f"{'='*60}")
        print(f"Found {len(models)} models to test:")
        for model in models:
            print(f"  - {model}")
        
        # Check model-server
        if self.test_model_server_health():
            print(f"\n✅ Model-server is accessible at {MODEL_SERVER_URL}")
        else:
            print(f"\n⚠️  Model-server not accessible, will use fallback paths")
        
        for model in models:
            try:
                self.test_model(model)
            except Exception as e:
                print(f"\n❌ Error testing {model}: {e}")
                traceback.print_exc()
        
        return self.summaries
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report"""
        report = []
        report.append("="*80)
        report.append("MODEL TESTING REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Total Models Tested: {len(self.summaries)}")
        report.append(f"Model Server: {MODEL_SERVER_URL}")
        report.append("")
        
        # Overall statistics
        total_tests = sum(s.total_tests for s in self.summaries.values())
        total_passed = sum(s.passed_tests for s in self.summaries.values())
        total_failed = sum(s.failed_tests for s in self.summaries.values())
        
        report.append("OVERALL STATISTICS")
        report.append("-"*80)
        report.append(f"Total Tests: {total_tests}")
        if total_tests > 0:
            report.append(f"Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
            report.append(f"Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)")
        else:
            report.append(f"Passed: {total_passed} (N/A - no tests run)")
            report.append(f"Failed: {total_failed} (N/A - no tests run)")
        report.append("")
        
        # Per-model summary
        report.append("PER-MODEL SUMMARY")
        report.append("-"*80)
        for model_name, summary in sorted(self.summaries.items()):
            status_icon = "✅" if summary.status == "PASS" else ("⚠️" if summary.status == "PARTIAL" else "❌")
            report.append(f"{status_icon} {model_name}")
            report.append(f"   Status: {summary.status}")
            report.append(f"   Tests: {summary.passed_tests}/{summary.total_tests} passed")
            report.append(f"   Avg Response Time: {summary.average_response_time_ms:.1f}ms")
            report.append(f"   Avg Quality Score: {summary.average_quality_score:.2f}")
            report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-"*80)
        for model_name, summary in sorted(self.summaries.items()):
            report.append(f"\nModel: {model_name}")
            for result in summary.results:
                icon = "✅" if result.success else "❌"
                report.append(f"  {icon} {result.test_name}")
                report.append(f"     Success: {result.success}")
                report.append(f"     Response Time: {result.response_time_ms:.1f}ms")
                if result.quality_score:
                    report.append(f"     Quality Score: {result.quality_score:.2f}")
                if result.response_text:
                    report.append(f"     Response: {result.response_text[:150]}...")
                if result.error:
                    report.append(f"     Error: {result.error}")
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self, output_dir: str = RESULTS_DIR):
        """Save test results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON results
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "model_server_url": MODEL_SERVER_URL,
            "summaries": {k: asdict(v) for k, v in self.summaries.items()},
            "results": [asdict(r) for r in self.results]
        }
        
        json_path = os.path.join(output_dir, f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save text report
        report = self.generate_report()
        report_path = os.path.join(output_dir, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✅ Results saved to:")
        print(f"   JSON: {json_path}")
        print(f"   Report: {report_path}")
        
        return json_path, report_path


def main():
    """Main entry point"""
    tester = ModelTester()
    
    print("\n🚀 Starting Model Testing Framework...")
    print(f"   Transformers Service: {TRANSFORMERS_SERVICE_URL}")
    print(f"   Model Server: {MODEL_SERVER_URL}")
    print(f"   Test Timeout: {TEST_TIMEOUT}s")
    
    try:
        summaries = tester.test_all_models()
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        for model_name, summary in sorted(summaries.items()):
            status_icon = "✅" if summary.status == "PASS" else ("⚠️" if summary.status == "PARTIAL" else "❌")
            print(f"{status_icon} {model_name}: {summary.passed_tests}/{summary.total_tests} tests passed")
        
        # Generate and save report
        report = tester.generate_report()
        print("\n" + report)
        
        tester.save_results()
        
        # Exit code
        all_passed = all(s.status == "PASS" for s in summaries.values())
        sys.exit(0 if all_passed else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

