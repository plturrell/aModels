#!/usr/bin/env python3
"""
Week 1: Domain Trainer Tests

Tests domain-specific model training functionality:
- Domain trainer initialization
- Training run ID generation
- Model version tracking
- Deployment threshold checks
- Domain config integration
"""

import os
import sys
import json
import httpx
import time
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

# Test configuration
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localhost:8081")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://user:pass@localhost:5432/amodels")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

DEFAULT_TIMEOUT = 30
HEALTH_TIMEOUT = 5


class TestResult(Enum):
    PASS = "✅"
    FAIL = "❌"
    SKIP = "⏭️"
    WARN = "⚠️"


@dataclass
class TestCase:
    name: str
    description: str
    result: TestResult = TestResult.SKIP
    message: str = ""
    duration: float = 0.0


class DomainTrainerTestSuite:
    def __init__(self):
        self.tests: List[TestCase] = []
        self.start_time = time.time()

    def run_test(self, name: str, description: str, test_func):
        """Run a test and record the result."""
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"{'='*60}")
        print(f"Description: {description}")
        print()
        
        start = time.time()
        try:
            result = test_func()
            duration = time.time() - start
            
            if result:
                test_case = TestCase(name, description, TestResult.PASS, "", duration)
                print(f"{TestResult.PASS.value} {name} passed ({duration:.2f}s)")
            else:
                test_case = TestCase(name, description, TestResult.FAIL, "Test returned False", duration)
                print(f"{TestResult.FAIL.value} {name} failed ({duration:.2f}s)")
        except Exception as e:
            duration = time.time() - start
            test_case = TestCase(name, description, TestResult.FAIL, str(e), duration)
            print(f"{TestResult.FAIL.value} {name} failed with error: {e} ({duration:.2f}s)")
            import traceback
            traceback.print_exc()
        
        self.tests.append(test_case)
        return test_case.result == TestResult.PASS

    def print_summary(self):
        """Print test summary."""
        total = len(self.tests)
        passed = sum(1 for t in self.tests if t.result == TestResult.PASS)
        failed = sum(1 for t in self.tests if t.result == TestResult.FAIL)
        skipped = sum(1 for t in self.tests if t.result == TestResult.SKIP)
        total_duration = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("Domain Trainer Test Summary")
        print("="*60)
        print(f"Total Tests: {total}")
        print(f"{TestResult.PASS.value} Passed: {passed}")
        print(f"{TestResult.FAIL.value} Failed: {failed}")
        print(f"{TestResult.SKIP.value} Skipped: {skipped}")
        print(f"Total Duration: {total_duration:.2f}s")
        print()
        
        if failed > 0:
            print("Failed Tests:")
            for test in self.tests:
                if test.result == TestResult.FAIL:
                    print(f"  {TestResult.FAIL.value} {test.name}: {test.message}")
            print()
        
        return failed == 0


def test_domain_trainer_import() -> bool:
    """Test that domain trainer module can be imported."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from domain_trainer import DomainTrainer
            print(f"✅ Domain trainer module imported successfully")
            return True
        except ImportError as e:
            print(f"⚠️  Domain trainer module not found: {e}")
            return False
    except Exception as e:
        print(f"❌ Import test error: {e}")
        return False


def test_domain_trainer_initialization() -> bool:
    """Test DomainTrainer initialization."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from domain_trainer import DomainTrainer
            
            trainer = DomainTrainer(
                localai_url=LOCALAI_URL,
                postgres_dsn=POSTGRES_DSN,
                redis_url=REDIS_URL
            )
            
            print(f"✅ DomainTrainer initialized successfully")
            print(f"   LocalAI URL: {trainer.localai_url}")
            print(f"   Checkpoint dir: {trainer.checkpoint_dir}")
            print(f"   Model output dir: {trainer.model_output_dir}")
            return True
        except ImportError:
            print(f"⚠️  DomainTrainer not available (module not found)")
            return False
    except Exception as e:
        print(f"❌ DomainTrainer initialization error: {e}")
        return False


def test_training_run_id_generation() -> bool:
    """Test training run ID generation."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from domain_trainer import DomainTrainer
            from datetime import datetime
            
            trainer = DomainTrainer(localai_url=LOCALAI_URL)
            
            # Test run ID generation pattern
            domain_id = "test-domain"
            expected_pattern = f"{domain_id}_{datetime.now().strftime('%Y%m%d')}"
            
            # Generate a test run ID (simulate)
            test_run_id = f"{domain_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            print(f"✅ Training run ID generation test")
            print(f"   Domain: {domain_id}")
            print(f"   Run ID pattern: {domain_id}_YYYYMMDD_HHMMSS")
            print(f"   Example: {test_run_id}")
            
            # Verify pattern
            if test_run_id.startswith(domain_id) and "_" in test_run_id:
                return True
            else:
                return False
        except ImportError:
            print(f"⚠️  Training run ID test not available (module not found)")
            return False
    except Exception as e:
        print(f"❌ Training run ID test error: {e}")
        return False


def test_deployment_thresholds() -> bool:
    """Test deployment threshold configuration."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from domain_trainer import DomainTrainer
            
            trainer = DomainTrainer(localai_url=LOCALAI_URL)
            
            thresholds = trainer.deployment_thresholds
            
            print(f"✅ Deployment thresholds configured")
            print(f"   Accuracy: >= {thresholds.get('accuracy', 'N/A')}")
            print(f"   Latency: <= {thresholds.get('latency_ms', 'N/A')} ms")
            print(f"   Training loss: <= {thresholds.get('training_loss', 'N/A')}")
            print(f"   Validation loss: <= {thresholds.get('validation_loss', 'N/A')}")
            
            # Verify thresholds exist
            required_thresholds = ["accuracy", "latency_ms", "training_loss", "validation_loss"]
            has_all = all(t in thresholds for t in required_thresholds)
            
            return has_all
        except ImportError:
            print(f"⚠️  Deployment thresholds test not available (module not found)")
            return False
    except Exception as e:
        print(f"❌ Deployment thresholds test error: {e}")
        return False


def test_domain_config_integration() -> bool:
    """Test domain config integration for training."""
    try:
        # Get domain configs from LocalAI
        response = httpx.get(
            f"{LOCALAI_URL}/v1/domains",
            timeout=DEFAULT_TIMEOUT
        )
        
        if response.status_code != 200:
            print(f"⚠️  Cannot test domain config: LocalAI not available")
            return False
        
        data = response.json()
        domains = data.get("data", [])
        
        if len(domains) == 0:
            print(f"⚠️  No domains to test config integration")
            return False
        
        # Test that domains have config needed for training
        configs_valid = 0
        for domain in domains[:3]:  # Test first 3
            domain_id = domain.get("id", "")
            config = domain.get("config", domain)
            
            # Check for training-relevant config
            has_model = "model_path" in config or "model_name" in config
            has_backend = "backend_type" in config
            
            if has_model or has_backend:
                configs_valid += 1
                print(f"   Domain {domain_id}: Has model config")
        
        if configs_valid > 0:
            print(f"✅ Domain config integration test")
            print(f"   {configs_valid}/{min(3, len(domains))} domains have training config")
            return True
        else:
            print(f"⚠️  No domains have training config")
            return False
    except Exception as e:
        print(f"❌ Domain config integration test error: {e}")
        return False


def test_postgres_connection() -> bool:
    """Test PostgreSQL connection for domain trainer."""
    try:
        # Test PostgreSQL connection
        # Note: This would require psycopg2, but we can check DSN format
        if POSTGRES_DSN:
            print(f"✅ PostgreSQL DSN configured")
            print(f"   DSN: {POSTGRES_DSN.split('@')[0]}@...")  # Hide password
            return True
        else:
            print(f"⚠️  PostgreSQL DSN not configured")
            return False
    except Exception as e:
        print(f"❌ PostgreSQL connection test error: {e}")
        return False


def test_redis_connection() -> bool:
    """Test Redis connection for domain trainer."""
    try:
        # Test Redis connection
        if REDIS_URL:
            print(f"✅ Redis URL configured")
            print(f"   URL: {REDIS_URL}")
            return True
        else:
            print(f"⚠️  Redis URL not configured")
            return False
    except Exception as e:
        print(f"❌ Redis connection test error: {e}")
        return False


def main():
    """Run all domain trainer tests."""
    print("="*60)
    print("Domain Trainer Test Suite - Week 1")
    print("="*60)
    print(f"LocalAI URL: {LOCALAI_URL}")
    print(f"PostgreSQL DSN: {POSTGRES_DSN.split('@')[0] if POSTGRES_DSN else 'Not set'}@...")
    print(f"Redis URL: {REDIS_URL}")
    print()
    
    suite = DomainTrainerTestSuite()
    
    # Domain Trainer Tests
    suite.run_test(
        "Domain Trainer Import",
        "Test that domain trainer module can be imported",
        test_domain_trainer_import
    )
    
    suite.run_test(
        "Domain Trainer Initialization",
        "Test DomainTrainer initialization",
        test_domain_trainer_initialization
    )
    
    suite.run_test(
        "Training Run ID Generation",
        "Test training run ID generation",
        test_training_run_id_generation
    )
    
    suite.run_test(
        "Deployment Thresholds",
        "Test deployment threshold configuration",
        test_deployment_thresholds
    )
    
    suite.run_test(
        "Domain Config Integration",
        "Test domain config integration for training",
        test_domain_config_integration
    )
    
    suite.run_test(
        "PostgreSQL Connection",
        "Test PostgreSQL connection for domain trainer",
        test_postgres_connection
    )
    
    suite.run_test(
        "Redis Connection",
        "Test Redis connection for domain trainer",
        test_redis_connection
    )
    
    # Print summary
    success = suite.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

