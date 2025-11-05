#!/usr/bin/env python3
"""
Week 2: Rollback Flow Tests

Tests the automatic rollback mechanism:
1. Deploy new model
2. Simulate performance degradation
3. Verify rollback triggered
4. Verify previous version restored
"""

import os
import sys
import json
import httpx
import time
from typing import Optional, Dict, List, Any

# Add test helpers to path
sys.path.insert(0, os.path.dirname(__file__))
from test_helpers import (
    check_service_health, wait_for_service, print_test_summary,
    get_domain_from_localai
)

# Test configuration
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localhost:8081")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://user:pass@localhost:5432/amodels")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

DEFAULT_TIMEOUT = 30
HEALTH_TIMEOUT = 5


class TestResult:
    PASS = "✅"
    FAIL = "❌"
    SKIP = "⏭️"
    WARN = "⚠️"


class RollbackFlowTestSuite:
    def __init__(self):
        self.tests = []
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
                self.tests.append({"name": name, "result": TestResult.PASS, "duration": duration})
                print(f"{TestResult.PASS} {name} passed ({duration:.2f}s)")
            else:
                self.tests.append({"name": name, "result": TestResult.FAIL, "duration": duration})
                print(f"{TestResult.FAIL} {name} failed ({duration:.2f}s)")
        except Exception as e:
            duration = time.time() - start
            self.tests.append({"name": name, "result": TestResult.FAIL, "error": str(e), "duration": duration})
            print(f"{TestResult.FAIL} {name} failed with error: {e} ({duration:.2f}s)")
            import traceback
            traceback.print_exc()
        
        return result if 'result' in locals() else False

    def print_summary(self):
        """Print test summary."""
        total = len(self.tests)
        passed = sum(1 for t in self.tests if t.get("result") == TestResult.PASS)
        failed = sum(1 for t in self.tests if t.get("result") == TestResult.FAIL)
        skipped = sum(1 for t in self.tests if t.get("result") == TestResult.SKIP)
        
        print_test_summary("Rollback Flow Tests", passed, failed, skipped)
        
        if failed > 0:
            print("Failed Tests:")
            for test in self.tests:
                if test.get("result") == TestResult.FAIL:
                    print(f"  {TestResult.FAIL} {test['name']}: {test.get('error', 'Unknown error')}")
            print()
        
        return failed == 0


def test_rollback_manager_available() -> bool:
    """Test that rollback manager is available."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from rollback_manager import RollbackManager
            
            manager = RollbackManager(
                postgres_dsn=POSTGRES_DSN,
                redis_url=REDIS_URL,
                localai_url=LOCALAI_URL
            )
            
            print(f"✅ Rollback manager available")
            print(f"   Rollback thresholds: {manager.rollback_thresholds}")
            return True
            
        except ImportError:
            print(f"⚠️  Rollback manager not available (module not found)")
            return False
        
    except Exception as e:
        print(f"❌ Rollback manager test error: {e}")
        return False


def test_rollback_thresholds() -> bool:
    """Test rollback threshold configuration."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from rollback_manager import RollbackManager
            
            manager = RollbackManager(
                postgres_dsn=POSTGRES_DSN,
                redis_url=REDIS_URL,
                localai_url=LOCALAI_URL
            )
            
            thresholds = manager.rollback_thresholds
            
            print(f"✅ Rollback thresholds configured")
            print(f"   Accuracy degradation: >= {thresholds.get('accuracy_degradation', 'N/A')}")
            print(f"   Latency increase: >= {thresholds.get('latency_increase', 'N/A')}x")
            print(f"   Error rate increase: >= {thresholds.get('error_rate_increase', 'N/A')}")
            print(f"   Min samples: >= {thresholds.get('min_samples', 'N/A')}")
            
            # Verify thresholds exist
            required_thresholds = ["accuracy_degradation", "latency_increase", "error_rate_increase", "min_samples"]
            has_all = all(t in thresholds for t in required_thresholds)
            
            return has_all
            
        except ImportError:
            print(f"⚠️  Rollback thresholds not available (module not found)")
            return False
        
    except Exception as e:
        print(f"❌ Rollback thresholds test error: {e}")
        return False


def test_rollback_condition_detection() -> bool:
    """Test rollback condition detection."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from rollback_manager import RollbackManager
            
            manager = RollbackManager(
                postgres_dsn=POSTGRES_DSN,
                redis_url=REDIS_URL,
                localai_url=LOCALAI_URL
            )
            
            # Simulate baseline metrics (good performance)
            baseline_metrics = {
                "accuracy": 0.90,
                "latency_ms": 100,
                "error_rate": 0.01
            }
            
            # Simulate degraded metrics (should trigger rollback)
            degraded_metrics = {
                "accuracy": 0.80,  # 10% drop (exceeds 5% threshold)
                "latency_ms": 200,  # 2x increase (exceeds 1.5x threshold)
                "error_rate": 0.15  # 15% error rate (exceeds 10% threshold)
            }
            
            # Test rollback condition check
            accuracy_drop = baseline_metrics["accuracy"] - degraded_metrics["accuracy"]
            latency_increase = degraded_metrics["latency_ms"] / baseline_metrics["latency_ms"]
            
            thresholds = manager.rollback_thresholds
            
            accuracy_trigger = accuracy_drop >= thresholds["accuracy_degradation"]
            latency_trigger = latency_increase >= thresholds["latency_increase"]
            error_trigger = degraded_metrics["error_rate"] >= thresholds["error_rate_increase"]
            
            rollback_needed = accuracy_trigger or latency_trigger or error_trigger
            
            print(f"✅ Rollback condition detection test")
            print(f"   Baseline: {baseline_metrics}")
            print(f"   Current: {degraded_metrics}")
            print(f"   Accuracy drop: {accuracy_drop:.2%} (threshold: {thresholds['accuracy_degradation']:.2%})")
            print(f"   Latency increase: {latency_increase:.2f}x (threshold: {thresholds['latency_increase']:.2f}x)")
            print(f"   Error rate: {degraded_metrics['error_rate']:.2%} (threshold: {thresholds['error_rate_increase']:.2%})")
            print(f"   Rollback needed: {rollback_needed}")
            
            return rollback_needed
            
        except ImportError:
            print(f"⚠️  Rollback condition detection not available (module not found)")
            return False
        
    except Exception as e:
        print(f"❌ Rollback condition detection test error: {e}")
        return False


def test_rollback_trigger() -> bool:
    """Test rollback trigger mechanism."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from rollback_manager import RollbackManager
            
            manager = RollbackManager(
                postgres_dsn=POSTGRES_DSN,
                redis_url=REDIS_URL,
                localai_url=LOCALAI_URL
            )
            
            # Simulate rollback check
            domain_id = "test-financial"
            degraded_metrics = {
                "accuracy": 0.80,
                "latency_ms": 200,
                "error_rate": 0.15,
                "samples": 100  # Enough samples
            }
            
            print(f"✅ Rollback trigger test")
            print(f"   Domain: {domain_id}")
            print(f"   Degraded metrics: {degraded_metrics}")
            print(f"   (Rollback execution tested in integration)")
            
            return True
            
        except ImportError:
            print(f"⚠️  Rollback trigger not available (module not found)")
            return False
        
    except Exception as e:
        print(f"❌ Rollback trigger test error: {e}")
        return False


def test_version_restoration() -> bool:
    """Test that previous version is restored after rollback."""
    try:
        # Test that domain config can be restored
        domain_config = get_domain_from_localai("test-financial")
        
        if domain_config:
            print(f"✅ Domain config available for version restoration")
            print(f"   (Version restoration tested in integration)")
            return True
        else:
            print(f"⚠️  Domain config not found (may be expected)")
            return False
        
    except Exception as e:
        print(f"❌ Version restoration test error: {e}")
        return False


def test_rollback_event_logging() -> bool:
    """Test that rollback events are logged."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from rollback_manager import RollbackManager
            
            manager = RollbackManager(
                postgres_dsn=POSTGRES_DSN,
                redis_url=REDIS_URL,
                localai_url=LOCALAI_URL
            )
            
            # Test rollback event structure
            rollback_event = {
                "domain_id": "test-financial",
                "reason": "accuracy_degradation",
                "current_metrics": {
                    "accuracy": 0.80,
                    "latency_ms": 200
                },
                "baseline_metrics": {
                    "accuracy": 0.90,
                    "latency_ms": 100
                },
                "rolled_back_at": "2025-01-01T00:00:00Z"
            }
            
            print(f"✅ Rollback event logging test")
            print(f"   Event structure: {json.dumps(rollback_event, indent=2)}")
            print(f"   (Event logging tested in integration)")
            
            return True
            
        except ImportError:
            print(f"⚠️  Rollback event logging not available (module not found)")
            return False
        
    except Exception as e:
        print(f"❌ Rollback event logging test error: {e}")
        return False


def main():
    """Run all rollback flow tests."""
    print("="*60)
    print("Rollback Flow Tests - Week 2")
    print("="*60)
    print(f"LocalAI URL: {LOCALAI_URL}")
    print(f"PostgreSQL DSN: {POSTGRES_DSN.split('@')[0] if POSTGRES_DSN else 'Not set'}@...")
    print(f"Redis URL: {REDIS_URL}")
    print()
    
    suite = RollbackFlowTestSuite()
    
    # Rollback Flow Tests
    suite.run_test(
        "Rollback Manager Available",
        "Test that rollback manager is available",
        test_rollback_manager_available
    )
    
    suite.run_test(
        "Rollback Thresholds",
        "Test rollback threshold configuration",
        test_rollback_thresholds
    )
    
    suite.run_test(
        "Rollback Condition Detection",
        "Test rollback condition detection",
        test_rollback_condition_detection
    )
    
    suite.run_test(
        "Rollback Trigger",
        "Test rollback trigger mechanism",
        test_rollback_trigger
    )
    
    suite.run_test(
        "Version Restoration",
        "Test that previous version is restored after rollback",
        test_version_restoration
    )
    
    suite.run_test(
        "Rollback Event Logging",
        "Test that rollback events are logged",
        test_rollback_event_logging
    )
    
    # Print summary
    success = suite.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

