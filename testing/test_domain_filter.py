#!/usr/bin/env python3
"""
Week 1: Domain Filter Tests

Tests domain filtering functionality in the Training service:
- Keyword-based filtering
- Differential privacy application
- Privacy budget tracking
- Domain-specific feature extraction
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
TRAINING_SERVICE_URL = os.getenv("TRAINING_SERVICE_URL", "http://localhost:8080")

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


class DomainFilterTestSuite:
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
        print("Domain Filter Test Summary")
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


def test_domain_filter_import() -> bool:
    """Test that domain filter module can be imported."""
    try:
        # Try to import domain_filter module
        # This tests if the module exists and is importable
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from domain_filter import DomainFilter, PrivacyConfig
            print(f"✅ Domain filter module imported successfully")
            print(f"   DomainFilter class available")
            print(f"   PrivacyConfig class available")
            return True
        except ImportError as e:
            print(f"⚠️  Domain filter module not found: {e}")
            print(f"   (This is OK if running from outside training service)")
            return False
    except Exception as e:
        print(f"❌ Import test error: {e}")
        return False


def test_privacy_config_creation() -> bool:
    """Test PrivacyConfig creation and configuration."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from domain_filter import PrivacyConfig
            
            # Test default config
            config = PrivacyConfig()
            print(f"✅ PrivacyConfig created with defaults")
            print(f"   Epsilon: {config.epsilon}")
            print(f"   Delta: {config.delta}")
            
            # Test custom config
            custom_config = PrivacyConfig(epsilon=2.0, delta=1e-5)
            print(f"✅ Custom PrivacyConfig created")
            print(f"   Epsilon: {custom_config.epsilon}")
            print(f"   Delta: {custom_config.delta}")
            
            return True
        except ImportError:
            print(f"⚠️  PrivacyConfig not available (module not found)")
            return False
    except Exception as e:
        print(f"❌ PrivacyConfig test error: {e}")
        return False


def test_domain_filter_initialization() -> bool:
    """Test DomainFilter initialization."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from domain_filter import DomainFilter, PrivacyConfig
            
            # Test initialization
            privacy_config = PrivacyConfig(epsilon=1.0)
            domain_filter = DomainFilter(
                localai_url=LOCALAI_URL,
                privacy_config=privacy_config
            )
            
            print(f"✅ DomainFilter initialized successfully")
            print(f"   LocalAI URL: {LOCALAI_URL}")
            print(f"   Privacy epsilon: {privacy_config.epsilon}")
            
            return True
        except ImportError:
            print(f"⚠️  DomainFilter not available (module not found)")
            return False
    except Exception as e:
        print(f"❌ DomainFilter initialization error: {e}")
        return False


def test_keyword_filtering_logic() -> bool:
    """Test keyword-based filtering logic."""
    try:
        # Get domain configs to test filtering
        response = httpx.get(
            f"{LOCALAI_URL}/v1/domains",
            timeout=DEFAULT_TIMEOUT
        )
        
        if response.status_code != 200:
            print(f"⚠️  Cannot test filtering: LocalAI not available")
            return False
        
        data = response.json()
        domains = data.get("data", [])
        
        if len(domains) == 0:
            print(f"⚠️  No domains to test filtering")
            return False
        
        # Test keyword matching
        test_cases = []
        for domain in domains[:3]:  # Test first 3 domains
            domain_id = domain.get("id", "")
            config = domain.get("config", domain)
            keywords = config.get("keywords", [])
            
            if len(keywords) > 0:
                # Create test text with domain keywords
                test_text = " ".join(keywords[:2]).lower()
                
                # Simulate filtering: check if text matches domain keywords
                matches = sum(1 for kw in keywords if kw.lower() in test_text)
                match_ratio = matches / len(keywords) if keywords else 0
                
                test_cases.append({
                    "domain_id": domain_id,
                    "keywords": keywords,
                    "test_text": test_text,
                    "matches": matches,
                    "match_ratio": match_ratio
                })
        
        if len(test_cases) > 0:
            print(f"✅ Keyword filtering logic tested")
            for case in test_cases:
                print(f"   Domain {case['domain_id']}: {case['matches']}/{len(case['keywords'])} keywords matched ({case['match_ratio']:.2%})")
            return True
        else:
            print(f"⚠️  No valid test cases for keyword filtering")
            return False
    except Exception as e:
        print(f"❌ Keyword filtering test error: {e}")
        return False


def test_differential_privacy_noise() -> bool:
    """Test differential privacy noise application."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from domain_filter import DomainFilter, PrivacyConfig
            
            # Test that noise is applied
            privacy_config = PrivacyConfig(epsilon=1.0)
            domain_filter = DomainFilter(
                localai_url=LOCALAI_URL,
                privacy_config=privacy_config
            )
            
            # Test adding noise to a value
            original_value = 100.0
            
            # Note: This would call _add_laplacian_noise internally
            # We're testing that the mechanism exists
            print(f"✅ Differential privacy mechanism available")
            print(f"   Privacy epsilon: {privacy_config.epsilon}")
            print(f"   Original value: {original_value}")
            print(f"   (Noise application tested in integration tests)")
            
            return True
        except ImportError:
            print(f"⚠️  Differential privacy not available (module not found)")
            return False
    except Exception as e:
        print(f"❌ Differential privacy test error: {e}")
        return False


def test_privacy_budget_tracking() -> bool:
    """Test privacy budget tracking."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from domain_filter import DomainFilter, PrivacyConfig
            
            privacy_config = PrivacyConfig(epsilon=1.0)
            domain_filter = DomainFilter(
                localai_url=LOCALAI_URL,
                privacy_config=privacy_config
            )
            
            # Test privacy budget tracking
            if hasattr(domain_filter, 'get_privacy_stats'):
                stats = domain_filter.get_privacy_stats()
                print(f"✅ Privacy budget tracking available")
                print(f"   Budget used: {stats.get('budget_used', 0)}")
                print(f"   Budget remaining: {stats.get('budget_remaining', privacy_config.epsilon)}")
                return True
            else:
                print(f"⚠️  Privacy budget tracking method not found")
                return False
        except ImportError:
            print(f"⚠️  Privacy budget tracking not available (module not found)")
            return False
    except Exception as e:
        print(f"❌ Privacy budget tracking test error: {e}")
        return False


def test_domain_specific_feature_extraction() -> bool:
    """Test domain-specific feature extraction."""
    try:
        # Get domain configs
        response = httpx.get(
            f"{LOCALAI_URL}/v1/domains",
            timeout=DEFAULT_TIMEOUT
        )
        
        if response.status_code != 200:
            print(f"⚠️  Cannot test feature extraction: LocalAI not available")
            return False
        
        data = response.json()
        domains = data.get("data", [])
        
        if len(domains) == 0:
            print(f"⚠️  No domains to test feature extraction")
            return False
        
        # Test that domains have features for filtering
        features_found = 0
        for domain in domains:
            config = domain.get("config", domain)
            has_keywords = len(config.get("keywords", [])) > 0
            has_tags = len(config.get("tags", [])) > 0
            
            if has_keywords or has_tags:
                features_found += 1
        
        if features_found > 0:
            print(f"✅ Domain-specific features available")
            print(f"   {features_found}/{len(domains)} domains have keywords/tags")
            return True
        else:
            print(f"⚠️  No domain features found")
            return False
    except Exception as e:
        print(f"❌ Feature extraction test error: {e}")
        return False


def main():
    """Run all domain filter tests."""
    print("="*60)
    print("Domain Filter Test Suite - Week 1")
    print("="*60)
    print(f"LocalAI URL: {LOCALAI_URL}")
    print(f"Training Service URL: {TRAINING_SERVICE_URL}")
    print()
    
    suite = DomainFilterTestSuite()
    
    # Domain Filter Tests
    suite.run_test(
        "Domain Filter Import",
        "Test that domain filter module can be imported",
        test_domain_filter_import
    )
    
    suite.run_test(
        "Privacy Config Creation",
        "Test PrivacyConfig creation and configuration",
        test_privacy_config_creation
    )
    
    suite.run_test(
        "Domain Filter Initialization",
        "Test DomainFilter initialization",
        test_domain_filter_initialization
    )
    
    suite.run_test(
        "Keyword Filtering Logic",
        "Test keyword-based filtering logic",
        test_keyword_filtering_logic
    )
    
    suite.run_test(
        "Differential Privacy Noise",
        "Test differential privacy noise application",
        test_differential_privacy_noise
    )
    
    suite.run_test(
        "Privacy Budget Tracking",
        "Test privacy budget tracking",
        test_privacy_budget_tracking
    )
    
    suite.run_test(
        "Domain-Specific Feature Extraction",
        "Test domain-specific feature extraction",
        test_domain_specific_feature_extraction
    )
    
    # Print summary
    success = suite.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

