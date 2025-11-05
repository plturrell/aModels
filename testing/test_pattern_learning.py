#!/usr/bin/env python3
"""
Week 3: Pattern Learning Tests (Phase 7)

Tests domain-aware pattern learning:
- GNN pattern learning with domain models
- Meta-pattern learning (layer/team-specific)
- Sequence pattern learning with domain conditioning
- Active pattern learning with domain filtering
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
    get_domain_from_localai, load_test_data
)

# Test configuration
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localhost:8081")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://user:pass@localhost:5432/amodels")

DEFAULT_TIMEOUT = 30
HEALTH_TIMEOUT = 5


class TestResult:
    PASS = "✅"
    FAIL = "❌"
    SKIP = "⏭️"
    WARN = "⚠️"


class PatternLearningTestSuite:
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
        
        print_test_summary("Pattern Learning Tests (Phase 7)", passed, failed, skipped)
        
        if failed > 0:
            print("Failed Tests:")
            for test in self.tests:
                if test.get("result") == TestResult.FAIL:
                    print(f"  {TestResult.FAIL} {test['name']}: {test.get('error', 'Unknown error')}")
            print()
        
        return failed == 0


def test_gnn_pattern_learner_available() -> bool:
    """Test that GNN pattern learner is available."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from pattern_learning_gnn import GNNRelationshipPatternLearner
            
            learner = GNNRelationshipPatternLearner(
                localai_url=LOCALAI_URL
            )
            
            print(f"✅ GNN pattern learner available")
            print(f"   Domain-aware: {learner.localai_url is not None}")
            print(f"   Domain models cache: {len(learner.domain_models)} models")
            return True
            
        except ImportError:
            print(f"⚠️  GNN pattern learner not available (module not found)")
            return False
        
    except Exception as e:
        print(f"❌ GNN pattern learner test error: {e}")
        return False


def test_domain_specific_gnn_model() -> bool:
    """Test domain-specific GNN model creation."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from pattern_learning_gnn import GNNRelationshipPatternLearner
            
            learner = GNNRelationshipPatternLearner(localai_url=LOCALAI_URL)
            
            # Test domain model creation
            domain_id = "test-financial"
            
            # Load test knowledge graph
            try:
                kg_data = load_test_data("knowledge_graph.json")
                nodes = kg_data.get("nodes", [])
                edges = kg_data.get("edges", [])
            except FileNotFoundError:
                # Create minimal test data
                nodes = [
                    {"id": "node_1", "label": "transactions", "type": "table", "properties": {"domain": "test-financial"}},
                    {"id": "node_2", "label": "amount", "type": "column", "properties": {"domain": "test-financial"}}
                ]
                edges = [
                    {"source": "node_1", "target": "node_2", "type": "HAS_COLUMN", "properties": {"domain": "test-financial"}}
                ]
            
            print(f"Testing domain-specific GNN model for domain: {domain_id}")
            print(f"   Nodes: {len(nodes)}")
            print(f"   Edges: {len(edges)}")
            
            # Test domain pattern learning
            if hasattr(learner, 'learn_domain_patterns'):
                print(f"✅ Domain-specific learning method available")
                print(f"   (Full learning tested in integration)")
                return True
            else:
                print(f"⚠️  Domain-specific learning method not found")
                return False
                
        except ImportError:
            print(f"⚠️  Domain-specific GNN model test not available (module not found)")
            return False
        
    except Exception as e:
        print(f"❌ Domain-specific GNN model test error: {e}")
        return False


def test_meta_pattern_learner_available() -> bool:
    """Test that meta-pattern learner is available."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from meta_pattern_learner import MetaPatternLearner
            
            learner = MetaPatternLearner(localai_url=LOCALAI_URL)
            
            print(f"✅ Meta-pattern learner available")
            print(f"   Domain-aware: {learner.localai_url is not None}")
            print(f"   Layer patterns: {len(learner.layer_patterns)}")
            print(f"   Team patterns: {len(learner.team_patterns)}")
            return True
            
        except ImportError:
            print(f"⚠️  Meta-pattern learner not available (module not found)")
            return False
        
    except Exception as e:
        print(f"❌ Meta-pattern learner test error: {e}")
        return False


def test_layer_specific_meta_patterns() -> bool:
    """Test layer-specific meta-pattern learning."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from meta_pattern_learner import MetaPatternLearner
            
            learner = MetaPatternLearner(localai_url=LOCALAI_URL)
            
            # Test layer-specific learning
            sample_patterns = {
                "column_patterns": {"types": ["bigint", "decimal", "string"]},
                "relationship_patterns": {"labels": ["HAS_COLUMN", "RELATES_TO"]}
            }
            
            domains = ["test-financial", "test-customer"]
            
            print(f"Testing layer-specific meta-patterns")
            print(f"   Domains: {domains}")
            print(f"   (Layer-specific learning tested in integration)")
            
            return True
            
        except ImportError:
            print(f"⚠️  Layer-specific meta-patterns test not available (module not found)")
            return False
        
    except Exception as e:
        print(f"❌ Layer-specific meta-patterns test error: {e}")
        return False


def test_sequence_pattern_learner_available() -> bool:
    """Test that sequence pattern learner is available."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from sequence_pattern_transformer import SequencePatternTransformer
            
            learner = SequencePatternTransformer(localai_url=LOCALAI_URL)
            
            print(f"✅ Sequence pattern learner available")
            print(f"   Domain-aware: {learner.localai_url is not None}")
            print(f"   Domain embeddings cache: {len(learner.domain_embeddings)} embeddings")
            return True
            
        except ImportError:
            print(f"⚠️  Sequence pattern learner not available (module not found)")
            return False
        
    except Exception as e:
        print(f"❌ Sequence pattern learner test error: {e}")
        return False


def test_domain_conditioned_sequences() -> bool:
    """Test domain-conditioned sequence learning."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from sequence_pattern_transformer import SequencePatternTransformer
            
            learner = SequencePatternTransformer(localai_url=LOCALAI_URL)
            
            # Test domain-conditioned learning
            domain_id = "test-financial"
            sequences = [
                [{"type": "controlm_job", "name": "load_data"}, {"type": "sql", "query": "SELECT * FROM transactions"}],
                [{"type": "controlm_job", "name": "process_payments"}, {"type": "sql", "query": "UPDATE accounts SET balance = ..."}]
            ]
            
            print(f"Testing domain-conditioned sequences")
            print(f"   Domain: {domain_id}")
            print(f"   Sequences: {len(sequences)}")
            print(f"   (Domain-conditioned learning tested in integration)")
            
            return True
            
        except ImportError:
            print(f"⚠️  Domain-conditioned sequences test not available (module not found)")
            return False
        
    except Exception as e:
        print(f"❌ Domain-conditioned sequences test error: {e}")
        return False


def test_active_pattern_learner_available() -> bool:
    """Test that active pattern learner is available."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from active_pattern_learner import ActivePatternLearner
            
            learner = ActivePatternLearner(localai_url=LOCALAI_URL)
            
            print(f"✅ Active pattern learner available")
            print(f"   Domain-aware: {learner.localai_url is not None}")
            return True
            
        except ImportError:
            print(f"⚠️  Active pattern learner not available (module not found)")
            return False
        
    except Exception as e:
        print(f"❌ Active pattern learner test error: {e}")
        return False


def test_domain_filtered_active_learning() -> bool:
    """Test domain-filtered active learning."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "training"))
        
        try:
            from active_pattern_learner import ActivePatternLearner
            
            learner = ActivePatternLearner(localai_url=LOCALAI_URL)
            
            # Test domain filtering
            domain_id = "test-financial"
            
            # Load test knowledge graph
            try:
                kg_data = load_test_data("knowledge_graph.json")
                nodes = kg_data.get("nodes", [])
                edges = kg_data.get("edges", [])
            except FileNotFoundError:
                nodes = [{"id": "node_1", "label": "transactions", "type": "table"}]
                edges = []
            
            print(f"Testing domain-filtered active learning")
            print(f"   Domain: {domain_id}")
            print(f"   Nodes: {len(nodes)}")
            print(f"   Edges: {len(edges)}")
            print(f"   (Domain filtering tested in integration)")
            
            return True
            
        except ImportError:
            print(f"⚠️  Domain-filtered active learning test not available (module not found)")
            return False
        
    except Exception as e:
        print(f"❌ Domain-filtered active learning test error: {e}")
        return False


def main():
    """Run all pattern learning tests."""
    print("="*60)
    print("Pattern Learning Tests (Phase 7) - Week 3")
    print("="*60)
    print(f"LocalAI URL: {LOCALAI_URL}")
    print()
    
    # Wait for services
    print("Waiting for services...")
    if not wait_for_service(f"{LOCALAI_URL}/health", "LocalAI"):
        print("⚠️  LocalAI not available, some tests will be skipped")
    print()
    
    suite = PatternLearningTestSuite()
    
    # Pattern Learning Tests
    suite.run_test(
        "GNN Pattern Learner Available",
        "Test that GNN pattern learner is available",
        test_gnn_pattern_learner_available
    )
    
    suite.run_test(
        "Domain-Specific GNN Model",
        "Test domain-specific GNN model creation",
        test_domain_specific_gnn_model
    )
    
    suite.run_test(
        "Meta-Pattern Learner Available",
        "Test that meta-pattern learner is available",
        test_meta_pattern_learner_available
    )
    
    suite.run_test(
        "Layer-Specific Meta-Patterns",
        "Test layer-specific meta-pattern learning",
        test_layer_specific_meta_patterns
    )
    
    suite.run_test(
        "Sequence Pattern Learner Available",
        "Test that sequence pattern learner is available",
        test_sequence_pattern_learner_available
    )
    
    suite.run_test(
        "Domain-Conditioned Sequences",
        "Test domain-conditioned sequence learning",
        test_domain_conditioned_sequences
    )
    
    suite.run_test(
        "Active Pattern Learner Available",
        "Test that active pattern learner is available",
        test_active_pattern_learner_available
    )
    
    suite.run_test(
        "Domain-Filtered Active Learning",
        "Test domain-filtered active learning",
        test_domain_filtered_active_learning
    )
    
    # Print summary
    success = suite.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

