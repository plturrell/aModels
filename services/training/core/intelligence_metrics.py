"""ARC-AGI style intelligence metrics for relational/ETL domain.

This module provides metrics to measure the model's level of intelligence
on domain-specific tasks, including pattern recognition, generalization,
and compositional reasoning.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


class DomainIntelligenceEvaluator:
    """Evaluates domain-specific intelligence using ARC-AGI style metrics."""
    
    def __init__(self):
        self.test_results = {}
        self.intelligence_levels = {
            "level_1": "Pattern Recognition",      # Can recognize basic patterns
            "level_2": "Pattern Generalization", # Can generalize to unseen patterns
            "level_3": "Compositional Reasoning", # Can compose patterns to solve new problems
            "level_4": "Transfer Learning",      # Can transfer knowledge across domains
            "level_5": "Abstract Reasoning",      # Can reason about abstract concepts
        }
    
    def evaluate_domain_intelligence(
        self,
        model_predictions: Dict[str, Any],
        test_cases: List[Dict[str, Any]],
        training_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate model intelligence on domain-specific tasks.
        
        Args:
            model_predictions: Model predictions for test cases
            test_cases: List of test cases with expected patterns
            training_context: Optional training context for learning rate calculation
        
        Returns:
            Dictionary with intelligence metrics:
            - intelligence_level: Overall intelligence level (1-5)
            - pattern_recognition_score: Level 1 score
            - generalization_score: Level 2 score
            - compositional_score: Level 3 score
            - transfer_score: Level 4 score
            - abstract_reasoning_score: Level 5 score
            - learning_rate: How quickly model learns new patterns
            - domain_expertise: Domain-specific expertise score
        """
        logger.info("Evaluating domain intelligence using ARC-AGI style metrics")
        
        intelligence = {
            "evaluated_at": datetime.now().isoformat(),
            "intelligence_level": 0,
            "pattern_recognition_score": 0.0,
            "generalization_score": 0.0,
            "compositional_score": 0.0,
            "transfer_score": 0.0,
            "abstract_reasoning_score": 0.0,
            "learning_rate": None,
            "domain_expertise": 0.0,
            "test_results": {},
        }
        
        # Level 1: Pattern Recognition
        pattern_recognition = self._evaluate_pattern_recognition(model_predictions, test_cases)
        intelligence["pattern_recognition_score"] = pattern_recognition["score"]
        intelligence["test_results"]["pattern_recognition"] = pattern_recognition
        
        # Level 2: Pattern Generalization
        generalization = self._evaluate_generalization(model_predictions, test_cases)
        intelligence["generalization_score"] = generalization["score"]
        intelligence["test_results"]["generalization"] = generalization
        
        # Level 3: Compositional Reasoning
        compositional = self._evaluate_compositional_reasoning(model_predictions, test_cases)
        intelligence["compositional_score"] = compositional["score"]
        intelligence["test_results"]["compositional_reasoning"] = compositional
        
        # Level 4: Transfer Learning
        transfer = self._evaluate_transfer_learning(model_predictions, test_cases, training_context)
        intelligence["transfer_score"] = transfer["score"]
        intelligence["test_results"]["transfer_learning"] = transfer
        
        # Level 5: Abstract Reasoning
        abstract = self._evaluate_abstract_reasoning(model_predictions, test_cases)
        intelligence["abstract_reasoning_score"] = abstract["score"]
        intelligence["test_results"]["abstract_reasoning"] = abstract
        
        # Calculate overall intelligence level
        scores = [
            intelligence["pattern_recognition_score"],
            intelligence["generalization_score"],
            intelligence["compositional_score"],
            intelligence["transfer_score"],
            intelligence["abstract_reasoning_score"],
        ]
        
        # Intelligence level is the highest level where score >= 0.7
        for level in range(5, 0, -1):
            if scores[level - 1] >= 0.7:
                intelligence["intelligence_level"] = level
                break
        
        # Calculate domain expertise (weighted average)
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Higher weights for lower levels
        intelligence["domain_expertise"] = sum(s * w for s, w in zip(scores, weights))
        
        # Calculate learning rate if training context available
        if training_context:
            learning_rate = self._calculate_learning_rate(training_context)
            intelligence["learning_rate"] = learning_rate
        
        logger.info(
            f"Intelligence evaluation completed: Level {intelligence['intelligence_level']}, "
            f"Expertise: {intelligence['domain_expertise']:.2f}"
        )
        
        return intelligence
    
    def _evaluate_pattern_recognition(
        self,
        predictions: Dict[str, Any],
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Level 1: Evaluate pattern recognition capability.
        
        Tests if model can recognize basic patterns:
        - Column type patterns
        - Relationship patterns
        - Naming conventions
        """
        correct = 0
        total = 0
        results = []
        
        for test_case in test_cases:
            if test_case.get("type") != "pattern_recognition":
                continue
            
            expected_pattern = test_case.get("expected_pattern")
            predicted_pattern = predictions.get(test_case.get("id", ""))
            
            if expected_pattern and predicted_pattern:
                total += 1
                
                # Check if pattern matches
                if self._patterns_match(expected_pattern, predicted_pattern):
                    correct += 1
                    results.append({"test_id": test_case.get("id"), "correct": True})
                else:
                    results.append({"test_id": test_case.get("id"), "correct": False})
        
        score = correct / total if total > 0 else 0.0
        
        return {
            "score": score,
            "correct": correct,
            "total": total,
            "results": results,
            "level": 1,
            "description": "Pattern Recognition - Can recognize basic patterns in data"
        }
    
    def _evaluate_generalization(
        self,
        predictions: Dict[str, Any],
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Level 2: Evaluate pattern generalization capability.
        
        Tests if model can generalize to unseen patterns:
        - Unseen column type combinations
        - Unseen relationship structures
        - Similar but different schemas
        """
        correct = 0
        total = 0
        results = []
        
        for test_case in test_cases:
            if test_case.get("type") != "generalization":
                continue
            
            expected_generalization = test_case.get("expected_generalization")
            predicted_generalization = predictions.get(test_case.get("id", ""))
            
            if expected_generalization and predicted_generalization:
                total += 1
                
                # Check if generalization is correct (similarity-based)
                similarity = self._calculate_generalization_similarity(
                    expected_generalization,
                    predicted_generalization
                )
                
                if similarity >= 0.7:  # 70% similarity threshold
                    correct += 1
                    results.append({
                        "test_id": test_case.get("id"),
                        "correct": True,
                        "similarity": similarity
                    })
                else:
                    results.append({
                        "test_id": test_case.get("id"),
                        "correct": False,
                        "similarity": similarity
                    })
        
        score = correct / total if total > 0 else 0.0
        
        return {
            "score": score,
            "correct": correct,
            "total": total,
            "results": results,
            "level": 2,
            "description": "Pattern Generalization - Can generalize to unseen patterns"
        }
    
    def _evaluate_compositional_reasoning(
        self,
        predictions: Dict[str, Any],
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Level 3: Evaluate compositional reasoning capability.
        
        Tests if model can compose patterns to solve new problems:
        - Combining multiple patterns
        - Reasoning about relationships
        - Inferring missing information
        """
        correct = 0
        total = 0
        results = []
        
        for test_case in test_cases:
            if test_case.get("type") != "compositional":
                continue
            
            expected_composition = test_case.get("expected_composition")
            predicted_composition = predictions.get(test_case.get("id", ""))
            
            if expected_composition and predicted_composition:
                total += 1
                
                # Check if composition is correct
                composition_score = self._evaluate_composition(
                    expected_composition,
                    predicted_composition
                )
                
                if composition_score >= 0.7:
                    correct += 1
                    results.append({
                        "test_id": test_case.get("id"),
                        "correct": True,
                        "composition_score": composition_score
                    })
                else:
                    results.append({
                        "test_id": test_case.get("id"),
                        "correct": False,
                        "composition_score": composition_score
                    })
        
        score = correct / total if total > 0 else 0.0
        
        return {
            "score": score,
            "correct": correct,
            "total": total,
            "results": results,
            "level": 3,
            "description": "Compositional Reasoning - Can compose patterns to solve new problems"
        }
    
    def _evaluate_transfer_learning(
        self,
        predictions: Dict[str, Any],
        test_cases: List[Dict[str, Any]],
        training_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Level 4: Evaluate transfer learning capability.
        
        Tests if model can transfer knowledge across domains:
        - Adapting patterns from one domain to another
        - Learning from limited examples
        - Cross-domain pattern recognition
        """
        correct = 0
        total = 0
        results = []
        
        for test_case in test_cases:
            if test_case.get("type") != "transfer":
                continue
            
            source_domain = test_case.get("source_domain")
            target_domain = test_case.get("target_domain")
            expected_transfer = test_case.get("expected_transfer")
            predicted_transfer = predictions.get(test_case.get("id", ""))
            
            if expected_transfer and predicted_transfer:
                total += 1
                
                # Check if transfer is successful
                transfer_score = self._evaluate_transfer(
                    source_domain,
                    target_domain,
                    expected_transfer,
                    predicted_transfer
                )
                
                if transfer_score >= 0.7:
                    correct += 1
                    results.append({
                        "test_id": test_case.get("id"),
                        "correct": True,
                        "transfer_score": transfer_score
                    })
                else:
                    results.append({
                        "test_id": test_case.get("id"),
                        "correct": False,
                        "transfer_score": transfer_score
                    })
        
        score = correct / total if total > 0 else 0.0
        
        return {
            "score": score,
            "correct": correct,
            "total": total,
            "results": results,
            "level": 4,
            "description": "Transfer Learning - Can transfer knowledge across domains"
        }
    
    def _evaluate_abstract_reasoning(
        self,
        predictions: Dict[str, Any],
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Level 5: Evaluate abstract reasoning capability.
        
        Tests if model can reason about abstract concepts:
        - Abstract patterns beyond concrete examples
        - Meta-patterns (patterns of patterns)
        - Theoretical reasoning about data structures
        """
        correct = 0
        total = 0
        results = []
        
        for test_case in test_cases:
            if test_case.get("type") != "abstract":
                continue
            
            expected_abstract = test_case.get("expected_abstract")
            predicted_abstract = predictions.get(test_case.get("id", ""))
            
            if expected_abstract and predicted_abstract:
                total += 1
                
                # Check if abstract reasoning is correct
                abstract_score = self._evaluate_abstract_reasoning_task(
                    expected_abstract,
                    predicted_abstract
                )
                
                if abstract_score >= 0.7:
                    correct += 1
                    results.append({
                        "test_id": test_case.get("id"),
                        "correct": True,
                        "abstract_score": abstract_score
                    })
                else:
                    results.append({
                        "test_id": test_case.get("id"),
                        "correct": False,
                        "abstract_score": abstract_score
                    })
        
        score = correct / total if total > 0 else 0.0
        
        return {
            "score": score,
            "correct": correct,
            "total": total,
            "results": results,
            "level": 5,
            "description": "Abstract Reasoning - Can reason about abstract concepts"
        }
    
    def _calculate_learning_rate(
        self,
        training_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate learning rate - how quickly model learns new patterns.
        
        Args:
            training_context: Training context with learning history
        
        Returns:
            Dictionary with learning rate metrics:
            - patterns_per_epoch: Average patterns learned per training epoch
            - convergence_rate: How quickly model converges
            - adaptation_rate: How quickly model adapts to new data
        """
        learned_patterns_history = training_context.get("learned_patterns_history", [])
        if not learned_patterns_history:
            return {
                "patterns_per_epoch": 0.0,
                "convergence_rate": 0.0,
                "adaptation_rate": 0.0,
                "insufficient_data": True
            }
        
        # Calculate patterns learned per epoch
        if len(learned_patterns_history) >= 2:
            pattern_counts = [
                len(epoch.get("patterns", []))
                for epoch in learned_patterns_history
            ]
            patterns_per_epoch = statistics.mean(pattern_counts) if pattern_counts else 0.0
            
            # Calculate convergence rate (how quickly pattern count stabilizes)
            if len(pattern_counts) >= 3:
                recent_avg = statistics.mean(pattern_counts[-3:])
                early_avg = statistics.mean(pattern_counts[:3])
                convergence_rate = 1.0 - abs(recent_avg - early_avg) / max(recent_avg, early_avg, 1)
            else:
                convergence_rate = 0.0
            
            # Calculate adaptation rate (how quickly model adapts to new patterns)
            if len(pattern_counts) >= 2:
                adaptation_scores = []
                for i in range(1, len(pattern_counts)):
                    if pattern_counts[i-1] > 0:
                        adaptation = pattern_counts[i] / pattern_counts[i-1]
                        adaptation_scores.append(min(adaptation, 2.0))  # Cap at 2x
                adaptation_rate = statistics.mean(adaptation_scores) if adaptation_scores else 0.0
            else:
                adaptation_rate = 0.0
        else:
            patterns_per_epoch = 0.0
            convergence_rate = 0.0
            adaptation_rate = 0.0
        
        return {
            "patterns_per_epoch": patterns_per_epoch,
            "convergence_rate": convergence_rate,
            "adaptation_rate": adaptation_rate,
            "total_epochs": len(learned_patterns_history),
            "insufficient_data": False
        }
    
    def _patterns_match(self, expected: Dict[str, Any], predicted: Dict[str, Any]) -> bool:
        """Check if predicted pattern matches expected pattern."""
        # Simple equality check - can be enhanced with fuzzy matching
        return expected == predicted
    
    def _calculate_generalization_similarity(
        self,
        expected: Dict[str, Any],
        predicted: Dict[str, Any]
    ) -> float:
        """Calculate similarity between expected and predicted generalizations."""
        # Simple similarity calculation - can be enhanced
        if expected == predicted:
            return 1.0
        
        # Calculate key overlap
        expected_keys = set(expected.keys()) if isinstance(expected, dict) else set()
        predicted_keys = set(predicted.keys()) if isinstance(predicted, dict) else set()
        
        if not expected_keys:
            return 0.0
        
        overlap = len(expected_keys & predicted_keys)
        total = len(expected_keys | predicted_keys)
        
        return overlap / total if total > 0 else 0.0
    
    def _evaluate_composition(
        self,
        expected: Dict[str, Any],
        predicted: Dict[str, Any]
    ) -> float:
        """Evaluate composition correctness."""
        # Similar to generalization but with composition-specific logic
        return self._calculate_generalization_similarity(expected, predicted)
    
    def _evaluate_transfer(
        self,
        source_domain: str,
        target_domain: str,
        expected: Dict[str, Any],
        predicted: Dict[str, Any]
    ) -> float:
        """Evaluate transfer learning correctness."""
        # Transfer-specific evaluation
        base_similarity = self._calculate_generalization_similarity(expected, predicted)
        
        # Bonus for domain transfer
        if source_domain != target_domain:
            # Transfer bonus if similarity is high
            return min(base_similarity * 1.1, 1.0)
        
        return base_similarity
    
    def _evaluate_abstract_reasoning_task(
        self,
        expected: Dict[str, Any],
        predicted: Dict[str, Any]
    ) -> float:
        """Evaluate abstract reasoning task correctness."""
        # Abstract reasoning is more lenient - looks for conceptual correctness
        return self._calculate_generalization_similarity(expected, predicted)
    
    def create_domain_test_cases(
        self,
        knowledge_graph: Dict[str, Any],
        learned_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create domain-specific test cases for intelligence evaluation.
        
        Args:
            knowledge_graph: Current knowledge graph
            learned_patterns: Learned patterns from training
        
        Returns:
            List of test cases for each intelligence level
        """
        test_cases = []
        
        # Level 1: Pattern Recognition tests
        nodes = knowledge_graph.get("nodes", [])
        for node in nodes[:10]:  # Sample of nodes
            if node.get("type") == "Column":
                test_cases.append({
                    "id": f"pattern_recognition_{node.get('id')}",
                    "type": "pattern_recognition",
                    "input": {"column": node},
                    "expected_pattern": {
                        "column_type": node.get("properties", {}).get("type"),
                        "nullable": node.get("properties", {}).get("nullable"),
                    }
                })
        
        # Level 2: Generalization tests
        column_patterns = learned_patterns.get("column_patterns", {})
        if column_patterns:
            test_cases.append({
                "id": "generalization_column_types",
                "type": "generalization",
                "input": {"unseen_column_types": ["new_type_1", "new_type_2"]},
                "expected_generalization": {
                    "can_generalize": True,
                    "similarity_to_known": 0.8
                }
            })
        
        # Level 3: Compositional tests
        edges = knowledge_graph.get("edges", [])
        if len(edges) >= 2:
            test_cases.append({
                "id": "compositional_relationship",
                "type": "compositional",
                "input": {
                    "table1": nodes[0] if nodes else {},
                    "table2": nodes[1] if len(nodes) > 1 else {},
                },
                "expected_composition": {
                    "can_compose_relationships": True
                }
            })
        
        # Level 4: Transfer tests
        test_cases.append({
            "id": "transfer_cross_domain",
            "type": "transfer",
            "source_domain": "sgmi",
            "target_domain": "new_domain",
            "input": {"source_pattern": column_patterns},
            "expected_transfer": {
                "can_transfer": True,
                "adaptation_score": 0.7
            }
        })
        
        # Level 5: Abstract reasoning tests
        test_cases.append({
            "id": "abstract_meta_pattern",
            "type": "abstract",
            "input": {"meta_pattern_query": "What patterns emerge across all schemas?"},
            "expected_abstract": {
                "can_reason_abstractly": True,
                "meta_pattern_insight": "High-level pattern insight"
            }
        })
        
        return test_cases

