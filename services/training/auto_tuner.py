"""Automated training optimization with hyperparameter tuning and model architecture selection.

This module implements hyperparameter auto-tuning using Optuna, automated model architecture
selection, training data quality auto-assessment, and automated feature engineering.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None

logger = logging.getLogger(__name__)


class AutoTuner:
    """Automated hyperparameter tuning and model optimization."""
    
    def __init__(
        self,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        n_trials: int = 50
    ):
        """Initialize auto-tuner.
        
        Args:
            study_name: Name for Optuna study
            storage: Storage backend for Optuna (e.g., "sqlite:///optuna.db")
            n_trials: Number of optimization trials
        """
        self.study_name = study_name or "amodels_training_optimization"
        self.storage = storage or os.getenv("OPTUNA_STORAGE", "sqlite:///optuna.db")
        self.n_trials = n_trials
        self.study = None
        
        if HAS_OPTUNA:
            try:
                self.study = optuna.create_study(
                    study_name=self.study_name,
                    storage=self.storage,
                    direction="maximize",  # Maximize validation score
                    load_if_exists=True
                )
                logger.info(f"Created Optuna study: {self.study_name}")
            except Exception as e:
                logger.warning(f"Failed to create Optuna study: {e}")
                self.study = None
        else:
            logger.warning("Optuna not available. Auto-tuning will be limited.")
    
    def optimize_hyperparameters(
        self,
        objective_func,
        search_space: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna.
        
        Args:
            objective_func: Function that takes a trial and returns a score
            search_space: Optional predefined search space
        
        Returns:
            Dictionary with best hyperparameters and score
        """
        if not HAS_OPTUNA or self.study is None:
            logger.warning("Optuna not available, using default hyperparameters")
            return self._get_default_hyperparameters()
        
        try:
            # Run optimization
            self.study.optimize(objective_func, n_trials=self.n_trials)
            
            # Get best parameters
            best_params = self.study.best_params
            best_score = self.study.best_value
            
            logger.info(f"Hyperparameter optimization complete: best_score={best_score:.4f}")
            
            return {
                "best_hyperparameters": best_params,
                "best_score": best_score,
                "n_trials": self.n_trials,
                "optimization_complete": True,
            }
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}", exc_info=True)
            return self._get_default_hyperparameters()
    
    def suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Dictionary with suggested hyperparameters
        """
        if not HAS_OPTUNA:
            return self._get_default_hyperparameters()
        
        hyperparameters = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512, 768]),
            "num_layers": trial.suggest_int("num_layers", 2, 8),
            "num_heads": trial.suggest_categorical("num_heads", [4, 8, 16]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "warmup_steps": trial.suggest_int("warmup_steps", 100, 1000),
        }
        
        return hyperparameters
    
    def _get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "hidden_dim": 256,
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.1,
            "weight_decay": 1e-4,
            "warmup_steps": 500,
            "optimization_complete": False,
        }
    
    def select_model_architecture(
        self,
        training_data_stats: Dict[str, Any],
        task_type: str = "pattern_learning"
    ) -> Dict[str, Any]:
        """Automatically select optimal model architecture.
        
        Args:
            training_data_stats: Statistics about training data
            task_type: Type of task (pattern_learning, classification, regression, etc.)
        
        Returns:
            Dictionary with recommended architecture
        """
        # Analyze data characteristics
        data_size = training_data_stats.get("total_samples", 0)
        feature_dim = training_data_stats.get("feature_dim", 0)
        num_classes = training_data_stats.get("num_classes", 0)
        
        # Architecture selection logic
        if data_size < 1000:
            # Small dataset: use simpler architecture
            architecture = {
                "hidden_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "recommendation": "small_dataset",
            }
        elif data_size < 10000:
            # Medium dataset: moderate architecture
            architecture = {
                "hidden_dim": 256,
                "num_layers": 4,
                "num_heads": 8,
                "recommendation": "medium_dataset",
            }
        else:
            # Large dataset: larger architecture
            architecture = {
                "hidden_dim": 512,
                "num_layers": 6,
                "num_heads": 16,
                "recommendation": "large_dataset",
            }
        
        # Adjust based on task type
        if task_type == "classification" and num_classes > 100:
            architecture["hidden_dim"] = max(architecture["hidden_dim"], 512)
        
        architecture["task_type"] = task_type
        architecture["data_size"] = data_size
        
        logger.info(f"Selected architecture: {architecture['recommendation']} for {task_type}")
        
        return architecture
    
    def assess_training_data_quality(
        self,
        training_data: Dict[str, Any],
        quality_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Automatically assess training data quality.
        
        Args:
            training_data: Training data dictionary
            quality_thresholds: Optional custom quality thresholds
        
        Returns:
            Dictionary with quality assessment
        """
        if quality_thresholds is None:
            quality_thresholds = {
                "min_samples": 100,
                "min_features": 10,
                "min_class_balance": 0.1,
                "max_missing_ratio": 0.1,
            }
        
        assessment = {
            "quality_score": 0.0,
            "issues": [],
            "recommendations": [],
            "passed": True,
        }
        
        # Check data size
        num_samples = training_data.get("num_samples", 0)
        if num_samples < quality_thresholds["min_samples"]:
            assessment["issues"].append(f"Insufficient samples: {num_samples} < {quality_thresholds['min_samples']}")
            assessment["recommendations"].append("Collect more training data")
            assessment["passed"] = False
        
        # Check feature count
        num_features = training_data.get("num_features", 0)
        if num_features < quality_thresholds["min_features"]:
            assessment["issues"].append(f"Insufficient features: {num_features} < {quality_thresholds['min_features']}")
            assessment["recommendations"].append("Extract more features")
            assessment["passed"] = False
        
        # Check class balance (for classification)
        if "class_distribution" in training_data:
            class_dist = training_data["class_distribution"]
            min_class_ratio = min(class_dist.values()) / sum(class_dist.values())
            if min_class_ratio < quality_thresholds["min_class_balance"]:
                assessment["issues"].append(f"Class imbalance: min_ratio={min_class_ratio:.3f}")
                assessment["recommendations"].append("Apply class balancing techniques")
        
        # Check missing values
        missing_ratio = training_data.get("missing_ratio", 0.0)
        if missing_ratio > quality_thresholds["max_missing_ratio"]:
            assessment["issues"].append(f"High missing ratio: {missing_ratio:.3f}")
            assessment["recommendations"].append("Handle missing values")
            assessment["passed"] = False
        
        # Calculate quality score
        score = 1.0
        score -= max(0, 1.0 - num_samples / quality_thresholds["min_samples"]) * 0.3
        score -= max(0, 1.0 - num_features / quality_thresholds["min_features"]) * 0.2
        score -= min(1.0, missing_ratio / quality_thresholds["max_missing_ratio"]) * 0.3
        score -= max(0, quality_thresholds["min_class_balance"] - min_class_ratio) * 0.2 if "class_distribution" in training_data else 0
        
        assessment["quality_score"] = max(0.0, min(1.0, score))
        
        logger.info(f"Data quality assessment: score={assessment['quality_score']:.2f}, passed={assessment['passed']}")
        
        return assessment
    
    def suggest_feature_engineering(
        self,
        training_data: Dict[str, Any],
        learned_patterns: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Suggest feature engineering improvements.
        
        Args:
            training_data: Current training data
            learned_patterns: Optional learned patterns
        
        Returns:
            List of feature engineering suggestions
        """
        suggestions = []
        
        # Suggest feature engineering based on data characteristics
        num_features = training_data.get("num_features", 0)
        
        if num_features < 50:
            suggestions.append({
                "type": "add_interaction_features",
                "description": "Add interaction features between columns",
                "priority": "high",
            })
        
        if "missing_ratio" in training_data and training_data["missing_ratio"] > 0:
            suggestions.append({
                "type": "handle_missing_values",
                "description": "Impute or encode missing values",
                "priority": "high",
            })
        
        if learned_patterns:
            # Suggest features based on learned patterns
            if "column_patterns" in learned_patterns:
                suggestions.append({
                    "type": "add_pattern_features",
                    "description": "Add features based on learned column patterns",
                    "priority": "medium",
                })
            
            if "relationship_patterns" in learned_patterns:
                suggestions.append({
                    "type": "add_relationship_features",
                    "description": "Add features based on learned relationships",
                    "priority": "medium",
                })
        
        return suggestions
    
    def get_best_hyperparameters(self) -> Dict[str, Any]:
        """Get the best hyperparameters from previous optimization."""
        if not HAS_OPTUNA or self.study is None:
            return self._get_default_hyperparameters()
        
        try:
            if self.study.best_params:
                return {
                    "best_hyperparameters": self.study.best_params,
                    "best_score": self.study.best_value,
                    "n_trials": len(self.study.trials),
                }
        except Exception as e:
            logger.warning(f"Failed to get best hyperparameters: {e}")
        
        return self._get_default_hyperparameters()
    
    def export_optimization_results(self, output_path: str) -> bool:
        """Export optimization results to file.
        
        Args:
            output_path: Path to export results
        
        Returns:
            True if successful
        """
        try:
            results = {
                "study_name": self.study_name,
                "best_hyperparameters": self.get_best_hyperparameters(),
                "exported_at": datetime.now().isoformat(),
            }
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Exported optimization results to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export optimization results: {e}")
            return False

