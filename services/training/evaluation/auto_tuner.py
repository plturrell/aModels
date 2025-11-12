"""Automated training optimization with hyperparameter tuning and model architecture selection.

This module implements hyperparameter auto-tuning using Optuna, automated model architecture
selection, training data quality auto-assessment, and automated feature engineering.

Domain-aware enhancements:
- Domain-specific Optuna studies per domain
- Domain-aware hyperparameter constraints
- Domain-specific architecture selection
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import httpx

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None

logger = logging.getLogger(__name__)


class AutoTuner:
    """Automated hyperparameter tuning and model optimization.
    
    Domain-aware enhancements:
    - Domain-specific Optuna studies
    - Domain-aware hyperparameter constraints
    - Domain-specific architecture selection
    """
    
    def __init__(
        self,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        n_trials: int = 50,
        localai_url: Optional[str] = None
    ):
        """Initialize auto-tuner.
        
        Args:
            study_name: Name for Optuna study
            storage: Storage backend for Optuna (e.g., "sqlite:///optuna.db")
            n_trials: Number of optimization trials
            localai_url: LocalAI URL for domain config fetching (optional)
        """
        self.study_name = study_name or "amodels_training_optimization"
        self.storage = storage or os.getenv("OPTUNA_STORAGE", "sqlite:///optuna.db")
        self.n_trials = n_trials
        self.study = None
        
        # Domain awareness
        self.localai_url = localai_url or os.getenv("LOCALAI_URL", "http://localai:8080")
        self.domain_configs = {}  # domain_id -> domain config
        self.domain_studies = {}  # domain_id -> Optuna study
        
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
    
    def _load_domain_config(self, domain_id: str) -> Optional[Dict[str, Any]]:
        """Load domain configuration from LocalAI.
        
        Args:
            domain_id: Domain identifier
        
        Returns:
            Domain configuration or None if not found
        """
        if domain_id in self.domain_configs:
            return self.domain_configs[domain_id]
        
        try:
            response = httpx.get(
                f"{self.localai_url}/v1/domains",
                timeout=5.0
            )
            if response.status_code == 200:
                domains_data = response.json()
                domains = domains_data.get("domains", {})
                
                if domain_id in domains:
                    domain_info = domains[domain_id]
                    config = domain_info.get("config", domain_info)
                    self.domain_configs[domain_id] = config
                    return config
        except Exception as e:
            logger.warning(f"Failed to load domain config for {domain_id}: {e}")
        
        return None
    
    def optimize_for_domain(
        self,
        domain_id: str,
        objective_func,
        training_data_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific domain.
        
        Args:
            domain_id: Domain identifier
            objective_func: Function that takes a trial and returns a score
            training_data_stats: Statistics about training data
        
        Returns:
            Dictionary with best hyperparameters and score for the domain
        """
        logger.info(f"Optimizing hyperparameters for domain: {domain_id}")
        
        # Get or create domain-specific study
        if domain_id not in self.domain_studies:
            if HAS_OPTUNA:
                try:
                    study_name = f"{self.study_name}_{domain_id}"
                    study = optuna.create_study(
                        study_name=study_name,
                        storage=self.storage,
                        direction="maximize",
                        load_if_exists=True
                    )
                    self.domain_studies[domain_id] = study
                    logger.info(f"Created domain-specific Optuna study: {study_name}")
                except Exception as e:
                    logger.warning(f"Failed to create domain study for {domain_id}: {e}")
                    self.domain_studies[domain_id] = None
        
        study = self.domain_studies.get(domain_id)
        if study is None:
            # Fallback to generic optimization
            logger.warning(f"Domain study not available for {domain_id}, using generic optimization")
            return self.optimize_hyperparameters(objective_func)
        
        # Get domain configuration for constraints
        domain_config = self._load_domain_config(domain_id)
        
        # Create domain-aware objective
        def domain_objective(trial):
            # Suggest hyperparameters with domain constraints
            params = self.suggest_hyperparameters_with_domain(
                trial, domain_config, training_data_stats
            )
            return objective_func(trial, params)
        
        # Run optimization
        try:
            study.optimize(domain_objective, n_trials=self.n_trials)
            
            return {
                "domain_id": domain_id,
                "best_hyperparameters": study.best_params,
                "best_score": study.best_value,
                "n_trials": self.n_trials,
                "optimization_complete": True,
            }
        except Exception as e:
            logger.error(f"Domain optimization failed for {domain_id}: {e}", exc_info=True)
            return {
                "domain_id": domain_id,
                "best_hyperparameters": self._get_default_hyperparameters(),
                "best_score": 0.0,
                "optimization_complete": False,
                "error": str(e),
            }
    
    def suggest_hyperparameters_with_domain(
        self,
        trial,
        domain_config: Optional[Dict[str, Any]],
        training_data_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest hyperparameters with domain-specific constraints.
        
        Args:
            trial: Optuna trial object
            domain_config: Optional domain configuration
            training_data_stats: Training data statistics
        
        Returns:
            Dictionary with suggested hyperparameters
        """
        if not HAS_OPTUNA:
            return self._get_default_hyperparameters()
        
        # Base hyperparameters
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
        
        # Apply domain-specific constraints
        if domain_config:
            layer = domain_config.get("layer", "")
            
            # Adjust based on domain layer
            if layer == "data":
                # Data layer: prefer smaller models, faster training
                hyperparameters["hidden_dim"] = trial.suggest_categorical(
                    "hidden_dim", [128, 256, 512]
                )
                hyperparameters["num_layers"] = trial.suggest_int("num_layers", 2, 4)
                hyperparameters["batch_size"] = trial.suggest_categorical(
                    "batch_size", [32, 64, 128]
                )
            elif layer == "application":
                # Application layer: balanced
                hyperparameters["hidden_dim"] = trial.suggest_categorical(
                    "hidden_dim", [256, 512, 768]
                )
            elif layer == "business":
                # Business layer: can use larger models
                hyperparameters["hidden_dim"] = trial.suggest_categorical(
                    "hidden_dim", [512, 768, 1024]
                )
                hyperparameters["num_layers"] = trial.suggest_int("num_layers", 4, 8)
            
            # Adjust based on domain keywords (semantic richness)
            keywords = domain_config.get("keywords", [])
            if len(keywords) > 10:
                # Semantic-rich domain: can benefit from more attention heads
                hyperparameters["num_heads"] = trial.suggest_categorical(
                    "num_heads", [8, 16]
                )
        
        return hyperparameters
    
    def suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial (generic, no domain).
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Dictionary with suggested hyperparameters
        """
        return self.suggest_hyperparameters_with_domain(trial, None, {})
    
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
        task_type: str = "pattern_learning",
        domain_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Automatically select optimal model architecture.
        
        Args:
            training_data_stats: Statistics about training data
            task_type: Type of task (pattern_learning, classification, regression, etc.)
            domain_id: Optional domain identifier for domain-specific architecture
        
        Returns:
            Dictionary with recommended architecture
        """
        # Analyze data characteristics
        data_size = training_data_stats.get("total_samples", 0)
        feature_dim = training_data_stats.get("feature_dim", 0)
        num_classes = training_data_stats.get("num_classes", 0)
        
        # Get domain config if domain_id provided
        domain_config = None
        if domain_id:
            domain_config = self._load_domain_config(domain_id)
        
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
        
        # Phase 9.1: Adjust based on domain configuration
        if domain_config:
            layer = domain_config.get("layer", "")
            
            # Adjust based on domain layer
            if layer == "data":
                # Data layer: prefer smaller, faster models
                architecture["hidden_dim"] = min(architecture["hidden_dim"], 256)
                architecture["num_layers"] = min(architecture["num_layers"], 4)
            elif layer == "business":
                # Business layer: can use larger models
                architecture["hidden_dim"] = max(architecture["hidden_dim"], 512)
                architecture["num_layers"] = max(architecture["num_layers"], 4)
            
            # Adjust based on semantic richness (keywords)
            keywords = domain_config.get("keywords", [])
            if len(keywords) > 10:
                # Semantic-rich: benefit from more attention
                architecture["num_heads"] = max(architecture["num_heads"], 8)
            
            architecture["domain_id"] = domain_id
            architecture["domain_layer"] = layer
        
        # Adjust based on task type
        if task_type == "classification" and num_classes > 100:
            architecture["hidden_dim"] = max(architecture["hidden_dim"], 512)
        
        architecture["task_type"] = task_type
        architecture["data_size"] = data_size
        
        logger.info(
            f"Selected architecture: {architecture['recommendation']} for {task_type}"
            + (f" (domain: {domain_id})" if domain_id else "")
        )
        
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

