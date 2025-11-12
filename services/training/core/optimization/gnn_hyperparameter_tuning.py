"""Hyperparameter Tuning for GNN Models.

This module provides automated hyperparameter optimization using grid search,
random search, and Bayesian optimization.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    from sklearn.model_selection import ParameterGrid, ParameterSampler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

logger = logging.getLogger(__name__)


class GNNHyperparameterTuner:
    """Hyperparameter tuner for GNN models."""
    
    def __init__(
        self,
        method: str = "grid_search",  # "grid_search", "random_search", "bayesian"
        n_trials: int = 20,
        cv_folds: int = 3
    ):
        """Initialize hyperparameter tuner.
        
        Args:
            method: Tuning method ("grid_search", "random_search", "bayesian")
            n_trials: Number of trials for random/bayesian search
            cv_folds: Number of cross-validation folds
        """
        self.method = method
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        
        if method == "bayesian" and not HAS_OPTUNA:
            logger.warning("Optuna not available, falling back to random search")
            self.method = "random_search"
        
        if method in ["grid_search", "random_search"] and not HAS_SKLEARN:
            logger.warning("scikit-learn not available, hyperparameter tuning disabled")
    
    def _get_default_param_grid(self) -> Dict[str, List[Any]]:
        """Get default parameter grid for GNN models.
        
        Returns:
            Dictionary of parameter names to value lists
        """
        return {
            "hidden_dim": [32, 64, 128],
            "num_layers": [2, 3, 4],
            "dropout": [0.0, 0.1, 0.2, 0.3],
            "lr": [0.001, 0.01, 0.1],
            "use_sage": [True, False],
            "use_gat": [True, False]
        }
    
    def tune_node_classifier(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        labels: Dict[str, str],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        epochs_per_trial: int = 50,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """Tune hyperparameters for node classifier.
        
        Args:
            nodes: Training nodes
            edges: Training edges
            labels: Node labels
            param_grid: Parameter grid (if None, uses default)
            epochs_per_trial: Number of epochs per trial
            device: Device to use
        
        Returns:
            Dictionary with best parameters and results
        """
        from .gnn_node_classifier import GNNNodeClassifier
        from .gnn_evaluation import GNNEvaluator
        
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        evaluator = GNNEvaluator()
        best_score = -1.0
        best_params = None
        best_result = None
        all_results = []
        
        if self.method == "grid_search":
            if not HAS_SKLEARN:
                return {"error": "scikit-learn required for grid search"}
            
            param_combinations = list(ParameterGrid(param_grid))
            logger.info(f"Grid search: {len(param_combinations)} combinations")
            
            for i, params in enumerate(param_combinations):
                logger.info(f"Trial {i+1}/{len(param_combinations)}: {params}")
                
                try:
                    # Train with these parameters
                    classifier = GNNNodeClassifier(
                        device=device,
                        hidden_dim=params.get("hidden_dim", 64),
                        num_layers=params.get("num_layers", 3),
                        dropout=params.get("dropout", 0.1),
                        use_sage=params.get("use_sage", True)
                    )
                    
                    result = classifier.train(
                        nodes, edges,
                        labels=labels,
                        epochs=epochs_per_trial,
                        lr=params.get("lr", 0.01)
                    )
                    
                    if "error" not in result:
                        score = result.get("accuracy", 0.0)
                        all_results.append({
                            "params": params,
                            "score": score,
                            "result": result
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_result = result
                
                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    continue
        
        elif self.method == "random_search":
            if not HAS_SKLEARN:
                return {"error": "scikit-learn required for random search"}
            
            param_combinations = list(ParameterSampler(param_grid, n_iter=self.n_trials, random_state=42))
            logger.info(f"Random search: {self.n_trials} trials")
            
            for i, params in enumerate(param_combinations):
                logger.info(f"Trial {i+1}/{self.n_trials}: {params}")
                
                try:
                    classifier = GNNNodeClassifier(
                        device=device,
                        hidden_dim=params.get("hidden_dim", 64),
                        num_layers=params.get("num_layers", 3),
                        dropout=params.get("dropout", 0.1),
                        use_sage=params.get("use_sage", True)
                    )
                    
                    result = classifier.train(
                        nodes, edges,
                        labels=labels,
                        epochs=epochs_per_trial,
                        lr=params.get("lr", 0.01)
                    )
                    
                    if "error" not in result:
                        score = result.get("accuracy", 0.0)
                        all_results.append({
                            "params": params,
                            "score": score,
                            "result": result
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_result = result
                
                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    continue
        
        elif self.method == "bayesian":
            if not HAS_OPTUNA:
                return {"error": "Optuna required for Bayesian optimization"}
            
            def objective(trial):
                # Suggest parameters
                params = {
                    "hidden_dim": trial.suggest_categorical("hidden_dim", param_grid.get("hidden_dim", [64])),
                    "num_layers": trial.suggest_int("num_layers", 2, 4),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.3),
                    "lr": trial.suggest_loguniform("lr", 0.001, 0.1),
                    "use_sage": trial.suggest_categorical("use_sage", [True, False])
                }
                
                try:
                    classifier = GNNNodeClassifier(
                        device=device,
                        hidden_dim=params["hidden_dim"],
                        num_layers=params["num_layers"],
                        dropout=params["dropout"],
                        use_sage=params["use_sage"]
                    )
                    
                    result = classifier.train(
                        nodes, edges,
                        labels=labels,
                        epochs=epochs_per_trial,
                        lr=params["lr"]
                    )
                    
                    if "error" not in result:
                        return result.get("accuracy", 0.0)
                    else:
                        return 0.0
                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    return 0.0
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials)
            
            best_params = study.best_params
            best_score = study.best_value
            
            # Train final model with best parameters
            classifier = GNNNodeClassifier(
                device=device,
                hidden_dim=best_params["hidden_dim"],
                num_layers=best_params["num_layers"],
                dropout=best_params["dropout"],
                use_sage=best_params["use_sage"]
            )
            best_result = classifier.train(
                nodes, edges,
                labels=labels,
                epochs=epochs_per_trial * 2,  # Train longer with best params
                lr=best_params["lr"]
            )
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_result": best_result,
            "all_results": all_results,
            "method": self.method,
            "num_trials": len(all_results) if all_results else self.n_trials
        }
    
    def tune_link_predictor(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        param_grid: Optional[Dict[str, List[Any]]] = None,
        epochs_per_trial: int = 50,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """Tune hyperparameters for link predictor.
        
        Args:
            nodes: Training nodes
            edges: Training edges
            param_grid: Parameter grid (if None, uses default)
            epochs_per_trial: Number of epochs per trial
            device: Device to use
        
        Returns:
            Dictionary with best parameters and results
        """
        from .gnn_link_predictor import GNNLinkPredictor
        from .gnn_evaluation import GNNEvaluator
        
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        evaluator = GNNEvaluator()
        best_score = -1.0
        best_params = None
        best_result = None
        all_results = []
        
        if self.method == "grid_search":
            if not HAS_SKLEARN:
                return {"error": "scikit-learn required for grid search"}
            
            param_combinations = list(ParameterGrid(param_grid))
            logger.info(f"Grid search: {len(param_combinations)} combinations")
            
            for i, params in enumerate(param_combinations):
                logger.info(f"Trial {i+1}/{len(param_combinations)}: {params}")
                
                try:
                    predictor = GNNLinkPredictor(
                        device=device,
                        hidden_dim=params.get("hidden_dim", 64),
                        num_layers=params.get("num_layers", 3),
                        dropout=params.get("dropout", 0.1),
                        use_gat=params.get("use_gat", True)
                    )
                    
                    result = predictor.train(
                        nodes, edges,
                        epochs=epochs_per_trial,
                        lr=params.get("lr", 0.01)
                    )
                    
                    if "error" not in result:
                        score = result.get("accuracy", 0.0)
                        all_results.append({
                            "params": params,
                            "score": score,
                            "result": result
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_result = result
                
                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    continue
        
        elif self.method == "random_search":
            if not HAS_SKLEARN:
                return {"error": "scikit-learn required for random search"}
            
            param_combinations = list(ParameterSampler(param_grid, n_iter=self.n_trials, random_state=42))
            logger.info(f"Random search: {self.n_trials} trials")
            
            for i, params in enumerate(param_combinations):
                logger.info(f"Trial {i+1}/{self.n_trials}: {params}")
                
                try:
                    predictor = GNNLinkPredictor(
                        device=device,
                        hidden_dim=params.get("hidden_dim", 64),
                        num_layers=params.get("num_layers", 3),
                        dropout=params.get("dropout", 0.1),
                        use_gat=params.get("use_gat", True)
                    )
                    
                    result = predictor.train(
                        nodes, edges,
                        epochs=epochs_per_trial,
                        lr=params.get("lr", 0.01)
                    )
                    
                    if "error" not in result:
                        score = result.get("accuracy", 0.0)
                        all_results.append({
                            "params": params,
                            "score": score,
                            "result": result
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_result = result
                
                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    continue
        
        elif self.method == "bayesian":
            if not HAS_OPTUNA:
                return {"error": "Optuna required for Bayesian optimization"}
            
            def objective(trial):
                params = {
                    "hidden_dim": trial.suggest_categorical("hidden_dim", param_grid.get("hidden_dim", [64])),
                    "num_layers": trial.suggest_int("num_layers", 2, 4),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.3),
                    "lr": trial.suggest_loguniform("lr", 0.001, 0.1),
                    "use_gat": trial.suggest_categorical("use_gat", [True, False])
                }
                
                try:
                    predictor = GNNLinkPredictor(
                        device=device,
                        hidden_dim=params["hidden_dim"],
                        num_layers=params["num_layers"],
                        dropout=params["dropout"],
                        use_gat=params["use_gat"]
                    )
                    
                    result = predictor.train(
                        nodes, edges,
                        epochs=epochs_per_trial,
                        lr=params["lr"]
                    )
                    
                    if "error" not in result:
                        return result.get("accuracy", 0.0)
                    else:
                        return 0.0
                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    return 0.0
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.n_trials)
            
            best_params = study.best_params
            best_score = study.best_value
            
            # Train final model with best parameters
            predictor = GNNLinkPredictor(
                device=device,
                hidden_dim=best_params["hidden_dim"],
                num_layers=best_params["num_layers"],
                dropout=best_params["dropout"],
                use_gat=best_params["use_gat"]
            )
            best_result = predictor.train(
                nodes, edges,
                epochs=epochs_per_trial * 2,
                lr=best_params["lr"]
            )
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_result": best_result,
            "all_results": all_results,
            "method": self.method,
            "num_trials": len(all_results) if all_results else self.n_trials
        }

