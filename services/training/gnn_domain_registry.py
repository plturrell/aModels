"""GNN Domain Model Registry.

This module provides a registry for domain-specific GNN models, enabling
domain-aware routing and model management.
"""

import os
import json
import logging
from typing import Dict, Optional, List, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class DomainModelInfo:
    """Information about a domain-specific GNN model."""
    domain_id: str
    model_type: str  # "embeddings", "classifier", "link_predictor", "anomaly_detector", "schema_matcher"
    model_path: str
    version: str
    created_at: str
    updated_at: str
    training_metrics: Optional[Dict[str, Any]] = None
    model_config: Optional[Dict[str, Any]] = None
    is_active: bool = True
    description: Optional[str] = None


class GNNDomainRegistry:
    """Registry for domain-specific GNN models.
    
    Manages storage, retrieval, and routing of domain-specific GNN models.
    Supports multiple model types per domain (embeddings, classifier, etc.).
    """
    
    def __init__(self, registry_dir: str = "./models/gnn_registry"):
        """Initialize the GNN domain registry.
        
        Args:
            registry_dir: Directory to store registry data
        """
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_dir / "domain_models.json"
        self.models: Dict[str, Dict[str, DomainModelInfo]] = {}  # domain_id -> model_type -> DomainModelInfo
        self.lock = threading.RLock()
        
        # Load existing registry
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk."""
        if not self.registry_file.exists():
            logger.info("No existing registry found, starting fresh")
            return
        
        try:
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
            
            self.models = {}
            for domain_id, domain_models in data.items():
                self.models[domain_id] = {}
                for model_type, model_data in domain_models.items():
                    self.models[domain_id][model_type] = DomainModelInfo(**model_data)
            
            logger.info(f"Loaded registry with {len(self.models)} domains")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self.models = {}
    
    def _save_registry(self):
        """Save registry to disk."""
        try:
            # Convert to JSON-serializable format
            data = {}
            for domain_id, domain_models in self.models.items():
                data[domain_id] = {}
                for model_type, model_info in domain_models.items():
                    data[domain_id][model_type] = asdict(model_info)
            
            # Write atomically
            temp_file = self.registry_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(self.registry_file)
            logger.debug(f"Saved registry with {len(self.models)} domains")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register_model(
        self,
        domain_id: str,
        model_type: str,
        model_path: str,
        version: Optional[str] = None,
        training_metrics: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        is_active: bool = True,
    ) -> DomainModelInfo:
        """Register a domain-specific GNN model.
        
        Args:
            domain_id: Domain identifier (e.g., "finance", "supply_chain")
            model_type: Type of model ("embeddings", "classifier", "link_predictor", etc.)
            model_path: Path to the model file
            version: Model version (auto-generated if not provided)
            training_metrics: Training metrics dictionary
            model_config: Model configuration dictionary
            description: Optional description
            is_active: Whether the model is active
        
        Returns:
            DomainModelInfo for the registered model
        """
        with self.lock:
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            now = datetime.now().isoformat()
            
            # Deactivate previous model of same type if exists
            if domain_id in self.models and model_type in self.models[domain_id]:
                old_model = self.models[domain_id][model_type]
                if old_model.is_active:
                    old_model.is_active = False
                    old_model.updated_at = now
            
            model_info = DomainModelInfo(
                domain_id=domain_id,
                model_type=model_type,
                model_path=model_path,
                version=version,
                created_at=now,
                updated_at=now,
                training_metrics=training_metrics,
                model_config=model_config,
                is_active=is_active,
                description=description,
            )
            
            if domain_id not in self.models:
                self.models[domain_id] = {}
            
            self.models[domain_id][model_type] = model_info
            self._save_registry()
            
            logger.info(f"Registered {model_type} model for domain {domain_id} (version: {version})")
            return model_info
    
    def get_model(
        self,
        domain_id: str,
        model_type: str,
        active_only: bool = True,
    ) -> Optional[DomainModelInfo]:
        """Get a domain-specific model.
        
        Args:
            domain_id: Domain identifier
            model_type: Type of model
            active_only: Only return active models
        
        Returns:
            DomainModelInfo if found, None otherwise
        """
        with self.lock:
            if domain_id not in self.models:
                return None
            
            if model_type not in self.models[domain_id]:
                return None
            
            model_info = self.models[domain_id][model_type]
            if active_only and not model_info.is_active:
                return None
            
            return model_info
    
    def list_domains(self) -> List[str]:
        """List all registered domains.
        
        Returns:
            List of domain IDs
        """
        with self.lock:
            return list(self.models.keys())
    
    def list_models_for_domain(
        self,
        domain_id: str,
        active_only: bool = True,
    ) -> Dict[str, DomainModelInfo]:
        """List all models for a domain.
        
        Args:
            domain_id: Domain identifier
            active_only: Only return active models
        
        Returns:
            Dictionary of model_type -> DomainModelInfo
        """
        with self.lock:
            if domain_id not in self.models:
                return {}
            
            if active_only:
                return {
                    model_type: model_info
                    for model_type, model_info in self.models[domain_id].items()
                    if model_info.is_active
                }
            else:
                return self.models[domain_id].copy()
    
    def deactivate_model(self, domain_id: str, model_type: str):
        """Deactivate a model.
        
        Args:
            domain_id: Domain identifier
            model_type: Type of model
        """
        with self.lock:
            if domain_id in self.models and model_type in self.models[domain_id]:
                self.models[domain_id][model_type].is_active = False
                self.models[domain_id][model_type].updated_at = datetime.now().isoformat()
                self._save_registry()
                logger.info(f"Deactivated {model_type} model for domain {domain_id}")
    
    def remove_model(self, domain_id: str, model_type: str):
        """Remove a model from the registry.
        
        Args:
            domain_id: Domain identifier
            model_type: Type of model
        """
        with self.lock:
            if domain_id in self.models and model_type in self.models[domain_id]:
                del self.models[domain_id][model_type]
                if not self.models[domain_id]:
                    del self.models[domain_id]
                self._save_registry()
                logger.info(f"Removed {model_type} model for domain {domain_id}")
    
    def get_model_info(self, domain_id: str) -> Dict[str, Any]:
        """Get information about all models for a domain.
        
        Args:
            domain_id: Domain identifier
        
        Returns:
            Dictionary with model information
        """
        with self.lock:
            if domain_id not in self.models:
                return {
                    "domain_id": domain_id,
                    "models_available": False,
                    "models": {},
                }
            
            models_info = {}
            for model_type, model_info in self.models[domain_id].items():
                if model_info.is_active:
                    models_info[model_type] = {
                        "version": model_info.version,
                        "model_path": model_info.model_path,
                        "created_at": model_info.created_at,
                        "updated_at": model_info.updated_at,
                        "description": model_info.description,
                        "has_metrics": model_info.training_metrics is not None,
                    }
            
            return {
                "domain_id": domain_id,
                "models_available": len(models_info) > 0,
                "models": models_info,
            }

