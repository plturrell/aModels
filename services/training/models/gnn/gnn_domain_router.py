"""GNN Domain Router.

This module provides domain-aware routing for GNN queries, automatically
detecting the appropriate domain and routing to domain-specific models.
"""

import os
import logging
from typing import Dict, Optional, List, Any, Tuple
import re

logger = logging.getLogger(__name__)


class GNNDomainRouter:
    """Router for domain-aware GNN model selection.
    
    Detects domains from queries, nodes, or content and routes to
    appropriate domain-specific models.
    """
    
    def __init__(self, registry=None):
        """Initialize the domain router.
        
        Args:
            registry: GNNDomainRegistry instance (optional)
        """
        self.registry = registry
        
        # Domain keywords for detection
        self.domain_keywords = {
            "finance": [
                "finance", "financial", "accounting", "ledger", "gl", "general ledger",
                "revenue", "expense", "asset", "liability", "equity", "balance sheet",
                "income statement", "cash flow", "audit", "compliance", "regulatory",
                "capital", "risk", "credit", "loan", "mortgage", "investment",
            ],
            "supply_chain": [
                "supply chain", "logistics", "inventory", "warehouse", "procurement",
                "purchase", "order", "shipment", "delivery", "vendor", "supplier",
                "manufacturing", "production", "distribution", "fulfillment",
            ],
            "healthcare": [
                "health", "medical", "patient", "diagnosis", "treatment", "hospital",
                "clinic", "pharmacy", "prescription", "medication", "doctor", "nurse",
                "healthcare", "wellness", "clinical", "therapeutic",
            ],
            "retail": [
                "retail", "store", "merchandise", "product", "catalog", "cart",
                "checkout", "payment", "customer", "sales", "inventory", "sku",
                "pricing", "promotion", "discount",
            ],
            "manufacturing": [
                "manufacturing", "production", "factory", "assembly", "quality",
                "defect", "maintenance", "equipment", "machinery", "process",
            ],
            "regulatory": [
                "regulatory", "compliance", "audit", "regulation", "standard",
                "policy", "governance", "risk", "control", "oversight",
            ],
        }
    
    def detect_domain_from_text(self, text: str) -> Optional[str]:
        """Detect domain from text content.
        
        Args:
            text: Text content to analyze
        
        Returns:
            Detected domain ID or None
        """
        if not text:
            return None
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain_id, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            if score > 0:
                domain_scores[domain_id] = score
        
        if not domain_scores:
            return None
        
        # Return domain with highest score
        best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        logger.debug(f"Detected domain '{best_domain}' from text (score: {domain_scores[best_domain]})")
        return best_domain
    
    def detect_domain_from_nodes(
        self,
        nodes: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Detect domain from graph nodes.
        
        Args:
            nodes: List of graph nodes
        
        Returns:
            Detected domain ID or None
        """
        if not nodes:
            return None
        
        # Collect text from node properties
        text_parts = []
        for node in nodes:
            # Add node ID and label
            if "id" in node:
                text_parts.append(str(node["id"]))
            if "label" in node:
                text_parts.append(str(node["label"]))
            if "type" in node:
                text_parts.append(str(node["type"]))
            
            # Add properties
            if "properties" in node and isinstance(node["properties"], dict):
                for key, value in node["properties"].items():
                    if isinstance(value, str):
                        text_parts.append(value)
                    elif isinstance(value, (int, float)):
                        text_parts.append(str(value))
            
            # Check for explicit domain property
            if "properties" in node and isinstance(node["properties"], dict):
                if "domain" in node["properties"]:
                    domain = node["properties"]["domain"]
                    if isinstance(domain, str):
                        logger.debug(f"Found explicit domain '{domain}' in node properties")
                        return domain
        
        # Analyze collected text
        combined_text = " ".join(text_parts)
        return self.detect_domain_from_text(combined_text)
    
    def detect_domain_from_query(self, query: str) -> Optional[str]:
        """Detect domain from query string.
        
        Args:
            query: Query string
        
        Returns:
            Detected domain ID or None
        """
        return self.detect_domain_from_text(query)
    
    def route_to_domain_model(
        self,
        domain_id: Optional[str],
        model_type: str,
        fallback_to_generic: bool = True,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Route to domain-specific model.
        
        Args:
            domain_id: Domain identifier (None for auto-detection)
            model_type: Type of model needed
            fallback_to_generic: Whether to fallback to generic model
        
        Returns:
            Tuple of (domain_id, model_path) or (None, None) if not found
        """
        if not self.registry:
            logger.warning("No registry available, cannot route to domain model")
            return (None, None)
        
        # If domain_id not provided, try to detect
        if domain_id is None:
            logger.debug("No domain_id provided, cannot route to domain model")
            if fallback_to_generic:
                return (None, None)  # Will use generic model
            return (None, None)
        
        # Get domain-specific model
        model_info = self.registry.get_model(domain_id, model_type, active_only=True)
        if model_info:
            logger.info(f"Routed to domain-specific model: {domain_id}/{model_type}")
            return (domain_id, model_info.model_path)
        
        # Fallback to generic if enabled
        if fallback_to_generic:
            logger.debug(f"No domain-specific model for {domain_id}/{model_type}, using generic")
            return (None, None)
        
        logger.warning(f"No model found for {domain_id}/{model_type}")
        return (None, None)
    
    def get_domain_model_info(self, domain_id: str) -> Dict[str, Any]:
        """Get information about domain models.
        
        Args:
            domain_id: Domain identifier
        
        Returns:
            Dictionary with model information
        """
        if not self.registry:
            return {
                "domain_id": domain_id,
                "models_available": False,
                "note": "Registry not available",
            }
        
        return self.registry.get_model_info(domain_id)
    
    def list_available_domains(self) -> List[str]:
        """List all domains with registered models.
        
        Returns:
            List of domain IDs
        """
        if not self.registry:
            return []
        
        return self.registry.list_domains()

