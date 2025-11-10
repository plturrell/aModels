"""Model quantization pipeline for Coral NPU."""

import os
import logging
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

from .coralnpu_client import CoralNPUClient

logger = logging.getLogger(__name__)


class CoralNPUQuantizer:
    """Handles model quantization for Coral NPU."""
    
    def __init__(
        self,
        client: Optional[CoralNPUClient] = None,
        enabled: Optional[bool] = None,
    ):
        """
        Initialize Coral NPU quantizer.
        
        Args:
            client: Optional Coral NPU client (auto-created if None)
            enabled: Whether quantization is enabled (from env if None)
        """
        self.client = client or CoralNPUClient()
        self.enabled = enabled if enabled is not None else (
            os.getenv("CORALNPU_QUANTIZE_MODELS", "false").lower() == "true"
        )
    
    def quantize_pytorch_model(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        representative_dataset: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Quantize a PyTorch model for Coral NPU.
        
        This converts PyTorch -> ONNX -> TensorFlow -> TFLite -> Edge TPU.
        
        Args:
            model_path: Path to PyTorch model
            output_path: Optional output path
            representative_dataset: Optional representative dataset
            
        Returns:
            Path to quantized model, or None if quantization failed
        """
        if not self.enabled:
            logger.info("Coral NPU quantization not enabled")
            return None
        
        try:
            # Step 1: Convert PyTorch to ONNX
            onnx_path = self._pytorch_to_onnx(model_path)
            if onnx_path is None:
                return None
            
            # Step 2: Convert ONNX to TensorFlow
            tf_path = self._onnx_to_tensorflow(onnx_path)
            if tf_path is None:
                return None
            
            # Step 3: Quantize TensorFlow to TFLite
            tflite_path = self.client.quantize_model(
                tf_path,
                output_path=output_path,
                representative_dataset=representative_dataset,
            )
            
            # Cleanup intermediate files
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)
            if os.path.exists(tf_path):
                import shutil
                shutil.rmtree(tf_path, ignore_errors=True)
            
            return tflite_path
        
        except Exception as e:
            logger.error(f"PyTorch model quantization failed: {e}")
            return None
    
    def _detect_model_type_and_shape(self, model_path: str) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """Detect model type and input shape from saved model.
        
        Args:
            model_path: Path to PyTorch model
            
        Returns:
            Tuple of (model_type, input_shape_dict, dynamic_axes_dict)
            model_type: "gnn", "transformer", "cnn", or "unknown"
            input_shape_dict: Dictionary with input shapes
            dynamic_axes_dict: Dictionary with dynamic axes for ONNX export
        """
        try:
            import torch
            
            # Load model state
            state = torch.load(model_path, map_location="cpu")
            
            # Detect model type based on saved state keys
            model_type = "unknown"
            input_shape = {}
            dynamic_axes = {}
            
            # Check for GNN model indicators
            if "num_node_features" in state:
                model_type = "gnn"
                num_node_features = state.get("num_node_features", 64)
                # GNN models typically take (num_nodes, num_features) and edge_index
                # Use dynamic batch size for nodes
                input_shape = {
                    "x": (1, num_node_features),  # (batch_size, num_features) - will be dynamic
                    "edge_index": (2, 10)  # (2, num_edges) - will be dynamic
                }
                dynamic_axes = {
                    "x": {0: "num_nodes"},  # Dynamic number of nodes
                    "edge_index": {1: "num_edges"}  # Dynamic number of edges
                }
                logger.info(f"Detected GNN model with {num_node_features} node features")
            
            # Check for Transformer model indicators
            elif "num_heads" in state or "d_model" in state or "n_layers" in state:
                model_type = "transformer"
                d_model = state.get("d_model", 512)
                seq_len = state.get("max_seq_len", 128)
                input_shape = {
                    "input_ids": (1, seq_len),  # (batch_size, seq_len)
                    "attention_mask": (1, seq_len)
                }
                dynamic_axes = {
                    "input_ids": {0: "batch_size", 1: "seq_len"},
                    "attention_mask": {0: "batch_size", 1: "seq_len"}
                }
                logger.info(f"Detected Transformer model with d_model={d_model}, seq_len={seq_len}")
            
            # Check for CNN model indicators
            elif "num_classes" in state and "num_node_features" not in state:
                # Could be CNN or other classifier
                # Try to infer from model structure if available
                model_type = "cnn"
                # Default CNN input shape (can be overridden)
                input_shape = {
                    "input": (1, 3, 224, 224)  # (batch_size, channels, height, width)
                }
                dynamic_axes = {
                    "input": {0: "batch_size"}  # Dynamic batch size
                }
                logger.info("Detected CNN model (using default shape)")
            
            else:
                # Unknown model type - use default
                model_type = "unknown"
                input_shape = {
                    "input": (1, 3, 224, 224)
                }
                dynamic_axes = {
                    "input": {0: "batch_size"}
                }
                logger.warning(f"Unknown model type, using default input shape")
            
            return model_type, input_shape, dynamic_axes
        
        except Exception as e:
            logger.error(f"Failed to detect model type and shape: {e}")
            # Return defaults
            return "unknown", {"input": (1, 3, 224, 224)}, {"input": {0: "batch_size"}}
    
    def _pytorch_to_onnx(self, model_path: str) -> Optional[str]:
        """Convert PyTorch model to ONNX with dynamic shapes.
        
        Args:
            model_path: Path to PyTorch model
            
        Returns:
            Path to ONNX model or None if conversion failed
        """
        try:
            import torch
            
            # Load PyTorch model
            model = torch.load(model_path, map_location="cpu")
            model.eval()
            
            # Detect model type and input shape
            model_type, input_shape, dynamic_axes = self._detect_model_type_and_shape(model_path)
            
            # Create dummy inputs based on detected model type
            dummy_inputs = {}
            
            if model_type == "gnn":
                # GNN models need node features and edge index
                num_node_features = input_shape["x"][1]
                num_nodes = 10  # Example number of nodes
                num_edges = 20  # Example number of edges
                
                dummy_inputs["x"] = torch.randn(num_nodes, num_node_features)
                dummy_inputs["edge_index"] = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
                
                # For ONNX export, we need to pass inputs as tuple or dict
                # PyTorch Geometric models typically take (x, edge_index) as positional args
                dummy_input = (dummy_inputs["x"], dummy_inputs["edge_index"])
                input_names = ["x", "edge_index"]
                output_names = ["output"]
                
            elif model_type == "transformer":
                # Transformer models need input_ids and attention_mask
                seq_len = input_shape["input_ids"][1]
                vocab_size = 10000  # Default vocab size
                
                dummy_inputs["input_ids"] = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
                dummy_inputs["attention_mask"] = torch.ones(1, seq_len, dtype=torch.long)
                
                dummy_input = (dummy_inputs["input_ids"], dummy_inputs["attention_mask"])
                input_names = ["input_ids", "attention_mask"]
                output_names = ["output"]
                
            else:
                # CNN or unknown - use default
                dummy_input = torch.randn(*input_shape["input"])
                input_names = ["input"]
                output_names = ["output"]
            
            onnx_path = str(Path(model_path).with_suffix(".onnx"))
            
            # Export to ONNX with dynamic axes
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes if dynamic_axes else None,
            )
            
            logger.info(f"PyTorch model converted to ONNX with dynamic shapes: {onnx_path} (model_type={model_type})")
            return onnx_path
        
        except Exception as e:
            logger.error(f"PyTorch to ONNX conversion failed: {e}", exc_info=True)
            return None
    
    def _onnx_to_tensorflow(self, onnx_path: str) -> Optional[str]:
        """Convert ONNX model to TensorFlow."""
        try:
            import onnx
            from onnx_tf.backend import prepare
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Convert to TensorFlow
            tf_rep = prepare(onnx_model)
            
            # Save TensorFlow model
            tf_path = str(Path(onnx_path).with_suffix(""))
            tf_rep.export_graph(tf_path)
            
            logger.info(f"ONNX model converted to TensorFlow: {tf_path}")
            return tf_path
        
        except Exception as e:
            logger.error(f"ONNX to TensorFlow conversion failed: {e}")
            return None
    
    def quantize_gnn_model(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        representative_dataset: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Quantize a GNN model for Coral NPU.
        
        Args:
            model_path: Path to GNN model
            output_path: Optional output path
            representative_dataset: Optional representative dataset
            
        Returns:
            Path to quantized model, or None if quantization failed
        """
        return self.quantize_pytorch_model(
            model_path,
            output_path=output_path,
            representative_dataset=representative_dataset,
        )
    
    def quantize_transformer_model(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        representative_dataset: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Quantize a transformer model for Coral NPU.
        
        Args:
            model_path: Path to transformer model
            output_path: Optional output path
            representative_dataset: Optional representative dataset
            
        Returns:
            Path to quantized model, or None if quantization failed
        """
        return self.quantize_pytorch_model(
            model_path,
            output_path=output_path,
            representative_dataset=representative_dataset,
        )


def create_quantizer(
    client: Optional[CoralNPUClient] = None,
    enabled: Optional[bool] = None,
) -> CoralNPUQuantizer:
    """
    Create a Coral NPU quantizer instance.
    
    Args:
        client: Optional Coral NPU client
        enabled: Whether quantization is enabled
        
    Returns:
        CoralNPUQuantizer instance
    """
    return CoralNPUQuantizer(client=client, enabled=enabled)

