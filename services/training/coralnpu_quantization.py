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
    
    def _pytorch_to_onnx(self, model_path: str) -> Optional[str]:
        """Convert PyTorch model to ONNX."""
        try:
            import torch
            
            # Load PyTorch model
            model = torch.load(model_path, map_location="cpu")
            model.eval()
            
            # Create dummy input
            # Note: This is a simplified example - actual implementation would need
            # to determine input shape from model architecture
            dummy_input = torch.randn(1, 3, 224, 224)  # Example shape
            
            onnx_path = str(Path(model_path).with_suffix(".onnx"))
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )
            
            logger.info(f"PyTorch model converted to ONNX: {onnx_path}")
            return onnx_path
        
        except Exception as e:
            logger.error(f"PyTorch to ONNX conversion failed: {e}")
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

